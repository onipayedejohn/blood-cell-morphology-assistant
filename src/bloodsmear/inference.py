"""Load the exported model and turn an image into a result the app can show.

The app only needs onnxruntime, numpy and Pillow. TensorFlow is used for
training but is not a dependency of the deployed app.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from .config import CLASSES, MODEL_DIR
from .explain import class_activation_map
from .preprocess import resize_uint8, to_model_input
from .quality import Issue, check_image


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class Prediction:
    label: str
    confidence: float
    probs: dict[str, float]
    top: list[tuple[str, float]]
    cam: np.ndarray
    image_uint8: np.ndarray
    unfamiliarity: float  # distance percentile relative to validation images, 0-100
    unfamiliar: bool
    low_confidence: bool
    issues: list[Issue] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.unfamiliar:
            return "unfamiliar"
        if self.low_confidence:
            return "review"
        return "confident"


class CellClassifier:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.meta = json.loads((model_dir / "model_meta.json").read_text())
        aux = np.load(model_dir / "model_aux.npz")
        self.cam_weights = aux["cam_weights"]
        self.ood_means = aux["ood_means"]
        self.ood_precision = aux["ood_precision"]
        self.val_distances = np.sort(aux["val_distances"])
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_dir / "bloodcell_cnn.onnx"), opts,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.size = int(self.meta["input_size"])
        self.temperature = float(self.meta["temperature"])
        self.conf_threshold = float(self.meta["confidence_threshold"])
        self.ood_threshold = float(self.meta["ood_threshold"])

    # -- core -----------------------------------------------------------------
    def _run(self, x: np.ndarray):
        logits, feats = self.session.run(None, {self.input_name: x})
        return logits, feats

    def distance(self, pooled: np.ndarray) -> np.ndarray:
        """Smallest class-conditional Mahalanobis distance of pooled features."""
        diff = pooled[:, None, :] - self.ood_means[None, :, :]
        d2 = np.einsum("nck,kl,ncl->nc", diff, self.ood_precision, diff)
        return np.sqrt(np.maximum(d2.min(axis=1), 0))

    def percentile(self, d: float) -> float:
        return float(np.searchsorted(self.val_distances, d) / len(self.val_distances) * 100)

    def predict_arrays(self, arrs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """uint8 arrays -> calibrated probabilities, feature maps, distances."""
        x = to_model_input(np.stack(arrs))
        logits, feats = self._run(x)
        probs = softmax(logits / self.temperature)
        dist = self.distance(feats.mean(axis=(1, 2)))
        return probs, feats, dist

    def predict(self, img: Image.Image) -> Prediction:
        issues = check_image(img)
        arr = resize_uint8(img, self.size)
        probs, feats, dist = self.predict_arrays([arr])
        return self._package(arr, probs[0], feats[0], float(dist[0]), issues)

    def predict_many(self, images: list[Image.Image], batch: int = 32) -> list[Prediction]:
        out: list[Prediction] = []
        for i in range(0, len(images), batch):
            chunk = images[i:i + batch]
            issues = [check_image(im) for im in chunk]
            arrs = [resize_uint8(im, self.size) for im in chunk]
            probs, feats, dist = self.predict_arrays(arrs)
            out += [self._package(a, p, f, float(d), iss)
                    for a, p, f, d, iss in zip(arrs, probs, feats, dist, issues)]
        return out

    def _package(self, arr, p, feats, dist, issues) -> Prediction:
        order = np.argsort(p)[::-1]
        cls = int(order[0])
        cam = class_activation_map(feats, self.cam_weights, cls)
        return Prediction(
            label=CLASSES[cls],
            confidence=float(p[cls]),
            probs={c: float(p[i]) for i, c in enumerate(CLASSES)},
            top=[(CLASSES[i], float(p[i])) for i in order[:3]],
            cam=cam,
            image_uint8=arr,
            unfamiliarity=self.percentile(dist),
            unfamiliar=dist > self.ood_threshold,
            low_confidence=float(p[cls]) < self.conf_threshold,
            issues=issues,
        )
