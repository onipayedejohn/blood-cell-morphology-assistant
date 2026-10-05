"""Load the exported models and turn an image into a result the app can show.

Version 2 adds a gatekeeper in front of the classifier. The cell type is only
given when the gatekeeper accepts the image as a stained white blood cell.
Whole-field photos are split into one crop per white cell first. The
classifier sees each cell in a standard framing and color balance
(see canonical.py); the gatekeeper sees the image as it was uploaded.

The app only needs onnxruntime, numpy, scipy and Pillow.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from .canonical import canonical_uint8
from .config import CLASSES, MODEL_DIR
from .detect import Detection, find_cells
from .explain import class_activation_map
from .preprocess import resize_uint8, to_model_input, to_rgb, trim_dark_border
from .quality import Issue, check_image

GATE_LABELS = ["white_cell", "smear_no_wbc", "not_smear"]
FIELD_MARGIN = 3.0  # crop side around a found nucleus, as a multiple of the nucleus size


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class Prediction:
    label: str | None             # None when the gatekeeper refused the image
    confidence: float
    probs: dict[str, float]
    top: list[tuple[str, float]]
    cam: np.ndarray
    image_uint8: np.ndarray
    unfamiliarity: float          # distance percentile relative to validation images, 0-100
    unfamiliar: bool
    low_confidence: bool
    gate: dict[str, float] = field(default_factory=dict)
    gate_ok: bool = True
    issues: list[Issue] = field(default_factory=list)

    @property
    def gate_verdict(self) -> str:
        return max(self.gate, key=self.gate.get) if self.gate else "white_cell"

    @property
    def status(self) -> str:
        if not self.gate_ok:
            return "no_white_cell" if self.gate_verdict == "smear_no_wbc" else "not_blood"
        if self.unfamiliar:
            return "unfamiliar"
        if self.low_confidence:
            return "review"
        return "confident"


@dataclass
class FieldResult:
    image: Image.Image
    detections: list[Detection]
    cells: list[Prediction]


@dataclass
class Analysis:
    mode: str                     # "single", "field" or "refused"
    single: Prediction | None = None
    field_result: FieldResult | None = None
    whole_image_gate: dict[str, float] = field(default_factory=dict)
    trimmed: bool = False


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
        self.session = ort.InferenceSession(str(model_dir / "bloodcell_cnn.onnx"), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        gate_path = model_dir / "gate.onnx"
        self.gate = (ort.InferenceSession(str(gate_path), opts, providers=["CPUExecutionProvider"])
                     if gate_path.exists() else None)
        self.size = int(self.meta["input_size"])
        self.temperature = float(self.meta["temperature"])
        self.conf_threshold = float(self.meta["confidence_threshold"])
        self.ood_threshold = float(self.meta["ood_threshold"])
        self.gate_threshold = float(self.meta.get("gate", {}).get("threshold_p_white", 0.5))
        self.framing = self.meta.get("framing", "center")  # "canonical" from version 2
        self.gate_size = int(self.meta.get("gate", {}).get("input_size", 64))

    def model_array(self, img: Image.Image) -> np.ndarray:
        """The exact array the classifier sees for this image."""
        return canonical_uint8(img, self.size) if self.framing == "canonical" else resize_uint8(img, self.size)

    # -- core ------------------------------------------------------------------------
    def distance(self, pooled: np.ndarray) -> np.ndarray:
        """Smallest class-conditional Mahalanobis distance of pooled features."""
        diff = pooled[:, None, :] - self.ood_means[None, :, :]
        d2 = np.einsum("nck,kl,ncl->nc", diff, self.ood_precision, diff)
        return np.sqrt(np.maximum(d2.min(axis=1), 0))

    def percentile(self, d: float) -> float:
        return float(np.searchsorted(self.val_distances, d) / len(self.val_distances) * 100)

    def gate_probs(self, arrs: list[np.ndarray]) -> np.ndarray:
        if self.gate is None:
            return np.tile([1.0, 0.0, 0.0], (len(arrs), 1))
        x = to_model_input(np.stack(arrs))
        return softmax(self.gate.run(None, {self.gate.get_inputs()[0].name: x})[0])

    def predict_arrays(self, arrs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """uint8 arrays -> calibrated probabilities, feature maps, distances."""
        x = to_model_input(np.stack(arrs))
        logits, feats = self.session.run(None, {self.input_name: x})
        probs = softmax(logits / self.temperature)
        dist = self.distance(feats.mean(axis=(1, 2)))
        return probs, feats, dist

    def predict(self, img: Image.Image, force: bool = False) -> Prediction:
        """Classify one cropped cell. With force=True the gatekeeper's refusal is overridden."""
        return self.predict_many([img], force=force)[0]

    def predict_many(self, images: list[Image.Image], batch: int = 32, force: bool = False) -> list[Prediction]:
        out: list[Prediction] = []
        for i in range(0, len(images), batch):
            chunk = [to_rgb(im) for im in images[i:i + batch]]
            issues = [check_image(im) for im in chunk]
            g = self.gate_probs([resize_uint8(im, self.gate_size) for im in chunk])
            arrs = [self.model_array(im) for im in chunk]
            probs, feats, dist = self.predict_arrays(arrs)
            for a, p, f, d, iss, gp in zip(arrs, probs, feats, dist, issues, g):
                ok = force or gp[0] >= self.gate_threshold
                out.append(self._package(a, p, f, float(d), iss, gp, ok))
        return out

    def _package(self, arr, p, feats, dist, issues, gp, gate_ok) -> Prediction:
        order = np.argsort(p)[::-1]
        cls = int(order[0])
        return Prediction(
            label=CLASSES[cls] if gate_ok else None,
            confidence=float(p[cls]),
            probs={c: float(p[i]) for i, c in enumerate(CLASSES)},
            top=[(CLASSES[i], float(p[i])) for i in order[:3]],
            cam=class_activation_map(feats, self.cam_weights, cls),
            image_uint8=arr,
            unfamiliarity=self.percentile(dist),
            unfamiliar=dist > self.ood_threshold,
            low_confidence=float(p[cls]) < self.conf_threshold,
            gate={k: float(v) for k, v in zip(GATE_LABELS, gp)},
            gate_ok=bool(gate_ok),
            issues=issues,
        )

    # -- whole images --------------------------------------------------------------------
    def analyze(self, img: Image.Image, mode: str = "auto", force: bool = False) -> Analysis:
        """mode: "auto", "single" (one cropped cell) or "field" (find every white cell)."""
        img = to_rgb(img)
        trimmed_img, trimmed = trim_dark_border(img)
        g = self.gate_probs([resize_uint8(trimmed_img, self.gate_size)])[0]
        whole = {k: float(v) for k, v in zip(GATE_LABELS, g)}
        not_blood = whole["not_smear"] >= 0.5 and whole["white_cell"] < self.gate_threshold
        if not_blood and not force:
            return Analysis("refused", whole_image_gate=whole, trimmed=trimmed)

        # crops wide enough that the standard framing rarely needs padding
        dets = find_cells(trimmed_img, margin=FIELD_MARGIN) if mode in ("auto", "field") else []
        looks_like_field = len(dets) >= 2 or (len(dets) == 1 and dets[0].area_fraction < 0.04)
        if mode == "field" or (mode == "auto" and looks_like_field and dets):
            crops = [trimmed_img.crop(d.box) for d in dets]
            cells = self.predict_many(crops, force=force) if crops else []
            return Analysis("field", field_result=FieldResult(trimmed_img, dets, cells), whole_image_gate=whole,
                            trimmed=trimmed)
        single = self.predict(trimmed_img, force=force)
        if mode == "auto" and single.status == "no_white_cell" and dets:
            # a smear with no white cell in the middle: look at the cells the finder picked up instead
            crops = [trimmed_img.crop(d.box) for d in dets]
            return Analysis("field", field_result=FieldResult(trimmed_img, dets, self.predict_many(crops, force=force)),
                            whole_image_gate=whole, trimmed=trimmed)
        return Analysis("single", single=single, whole_image_gate=whole, trimmed=trimmed)
