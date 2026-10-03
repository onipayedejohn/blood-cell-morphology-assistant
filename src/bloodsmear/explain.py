"""Class activation maps and the heatmap overlay.

The network ends in global average pooling and one dense layer, so the score
for class c is a weighted sum of the last feature maps. Applying the same
weights to the maps, before pooling, shows where that evidence sits in the
image (Zhou et al., CVPR 2016).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# A single hue, so brighter always means "more influence". Cyan stands out
# against both the violet nuclei and the pink red cells of a stained smear.
HEAT_RGB = np.array([0, 214, 230], dtype=np.float32)


def class_activation_map(features: np.ndarray, weights: np.ndarray, cls: int) -> np.ndarray:
    """features (h, w, k), weights (k, n_classes) -> map in [0, 1] of shape (h, w)."""
    cam = features @ weights[:, cls]
    cam = np.maximum(cam, 0)
    peak = cam.max()
    return cam / peak if peak > 0 else cam


def overlay(image_uint8: np.ndarray, cam: np.ndarray, strength: float = 0.65) -> np.ndarray:
    """Blend the map onto the image. Weak evidence is left transparent."""
    h, w = image_uint8.shape[:2]
    m = Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BICUBIC)
    m = np.asarray(m, dtype=np.float32) / 255.0
    m = np.clip((m - 0.2) / 0.8, 0, 1) ** 1.2  # hide the weakest 20%
    alpha = (strength * m)[..., None]
    base = image_uint8.astype(np.float32) * 0.85  # dim slightly so the map reads
    out = base * (1 - alpha) + HEAT_RGB * alpha
    return np.clip(out, 0, 255).astype(np.uint8)
