"""Cheap checks on an uploaded image before the model sees it.

These catch problems a person can fix (a tiny crop, a black frame, a
grayscale scan) and explain them in plain words.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

MIN_SIDE = 64


@dataclass
class Issue:
    level: str  # "block" stops the prediction, "warn" is shown next to it
    message: str


def check_image(img: Image.Image) -> list[Issue]:
    issues: list[Issue] = []
    w, h = img.size
    if min(w, h) < MIN_SIDE:
        issues.append(Issue("block", f"The image is {w}x{h} pixels. Use a crop at least {MIN_SIDE} pixels on each side."))
        return issues
    ratio = max(w, h) / min(w, h)
    if ratio > 1.6:
        issues.append(Issue("warn", "The image is far from square. The model looks at the center square only, "
                                    "so crop around one cell for a fair result."))
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    mean, std = arr.mean(), arr.std()
    if mean < 40:
        issues.append(Issue("warn", "The image is very dark. Increase the microscope light or exposure."))
    elif mean > 245:
        issues.append(Issue("warn", "The image is almost white. It may be over-exposed or show an empty field."))
    if std < 12:
        issues.append(Issue("warn", "The image has very little contrast, so cell detail may be lost."))
    chroma = np.abs(arr[..., 0] - arr[..., 1]).mean() + np.abs(arr[..., 1] - arr[..., 2]).mean()
    if chroma < 4:
        issues.append(Issue("warn", "The image looks grayscale. The model relies on stain color, "
                                    "for example eosinophil granules, so use a color image."))
    return issues
