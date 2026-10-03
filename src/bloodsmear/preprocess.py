"""The one preprocessing function used at training time and in the app.

Keeping this in a single place is what stops the app from feeding the model
images that look different from the ones it learned on.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

INPUT_SIZE = 112


def center_square(img: Image.Image) -> Image.Image:
    """Crop the largest centered square. PBC images are 360x363, so this trims 3 rows."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def to_rgb(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        rgba = img.convert("RGBA")
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return img.convert("RGB")


def resize_uint8(img: Image.Image, size: int = INPUT_SIZE) -> np.ndarray:
    """RGB image -> centered square -> size x size uint8 array."""
    img = center_square(to_rgb(img))
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def to_model_input(arr_uint8: np.ndarray) -> np.ndarray:
    """uint8 (H, W, 3) or (N, H, W, 3) -> float32 in [0, 1] with a batch axis."""
    x = arr_uint8.astype(np.float32) / 255.0
    if x.ndim == 3:
        x = x[None, ...]
    return x
