"""The one preprocessing function used at training time and in the app.

Keeping this in a single place is what stops the app from feeding the model
images that look different from the ones it learned on.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

INPUT_SIZE = 64


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


def trim_dark_border(img: Image.Image, dark: int = 45, min_border: float = 0.12) -> tuple[Image.Image, bool]:
    """Crop away a dark surround, such as the black ring of a phone photo through an eyepiece.

    Only acts when at least `min_border` of the image is dark and the lit area is
    a clear inner region. Returns the (possibly) cropped image and whether it was cropped.
    """
    rgb = to_rgb(img)
    a = np.asarray(rgb.resize((256, max(1, int(256 * rgb.height / rgb.width)))), dtype=np.float32).mean(axis=2)
    lit = a > dark
    if lit.mean() > 1 - min_border or lit.mean() < 0.15:
        return rgb, False
    rows, cols = np.nonzero(lit.mean(axis=1) > 0.2)[0], np.nonzero(lit.mean(axis=0) > 0.2)[0]
    if len(rows) < 8 or len(cols) < 8:
        return rgb, False
    sy, sx = rgb.height / a.shape[0], rgb.width / a.shape[1]
    y0, y1, x0, x1 = rows[0] * sy, (rows[-1] + 1) * sy, cols[0] * sx, (cols[-1] + 1) * sx
    # keep the central part of a round field so the corners are not black
    cy, cx, r = (y0 + y1) / 2, (x0 + x1) / 2, min(y1 - y0, x1 - x0) / 2 * 0.72
    box = (int(max(0, cx - r)), int(max(0, cy - r)), int(min(rgb.width, cx + r)), int(min(rgb.height, cy + r)))
    if (box[2] - box[0]) < 64 or (box[3] - box[1]) < 64:
        return rgb, False
    return rgb.crop(box), True


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
