"""Put every white cell image into the same framing and color balance before the model sees it.

Version 2's first leave-one-lab-out run reached only 50% accuracy on a lab it
had never seen. Looking at the images showed two differences that color
augmentation alone does not remove:

- **Scale.** In PBC images the cell fills about a third of the frame. BCCD and
  JTSC crops are tight, with the cell filling the frame, and a phone photo can
  be at any magnification.
- **Background color.** The background is pinkish in PBC, yellow in JTSC and
  gray-blue in BCCD, depending on lamp, camera and stain.

`canonical_cell` fixes both the same way for every source, at training time and
in the app:

1. estimate the background color (the brightest pixels) and balance it out,
   then find the nucleus (the darkest, bluest object near the center),
2. crop a square `FRAME` times the nucleus size around it, padding with the
   background color where the crop runs past the image edge,
3. divide each channel by the background color, so the background becomes
   the same neutral gray everywhere (the same balance as in step 1).

Stain strength is left alone. Scaling every nucleus to the same optical
density was tried and lowered accuracy on an unseen lab (see docs/decisions.md).

If no nucleus is found, the image is only center-cropped and color-balanced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .detect import nucleus_score, otsu
from .preprocess import INPUT_SIZE, center_square, to_rgb

FRAME = 2.4          # crop side as a multiple of the nucleus extent
BACKGROUND = 0.92    # neutral gray the background is mapped to
WORK = 192           # segment at this size for speed


@dataclass
class Framing:
    found: bool
    nucleus_fraction: float  # nucleus extent / original image side
    padded: bool


def background_color(rgb: np.ndarray) -> np.ndarray:
    """Median color of the brightest 15% of pixels that are not nucleus-like (float, 0-1)."""
    f = rgb.reshape(-1, 3).astype(np.float32) / 255.0
    bright = f.mean(axis=1)
    cut = np.quantile(bright, 0.85)
    bg = np.median(f[bright >= cut], axis=0)
    return np.clip(bg, 0.25, 1.0)


def balance(arr: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Divide out the background color so the background becomes neutral gray."""
    f = arr.astype(np.float32) / 255.0 * (BACKGROUND / bg)
    return (np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8)


def locate_nucleus(rgb_small: np.ndarray) -> tuple[float, float, float] | None:
    """Return (cx, cy, extent) of the central nucleus in pixel units of `rgb_small`, or None."""
    h, w = rgb_small.shape[:2]
    s = ndi.gaussian_filter(nucleus_score(rgb_small), 1.0)
    t = max(otsu(s), float(np.median(s)) + 0.6 * float(s.std()))
    mask = ndi.binary_opening(s > t, iterations=1)
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)), iterations=4)
    mask = ndi.binary_fill_holes(mask)
    lab, n = ndi.label(mask)
    if n == 0:
        return None
    idx = np.arange(1, n + 1)
    areas = ndi.sum(mask, lab, index=idx)
    com = ndi.center_of_mass(mask, lab, idx)
    f = rgb_small.astype(np.float32) / 255.0
    blue = ndi.mean(f[..., 2] - f[..., 0], lab, index=idx)
    side = max(h, w)
    # prefer large, blue objects near the center (single-cell images are roughly centerd)
    d2 = np.array([((cy - h / 2) ** 2 + (cx - w / 2) ** 2) for cy, cx in com]) / (0.35 * side) ** 2
    score = areas * np.exp(-0.5 * d2) * (blue > 0)
    if score.max() <= 0 or areas[score.argmax()] < 0.004 * h * w:
        return None
    best = int(score.argmax()) + 1
    # merge nearby pieces of the same nucleus (separate lobes of a neutrophil)
    ys, xs = np.nonzero(lab == best)
    ext = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
    cy0, cx0 = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    keep = [best]
    for i, (cy, cx) in enumerate(com, start=1):
        if i != best and blue[i - 1] > 0 and areas[i - 1] >= 0.08 * areas[best - 1] \
                and np.hypot(cy - cy0, cx - cx0) < 0.9 * ext:
            keep.append(i)
    ys, xs = np.nonzero(np.isin(lab, keep))
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return (x0 + x1) / 2, (y0 + y1) / 2, float(max(y1 - y0, x1 - x0))


def canonical_cell(img: Image.Image, size: int = INPUT_SIZE) -> tuple[np.ndarray, Framing]:
    """Any single-cell image -> (size x size uint8 array in the standard framing, info)."""
    rgb = to_rgb(img)
    W, H = rgb.size
    k = WORK / max(W, H)
    small = np.asarray(rgb.resize((max(1, round(W * k)), max(1, round(H * k))), Image.Resampling.BILINEAR))
    bg = background_color(small)
    loc = locate_nucleus(balance(small, bg))  # balanced first, so odd stain casts do not hide the nucleus
    if loc is None:
        out = center_square(rgb).resize((size, size), Image.Resampling.LANCZOS)
        return balance(np.asarray(out), bg), Framing(False, 0.0, False)
    cx, cy, ext = (v / k for v in loc)
    side = max(ext * FRAME, 16.0)
    x0, y0 = cx - side / 2, cy - side / 2
    box = (int(round(x0)), int(round(y0)), int(round(x0 + side)), int(round(y0 + side)))
    padded = box[0] < 0 or box[1] < 0 or box[2] > W or box[3] > H
    if padded:
        fill = tuple(int(c * 255) for c in bg)
        pw = max(0, -box[0], box[2] - W)
        ph = max(0, -box[1], box[3] - H)
        canvas = Image.new("RGB", (W + 2 * pw, H + 2 * ph), fill)
        canvas.paste(rgb, (pw, ph))
        crop = canvas.crop((box[0] + pw, box[1] + ph, box[2] + pw, box[3] + ph))
    else:
        crop = rgb.crop(box)
    out = crop.resize((size, size), Image.Resampling.LANCZOS)
    return balance(np.asarray(out), bg), Framing(True, float(ext / max(W, H)), padded)


def canonical_uint8(img: Image.Image, size: int = INPUT_SIZE) -> np.ndarray:
    return canonical_cell(img, size)[0]
