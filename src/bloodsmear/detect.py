"""Find white cells in a whole-field smear photo.

White cell nuclei are the darkest and bluest objects on a Romanowsky-stained
smear: red cells are pink and paler, and the background is pale. The finder

1. scores every pixel for "nucleus-ness" (darkness plus blue over green),
2. thresholds the score with Otsu's method,
3. closes small gaps, and joins nearby pieces, so the lobes of one neutrophil
   count as one object,
4. keeps objects that are large compared with the biggest one and with the
   image (this drops platelets and stain specks), and
5. returns a square crop around each object, sized to include the cytoplasm.

It is a classical image-processing step with no learned weights, so it is fast
and transparent, but it can merge touching cells or miss pale nuclei.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .preprocess import to_rgb

WORK_SIDE = 512  # analyze at this size for speed


@dataclass
class Detection:
    box: tuple[int, int, int, int]  # crop square in original image pixels (x0, y0, x1, y1)
    nucleus_box: tuple[int, int, int, int]
    area_fraction: float

    def display_box(self, factor: float = 1.7) -> tuple[int, int, int, int]:
        """A box about the size of the whole cell, for drawing (the crop box is wider)."""
        x0, y0, x1, y1 = self.nucleus_box
        cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * factor / 2
        return int(cx - half), int(cy - half), int(cx + half), int(cy + half)


def nucleus_score(rgb: np.ndarray) -> np.ndarray:
    """High for dark blue-violet pixels (nuclei), low for pink red cells and pale background."""
    f = rgb.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    dark = 1.0 - (r + g + b) / 3.0
    return 0.5 * dark + (b - r) + 0.5 * (b - g)


def otsu(values: np.ndarray, bins: int = 128) -> float:
    hist, edges = np.histogram(values, bins=bins)
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    mt = (hist * centers).sum()
    m1 = (mt - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[np.argmax(between)])


def find_cells(img: Image.Image, min_rel_area: float = 0.25, min_img_fraction: float = 0.002,
               margin: float = 1.9, max_cells: int = 30, min_blue: float = 0.0) -> list[Detection]:
    img = to_rgb(img)
    W, H = img.size
    scale = WORK_SIDE / max(W, H)
    small = img.resize((max(1, int(W * scale)), max(1, int(H * scale))), Image.Resampling.BILINEAR)
    s = nucleus_score(np.asarray(small))
    s = ndi.gaussian_filter(s, 1.0)
    t = otsu(s)
    # Otsu on a field that is mostly background can sit too low; require a clear step above the median
    t = max(t, float(np.median(s)) + 0.6 * float(s.std()))
    mask = s > t
    mask = ndi.binary_opening(mask, iterations=1)
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)), iterations=3)
    mask = ndi.binary_fill_holes(mask)
    lab, n = ndi.label(mask)
    if n == 0:
        return []
    idx = np.arange(1, n + 1)
    areas = ndi.sum(mask, lab, index=idx)
    f = np.asarray(small).astype(np.float32) / 255.0
    b_minus_r = ndi.mean(f[..., 2] - f[..., 0], lab, index=idx)  # nuclei are bluer than red
    keep = [i + 1 for i, (a, br) in enumerate(zip(areas, b_minus_r))
            if a >= min_rel_area * areas.max() and a >= min_img_fraction * mask.size and br > min_blue]
    boxes = []
    for sl_idx in keep:
        sl = ndi.find_objects((lab == sl_idx).astype(int))[0]
        boxes.append([sl[1].start, sl[0].start, sl[1].stop, sl[0].stop, float(areas[sl_idx - 1])])
    boxes = merge_nearby(boxes)
    out: list[Detection] = []
    for x0, y0, x1, y1, area in boxes:
        cx, cy = (x0 + x1) / 2 / scale, (y0 + y1) / 2 / scale
        side = max(x1 - x0, y1 - y0) / scale * margin
        side = max(side, 0.06 * max(W, H))
        bx = (int(max(0, cx - side / 2)), int(max(0, cy - side / 2)),
              int(min(W, cx + side / 2)), int(min(H, cy + side / 2)))
        nb = (int(x0 / scale), int(y0 / scale), int(x1 / scale), int(y1 / scale))
        out.append(Detection(bx, nb, float(area / mask.size)))
    out.sort(key=lambda d: -d.area_fraction)
    return out[:max_cells]


def merge_nearby(boxes: list[list[float]], gap: float = 0.35) -> list[list[float]]:
    """Join pieces of one nucleus (separate lobes) whose boxes are closer than `gap` x their size."""
    boxes = [b[:] for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                size = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1])
                dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
                dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
                if np.hypot(dx, dy) < gap * size:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]), a[4] + b[4]]
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return boxes


def crops(img: Image.Image, dets: list[Detection]) -> list[Image.Image]:
    img = to_rgb(img)
    return [img.crop(d.box) for d in dets]
