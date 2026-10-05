"""Build the multi-lab and gatekeeper datasets (version 2).

Labeled white cells from labs other than the PBC training lab:
- bccd:   BCCD cleaned crops (apple2373/BCCD, MIT): 351 cells, 5 classes
- jtsc:   Zheng et al. Dataset 1, Jiangxi Tecom, China (zxaoyou/segmentation_WBC, GPL-3): 300 cells
- cvblog: Zheng et al. Dataset 2, CellaVision blog images: 100 cells
Each external lab is split 50/50 (per class) into train and test halves for
the final model. Leave-one-lab-out runs use the whole held-out lab as test.

Gatekeeper data (is this a stained blood cell at all?):
- smear_no_wbc: smear patches with red cells but no white cell, cut from BCCD
  fields, the draaslan detection images and PBC image corners, away from any
  white cell box.
- not_smear: CIFAR-10 photos plus synthetic paper sheets, printed documents,
  screenshots, gradients and noise. Held-out test negatives use other CIFAR
  images, a different generator seed, one pattern type never used in training,
  and 19 scikit-image photos.

Whole-field images with white cell boxes (BCCD, draaslan) are listed for the
cell-finder evaluation.

Run from the project root:  python scripts/prepare_external.py
"""

from __future__ import annotations

import io
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bloodsmear.config import CLASSES  # noqa: E402
from bloodsmear.preprocess import resize_uint8  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
SIZE = 64
SEED = 42
ZHENG = {1: "neutrophil", 2: "lymphocyte", 3: "monocyte", 4: "eosinophil", 5: "basophil"}


# -- labeled external cells -------------------------------------------------------------
def external_cells() -> pd.DataFrame:
    rows = []
    for folder in sorted((RAW / "bccd_labeled" / "BCCD_cropped").iterdir()):
        for f in sorted(folder.glob("*")):
            rows.append({"lab": "bccd", "path": str(f.relative_to(RAW)), "class": folder.name.lower()})
    for ds, lab in [(1, "jtsc"), (2, "cvblog")]:
        lb = pd.read_csv(RAW / "zheng" / f"Class Labels of Dataset {ds}.csv")
        for i, c in zip(lb.iloc[:, 0], lb.iloc[:, 1]):
            rows.append({"lab": lab, "path": f"zheng/Dataset {ds}/{int(i):03d}.bmp", "class": ZHENG[int(c)]})
    df = pd.DataFrame(rows)
    df["label"] = df["class"].map(CLASSES.index)
    rng = np.random.default_rng(SEED)
    df["split"] = "test"
    for (_, _), g in df.groupby(["lab", "class"]):
        idx = rng.permutation(g.index.to_numpy())
        df.loc[idx[: len(idx) // 2], "split"] = "train"
    return df


# -- smear patches without a white cell ------------------------------------------------------
def overlaps(box, boxes, pad=0):
    x0, y0, x1, y1 = box
    return any(not (x1 < b[0] - pad or x0 > b[2] + pad or y1 < b[1] - pad or y0 > b[3] + pad) for b in boxes)


def bccd_fields():
    ann = RAW / "bccd" / "BCCD" / "Annotations"
    out = []
    for xml in sorted(ann.glob("*.xml")):
        t = ET.parse(xml)
        img = RAW / "bccd" / "BCCD" / "JPEGImages" / (t.findtext("filename").split(".")[0] + ".jpg")
        if not img.exists():
            continue
        wbc = [[int(float(o.find("bndbox").findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax")]
               for o in t.findall("object") if o.findtext("name") == "WBC"]
        out.append({"source": "bccd", "path": str(img.relative_to(RAW)), "wbc_boxes": wbc})
    return out


def draaslan_fields():
    a = pd.read_csv(RAW / "draaslan" / "annotations.csv")
    out = []
    for name, g in a.groupby("image"):
        p = RAW / "draaslan" / "images" / name
        w, h = Image.open(p).size
        wbc = [[int(r.xmin * w), int(r.ymin * h), int(r.xmax * w), int(r.ymax * h)] for r in g.itertuples() if r.label == "wbc"]
        out.append({"source": "draaslan", "path": str(p.relative_to(RAW)), "wbc_boxes": wbc})
    return out


def smear_patches(fields, per_image, rng, side_range):
    patches = []
    for f in fields:
        img = Image.open(RAW / f["path"]).convert("RGB")
        W, H = img.size
        for _ in range(per_image * 6):
            if sum(1 for p in patches if p[1] == f["path"]) >= per_image:
                break
            s = int(rng.integers(side_range[0], min(side_range[1], W, H) + 1))
            x, y = int(rng.integers(0, W - s + 1)), int(rng.integers(0, H - s + 1))
            if not overlaps((x, y, x + s, y + s), f["wbc_boxes"], pad=8):
                patches.append((resize_uint8(img.crop((x, y, x + s, y + s)), SIZE), f["path"]))
    return patches


def pbc_corner_patches(rng, n):
    man = pd.read_csv(ROOT / "data" / "manifest.csv").sample(n * 2, random_state=SEED)
    out = []
    for r in man.itertuples():
        if len(out) >= n:
            break
        lab = (RAW / r.path).with_suffix(".txt").as_posix().replace("/images/", "/labels/")
        cx, cy, bw, bh = map(float, open(lab).read().split()[1:5])
        img = Image.open(RAW / r.path).convert("RGB")
        W, H = img.size
        box = ((cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H)
        for corner in [(0, 0), (W - 110, 0), (0, H - 110), (W - 110, H - 110)]:
            pb = (corner[0], corner[1], corner[0] + 110, corner[1] + 110)
            if not overlaps(pb, [box], pad=4):
                out.append((resize_uint8(img.crop(pb), SIZE), r.split))
                break
    return out


# -- negatives ------------------------------------------------------------------------------
def jpeg(img: Image.Image, q: int) -> Image.Image:
    b = io.BytesIO()
    img.save(b, "JPEG", quality=int(q))
    return Image.open(b).convert("RGB")


def lighting(arr, rng):
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
    v = 1 - rng.uniform(0.1, 0.5) * (((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2) * 2
    return arr * v[..., None]


def synthetic(kind: str, rng: np.random.Generator) -> Image.Image:
    W = H = int(rng.integers(200, 700))
    if kind == "sheet":
        base = rng.choice([rng.uniform(5, 50), rng.uniform(200, 250), rng.uniform(60, 200)])
        tint = rng.normal(0, 12, 3)
        arr = np.ones((H, W, 3)) * (base + tint)
        arr = lighting(arr, rng) + rng.normal(0, rng.uniform(1, 8), (H, W, 3))
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    elif kind == "document":
        paper = tuple(int(v) for v in rng.uniform(200, 255, 3))
        img = Image.new("RGB", (W, H), paper)
        d = ImageDraw.Draw(img)
        ink = tuple(int(v) for v in rng.uniform(0, 90, 3))
        step = int(rng.integers(8, 26))
        for y in range(int(rng.integers(5, 30)), H, step):
            x = int(rng.integers(5, 40))
            while x < W - 20:
                L = int(rng.integers(10, 60))
                d.rectangle((x, y, x + L, y + max(2, step // 3)), fill=ink)
                x += L + int(rng.integers(4, 12))
        img = Image.fromarray(np.clip(lighting(np.asarray(img, float), rng), 0, 255).astype(np.uint8))
    elif kind == "screen":
        img = Image.new("RGB", (W, H), tuple(int(v) for v in rng.uniform(180, 255, 3)))
        d = ImageDraw.Draw(img)
        for _ in range(int(rng.integers(3, 14))):
            x0, y0 = int(rng.integers(0, W)), int(rng.integers(0, H))
            d.rectangle((x0, y0, x0 + int(rng.integers(20, W // 2)), y0 + int(rng.integers(10, H // 3))),
                        fill=tuple(int(v) for v in rng.uniform(0, 255, 3)))
    elif kind == "gradient":
        a, b = rng.uniform(0, 255, 3), rng.uniform(0, 255, 3)
        t = np.linspace(0, 1, W)[None, :, None] if rng.random() < 0.5 else np.linspace(0, 1, H)[:, None, None]
        arr = a * (1 - t) + b * t
        arr = np.broadcast_to(arr, (H, W, 3)) + rng.normal(0, 3, (H, W, 3))
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    elif kind == "noise":
        s = int(rng.integers(4, 64))
        small = rng.integers(0, 256, (s, s, 3), dtype=np.uint8)
        img = Image.fromarray(small).resize((W, H), Image.Resampling.BICUBIC)
    elif kind == "texture":  # fabric, wood, skin-like filtered noise
        base = rng.uniform(40, 230, 3)
        n = rng.normal(0, 1, (H, W, 1)) * rng.uniform(10, 40)
        img = Image.fromarray(np.clip(base + n, 0, 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(float(rng.uniform(1, 6))))
    elif kind == "pattern":  # held out of training entirely
        img = Image.new("RGB", (W, H), tuple(int(v) for v in rng.uniform(0, 255, 3)))
        d = ImageDraw.Draw(img)
        s = int(rng.integers(10, 60))
        col = tuple(int(v) for v in rng.uniform(0, 255, 3))
        for y in range(0, H, s):
            for x in range(0, W, s):
                if (x // s + y // s) % 2:
                    d.ellipse((x, y, x + s, y + s), fill=col)
    else:
        raise ValueError(kind)
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.3, 2.0))))
    return jpeg(img, rng.integers(40, 95))


TRAIN_KINDS = ["sheet", "document", "screen", "gradient", "noise", "texture"]


HELDOUT_SKIMAGE = ["astronaut", "brick", "camera", "cat", "checkerboard", "chelsea", "clock", "coffee", "coins",
                   "colorwheel", "grass", "gravel", "hubble_deep_field", "immunohistochemistry", "logo", "moon", "page",
                   "rocket", "text"]


def heldout_photos() -> list[Path]:
    """19 images bundled with scikit-image (never used in training), written once to data/raw/heldout_negatives."""
    out = RAW / "heldout_negatives"
    if not out.exists() or not any(out.glob("*.png")):
        from skimage import data as skdata  # only needed the first time
        out.mkdir(parents=True, exist_ok=True)
        for name in HELDOUT_SKIMAGE:
            arr = getattr(skdata, name)()
            Image.fromarray(arr).convert("RGB").save(out / f"skimage_{name}.png")
    return sorted(out.glob("*.png"))


def cifar(split: str, n_per_class: int) -> list[np.ndarray]:
    root = RAW / "cifar" / split
    out = []
    for cls in sorted(root.iterdir()):
        for f in sorted(cls.glob("*"))[:n_per_class]:
            try:
                out.append(resize_uint8(Image.open(f), SIZE))
            except OSError:
                continue  # an incomplete download
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    ext = external_cells()
    X = np.stack([resize_uint8(Image.open(RAW / p), SIZE) for p in ext["path"]])
    np.savez_compressed(OUT / f"external_{SIZE}.npz", X=X, y=ext["label"].to_numpy(), lab=ext["lab"].to_numpy(),
                        split=ext["split"].to_numpy())
    ext.to_csv(ROOT / "data" / "manifest_external.csv", index=False)
    print(pd.crosstab([ext["lab"], ext["class"]], ext["split"]))

    fields = bccd_fields() + draaslan_fields()
    rs = np.random.default_rng(SEED)  # half of the fields for patch training, the rest for testing
    for f in fields:
        f["split"] = "train" if rs.random() < 0.5 else "test"
    (ROOT / "data" / "fields.json").write_text(json.dumps(fields, indent=1))

    smear = {"train": [], "test": []}
    smear_src = {"train": [], "test": []}
    for split in ("train", "test"):
        fs = [f for f in fields if f["split"] == split]
        for src, per, rng_side in [("bccd", 3, (90, 260)), ("draaslan", 2, (60, 150))]:
            got = [p for p, _ in smear_patches([f for f in fs if f["source"] == src], per, rng, rng_side)]
            smear[split] += got
            smear_src[split] += [src] * len(got)
    for p, sp in pbc_corner_patches(rng, 1600):
        smear["test" if sp == "test" else "train"].append(p)
        smear_src["test" if sp == "test" else "train"].append("pbc")

    neg_train = cifar("train", 300)
    neg_train += [resize_uint8(synthetic(TRAIN_KINDS[i % len(TRAIN_KINDS)], rng), SIZE) for i in range(3000)]
    rt = np.random.default_rng(SEED + 1000)
    neg_test = {
        "cifar_unseen_photos": cifar("test", 100),
        "synthetic_new_seed": [resize_uint8(synthetic(TRAIN_KINDS[i % len(TRAIN_KINDS)], rt), SIZE) for i in range(600)],
        "pattern_type_never_trained": [resize_uint8(synthetic("pattern", rt), SIZE) for _ in range(200)],
        "scikit_image_photos": [resize_uint8(Image.open(f), SIZE) for f in heldout_photos()],
    }
    np.savez_compressed(OUT / "gate_64.npz",
                        smear_train=np.stack(smear["train"]), smear_test=np.stack(smear["test"]),
                        smear_train_source=np.array(smear_src["train"]), smear_test_source=np.array(smear_src["test"]),
                        neg_train=np.stack(neg_train),
                        **{f"negtest_{k}": np.stack(v) for k, v in neg_test.items()})
    summary = {"external": ext.groupby(["lab", "split"]).size().unstack().to_dict(),
               "smear_no_wbc": {k: len(v) for k, v in smear.items()},
               "not_smear_train": len(neg_train), "not_smear_test": {k: len(v) for k, v in neg_test.items()},
               "fields": pd.DataFrame(fields).groupby(["source", "split"]).size().unstack().to_dict()}
    (ROOT / "reports" / "metrics" / "data_summary_v2.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
