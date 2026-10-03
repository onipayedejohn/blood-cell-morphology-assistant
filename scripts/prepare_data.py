"""Build the processed dataset from the raw PBC and BCCD downloads.

Steps
1. Read every PBC image and its label file, and check the label agrees with the
   filename prefix (BA, EO, ERB, MY, ... ).
2. Find exact duplicates (same decoded pixels) and near-duplicates (perceptual
   hash within a few bits) and keep each duplicate group inside one split.
3. Make a stratified, group-aware 70/10/20 train/validation/test split.
4. Save 64 px and 112 px arrays for training, a manifest CSV, sample images for
   the app, a demo batch for the differential count, and BCCD crops from a
   different laboratory for the out-of-distribution check.

Run from the project root:  python scripts/prepare_data.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bloodsmear.config import CLASSES, PREFIX_TO_SUBTYPE  # noqa: E402
from bloodsmear.preprocess import center_square, resize_uint8  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data"
SEED = 42
PREFIX_CLASS = {"BA": 0, "EO": 1, "ERB": 2, "PMY": 3, "MY": 3, "MMY": 3, "IG": 3,
                "LY": 4, "MO": 5, "SNE": 6, "BNE": 6, "NEUTROPHIL": 6}


def ahash(img: Image.Image, n: int = 16) -> np.ndarray:
    g = np.asarray(img.convert("L").resize((n, n), Image.Resampling.BILINEAR), dtype=np.float32)
    return (g > g.mean()).flatten()


def read_pbc() -> pd.DataFrame:
    rows = []
    for split_dir in ["train", "val", "test"]:
        img_dir = RAW / "pbc" / split_dir / "images"
        for f in sorted(img_dir.glob("*.jpg")):
            lab = RAW / "pbc" / split_dir / "labels" / (f.stem + ".txt")
            lines = [ln for ln in lab.read_text().splitlines() if ln.strip()]
            if len(lines) != 1:
                raise ValueError(f"{f.name}: expected one box, found {len(lines)}")
            cid = int(lines[0].split()[0])
            prefix = f.stem.split("_")[0]
            if PREFIX_CLASS[prefix] != cid:
                raise ValueError(f"{f.name}: label {cid} disagrees with prefix {prefix}")
            rows.append({"path": str(f.relative_to(RAW)), "file": f.name, "label": cid,
                         "class": CLASSES[cid], "subtype": PREFIX_TO_SUBTYPE[prefix]})
    return pd.DataFrame(rows)


def find_groups(df: pd.DataFrame, near_bits: int = 6) -> pd.DataFrame:
    """Union exact and near duplicates into groups so they never cross splits."""
    hashes, md5s = [], []
    for p in df["path"]:
        img = center_square(Image.open(RAW / p).convert("RGB"))
        md5s.append(hashlib.md5(np.asarray(img).tobytes()).hexdigest())
        hashes.append(ahash(img))
    H = np.array(hashes)
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_md5: dict[str, int] = {}
    exact = 0
    for i, m in enumerate(md5s):
        if m in by_md5:
            parent[find(i)] = find(by_md5[m])
            exact += 1
        else:
            by_md5[m] = i
    near = 0
    for i in range(len(H)):  # O(n^2 / 2) popcount on 256-bit hashes, fine for 13k
        d = (H[i + 1:] != H[i]).sum(axis=1)
        for j in np.nonzero(d <= near_bits)[0] + i + 1:
            if find(i) != find(j):
                parent[find(j)] = find(i)
                near += 1
    df = df.copy()
    df["md5"] = md5s
    df["group"] = [find(i) for i in range(len(df))]
    df.attrs["exact_duplicates"] = exact
    df.attrs["near_duplicate_links"] = near
    return df


def split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    dev_idx, test_idx = next(outer.split(df, df["label"], df["group"]))
    df["split"] = "train"
    df.loc[df.index[test_idx], "split"] = "test"
    dev = df.iloc[dev_idx]
    inner = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=SEED)
    _, val_idx = next(inner.split(dev, dev["label"], dev["group"]))
    df.loc[dev.index[val_idx], "split"] = "val"
    return df


def save_arrays(df: pd.DataFrame, size: int):
    X = np.stack([resize_uint8(Image.open(RAW / p), size) for p in df["path"]])
    np.savez_compressed(OUT / "processed" / f"pbc_{size}.npz", X=X, y=df["label"].to_numpy(),
                        split=df["split"].to_numpy(), files=df["file"].to_numpy())


def save_samples(df: pd.DataFrame, rng: np.random.Generator):
    """Test-set images only, so the app never shows a cell the model trained on."""
    test = df[df["split"] == "test"]
    sample_dir = OUT / "samples"
    for f in sample_dir.glob("*.jpg"):
        f.unlink()
    rows = []
    for cls in CLASSES:
        pick = test[test["class"] == cls].sample(3, random_state=SEED)
        for k, r in enumerate(pick.itertuples()):
            name = f"{cls}_{k + 1}.jpg"
            center_square(Image.open(RAW / r.path).convert("RGB")).save(sample_dir / name, quality=92)
            rows.append({"file": name, "class": cls, "subtype": r.subtype, "source": r.file})
    pd.DataFrame(rows).to_csv(sample_dir / "samples.csv", index=False)

    # Demo batch for the differential count: 120 test cells in proportions that
    # resemble a mildly left-shifted adult differential with a few nucleated red
    # cells, large enough to leave about 100 white cells after review. Not a real patient.
    mix = {"neutrophil": 68, "lymphocyte": 33, "monocyte": 7, "eosinophil": 4,
           "basophil": 1, "immature_granulocyte": 4, "erythroblast": 3}
    demo_dir = OUT / "demo_batch"
    for f in demo_dir.glob("*.jpg"):
        f.unlink()
    used = set(r["source"] for r in rows)
    demo_rows = []
    order = rng.permutation(sum(mix.values()))
    k = 0
    for cls, n in mix.items():
        pool = test[(test["class"] == cls) & (~test["file"].isin(used))]
        for r in pool.sample(n, random_state=SEED).itertuples():
            name = f"cell_{order[k] + 1:03d}.jpg"
            k += 1
            center_square(Image.open(RAW / r.path).convert("RGB")).save(demo_dir / name, quality=90)
            demo_rows.append({"file": name, "class": cls, "source": r.file})
    pd.DataFrame(demo_rows).sort_values("file").to_csv(demo_dir / "demo_batch_labels.csv", index=False)


def bccd_crops(size_margin: float = 0.2):
    """White cell and red-cell-only crops from BCCD, a different lab and camera.

    BCCD has no white cell subtype labels, so these crops are used only to test
    whether the app notices that an image does not look like its training data.
    """
    ann_dir = RAW / "bccd" / "BCCD" / "Annotations"
    img_dir = RAW / "bccd" / "BCCD" / "JPEGImages"
    ext = OUT / "external"
    (ext / "bccd_wbc").mkdir(parents=True, exist_ok=True)
    (ext / "bccd_rbc_only").mkdir(parents=True, exist_ok=True)
    n_wbc = n_rbc = 0
    for xml in sorted(ann_dir.glob("*.xml")):
        tree = ET.parse(xml)
        img_path = img_dir / (tree.findtext("filename").split(".")[0] + ".jpg")
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        objs = [(o.findtext("name"), [int(float(o.find("bndbox").findtext(k)))
                                     for k in ("xmin", "ymin", "xmax", "ymax")])
                for o in tree.findall("object")]
        wbcs = [b for n, b in objs if n == "WBC"]
        for b in wbcs[:1]:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            side = max(b[2] - b[0], b[3] - b[1]) * (1 + 2 * size_margin)
            box = (int(max(0, cx - side / 2)), int(max(0, cy - side / 2)),
                   int(min(W, cx + side / 2)), int(min(H, cy + side / 2)))
            img.crop(box).save(ext / "bccd_wbc" / f"{xml.stem}.jpg", quality=92)
            n_wbc += 1
        # A patch of red cells well away from any white cell
        if wbcs and n_rbc < 150:
            wx = (wbcs[0][0] + wbcs[0][2]) / 2
            x0 = 20 if wx > W / 2 else W - 20 - 220
            patch = (x0, 130, x0 + 220, 350)
            if all(b[2] < patch[0] or b[0] > patch[2] or b[3] < patch[1] or b[1] > patch[3] for b in wbcs):
                img.crop(patch).save(ext / "bccd_rbc_only" / f"{xml.stem}.jpg", quality=92)
                n_rbc += 1
    return n_wbc, n_rbc


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-only", action="store_true",
                    help="Rebuild the app samples and demo batch from the existing manifest.")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    if args.samples_only:
        save_samples(pd.read_csv(OUT / "manifest.csv"), rng)
        print("Rebuilt samples and demo batch")
        return
    (OUT / "processed").mkdir(parents=True, exist_ok=True)
    df = read_pbc()
    print(f"Read {len(df)} PBC images; every label matches its filename prefix.")
    df = find_groups(df)
    n_groups = df["group"].nunique()
    print(f"Exact duplicates: {df.attrs['exact_duplicates']}, near-duplicate links: "
          f"{df.attrs['near_duplicate_links']}, groups: {n_groups}")
    df = split(df)
    assert not set(df[df.split == "train"].group) & set(df[df.split == "test"].group)
    assert not set(df[df.split == "val"].group) & set(df[df.split == "test"].group)
    print(pd.crosstab(df["class"], df["split"]))
    df[["file", "path", "label", "class", "subtype", "split", "group", "md5"]].to_csv(OUT / "manifest.csv", index=False)
    for size in (64, 112):
        save_arrays(df, size)
        print(f"Saved {size}px arrays")
    save_samples(df, rng)
    n_wbc, n_rbc = bccd_crops()
    print(f"BCCD crops: {n_wbc} white cell, {n_rbc} red-cell-only")
    summary = {
        "images": int(len(df)),
        "exact_duplicates": int(df.attrs["exact_duplicates"]),
        "near_duplicate_links": int(df.attrs["near_duplicate_links"]),
        "groups": int(n_groups),
        "split_counts": df["split"].value_counts().to_dict(),
        "class_counts": df["class"].value_counts().to_dict(),
        "subtype_counts": df["subtype"].value_counts().to_dict(),
        "bccd_wbc_crops": n_wbc,
        "bccd_rbc_only_crops": n_rbc,
    }
    (ROOT / "reports" / "metrics" / "data_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
