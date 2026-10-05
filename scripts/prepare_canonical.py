"""Rebuild the classifier arrays in the standard framing and color balance (version 2).

Same images, labels and splits as pbc_64.npz and external_64.npz, but every
cell goes through bloodsmear.canonical.canonical_cell, the function the app
uses. The gatekeeper keeps the plain arrays, because in the app it sees images
as uploaded.

Run from the project root after prepare_data.py and prepare_external.py:
    python scripts/prepare_canonical.py            # 64 px
    python scripts/prepare_canonical.py --size 96
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bloodsmear.canonical import canonical_cell  # noqa: E402

RAW = ROOT / "data" / "raw"
P = ROOT / "data" / "processed"


def run(paths, size=64):
    X, info = [], []
    for p in paths:
        a, f = canonical_cell(Image.open(RAW / p), size)
        X.append(a)
        info.append((f.found, f.padded))
    info = np.array(info)
    return np.stack(X), {"n": len(paths), "nucleus_found": float(info[:, 0].mean()), "padded": float(info[:, 1].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=64)
    size = ap.parse_args().size
    m = pd.read_csv(ROOT / "data" / "manifest.csv")
    old = np.load(P / "pbc_64.npz", allow_pickle=True)
    assert (old["files"] == m["file"].to_numpy()).all(), "manifest and pbc_64.npz are out of step"
    X, s_pbc = run(m["path"], size)
    np.savez_compressed(P / f"pbc_{size}c.npz", X=X, y=old["y"], split=old["split"], files=old["files"])

    e = pd.read_csv(ROOT / "data" / "manifest_external.csv")
    old = np.load(P / "external_64.npz", allow_pickle=True)
    assert (old["lab"] == e["lab"].to_numpy()).all() and (old["split"] == e["split"].to_numpy()).all()
    Xe, s_ext = run(e["path"], size)
    np.savez_compressed(P / f"external_{size}c.npz", X=Xe, y=old["y"], lab=old["lab"], split=old["split"])
    if size != 64:
        return
    per_lab = {lab: run(e.loc[e["lab"] == lab, "path"])[1] for lab in sorted(e["lab"].unique())}
    summary = {"pbc": s_pbc, "external": s_ext, "external_by_lab": per_lab}
    (ROOT / "reports" / "metrics" / "framing_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
