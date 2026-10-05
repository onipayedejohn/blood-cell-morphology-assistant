"""Score the white cell finder on whole-field images with expert boxes.

Parameters were tuned on the "train" half of the BCCD and draaslan fields and
on PBC training images; this script reports the held-out halves.
A white cell counts as found when the center of a detected nucleus falls inside
its box (5 px tolerance). Any other detection is a false alarm.

Run from the project root:  python scripts/evaluate_finder.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bloodsmear.detect import find_cells  # noqa: E402

RAW = ROOT / "data" / "raw"


def pbc_fields(split, n=300):
    man = pd.read_csv(ROOT / "data" / "manifest.csv")
    out = []
    for r in man[man["split"] == split].sample(n, random_state=0).itertuples():
        lab = (RAW / r.path).as_posix().replace("/images/", "/labels/").replace(".jpg", ".txt")
        cx, cy, bw, bh = map(float, open(lab).read().split()[1:5])
        W, H = Image.open(RAW / r.path).size
        out.append({"path": r.path, "wbc_boxes": [[(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H]]})
    return out


def score(fields):
    tp = fn = fp = 0
    for f in fields:
        dets = find_cells(Image.open(RAW / f["path"]))
        gts, matched = f["wbc_boxes"], set()
        for d in dets:
            nx, ny = (d.nucleus_box[0] + d.nucleus_box[2]) / 2, (d.nucleus_box[1] + d.nucleus_box[3]) / 2
            hit = [i for i, g in enumerate(gts) if g[0] - 5 <= nx <= g[2] + 5 and g[1] - 5 <= ny <= g[3] + 5]
            matched.update(hit) if hit else None
            fp += 0 if hit else 1
        tp += len(matched)
        fn += len(gts) - len(matched)
    return {"images": len(fields), "white_cells": tp + fn, "found": tp, "recall": tp / max(tp + fn, 1),
            "false_alarms_per_image": fp / max(len(fields), 1)}


def main():
    fields = json.loads((ROOT / "data" / "fields.json").read_text())
    res = {}
    for split in ("train", "test"):
        res[split] = {src: score([f for f in fields if f["source"] == src and f["split"] == split])
                      for src in ("bccd", "draaslan")}
        res[split]["pbc"] = score(pbc_fields(split))
    (ROOT / "reports" / "metrics" / "finder_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["test"], indent=2))


if __name__ == "__main__":
    main()
