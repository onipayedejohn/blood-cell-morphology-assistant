"""Turn a batch of per-cell predictions into a differential count summary.

It follows the way a manual differential is reported:
- percentages are over white cells only;
- nucleated red cells (erythroblasts) are reported per 100 white cells;
- cells the model is unsure about, or that look unlike its training images,
  are held back for review instead of being counted.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import CELL_INFO, LEUKOCYTES, REFERENCE_RANGES


@dataclass
class Differential:
    table: pd.DataFrame           # one row per cell type
    counted_wbc: int
    nrbc_per_100_wbc: float | None
    held_for_review: int
    flags: list[str]


def summarize(labels: list[str], statuses: list[str]) -> Differential:
    """labels: predicted class per cell; statuses: 'confident' | 'review' | 'unfamiliar'."""
    df = pd.DataFrame({"label": labels, "status": statuses})
    counted = df[df["status"] == "confident"]
    held = int((df["status"] != "confident").sum())
    wbc = counted[counted["label"].isin(LEUKOCYTES)]
    n_wbc = len(wbc)
    n_nrbc = int((counted["label"] == "erythroblast").sum())

    rows, flags = [], []
    for cls in LEUKOCYTES:
        n = int((wbc["label"] == cls).sum())
        pct = 100 * n / n_wbc if n_wbc else 0.0
        lo, hi = REFERENCE_RANGES[cls]
        if not n_wbc:
            state = ""
        elif pct > hi:
            state = "Above range"
        elif pct < lo:
            state = "Below range"
        else:
            state = "Within range"
        rows.append({"Cell type": CELL_INFO[cls].name, "Count": n, "Percent of white cells": round(pct, 1),
                     "Typical adult range": CELL_INFO[cls].adult_range or "Not expected", "Flag": state})
        if cls == "neutrophil" and state == "Below range":
            flags.append("A low neutrophil share can be normal in people with Duffy-null associated neutrophil count, "
                         "which is common in West African ancestry. Compare with local reference ranges.")
        if state in ("Above range", "Below range"):
            if cls == "immature_granulocyte":
                flags.append(f"Immature granulocytes {pct:.1f}% ({n} of {n_wbc} white cells), above about 1%, which may "
                             "point to a left shift. Band forms are counted with neutrophils here, so a left shift "
                             "made up mainly of bands will not show.")
            else:
                flags.append(f"{CELL_INFO[cls].name}s {pct:.0f}%, {state.lower()} ({CELL_INFO[cls].adult_range}).")

    nrbc = round(100 * n_nrbc / n_wbc, 1) if n_wbc else None
    if n_nrbc:
        flags.append(f"Nucleated red cells seen: {n_nrbc} ({nrbc} per 100 white cells).")
    if n_wbc and n_wbc < 100:
        flags.append(f"Only {n_wbc} white cells were counted. A manual differential counts at least 100, "
                     "so treat these percentages as rough.")
    return Differential(pd.DataFrame(rows), n_wbc, nrbc, held, flags)
