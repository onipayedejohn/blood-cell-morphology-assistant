"""Version 2 figures for the README, drawn from reports/metrics/v2_results.json.

Run after evaluate_v2.py:  python scripts/make_figures_v2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import make_figures as v1fig  # noqa: E402  (shared style and the confusion / recall plots)
from bloodsmear.canonical import canonical_uint8  # noqa: E402
from bloodsmear.explain import overlay  # noqa: E402
from bloodsmear.inference import CellClassifier  # noqa: E402
from bloodsmear.preprocess import resize_uint8  # noqa: E402

FIG = ROOT / "reports" / "figures"
MET = ROOT / "reports" / "metrics"
BLUE, ORANGE, GREY, INK, MUTED = v1fig.BLUE, v1fig.ORANGE, v1fig.GREY, v1fig.INK, v1fig.MUTED
LAB = {"bccd": "BCCD", "jtsc": "Jiangxi Tecom"}


def unseen_labs(r):
    labs = list(r["unseen_labs"])
    rows = [("Version 1 (one lab)", "v1", "#c9ccd9"),
            ("Version 2, first attempt", "v2_first_attempt_no_framing", "#86b6ef"),
            ("Version 2 (final method)", "v2_never_saw_this_lab", BLUE)]
    rows = [x for x in rows if all(x[1] in r["unseen_labs"][lab] for lab in labs)]
    vals_of = {key: [r["unseen_labs"][lab][key]["accuracy"] for lab in labs] for _, key, _ in rows}
    adapt_path = MET / "local_adaptation.json"
    if adapt_path.exists():
        a = json.loads(adapt_path.read_text())["labs"]
        if all(lab in a for lab in labs):
            n_tr = sorted(a[lab]["local_train"] for lab in labs)
            rows.append((f"Version 2 adapted with {n_tr[0]} to {n_tr[-1]} local cells*", "adapted", "#0d366b"))
            vals_of["adapted"] = [a[lab]["after"]["accuracy"] for lab in labs]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    w = 0.8 / len(rows)
    for i, (name, key, col) in enumerate(rows):
        vals = vals_of[key]
        xs = np.arange(len(labs)) + (i - (len(rows) - 1) / 2) * w
        ax.bar(xs, vals, w * 0.92, color=col, label=name)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.015, f"{v:.0%}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(labs)), [f"{LAB[lab]}\n(n = {r['unseen_labs'][lab]['v1']['n']})" for lab in labs])
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy on a lab the model never saw", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    if "adapted" in vals_of:
        fig.text(0.01, 0.01, "*Scored on the 44 to 52 cells of that lab held back from adaptation, not on the whole lab.",
                 fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "unseen_labs.png")
    plt.close(fig)


def refusals(r):
    p = r["app_pipeline"]
    order = [("pbc_test_cells", "White cells, training lab", True), ("bccd_test_cells", "White cells, BCCD", True),
             ("jtsc_test_cells", "White cells, Jiangxi Tecom", True), ("cvblog_test_cells", "White cells, CellaVision blog", True),
             ("smear_without_white_cell", "Smear, no white cell", False),
             ("cifar_unseen_photos", "Everyday photos (unseen)", False),
             ("synthetic_new_seed", "Paper, documents, screens", False),
             ("pattern_type_never_trained", "Pattern type never trained", False),
             ("scikit_image_photos", "Other photos and pages", False)]
    order = [o for o in order if o[0] in p]
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    y = np.arange(len(order))[::-1]
    vals = [1 - p[k]["refused_by_gatekeeper"] for k, _, _ in order]
    ax.barh(y, vals, color=[BLUE if pos else ORANGE for _, _, pos in order], height=0.6)
    for yi, v in zip(y, vals):
        if v > 0.8:
            ax.text(v - 0.01, yi, f"{v:.1%} passed", va="center", ha="right", fontsize=8.5, color="white")
        else:
            ax.text(v + 0.01, yi, f"{v:.1%} passed", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, [f"{name} (n = {p[k]['n']:,})" for k, name, _ in order], fontsize=8.5)
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Share passed on to the cell-type classifier")
    ax.set_title("Blood cell check (blue should pass, orange should not)", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "blood_cell_check.png")
    plt.close(fig)


def framing_examples():
    raw = ROOT / "data" / "raw"
    if not raw.exists():
        return
    man = pd.read_csv(ROOT / "data" / "manifest.csv")
    ext = pd.read_csv(ROOT / "data" / "manifest_external.csv")
    picks = [("PBC (training lab)", man[man["split"] == "test"].sample(3, random_state=4)["path"])]
    for lab, name in [("bccd", "BCCD"), ("jtsc", "Jiangxi Tecom"), ("cvblog", "CellaVision blog")]:
        picks.append((name, ext[(ext["lab"] == lab) & (ext["split"] == "test")].sample(3, random_state=4)["path"]))
    fig, axes = plt.subplots(2, 12, figsize=(13, 2.7))
    for g, (name, paths) in enumerate(picks):
        for k, path in enumerate(paths):
            img = Image.open(raw / path)
            axes[0, g * 3 + k].imshow(resize_uint8(img))
            axes[1, g * 3 + k].imshow(canonical_uint8(img))
        axes[0, g * 3].set_title(name, loc="left", fontsize=9, color=INK)
    for ax in axes.flat:
        ax.axis("off")
    fig.text(0.003, 0.72, "As\nuploaded", fontsize=8, color=MUTED, va="center")
    fig.text(0.003, 0.27, "What the\nmodel sees", fontsize=8, color=MUTED, va="center")
    fig.suptitle("Standard framing: recentered on the nucleus at a fixed scale, background color balanced",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0.04, 0, 1, 1))
    fig.savefig(FIG / "standard_framing.png")
    plt.close(fig)


def heatmaps(clf):
    s = pd.read_csv(ROOT / "data" / "samples" / "samples.csv")
    s = s[s["file"].str.endswith("_1.jpg")]
    fig, axes = plt.subplots(2, 7, figsize=(13, 4.2))
    for k, r in enumerate(s.itertuples()):
        p = clf.predict(Image.open(ROOT / "data" / "samples" / r.file))
        axes[0, k].imshow(p.image_uint8)
        axes[0, k].set_title(v1fig.short(r._2), fontsize=9)
        axes[1, k].imshow(overlay(p.image_uint8, p.cam))
        axes[1, k].set_title(f"{v1fig.short(p.label)} {v1fig.conf(p.confidence)}", fontsize=9,
                             color=INK if p.label == r._2 else ORANGE)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("Test-set cells as the model sees them (top, expert label) and class activation maps (bottom, answer)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "heatmap_examples.png")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    r = json.loads((MET / "v2_results.json").read_text())
    v1fig.FIG = FIG  # reuse the version 1 plotting functions, writing here
    v1fig.confusion(r["final"]["pbc_test"])
    v1fig.per_class(r["final"]["pbc_test"])
    unseen_labs(r)
    refusals(r)
    framing_examples()
    heatmaps(CellClassifier())
    print("Figures written to", FIG)


if __name__ == "__main__":
    main()
