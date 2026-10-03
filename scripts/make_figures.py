"""Static figures for the README and the report, drawn from the saved metrics.

Run after evaluate.py:  python scripts/make_figures.py
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bloodsmear.config import CLASSES, display_name  # noqa: E402
from bloodsmear.explain import overlay  # noqa: E402
from bloodsmear.inference import CellClassifier  # noqa: E402
from bloodsmear.preprocess import resize_uint8  # noqa: E402

FIG = ROOT / "reports" / "figures"
MET = ROOT / "reports" / "metrics"
BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#c9ccd9", "#1c1d2b", "#5b5e70"
RAMP = LinearSegmentedColormap.from_list("blue", ["#f6f7fb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
SHORT = {"immature_granulocyte": "Immature gran.", "erythroblast": "Erythroblast"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GREY, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlecolor": INK, "figure.dpi": 150})


def conf(x):
    return ">99%" if x > 0.995 else f"{x:.0%}"


def short(c):
    return SHORT.get(c, display_name(c))


def confusion(test):
    cm = np.array(test["confusion_matrix"])
    share = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.imshow(share, cmap=RAMP, vmin=0, vmax=1)
    names = [short(c) for c in CLASSES]
    ax.set_xticks(range(7), names, rotation=35, ha="right")
    ax.set_yticks(range(7), names)
    for i in range(7):
        for j in range(7):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9, color="white" if share[i, j] > 0.5 else INK)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Expert label")
    ax.set_title(f"Test set confusion matrix (n = {cm.sum():,})", loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrix.png")
    plt.close(fig)


def per_class(test):
    pc = pd.DataFrame(test["per_class"]).sort_values("recall")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    y = np.arange(len(pc))
    ax.barh(y, pc["recall"], color=BLUE, height=0.55)
    ax.set_yticks(y, [short(c) for c in pc["class"]])
    for yi, r, n in zip(y, pc["recall"], pc["support"]):
        ax.text(r + 0.002, yi, f"{r:.1%}  (n={n})", va="center", fontsize=9, color=INK)
    lo = max(0.0, pc["recall"].min() - 0.05)
    ax.set_xlim(lo, 1.04)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Recall by cell type, test set", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "per_class_recall.png")
    plt.close(fig)


def reliability(test):
    rel = pd.DataFrame(test["reliability"])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color=GREY, lw=2, label="Perfect calibration")
    ax.plot(rel["mean_confidence"], rel["accuracy"], color=BLUE, lw=2, marker="o", ms=5, label="This model (calibrated)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Stated confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title(f"Calibration, ECE {test['ece_uncalibrated']:.1%} to {test['ece_calibrated']:.1%}", loc="left")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "reliability.png")
    plt.close(fig)


def coverage(test):
    cur = pd.DataFrame([c for c in test["selective"]["curve"] if c["accuracy"] is not None])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(cur["coverage"], cur["accuracy"], color=BLUE, lw=2, marker="o", ms=5)
    sel = test["selective"]
    ax.scatter([sel["coverage"]], [sel["accuracy_on_accepted"]], s=90, facecolor="white", edgecolor=ORANGE, lw=2, zorder=3,
               label=f"App setting: confidence at least {sel['confidence_threshold']:.0%}")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Share of cells answered")
    ax.set_ylabel("Accuracy on answered cells")
    ax.set_title("Answering fewer cells, more accurately", loc="left")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "selective_prediction.png")
    plt.close(fig)


def training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for size, color in [(64, BLUE), (112, ORANGE)]:
        p = ROOT / "models" / "candidates" / f"cnn_{size}_train.json"
        if not p.exists():
            continue
        h = pd.DataFrame(json.loads(p.read_text())["history"])
        axes[0].plot(h["epoch"], h["loss"], color=color, lw=2, label=f"{size} px train")
        axes[0].plot(h["epoch"], h["val_loss"], color=color, lw=2, ls="--", label=f"{size} px validation")
        axes[1].plot(h["epoch"], h["val_macro_f1"], color=color, lw=2, marker="o", ms=3, label=f"{size} px")
    axes[0].set_title("Loss", loc="left")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_title("Validation macro-F1", loc="left")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "training_curves.png")
    plt.close(fig)


def heatmaps(clf):
    s = pd.read_csv(ROOT / "data" / "samples" / "samples.csv")
    s = s[s["file"].str.endswith("_1.jpg")]
    fig, axes = plt.subplots(2, 7, figsize=(13, 4.2))
    for k, r in enumerate(s.itertuples()):
        p = clf.predict(Image.open(ROOT / "data" / "samples" / r.file))
        axes[0, k].imshow(p.image_uint8)
        axes[0, k].set_title(short(r._2), fontsize=9)
        axes[1, k].imshow(overlay(p.image_uint8, p.cam))
        axes[1, k].set_title(f"{short(p.label)} {conf(p.confidence)}", fontsize=9,
                             color=INK if p.label == r._2 else ORANGE)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("Test-set cells (top, expert label) and class activation maps (bottom, model answer)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "heatmap_examples.png")
    plt.close(fig)


def errors(clf):
    err = pd.read_csv(MET / "test_errors.csv").sort_values("confidence", ascending=False).head(12)
    man = pd.read_csv(ROOT / "data" / "manifest.csv").set_index("file")
    raw = ROOT / "data" / "raw"
    if not raw.exists():
        return
    fig, axes = plt.subplots(2, 6, figsize=(12, 4.6))
    for ax, r in zip(axes.flat, err.itertuples()):
        ax.imshow(resize_uint8(Image.open(raw / man.loc[r.file, "path"]), 112))
        ax.set_title(f"{r.subtype}\nread as {short(r.predicted)} {conf(r.confidence)}", fontsize=8)
        ax.axis("off")
    for ax in axes.flat[len(err):]:
        ax.axis("off")
    fig.suptitle("The most confident mistakes on the test set", x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "confident_errors.png")
    plt.close(fig)


def distances(clf):
    d = np.load(ROOT / "data" / "processed" / f"pbc_{clf.size}.npz", allow_pickle=True)
    Xte = d["X"][d["split"] == "test"]
    dist_te = np.concatenate([clf.predict_arrays(list(Xte[i:i + 128]))[2] for i in range(0, len(Xte), 128)])
    ext = sorted((ROOT / "data" / "external" / "bccd_wbc").glob("*.jpg"))
    arrs = [resize_uint8(Image.open(f), clf.size) for f in ext]
    dist_ex = np.concatenate([clf.predict_arrays(arrs[i:i + 128])[2] for i in range(0, len(arrs), 128)])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bins = np.linspace(0, max(dist_te.max(), dist_ex.max()) * 1.02, 60)
    ax.hist(dist_te, bins=bins, color=BLUE, alpha=0.85, density=True, label=f"PBC test cells (n={len(dist_te):,})")
    ax.hist(dist_ex, bins=bins, color=ORANGE, alpha=0.75, density=True, label=f"BCCD white cells, other lab (n={len(dist_ex)})")
    ax.axvline(clf.ood_threshold, color=INK, lw=1.5, ls="--")
    ax.text(clf.ood_threshold, ax.get_ylim()[1] * 0.92, "  warning level", color=INK, fontsize=9)
    ax.set_xlabel("Distance from the training cells (Mahalanobis, model features)")
    ax.set_yticks([])
    ax.set_title("How different the other lab's images look to the model", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "unfamiliar_distances.png")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    test = json.loads((MET / "test_metrics.json").read_text())
    clf = CellClassifier()
    confusion(test)
    per_class(test)
    reliability(test)
    coverage(test)
    training_curves()
    heatmaps(clf)
    errors(clf)
    distances(clf)
    print("Figures written to", FIG)


if __name__ == "__main__":
    main()
