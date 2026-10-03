"""Write and execute notebooks/blood_cell_analysis.ipynb from the saved outputs.

Run after evaluate.py and make_figures.py:  python scripts/build_notebook.py
"""

from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# Blood cell morphology: from smear images to a checked classifier

This notebook walks through the whole project using the saved outputs, so it runs in a minute without retraining. Each step links to the script that did the work.

1. The data and how it was split
2. Preprocessing that is identical in training and in the app
3. Model comparison and the choice of input size
4. Test-set results, calibration and the "confident" rule
5. Where the model looks (class activation maps)
6. Images from another lab
7. What the mistakes look like, and what this model should not be used for"""),
    code("""import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from bloodsmear.config import CLASSES, CELL_INFO, display_name
from bloodsmear.preprocess import resize_uint8
from bloodsmear.inference import CellClassifier
from bloodsmear.explain import overlay
MET = ROOT / "reports" / "metrics"
load = lambda n: json.loads((MET / n).read_text())
pd.set_option("display.precision", 3)"""),
    md("""## 1. The data

PBC dataset (Acevedo et al., *Data in Brief*, 2020, CC BY 4.0): single cells photographed with a CellaVision DM96 in Barcelona, from donors without infection or blood disease. The copy used here has 13,344 images in 7 classes. `scripts/prepare_data.py` checks that every label matches its filename prefix, finds duplicates, and makes the split."""),
    code("""man = pd.read_csv(ROOT / "data" / "manifest.csv")
summary = load("data_summary.json")
print(f"{summary['images']:,} images, {summary['exact_duplicates']} exact duplicates, {summary['groups']:,} duplicate groups")
counts = pd.crosstab(man["class"].map(display_name), man["split"], margins=True, margins_name="Total")[["train", "val", "test", "Total"]]
counts"""),
    code("""# No duplicate group and no identical image crosses a split boundary
assert (man.groupby("group")["split"].nunique() == 1).all()
assert (man.groupby("md5")["split"].nunique() == 1).all()
print("Duplicate groups stay inside one split.")
man.groupby("class")["subtype"].value_counts().rename("images").to_frame()"""),
    code("""samples = pd.read_csv(ROOT / "data" / "samples" / "samples.csv")
first = samples[samples["file"].str.endswith("_1.jpg")]
fig, axes = plt.subplots(1, 7, figsize=(14, 2.6))
for ax, r in zip(axes, first.itertuples()):
    ax.imshow(Image.open(ROOT / "data" / "samples" / r.file)); ax.set_title(display_name(r._2).split(" (")[0], fontsize=9); ax.axis("off")
plt.suptitle("One test-set cell per class", x=0.01, ha="left", fontweight="bold"); plt.tight_layout(); plt.show()"""),
    md("""## 2. Preprocessing

One function, `bloodsmear.preprocess.resize_uint8`, crops the centre square and resizes with Lanczos filtering. The training arrays were built with it and the app calls it too, so the model never sees an image prepared differently. A test (`tests/test_data_and_model.py`) checks that re-running it on raw files reproduces the training arrays exactly."""),
    code("""meta = json.loads((ROOT / "models" / "model_meta.json").read_text())
size = meta["input_size"]
img = Image.open(ROOT / "data" / "samples" / "eosinophil_1.jpg")
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].imshow(img); axes[0].set_title(f"Source {img.size[0]}x{img.size[1]}", fontsize=9)
axes[1].imshow(resize_uint8(img, size)); axes[1].set_title(f"Model input {size}x{size}", fontsize=9)
for a in axes: a.axis("off")
plt.tight_layout(); plt.show()"""),
    md("""## 3. Which model

Three candidates, compared on the **validation** set only:

- logistic regression on 24 x 24 pixels, as a no-learned-features baseline;
- a CNN at 64 x 64, the size used in the bootcamp notebook;
- a CNN at 112 x 112, to see whether finer granule detail helps.

The selection rule was written before training: keep 112 px only if it beats 64 px by more than 0.005 macro-F1. Batch-norm statistics are recomputed on clean images before every validation check (see `docs/decisions.md`, section 5)."""),
    code("""cmp = pd.DataFrame(load("model_comparison.json"))
print(meta["selection"]["reason"])
cmp[["name", "val_macro_f1", "val_accuracy", "params", "train_minutes", "train_energy_wh"]]"""),
    code("""from IPython.display import Image as Show
Show(filename=str(ROOT / "reports" / "figures" / "training_curves.png"), width=820)"""),
    md("""## 4. Test-set results

The test set was split off before training and used only for final scoring by `scripts/evaluate.py` (twice: before and after the confidence-rule change in `docs/decisions.md`; the metrics below are the same in both runs). Confidence intervals are bootstrap percentiles (1,000 resamples)."""),
    code("""test = load("test_metrics.json")
rows = []
for k, name in [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"), ("macro_f1", "Macro-F1")]:
    lo, hi = test[k + "_ci"]; rows.append({"Metric": name, "Value": test[k], "95% CI low": lo, "95% CI high": hi})
rows.append({"Metric": "ECE before calibration", "Value": test["ece_uncalibrated"]})
rows.append({"Metric": "ECE after temperature scaling", "Value": test["ece_calibrated"]})
pd.DataFrame(rows)"""),
    code("""pc = pd.DataFrame(test["per_class"]); pc["class"] = pc["class"].map(display_name)
pc.sort_values("recall")"""),
    code("""Show(filename=str(ROOT / "reports" / "figures" / "confusion_matrix.png"), width=560)"""),
    code("""pd.DataFrame(test["subtypes"]).sort_values("recall")"""),
    md("""The largest group of errors sits between neighboring maturation stages: band neutrophils against metamyelocytes, and monocytes against immature granulocytes. Those boundaries are also where people disagree on a real smear.

### Calibration and the "confident" rule"""),
    code("""val = load("validation_choices.json"); sel = test["selective"]
print(f"Temperature {val['temperature']:.2f}. Confidence level {sel['confidence_threshold']:.0%}: sends the least confident "
      f"{val['review_budget']:.0%} of validation cells for review.")
print(f"Test: held back {sel['held_back']} cells ({sel['held_by_confidence']} low confidence, {sel['held_by_unfamiliar']} unfamiliar), "
      f"caught {sel['errors_held_back']} of {sel['errors']} errors; accepted cells {sel['accuracy_on_accepted']:.2%} correct.")
print(f"Still marked Confident: {sel['errors_marked_confident']} errors, {sel['errors_marked_confident_above_90']} of them above 90%.")
print("Rule history:", val["first_rule_replaced"]["test_effect"])"""),
    code("""fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, f in zip(axes, ["reliability.png", "selective_prediction.png"]):
    ax.imshow(Image.open(ROOT / "reports" / "figures" / f)); ax.axis("off")
plt.tight_layout(); plt.show()"""),
    md("""## 5. Where the model looks

The network ends in global average pooling and one dense layer, so the class activation map is exact: the last feature maps weighted by the dense weights for the predicted class. The app computes it from the ONNX model's second output."""),
    code("""clf = CellClassifier()
fig, axes = plt.subplots(2, 7, figsize=(14, 4.4))
for k, r in enumerate(first.itertuples()):
    p = clf.predict(Image.open(ROOT / "data" / "samples" / r.file))
    axes[0, k].imshow(p.image_uint8); axes[0, k].set_title(display_name(r._2).split(" (")[0], fontsize=8)
    axes[1, k].imshow(overlay(p.image_uint8, p.cam)); axes[1, k].set_title(f"{display_name(p.label).split(' (')[0]} {p.confidence:.0%}", fontsize=8)
for a in axes.flat: a.axis("off")
plt.tight_layout(); plt.show()"""),
    md("""## 6. Images from another lab

BCCD photos come from a different lab, stain and camera, and have no subtype labels. So accuracy cannot be measured on them. What can be measured is whether the app notices they are different."""),
    code("""ext = load("external_checks.json")
pd.DataFrame(ext["checks"]).T[["n", "flagged_unfamiliar"]]"""),
    code("""print("What the model called the BCCD white cells:", ext["bccd_predicted_classes"])
Show(filename=str(ROOT / "reports" / "figures" / "unfamiliar_distances.png"), width=620)"""),
    md("""## 7. Mistakes and limits"""),
    code("""Show(filename=str(ROOT / "reports" / "figures" / "confident_errors.png"), width=900)"""),
    md("""**What this model should not be used for**

- Patient care. It is a research and teaching prototype, not a medical device.
- Images from other labs, stains, cameras or magnifications without checking the status line, and ideally without retraining.
- Abnormal cells: blasts, atypical lymphocytes, parasites and abnormal red cells are not classes here, and the model will force them into one of the seven.
- Whole smears: it classifies one cropped cell at a time.

The dataset has no patient identifiers, so test cells may come from the same people as training cells. Results on new patients are likely to be lower than those above."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = ROOT / "notebooks" / "blood_cell_analysis.ipynb"
ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(ROOT / "notebooks")}})
nbf.write(nb, out)
print("Wrote", out)
