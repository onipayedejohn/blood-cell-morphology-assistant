"""Adapt the classifier to your own laboratory with your own labeled cells (about 100 helped in testing).

On a lab it has never seen, the model is right only 34% to 51% of the time.
In a test on two such labs, this script trained on 84 and 99 of a lab's own
labeled cells and raised accuracy on that lab's held-back cells from 30% to 86%
and from 42% to 81%, in about 3 minutes on 2 CPU cores
(reports/metrics/local_adaptation.json). Rare cell types can stay weak.

1. Put single-cell crops (one white cell near the middle of each image) in one
   folder per class, named after the class:

       data/local/neutrophil/*.jpg
       data/local/lymphocyte/*.jpg
       ...

   Folder names: basophil, eosinophil, erythroblast (or nrbc),
   immature_granulocyte (or ig), lymphocyte, monocyte, neutrophil.
   Classes you have no cells for can be left out. Aim for at least 20 cells
   per class you include; more is better.

2. Run (a few minutes on a CPU for a few hundred cells):

       pip install -r requirements-dev.txt
       python scripts/finetune_local.py --images data/local --name mylab

   30% of your cells (per type) are held back and never trained on, a fifth
   of the rest picks the best epoch, and the remainder is trained on. The
   script reports accuracy on the held-back cells before and after.

3. Try the adapted model in the app:

       BLOODSMEAR_MODEL_DIR=models/local_mylab streamlit run app/streamlit_app.py

The adapted model keeps seeing a replay set of the original training cells
(models/train/replay_64c.npz) so it does not forget them.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", message=".*shuffle=True.*")

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from sklearn.covariance import LedoitWolf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from augment import classifier_augment_batch  # noqa: E402
from bloodsmear.canonical import canonical_uint8  # noqa: E402
from bloodsmear.config import CLASSES  # noqa: E402
from evaluate import fit_temperature, logits_and_features, softmax  # noqa: E402
from train import precise_bn, to_float  # noqa: E402

SEED = 42
ALIASES = {"nrbc": "erythroblast", "ig": "immature_granulocyte", "immature granulocyte": "immature_granulocyte",
           "neutrophils": "neutrophil", "lymphocytes": "lymphocyte", "monocytes": "monocyte",
           "eosinophils": "eosinophil", "basophils": "basophil", "erythroblasts": "erythroblast"}
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def read_folder(folder: Path, size: int):
    X, y, files = [], [], []
    unknown = []
    for d in sorted(p for p in folder.iterdir() if p.is_dir()):
        name = ALIASES.get(d.name.lower().replace("-", " ").strip(), d.name.lower().replace(" ", "_"))
        if name not in CLASSES:
            unknown.append(d.name)
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in IMAGE_TYPES:
                continue
            try:
                X.append(canonical_uint8(Image.open(f), size))
            except (UnidentifiedImageError, OSError):
                print(f"skipped unreadable file {f}")
                continue
            y.append(CLASSES.index(name))
            files.append(str(f))
    if unknown:
        print(f"ignored folders that are not class names: {', '.join(unknown)}")
    if not X:
        raise SystemExit(f"No images found in class folders under {folder}.")
    return np.stack(X), np.array(y, "int32"), np.array(files)


def split(y, share, rng):
    """Per class: `share` to the held-back test set, then 20% of the rest to validation."""
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = rng.permutation(np.nonzero(y == c)[0])
        n_te = max(1, int(round(len(idx) * share))) if len(idx) >= 3 else 0
        rest = idx[n_te:]
        n_va = max(1, int(round(len(rest) * 0.2))) if len(rest) >= 5 else 0
        te += list(idx[:n_te])
        va += list(rest[:n_va])
        tr += list(rest[n_va:])
    return np.array(tr, int), np.array(va, int), np.array(te, int)


def metrics(y, pred):
    present = sorted(set(y))
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=present, average="macro")),
            "recall_by_class": {CLASSES[c]: float((pred[y == c] == c).mean()) for c in present}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, required=True, help="folder with one subfolder per class")
    ap.add_argument("--name", default="mylab")
    ap.add_argument("--base", type=Path, default=ROOT / "models" / "train" / "bloodcell_cnn_v2.keras")
    ap.add_argument("--replay", type=Path, default=ROOT / "models" / "train" / "replay_64c.npz")
    ap.add_argument("--test-share", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--repeat", type=int, default=0, help="copies of each local cell per epoch (0: automatic)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    tf.keras.utils.set_random_seed(SEED)
    rng = np.random.default_rng(SEED)
    out = args.out or ROOT / "models" / f"local_{args.name}"

    model = tf.keras.models.load_model(args.base)
    size = int(model.input_shape[1])
    X, y, files = read_folder(args.images, size)
    tr, va, te = split(y, args.test_share, rng)
    print(f"{len(y)} local cells: {len(tr)} train, {len(va)} validation, {len(te)} held back for testing")
    print("per class:", {CLASSES[c]: int((y == c).sum()) for c in np.unique(y)})

    def predict(A):
        return logits_and_features(model, A)[0].argmax(1) if len(A) else np.zeros(0, int)

    before = metrics(y[te], predict(X[te])) if len(te) else None
    if before:
        print(f"before adaptation, held-back local cells: accuracy {before['accuracy']:.1%}")

    rp = np.load(args.replay, allow_pickle=True)
    Xr, yr, sr = rp["X"], rp["y"].astype("int32"), rp["split"]
    Xrt, yrt, Xrv, yrv = Xr[sr == "train"], yr[sr == "train"], Xr[sr == "val"], yr[sr == "val"]
    repeat = args.repeat or max(1, int(round(0.35 * len(Xrt) / max(len(tr), 1))))  # local cells ~ a quarter of each batch
    Xtrain = np.concatenate([Xrt] + [X[tr]] * repeat)
    ytrain = np.concatenate([yrt] + [y[tr]] * repeat)
    print(f"training on {len(Xrt)} replay cells and {len(tr)} local cells x {repeat}")

    ds = (tf.data.Dataset.from_tensor_slices((Xtrain, ytrain)).shuffle(len(Xtrain), seed=SEED)
          .map(to_float, num_parallel_calls=tf.data.AUTOTUNE).batch(64)
          .map(classifier_augment_batch, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE))
    steps = int(np.ceil(len(Xtrain) / 64)) * args.epochs
    model.compile(tf.keras.optimizers.AdamW(tf.keras.optimizers.schedules.CosineDecay(args.lr, steps, alpha=0.05),
                                            weight_decay=1e-4),
                  tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    X_bn = Xtrain[rng.choice(len(Xtrain), min(2048, len(Xtrain)), replace=False)]
    best, best_w, log = -1.0, None, []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.fit(ds, epochs=1, verbose=0)
        precise_bn(model, X_bn)
        local_acc = float((predict(X[va]) == y[va]).mean()) if len(va) else 0.0
        replay_f1 = float(f1_score(yrv, predict(Xrv), average="macro"))
        score = 0.5 * local_acc + 0.5 * replay_f1 if len(va) else replay_f1
        log.append({"epoch": epoch + 1, "local_val_accuracy": local_acc, "replay_val_macro_f1": replay_f1})
        print(f"epoch {epoch + 1}: local validation accuracy {local_acc:.1%}, original cells macro-F1 {replay_f1:.3f}",
              flush=True)
        if score > best:
            best, best_w = score, model.get_weights()
    model.set_weights(best_w)
    minutes = (time.time() - t0) / 60

    after = metrics(y[te], predict(X[te])) if len(te) else None
    if after:
        print(f"after adaptation, held-back local cells: accuracy {after['accuracy']:.1%} "
              f"(before {before['accuracy']:.1%})")

    # Calibration and the status line, set on validation cells (local and replay)
    Xv = np.concatenate([X[va], Xrv])
    yv = np.concatenate([y[va], yrv])
    L_va, F_va, feat_model = logits_and_features(model, Xv)
    T = fit_temperature(L_va, yv)
    conf_threshold = float(np.quantile(softmax(L_va / T).max(1), 0.02))
    _, F_tr, _ = logits_and_features(model, np.concatenate([Xrt, X[tr]]))
    y_tr = np.concatenate([yrt, y[tr]])
    present = [c for c in range(len(CLASSES)) if (y_tr == c).any()]
    means = np.stack([F_tr[y_tr == c].mean(0) if (y_tr == c).any() else np.full(F_tr.shape[1], 1e6)
                      for c in range(len(CLASSES))])
    precision = LedoitWolf().fit(F_tr - means[y_tr]).precision_
    diff = F_va[:, None, :] - means[None, present]
    d_va = np.sqrt(np.maximum(np.einsum("nck,kl,ncl->nc", diff, precision, diff).min(1), 0))

    # Export next to a copy of the gatekeeper and the shipped metadata
    import onnxruntime as ort
    import tf2onnx
    out.mkdir(parents=True, exist_ok=True)
    feat_model.output_names = ["logits", "features"]
    spec = (tf.TensorSpec((None, size, size, 3), tf.float32, name="image"),)
    tf2onnx.convert.from_keras(feat_model, input_signature=spec, opset=17, output_path=str(out / "bloodcell_cnn.onnx"))
    sess = ort.InferenceSession(str(out / "bloodcell_cnn.onnx"), providers=["CPUExecutionProvider"])
    probe = X[te][:16].astype("float32") / 255.0 if len(te) else Xrv[:16].astype("float32") / 255.0
    assert (sess.run(None, {"image": probe})[0].argmax(1) == feat_model(probe, training=False)[0].numpy().argmax(1)).all()
    W, b = model.get_layer("logits").get_weights()
    np.savez_compressed(out / "model_aux.npz", cam_weights=W.astype("float32"), cam_bias=b.astype("float32"),
                        ood_means=means.astype("float32"), ood_precision=precision.astype("float32"),
                        val_distances=np.sort(d_va).astype("float32"))
    shipped = ROOT / "models"
    shutil.copy(shipped / "gate.onnx", out / "gate.onnx")
    meta = json.loads((shipped / "model_meta.json").read_text())
    meta.update({"temperature": T, "confidence_threshold": conf_threshold,
                 "ood_threshold": float(np.percentile(d_va, 99)),
                 "adapted": {"name": args.name, "local_cells": int(len(y)), "local_train": int(len(tr)),
                             "local_validation": int(len(va)), "local_test": int(len(te)), "epochs": args.epochs,
                             "minutes": round(minutes, 1), "before": before, "after": after, "history": log}})
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2))
    (out / "adaptation_report.json").write_text(json.dumps(meta["adapted"], indent=2))
    print(f"\nSaved the adapted model to {out}")
    print(f"Try it:  BLOODSMEAR_MODEL_DIR={out} streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
