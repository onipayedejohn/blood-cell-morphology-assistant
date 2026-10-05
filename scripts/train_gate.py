"""Train and test the gatekeeper: is this image a stained white blood cell at all?

Three classes:
  0 white_cell     a cropped white cell from a stained smear (any lab)
  1 smear_no_wbc   a stained smear with red cells but no white cell in view
  2 not_smear      anything else: photos, paper, documents, screens, noise

The app only classifies the cell type when the gatekeeper says white_cell with
enough probability. The acceptance level is set on validation data from the
non-blood side: the lowest level at which no more than 0.5% of validation
images without a white cell (smear patches and non-blood images) would pass.
A first rule (let 99% of validation cells pass) hit its 0.9 ceiling and
refused 26% of cells from a lab the gatekeeper never saw, so it was replaced.
Test sets include image types the gatekeeper never saw in training.

Usage:
    python scripts/train_gate.py --name gate_final
    python scripts/train_gate.py --holdout bccd --name gate_lolo_bccd   # never sees BCCD
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from augment import strong_augment_batch  # noqa: E402
from train import precise_bn, to_float  # noqa: E402

layers = tf.keras.layers
SEED = 42
GATE_CLASSES = ["white_cell", "smear_no_wbc", "not_smear"]
P = ROOT / "data" / "processed"


def build_gate(size=64, widths=(16, 32, 64, 128)):
    inp = layers.Input((size, size, 3), name="image")
    x = inp
    for i, w in enumerate(widths):
        x = layers.Conv2D(w, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(w, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        if i < len(widths) - 1:
            x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    return tf.keras.Model(inp, layers.Dense(3, name="gate_logits")(x), name="gate")


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def split_off(X, frac, rng):
    idx = rng.permutation(len(X))
    n = int(len(X) * frac)
    return X[idx[n:]], X[idx[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--name", default="gate_final")
    args = ap.parse_args()
    tf.keras.utils.set_random_seed(SEED)
    rng = np.random.default_rng(SEED)

    pbc = np.load(P / "pbc_64.npz", allow_pickle=True)
    ext = np.load(P / "external_64.npz", allow_pickle=True)
    gate = np.load(P / "gate_64.npz", allow_pickle=True)

    # Positives: PBC training cells (a sample) plus external training halves
    pbc_tr = pbc["X"][pbc["split"] == "train"]
    pbc_tr = pbc_tr[rng.choice(len(pbc_tr), 3000, replace=False)]
    keep = (ext["split"] == "train") & (ext["lab"] != (args.holdout or ""))
    ext_tr = ext["X"][keep]
    ext_tr, ext_va = split_off(ext_tr, 0.2, rng)
    white_tr = np.concatenate([pbc_tr] + [ext_tr] * 6)
    white_va = np.concatenate([pbc["X"][pbc["split"] == "val"][:600], ext_va])

    smear_all = gate["smear_train"]
    if args.holdout:  # a held-out lab contributes nothing, not even background patches
        smear_all = smear_all[gate["smear_train_source"] != args.holdout]
    smear_tr, smear_va = split_off(smear_all, 0.15, rng)
    neg_tr, neg_va = split_off(gate["neg_train"], 0.15, rng)

    X = np.concatenate([white_tr, smear_tr, neg_tr])
    y = np.concatenate([np.zeros(len(white_tr)), np.ones(len(smear_tr)), np.full(len(neg_tr), 2)]).astype("int32")
    Xv = np.concatenate([white_va, smear_va, neg_va])
    yv = np.concatenate([np.zeros(len(white_va)), np.ones(len(smear_va)), np.full(len(neg_va), 2)]).astype("int32")
    cw = compute_class_weight("balanced", classes=np.arange(3), y=y)
    print(f"gate training: white {len(white_tr)}, smear {len(smear_tr)}, not smear {len(neg_tr)}", flush=True)

    ds = (tf.data.Dataset.from_tensor_slices((X, y)).shuffle(len(X), seed=SEED)
          .map(to_float, num_parallel_calls=tf.data.AUTOTUNE).batch(64)
          .map(strong_augment_batch, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE))
    model = build_gate()
    steps = int(np.ceil(len(X) / 64)) * args.epochs
    model.compile(tf.keras.optimizers.AdamW(tf.keras.optimizers.schedules.CosineDecay(2e-3, steps, alpha=0.02), weight_decay=1e-4),
                  tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    t0 = time.time()
    model.fit(ds, epochs=args.epochs, class_weight={i: float(w) for i, w in enumerate(cw)}, verbose=2)
    precise_bn(model, X[rng.choice(len(X), 2048, replace=False)])
    minutes = (time.time() - t0) / 60

    def probs(A):
        out = []
        for i in range(0, len(A), 256):
            out.append(softmax(model(A[i:i + 256].astype("float32") / 255.0, training=False).numpy()))
        return np.concatenate(out) if out else np.zeros((0, 3))

    pv = probs(Xv)
    p_white_other = pv[yv != 0, 0]
    tau = float(np.quantile(p_white_other, 0.995))  # at most 0.5% of non-cells pass
    tau = min(max(tau, 0.3), 0.9)
    np.savez_compressed(ROOT / "models" / "candidates" / f"{args.name}_val_probs.npz", p=pv, y=yv)

    def accept_rate(A):
        p = probs(A)
        return float((p[:, 0] >= tau).mean()) if len(A) else None, p

    results = {"threshold_p_white": tau, "train_minutes": round(minutes, 1),
               "validation_accuracy": float((pv.argmax(1) == yv).mean()),
               "validation_cells_accepted": float((pv[yv == 0, 0] >= tau).mean()),
               "validation_non_cells_accepted": float((p_white_other >= tau).mean())}
    cells = {"pbc_test": pbc["X"][pbc["split"] == "test"]}
    for lab in ["bccd", "jtsc", "cvblog"]:
        m = ext["lab"] == lab
        cells[f"{lab}_test_half"] = ext["X"][m & (ext["split"] == "test")]
        if lab == args.holdout:
            cells[f"{lab}_all_unseen_lab"] = ext["X"][m]
    results["white_cells_accepted"] = {k: accept_rate(v)[0] for k, v in cells.items()}
    a, p = accept_rate(gate["smear_test"])
    results["smear_patches_without_white_cell"] = {"accepted_as_white_cell": a,
                                                   "called_smear_no_wbc": float((p.argmax(1) == 1).mean())}
    results["not_smear_accepted_as_white_cell"] = {k.replace("negtest_", ""): accept_rate(gate[k])[0]
                                                   for k in gate.files if k.startswith("negtest_")}

    out = ROOT / "models" / "candidates"
    model.save(out / f"{args.name}.keras")
    (out / f"{args.name}_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
