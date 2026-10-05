"""Train one CNN candidate on the prepared PBC arrays.

The architecture ends in global average pooling followed by one dense layer.
That shape lets the app compute a class activation map (Zhou et al., 2016)
from a single forward pass, so the heatmap needs no gradients at run time.

Usage (from the project root):
    python scripts/train.py --size 64  --widths 32 64 128 256
    python scripts/train.py --size 112 --widths 24 48 96 192
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
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bloodsmear.config import CLASSES  # noqa: E402
from augment import classifier_augment_batch  # noqa: E402

EXTERNAL_LABS = ["bccd", "jtsc", "cvblog"]

layers = tf.keras.layers
SEED = 42


def build_model(size: int, widths: list[int], n_classes: int = len(CLASSES), dropout: float = 0.3):
    """Four conv blocks (two 3x3 convs each), max pooling between blocks, GAP head."""
    inputs = layers.Input((size, size, 3), name="image")
    x = inputs
    for i, w in enumerate(widths):
        for j in range(2):
            x = layers.Conv2D(w, 3, padding="same", use_bias=False, name=f"b{i}_conv{j}")(x)
            x = layers.BatchNormalization(name=f"b{i}_bn{j}")(x)
            x = layers.ReLU(name=f"b{i}_relu{j}")(x)
        if i < len(widths) - 1:
            x = layers.MaxPooling2D(name=f"b{i}_pool")(x)
    features = x  # (size/8, size/8, widths[-1]) feature maps used for the heatmap
    x = layers.GlobalAveragePooling2D(name="gap")(features)
    x = layers.Dropout(dropout, name="dropout")(x)
    logits = layers.Dense(n_classes, name="logits")(x)
    return tf.keras.Model(inputs, logits, name=f"cnn_{size}")


def build_mobilenet(size: int, weights: str | None, n_classes: int = len(CLASSES), dropout: float = 0.3,
                    cut: str = "conv_pw_11_relu", freeze_until: str = "conv_pw_5_relu"):
    """MobileNet (v1, width 1.0) pretrained on ImageNet, cut after block 11.

    Cutting there drops the two widest blocks and keeps a finer feature map for
    the heatmap (6 x 6 at 96 px input). Blocks up to `freeze_until` keep their
    ImageNet weights: generic edge and texture filters, which also makes each
    training step cheaper on a CPU.
    """
    inputs = layers.Input((size, size, 3), name="image")
    x = layers.Rescaling(2.0, -1.0, name="to_unit_range")(inputs)   # MobileNet expects [-1, 1]
    base = tf.keras.applications.MobileNet(input_tensor=x, include_top=False, weights=weights, alpha=1.0)
    if weights and freeze_until:
        for layer in base.layers:
            layer.trainable = False
            if layer.name == freeze_until:
                break
    features = layers.Activation("linear", name="features")(base.get_layer(cut).output)
    x = layers.GlobalAveragePooling2D(name="gap")(features)
    x = layers.Dropout(dropout, name="dropout")(x)
    logits = layers.Dense(n_classes, name="logits")(x)
    return tf.keras.Model(inputs, logits, name=f"mobilenet_{size}")


def augment(img, label):
    """Cells have no fixed orientation, and stain intensity varies between slides."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    img = tf.image.random_brightness(img, 0.08)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    img = tf.image.random_saturation(img, 0.9, 1.1)
    img = tf.image.random_hue(img, 0.02)
    return tf.clip_by_value(img, 0.0, 1.0), label


def to_float(img, label):
    return tf.cast(img, tf.float32) / 255.0, label


def make_ds(X, y, batch, training, aug="strong"):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(len(X), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(to_float, num_parallel_calls=tf.data.AUTOTUNE)
    if training and aug != "strong":
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch)
    if training and aug == "strong":
        ds = ds.map(classifier_augment_batch, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def external_splits(labs, holdout, size, seed=SEED, suffix=""):
    """External cells for training and validation.

    Each lab's train half is split again: 75% to train, 25% to validation. The
    test half is never touched here. A held-out lab is left out completely.
    """
    p = ROOT / "data" / "processed" / f"external_{size}{suffix}.npz"
    if not labs or not p.exists():
        return (np.zeros((0, size, size, 3), np.uint8), np.zeros(0, "int32"), np.zeros(0, object)), \
               (np.zeros((0, size, size, 3), np.uint8), np.zeros(0, "int32"), np.zeros(0, object))
    d = np.load(p, allow_pickle=True)
    X, y, lab, split = d["X"], d["y"].astype("int32"), d["lab"], d["split"]
    rng = np.random.default_rng(seed)
    tr_idx, va_idx = [], []
    for L in labs:
        if L == holdout:
            continue
        for c in np.unique(y[lab == L]):
            idx = np.nonzero((lab == L) & (y == c) & (split == "train"))[0]
            idx = rng.permutation(idx)
            n_va = int(round(len(idx) * 0.25))
            va_idx += list(idx[:n_va])
            tr_idx += list(idx[n_va:])
    tr_idx, va_idx = np.array(tr_idx, int), np.array(va_idx, int)
    return (X[tr_idx], y[tr_idx], lab[tr_idx]), (X[va_idx], y[va_idx], lab[va_idx])


def precise_bn(model, X_clean: np.ndarray, batch: int = 128):
    """Recompute batch-norm running averages on clean (un-augmented) images.

    During training the running averages are collected on augmented batches
    while the weights are still moving. In the first run (epoch 4) they drifted
    far enough that inference-mode validation macro-F1 was 0.10, while the same
    weights scored 0.96 after this step (figures from the debugging session). Averaging exact per-batch statistics over clean
    training images ("precise BN", Wu & Johnson 2021) fixes the mismatch.
    """
    bns = [l for l in model.layers if isinstance(l, layers.BatchNormalization)]
    saved = [l.momentum for l in bns]
    for l in bns:
        l.momentum = 0.0  # the running value becomes exactly this batch's statistic
    sums = {l.name: [0.0, 0.0] for l in bns}
    n = 0
    for i in range(0, len(X_clean), batch):
        model(tf.cast(X_clean[i:i + batch], tf.float32) / 255.0, training=True)
        for l in bns:
            sums[l.name][0] += l.moving_mean.numpy()
            sums[l.name][1] += l.moving_variance.numpy()
        n += 1
    for l, m in zip(bns, saved):
        l.moving_mean.assign(sums[l.name][0] / n)
        l.moving_variance.assign(sums[l.name][1] / n)
        l.momentum = m


class MacroF1Checkpoint(tf.keras.callbacks.Callback):
    """Keep the weights with the best validation macro-F1 and stop when it stalls."""

    def __init__(self, val_ds, y_val, path, patience, X_bn, ext_val=None):
        super().__init__()
        self.val_ds, self.y_val, self.path, self.patience = val_ds, y_val, path, patience
        self.X_bn = X_bn
        self.ext_val = ext_val  # (dataset, labels) from other labs, or None
        self.best, self.wait, self.log = -1.0, 0, []

    def on_epoch_end(self, epoch, logs=None):
        precise_bn(self.model, self.X_bn)
        logits = self.model.predict(self.val_ds, verbose=0)
        pred = logits.argmax(1)
        f1 = f1_score(self.y_val, pred, average="macro")
        logs = dict(logs or {})
        logs["val_macro_f1"] = float(f1)
        logs["val_loss"] = float(tf.keras.losses.sparse_categorical_crossentropy(
            self.y_val, logits, from_logits=True).numpy().mean())
        logs["val_accuracy"] = float((pred == self.y_val).mean())
        score = f1
        if self.ext_val is not None:
            ext_pred = self.model.predict(self.ext_val[0], verbose=0).argmax(1)
            logs["ext_val_accuracy"] = float((ext_pred == self.ext_val[1]).mean())
            score = 0.5 * f1 + 0.5 * logs["ext_val_accuracy"]
        logs["selection_score"] = float(score)
        f1 = score
        self.log.append({k: float(v) for k, v in logs.items()} | {"epoch": epoch + 1})
        print(f"epoch {epoch + 1}: loss {logs['loss']:.4f}  val_loss {logs.get('val_loss', float('nan')):.4f}  "
              f"val_macro_f1 {logs['val_macro_f1']:.4f}  ext_val_acc {logs.get('ext_val_accuracy', float('nan')):.4f}  "
              f"score {f1:.4f}", flush=True)
        if f1 > self.best:
            self.best, self.wait = f1, 0
            self.model.save(self.path)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--widths", type=int, nargs=4, default=[24, 48, 96, 192])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--labs", nargs="*", default=EXTERNAL_LABS, help="external labs to train on")
    ap.add_argument("--holdout", default=None, help="external lab to leave out entirely")
    ap.add_argument("--aug", choices=["basic", "strong"], default="strong")
    ap.add_argument("--ext-repeat", type=int, default=10, help="how often each external cell is repeated per epoch")
    ap.add_argument("--name", default=None)
    ap.add_argument("--framing", choices=["center", "canonical"], default="canonical",
                    help="canonical: arrays from prepare_canonical.py (standard framing and color balance)")
    ap.add_argument("--arch", choices=["cnn", "mobilenet"], default="cnn")
    ap.add_argument("--pretrained", default=None, help="ImageNet weights file for --arch mobilenet")
    ap.add_argument("--init", default=None, help="start from these weights (a .keras file of the same architecture)")
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    suffix = "c" if args.framing == "canonical" else ""
    d = np.load(ROOT / "data" / "processed" / f"pbc_{args.size}{suffix}.npz", allow_pickle=True)
    X, y, split = d["X"], d["y"].astype("int32"), d["split"]
    Xtr, ytr = X[split == "train"], y[split == "train"]
    Xva, yva = X[split == "val"], y[split == "val"]
    (Xe, ye, le), (Xev, yev, lev) = external_splits(args.labs, args.holdout, args.size, suffix=suffix)
    n_ext_train = len(Xe)
    if len(Xe):
        Xtr = np.concatenate([Xtr] + [Xe] * args.ext_repeat)
        ytr = np.concatenate([ytr] + [ye] * args.ext_repeat)
    print(f"training cells: PBC {int((split == 'train').sum())}, external {n_ext_train} x {args.ext_repeat}; "
          f"external validation {len(Xev)} ({sorted(set(lev))})", flush=True)

    weights = compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
    class_weight = {int(c): float(w) for c, w in zip(np.unique(ytr), weights)}

    model = (build_mobilenet(args.size, str(ROOT / args.pretrained) if args.pretrained else None)
             if args.arch == "mobilenet" else build_model(args.size, args.widths))
    if args.init:  # e.g. the version 1 model, which saw PBC training cells only
        model.set_weights(tf.keras.models.load_model(ROOT / args.init, compile=False).get_weights())
        print(f"initialized from {args.init}", flush=True)
    steps = int(np.ceil(len(Xtr) / args.batch)) * args.epochs
    schedule = tf.keras.optimizers.schedules.CosineDecay(args.lr, decay_steps=steps, alpha=0.02)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=schedule, weight_decay=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    out = ROOT / "models" / "candidates"
    out.mkdir(parents=True, exist_ok=True)
    name = args.name or f"cnn_{args.size}"
    bn_idx = np.random.default_rng(SEED).choice(len(Xtr), size=min(2048, len(Xtr)), replace=False)
    ext_val = (make_ds(Xev, yev, 256, False), yev) if len(Xev) else None
    ckpt = MacroF1Checkpoint(make_ds(Xva, yva, 256, False), yva, out / f"{name}.keras", args.patience,
                             Xtr[bn_idx], ext_val)

    from codecarbon import OfflineEmissionsTracker
    tracker = OfflineEmissionsTracker(country_iso_code="GHA", project_name=name, log_level="error",
                                      save_to_file=False)
    tracker.start()
    t0 = time.time()
    model.fit(make_ds(Xtr, ytr, args.batch, True, args.aug),
              epochs=args.epochs, class_weight=class_weight, callbacks=[ckpt], verbose=0)
    minutes = (time.time() - t0) / 60
    kg = tracker.stop()
    energy_kwh = float(tracker.final_emissions_data.energy_consumed)

    info = {
        "name": name, "size": args.size, "widths": args.widths, "params": int(model.count_params()),
        "epochs_run": len(ckpt.log), "best_selection_score": ckpt.best, "train_minutes": round(minutes, 1),
        "train_energy_wh": round(energy_kwh * 1000, 2), "train_co2_g": round(float(kg) * 1000, 2),
        "lr": args.lr, "batch": args.batch, "class_weight": class_weight, "aug": args.aug,
        "labs": [l for l in args.labs if l != args.holdout], "holdout": args.holdout, "ext_repeat": args.ext_repeat,
        "n_external_train": n_ext_train, "init": args.init, "arch": args.arch, "pretrained": args.pretrained, "framing": args.framing, "history": ckpt.log,
    }
    (out / f"{name}_train.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
