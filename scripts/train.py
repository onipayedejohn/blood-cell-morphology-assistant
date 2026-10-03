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


def make_ds(X, y, batch, training):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(len(X), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(to_float, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


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

    def __init__(self, val_ds, y_val, path, patience, X_bn):
        super().__init__()
        self.val_ds, self.y_val, self.path, self.patience = val_ds, y_val, path, patience
        self.X_bn = X_bn
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
        self.log.append({k: float(v) for k, v in logs.items()} | {"epoch": epoch + 1})
        print(f"epoch {epoch + 1}: loss {logs['loss']:.4f}  val_loss {logs.get('val_loss', float('nan')):.4f}  "
              f"val_macro_f1 {f1:.4f}", flush=True)
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
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    d = np.load(ROOT / "data" / "processed" / f"pbc_{args.size}.npz", allow_pickle=True)
    X, y, split = d["X"], d["y"].astype("int32"), d["split"]
    Xtr, ytr = X[split == "train"], y[split == "train"]
    Xva, yva = X[split == "val"], y[split == "val"]

    weights = compute_class_weight("balanced", classes=np.arange(len(CLASSES)), y=ytr)
    class_weight = {i: float(w) for i, w in enumerate(weights)}

    model = build_model(args.size, args.widths)
    steps = int(np.ceil(len(Xtr) / args.batch)) * args.epochs
    schedule = tf.keras.optimizers.schedules.CosineDecay(args.lr, decay_steps=steps, alpha=0.02)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=schedule, weight_decay=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    out = ROOT / "models" / "candidates"
    out.mkdir(parents=True, exist_ok=True)
    name = f"cnn_{args.size}"
    bn_idx = np.random.default_rng(SEED).choice(len(Xtr), size=min(2048, len(Xtr)), replace=False)
    ckpt = MacroF1Checkpoint(make_ds(Xva, yva, 256, False), yva, out / f"{name}.keras", args.patience,
                             Xtr[bn_idx])

    from codecarbon import OfflineEmissionsTracker
    tracker = OfflineEmissionsTracker(country_iso_code="GHA", project_name=name, log_level="error",
                                      save_to_file=False)
    tracker.start()
    t0 = time.time()
    model.fit(make_ds(Xtr, ytr, args.batch, True),
              epochs=args.epochs, class_weight=class_weight, callbacks=[ckpt], verbose=0)
    minutes = (time.time() - t0) / 60
    kg = tracker.stop()
    energy_kwh = float(tracker.final_emissions_data.energy_consumed)

    info = {
        "name": name, "size": args.size, "widths": args.widths, "params": int(model.count_params()),
        "epochs_run": len(ckpt.log), "best_val_macro_f1": ckpt.best, "train_minutes": round(minutes, 1),
        "train_energy_wh": round(energy_kwh * 1000, 2), "train_co2_g": round(float(kg) * 1000, 2),
        "lr": args.lr, "batch": args.batch, "class_weight": class_weight, "history": ckpt.log,
    }
    (out / f"{name}_train.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
