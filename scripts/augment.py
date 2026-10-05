"""Training-time augmentation that imitates other labs, stains and cameras.

Version 1 only flipped, rotated and nudged colors, and it failed completely on
other labs (0% to 30% accuracy). These transforms imitate the ways images from
another laboratory differ:

- stain: Romanowsky stains vary in strength and balance, so the image is
  converted to optical density, split into stain components with the standard
  color-deconvolution matrix (Ruifrok & Johnston 2001), and each component is
  scaled and shifted at random;
- white balance, brightness, contrast, saturation, hue and gamma (lamp, camera);
- zoom 0.7x to 1.35x and free rotation (magnification and how the cell was cropped);
- uneven illumination (phone through an eyepiece, dirty optics);
- blur, lower resolution, sensor noise and JPEG compression (focus and camera);
- for the classifier only, flat gray borders like the padding the standard
  framing adds to tightly cropped cells (see bloodsmear/canonical.py), so that
  padding is not a clue to which lab an image came from.

Everything runs on whole batches with per-image random settings, which is
several times faster on a CPU than augmenting one image at a time.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

# Color deconvolution (haematoxylin-like, eosin-like, residual), from scikit-image's rgb_from_hed
RGB_FROM_STAIN = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]], dtype=np.float32)
STAIN_FROM_RGB = np.linalg.inv(RGB_FROM_STAIN).astype(np.float32)

_geometry = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.5, fill_mode="reflect"),
    tf.keras.layers.RandomZoom((-0.35, 0.3), fill_mode="reflect"),
    tf.keras.layers.RandomTranslation(0.06, 0.06, fill_mode="reflect"),
])


def _u(shape, lo, hi):
    return tf.random.uniform(shape, lo, hi)


def _where(p, b, new, old):
    """Per-image choice between augmented and original."""
    m = tf.cast(tf.random.uniform((b, 1, 1, 1)) < p, tf.float32)
    return m * new + (1 - m) * old


def stain_jitter(x, strength=0.3, shift=0.05):
    b = tf.shape(x)[0]
    od = -tf.math.log(tf.clip_by_value(x, 1 / 255, 1.0))
    st = tf.einsum("bhwc,cd->bhwd", od, STAIN_FROM_RGB)
    st = st * _u((b, 1, 1, 3), 1 - strength, 1 + strength) + _u((b, 1, 1, 3), -shift, shift)
    return tf.exp(-tf.maximum(tf.einsum("bhwc,cd->bhwd", st, RGB_FROM_STAIN), 0.0))


def blur(x, sigma):
    r = 3
    t = tf.range(-r, r + 1, dtype=tf.float32)
    k = tf.exp(-(t ** 2) / (2 * sigma ** 2))
    k = k / tf.reduce_sum(k)
    kx = tf.tile(tf.reshape(k, (1, -1, 1, 1)), (1, 1, 3, 1))
    ky = tf.tile(tf.reshape(k, (-1, 1, 1, 1)), (1, 1, 3, 1))
    return tf.nn.depthwise_conv2d(tf.nn.depthwise_conv2d(x, kx, [1, 1, 1, 1], "SAME"), ky, [1, 1, 1, 1], "SAME")


PAD_GREY = 0.92  # bloodsmear.canonical.BACKGROUND


def pad_borders(x, p=0.5, max_frac=0.3):
    """Replace a random strip on each side with flat background gray, per image."""
    b, size = tf.shape(x)[0], tf.shape(x)[1]
    n = tf.cast(size, tf.float32)
    on = tf.cast(tf.random.uniform((b, 4)) < p, tf.float32)
    m = on * tf.random.uniform((b, 4), 0, max_frac) * n          # left, right, top, bottom in pixels
    r = tf.range(n)
    cols = (r[None, :] >= m[:, 0:1]) & (r[None, :] < n - m[:, 1:2])
    rows = (r[None, :] >= m[:, 2:3]) & (r[None, :] < n - m[:, 3:4])
    keep = tf.cast(rows[:, :, None] & cols[:, None, :], tf.float32)[..., None]
    return keep * x + (1 - keep) * PAD_GREY


def classifier_augment_batch(x, y):
    """Strong augmentation for 75% of images; the rest are only flipped and rotated,
    so the classifier also keeps seeing clean cells like the ones it is judged on."""
    b = tf.shape(x)[0]
    strong, _ = strong_augment_batch(pad_borders(x), y)
    mild = tf.image.rot90(tf.image.random_flip_left_right(x), k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    return _where(0.75, b, strong, mild), y


def strong_augment_batch(x, y):
    """x: float32 (B, H, W, 3) in [0, 1]."""
    b, size = tf.shape(x)[0], tf.shape(x)[1]
    x = tf.image.random_flip_left_right(x)
    x = tf.image.rot90(x, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    x = _geometry(x, training=True)
    x = _where(0.8, b, stain_jitter(x), x)
    x = x * _u((b, 1, 1, 3), 0.88, 1.12)                                     # white balance
    x = x + _u((b, 1, 1, 1), -0.12, 0.12)                                    # brightness
    mean = tf.reduce_mean(x, axis=(1, 2, 3), keepdims=True)
    x = (x - mean) * _u((b, 1, 1, 1), 0.75, 1.25) + mean                     # contrast
    gray = tf.reduce_mean(x, axis=3, keepdims=True)
    x = gray + (x - gray) * _u((b, 1, 1, 1), 0.6, 1.4)                       # saturation
    x = tf.image.adjust_hue(tf.clip_by_value(x, 0, 1), tf.random.uniform([], -0.04, 0.04))
    x = tf.clip_by_value(x, 1e-4, 1.0) ** _u((b, 1, 1, 1), 0.75, 1.35)       # gamma
    yy, xx = tf.meshgrid(tf.linspace(-1.0, 1.0, size), tf.linspace(-1.0, 1.0, size), indexing="ij")
    cx, cy = _u((b, 1, 1, 1), -0.6, 0.6), _u((b, 1, 1, 1), -0.6, 0.6)
    light = 1 - _u((b, 1, 1, 1), 0.05, 0.35) * ((xx[None, ..., None] - cx) ** 2 + (yy[None, ..., None] - cy) ** 2)
    x = _where(0.35, b, x * tf.clip_by_value(light, 0.45, 1.0), x)           # uneven illumination
    x = _where(0.35, b, blur(x, tf.random.uniform([], 0.4, 1.1)), x)        # focus
    s = tf.cast(tf.cast(size, tf.float32) * tf.random.uniform([], 0.45, 0.85), tf.int32)
    x = _where(0.3, b, tf.image.resize(tf.image.resize(x, (s, s)), (size, size)), x)  # lower resolution
    x = _where(0.4, b, x + tf.random.normal(tf.shape(x)) * _u((b, 1, 1, 1), 0.005, 0.035), x)  # sensor noise
    x = tf.clip_by_value(x, 0.0, 1.0)

    def jpeg_all(z):
        q = tf.random.uniform([], 35, 95, dtype=tf.int32)
        return tf.map_fn(lambda im: tf.image.adjust_jpeg_quality(im, q), z)

    x = tf.cond(tf.random.uniform(()) < 0.4, lambda: jpeg_all(x), lambda: x)  # compression, whole batch
    return tf.reshape(tf.clip_by_value(x, 0.0, 1.0), (b, size, size, 3)), y


def strong_augment(img, label):
    """Single-image wrapper (used by the preview)."""
    out, lab = strong_augment_batch(img[None], label)
    return out[0], lab


def preview_grid(X: np.ndarray, n: int = 8, seed: int = 0) -> np.ndarray:
    """Rows of original then augmented copies, for checking the transforms by eye."""
    tf.random.set_seed(seed)
    rows = []
    for x in X[:n]:
        f = x.astype("float32") / 255
        aug = strong_augment_batch(tf.constant(np.repeat(f[None], 7, 0)), 0)[0].numpy()
        rows.append(np.concatenate([f] + list(aug), axis=1))
    return (np.concatenate(rows, axis=0) * 255).astype(np.uint8)
