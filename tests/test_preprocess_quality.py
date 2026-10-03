import numpy as np
from PIL import Image

from bloodsmear.explain import class_activation_map, overlay
from bloodsmear.preprocess import center_square, resize_uint8, to_model_input
from bloodsmear.quality import check_image


def colour_image(w=200, h=200, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.zeros((h, w, 3), np.uint8)
    arr[..., 0] = rng.integers(150, 230, (h, w))
    arr[..., 1] = rng.integers(90, 160, (h, w))
    arr[..., 2] = rng.integers(140, 220, (h, w))
    return Image.fromarray(arr)


def test_center_square_takes_the_middle():
    img = Image.new("RGB", (360, 363))
    sq = center_square(img)
    assert sq.size == (360, 360)


def test_resize_gives_uint8_square():
    arr = resize_uint8(colour_image(300, 240), 112)
    assert arr.shape == (112, 112, 3) and arr.dtype == np.uint8


def test_model_input_is_scaled_with_batch_axis():
    x = to_model_input(np.full((112, 112, 3), 255, np.uint8))
    assert x.shape == (1, 112, 112, 3) and x.dtype == np.float32
    assert x.max() == 1.0


def test_transparent_png_is_put_on_white():
    rgba = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    arr = resize_uint8(rgba, 64)
    assert arr.mean() == 255


def test_tiny_image_is_blocked():
    issues = check_image(colour_image(40, 40))
    assert issues and issues[0].level == "block"


def test_normal_colour_image_passes():
    assert check_image(colour_image()) == []


def test_greyscale_dark_and_flat_images_warn():
    grey = Image.fromarray(np.full((200, 200, 3), 120, np.uint8))
    msgs = " ".join(i.message for i in check_image(grey))
    assert "grayscale" in msgs and "contrast" in msgs
    dark = Image.fromarray(np.full((200, 200, 3), 10, np.uint8))
    assert any("dark" in i.message for i in check_image(dark))


def test_long_thin_image_warns():
    assert any("square" in i.message for i in check_image(colour_image(400, 150)))


def test_cam_is_normalised_and_overlay_keeps_shape():
    rng = np.random.default_rng(1)
    feats = rng.random((14, 14, 8)).astype(np.float32)
    weights = rng.normal(size=(8, 7)).astype(np.float32)
    cam = class_activation_map(feats, weights, 3)
    assert cam.shape == (14, 14) and cam.min() >= 0 and np.isclose(cam.max(), 1)
    out = overlay(np.zeros((112, 112, 3), np.uint8), cam)
    assert out.shape == (112, 112, 3) and out.dtype == np.uint8


def test_cam_with_no_positive_evidence_is_zero():
    cam = class_activation_map(np.ones((4, 4, 2), np.float32), -np.ones((2, 7), np.float32), 0)
    assert cam.max() == 0
