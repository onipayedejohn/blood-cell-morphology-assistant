"""Checks on the prepared data and the exported model.

Tests that need the trained model or the raw download skip themselves when
those files are absent, so the suite still runs on a fresh clone.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from bloodsmear.config import CLASSES, MODEL_DIR

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
has_model = (MODEL_DIR / "bloodcell_cnn.onnx").exists()
needs_model = pytest.mark.skipif(not has_model, reason="trained model not present")


@pytest.fixture(scope="module")
def manifest():
    return pd.read_csv(MANIFEST)


def test_no_duplicate_group_crosses_splits(manifest):
    splits_per_group = manifest.groupby("group")["split"].nunique()
    assert (splits_per_group == 1).all()
    splits_per_md5 = manifest.groupby("md5")["split"].nunique()
    assert (splits_per_md5 == 1).all()


def test_labels_match_filename_prefixes(manifest):
    prefix_class = {"BA": 0, "EO": 1, "ERB": 2, "PMY": 3, "MY": 3, "MMY": 3, "IG": 3,
                    "LY": 4, "MO": 5, "SNE": 6, "BNE": 6, "NEUTROPHIL": 6}
    assert (manifest["file"].str.split("_").str[0].map(prefix_class) == manifest["label"]).all()


def test_split_is_roughly_70_10_20_and_stratified(manifest):
    share = manifest["split"].value_counts(normalize=True)
    assert abs(share["train"] - 0.7) < 0.02 and abs(share["val"] - 0.1) < 0.02 and abs(share["test"] - 0.2) < 0.02
    by_class = pd.crosstab(manifest["class"], manifest["split"], normalize="index")
    assert (abs(by_class["test"] - 0.2) < 0.03).all()


def test_app_samples_and_demo_come_from_test_split(manifest):
    split_of = manifest.set_index("file")["split"]
    samples = pd.read_csv(ROOT / "data" / "samples" / "samples.csv")
    demo = pd.read_csv(ROOT / "data" / "demo_batch" / "demo_batch_labels.csv")
    assert (split_of[samples["source"]] == "test").all()
    assert (split_of[demo["source"]] == "test").all()
    assert not set(samples["source"]) & set(demo["source"])


@pytest.mark.skipif(not (ROOT / "data" / "processed" / "pbc_64.npz").exists()
                    or not (ROOT / "data" / "raw" / "pbc").exists(), reason="processed arrays or raw data absent")
@pytest.mark.parametrize("size", [64, 112])
def test_app_preprocessing_matches_training_arrays(manifest, size):
    """Only runs where the raw download and processed arrays exist (they are not in git)."""
    from bloodsmear.preprocess import resize_uint8
    d = np.load(ROOT / "data" / "processed" / f"pbc_{size}.npz", allow_pickle=True)
    for i in np.random.default_rng(0).choice(len(manifest), 25, replace=False):
        raw = Image.open(ROOT / "data" / "raw" / manifest.loc[i, "path"])
        assert np.array_equal(resize_uint8(raw, size), d["X"][i])


def test_deployed_model_input_size_matches_training():
    import json
    meta = json.loads((ROOT / "models" / "model_meta.json").read_text())
    from bloodsmear.preprocess import resize_uint8
    arr = resize_uint8(Image.open(ROOT / "data" / "samples" / "basophil_1.jpg"), meta["input_size"])
    assert arr.shape == (meta["input_size"], meta["input_size"], 3)


@pytest.fixture(scope="module")
def clf():
    from bloodsmear.inference import CellClassifier
    return CellClassifier()


@needs_model
def test_probabilities_are_valid_and_repeatable(clf):
    img = Image.open(ROOT / "data" / "samples" / "neutrophil_1.jpg")
    a, b = clf.predict(img), clf.predict(img)
    assert abs(sum(a.probs.values()) - 1) < 1e-4
    assert a.label == b.label and abs(a.confidence - b.confidence) < 1e-6
    assert a.label in CLASSES and 0 <= a.unfamiliarity <= 100


@needs_model
def test_batch_matches_single(clf):
    imgs = [Image.open(p) for p in sorted((ROOT / "data" / "samples").glob("*_1.jpg"))]
    many = clf.predict_many(imgs, batch=4)
    for img, m in zip(imgs, many):
        one = clf.predict(img)
        assert one.label == m.label and abs(one.confidence - m.confidence) < 1e-4


@needs_model
def test_sample_cells_are_mostly_right(clf):
    s = pd.read_csv(ROOT / "data" / "samples" / "samples.csv")
    preds = clf.predict_many([Image.open(ROOT / "data" / "samples" / f) for f in s["file"]])
    right = sum(p.label == c for p, c in zip(preds, s["class"]))
    assert right >= len(s) - 3


@needs_model
def test_noise_is_not_reported_as_confident(clf):
    p = clf.predict(Image.open(ROOT / "data" / "samples" / "other" / "random_noise.png"))
    assert p.status != "confident"


@needs_model
def test_thresholds_are_sane(clf):
    assert 0.3 < clf.conf_threshold < 1
    assert clf.ood_threshold > 0 and 0.2 < clf.temperature < 5
    assert clf.cam_weights.shape[1] == len(CLASSES)


# -- version 2: gatekeeper, cell finder, external labs ------------------------------------------

OTHER = ROOT / "data" / "samples" / "other"


@needs_model
@pytest.mark.parametrize("name", ["black_a4_sheet.jpg", "printed_document.jpg", "everyday_photo.jpg", "random_noise.png"])
def test_images_that_are_not_blood_get_no_cell_type(clf, name):
    an = clf.analyze(Image.open(OTHER / name))
    assert an.mode == "refused" or (an.single is not None and an.single.label is None)


@needs_model
def test_real_cells_pass_the_blood_cell_check(clf):
    s = pd.read_csv(ROOT / "data" / "samples" / "samples.csv")
    preds = clf.predict_many([Image.open(ROOT / "data" / "samples" / f) for f in s["file"]])
    assert sum(p.gate_ok for p in preds) >= len(s) - 1


@needs_model
def test_whole_field_samples_are_split_into_cells(clf):
    other = pd.read_csv(OTHER / "other.csv")
    for f in other[other["group"] == "Whole microscope fields"]["file"]:
        an = clf.analyze(Image.open(OTHER / f), mode="field")
        assert an.mode == "field" and len(an.field_result.cells) >= 1


@needs_model
def test_forcing_gives_a_label_with_gate_scores_kept(clf):
    p = clf.predict(Image.open(OTHER / "black_a4_sheet.jpg"), force=True)
    assert p.label in CLASSES and p.gate["white_cell"] < clf.gate_threshold


def test_external_cells_sit_in_one_split():
    ext = pd.read_csv(ROOT / "data" / "manifest_external.csv")
    assert ext["path"].is_unique
    assert set(ext["split"]) == {"train", "test"}
    other = pd.read_csv(OTHER / "other.csv")
    lab_cells = other[other["group"] == "Single cells from another lab"]
    assert len(lab_cells) >= 1


def test_trim_dark_border_crops_an_eyepiece_ring():
    from bloodsmear.preprocess import trim_dark_border
    arr = np.zeros((600, 800, 3), np.uint8)
    yy, xx = np.mgrid[0:600, 0:800]
    arr[(yy - 300) ** 2 + (xx - 400) ** 2 < 260 ** 2] = (230, 200, 205)
    out, trimmed = trim_dark_border(Image.fromarray(arr))
    assert trimmed and max(out.size) < 600
    _, black_trimmed = trim_dark_border(Image.fromarray(np.full((400, 300, 3), 20, np.uint8)))
    assert not black_trimmed


def test_finder_finds_purple_nuclei_and_ignores_pink_cells():
    from bloodsmear.detect import find_cells
    arr = np.full((400, 400, 3), (238, 225, 228), np.uint8)
    yy, xx = np.mgrid[0:400, 0:400]
    for cy, cx in [(80, 80), (90, 300), (300, 120), (320, 320), (200, 210)]:
        arr[(yy - cy) ** 2 + (xx - cx) ** 2 < 30 ** 2] = (215, 130, 140)   # red cells
    for cy, cx in [(100, 190), (280, 250)]:
        arr[(yy - cy) ** 2 + (xx - cx) ** 2 < 22 ** 2] = (95, 60, 150)     # nuclei
    dets = find_cells(Image.fromarray(arr))
    assert len(dets) == 2


# -- standard framing ------------------------------------------------------------------------------

def _cell_image(side, cx, cy, r_nuc, bg):
    arr = np.full((side, side, 3), bg, np.uint8)
    yy, xx = np.mgrid[0:side, 0:side]
    arr[(yy - cy) ** 2 + (xx - cx) ** 2 < (1.6 * r_nuc) ** 2] = (200, 170, 200)   # cytoplasm
    arr[(yy - cy) ** 2 + (xx - cx) ** 2 < r_nuc ** 2] = (90, 50, 140)            # nucleus
    return Image.fromarray(arr)


def test_framing_removes_scale_and_background_color():
    """The same cell, tight or loose, on pink or yellow background, gives nearly the same array."""
    from bloodsmear.canonical import canonical_cell
    loose, f1 = canonical_cell(_cell_image(360, 180, 180, 30, (245, 225, 210)))
    tight, f2 = canonical_cell(_cell_image(120, 60, 60, 30, (240, 235, 150)))
    off, f3 = canonical_cell(_cell_image(360, 120, 230, 30, (245, 225, 210)))
    assert f1.found and f2.found and f3.found and f2.padded
    for a in (tight, off):
        assert np.abs(a.astype(int) - loose.astype(int)).mean() < 12
    corner = loose[:4, :4].reshape(-1, 3).mean(0)
    assert np.all(np.abs(corner - 0.92 * 255) < 6)  # background is neutral gray


def test_framing_without_a_nucleus_falls_back_to_center_crop():
    from bloodsmear.canonical import canonical_cell
    arr, f = canonical_cell(Image.fromarray(np.full((200, 300, 3), (240, 230, 220), np.uint8)))
    assert not f.found and arr.shape == (64, 64, 3)


def _deployed_size():
    import json
    return json.loads((ROOT / "models" / "model_meta.json").read_text())["input_size"]


@pytest.mark.skipif(not (ROOT / "data" / "raw" / "pbc").exists(), reason="raw data absent (not in git)")
def test_app_framing_matches_training_arrays(manifest):
    """The app's framing function reproduces the arrays the deployed model was trained on."""
    from bloodsmear.canonical import canonical_uint8
    size = _deployed_size()
    path = ROOT / "data" / "processed" / f"pbc_{size}c.npz"
    if not path.exists():
        pytest.skip("processed arrays absent")
    d = np.load(path, allow_pickle=True)
    for i in np.random.default_rng(1).choice(len(manifest), 25, replace=False):
        assert np.array_equal(canonical_uint8(Image.open(ROOT / "data" / "raw" / manifest.loc[i, "path"]), size), d["X"][i])
