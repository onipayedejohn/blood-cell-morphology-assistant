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
    assert right >= len(s) - 2


@needs_model
def test_noise_is_not_reported_as_confident(clf):
    p = clf.predict(Image.open(ROOT / "data" / "samples" / "other" / "random_noise.png"))
    assert p.status != "confident"


@needs_model
def test_thresholds_are_sane(clf):
    assert 0.3 < clf.conf_threshold < 1
    assert clf.ood_threshold > 0 and 0.2 < clf.temperature < 5
    assert clf.cam_weights.shape[1] == len(CLASSES)
