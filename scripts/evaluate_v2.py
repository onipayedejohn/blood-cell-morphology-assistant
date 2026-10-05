"""Evaluate version 2: multi-lab classifier plus gatekeeper.

1. Unseen labs. The two leave-one-lab-out classifiers are scored on every cell
   of the lab they never saw (BCCD, Jiangxi Tecom). Version 1 is scored on the
   same cells for comparison.
2. Final classifier. Temperature, review level (least confident 2% of
   validation cells) and unfamiliar-image level (99th percentile of validation
   distances) are set on validation data from all labs. Then it is scored once
   on the PBC test set and on the held-back test half of each external lab.
3. The whole app. Gatekeeper and classifier together, on real cells, on smear
   patches without a white cell, on non-blood images including image types the
   gatekeeper never saw, and on the kinds of images users reported (a photo of
   a black A4 sheet, a white page, a document, a screenshot).
4. Export both models to ONNX, check the exports match Keras, and time them.

The classifier is scored on the standard-framing arrays (prepare_canonical.py),
the gatekeeper on plain arrays, exactly as the app feeds them.

Run from the project root after run_training_v2.sh and train_gate.py:
    python scripts/evaluate_v2.py
"""

from __future__ import annotations

import io
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFilter
from sklearn.covariance import LedoitWolf
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bloodsmear.config import CLASSES  # noqa: E402
from bloodsmear.preprocess import resize_uint8  # noqa: E402
from evaluate import bootstrap_ci, ece, fit_temperature, logits_and_features, macro_f1, softmax  # noqa: E402
from train import external_splits  # noqa: E402

P = ROOT / "data" / "processed"
CAND = ROOT / "models" / "candidates"
MODELS = ROOT / "models"
METRICS = ROOT / "reports" / "metrics"
SEED = 42
REVIEW_BUDGET = 0.02
EXT_CLASSES = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]


def lab_metrics(y, pred):
    present = sorted(set(y))
    prec, rec, f1, sup = precision_recall_fscore_support(y, pred, labels=present, zero_division=0)
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
            "accuracy_ci": bootstrap_ci(y, pred, accuracy_score) if len(y) > 20 else None,
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "per_class": [{"class": CLASSES[c], "recall": float(r), "precision": float(p), "support": int(s)}
                          for c, p, r, s in zip(present, prec, rec, sup)],
            "predicted": {CLASSES[k]: int(v) for k, v in zip(*np.unique(pred, return_counts=True))}}


def user_report_images():
    """The kinds of images people uploaded that v1 still classified."""
    rng = np.random.default_rng(7)

    def phone(arr, blur=1.2, q=70):
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur))
        b = io.BytesIO()
        im.save(b, "JPEG", quality=q)
        return Image.open(b).convert("RGB")

    H, W = 1600, 1200
    yy, xx = np.mgrid[0:H, 0:W]
    vign = 1 - 0.7 * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2)
    out = {}
    for name, rgb in [("black A4 sheet photo", (22, 20, 24)), ("black A4 sheet, warm light", (45, 30, 28)),
                      ("white A4 sheet photo", (235, 232, 225)), ("blue folder photo", (40, 70, 140))]:
        out[name] = phone(np.dstack([np.full((H, W), v) for v in rgb]) * vign[..., None] + rng.normal(0, 5, (H, W, 3)))
    doc = Image.new("RGB", (W, H), (246, 244, 238))
    d = ImageDraw.Draw(doc)
    for i in range(40):
        d.text((80, 60 + i * 36), "Lorem ipsum dolor sit amet, consectetur adipiscing elit " * 2, fill=(30, 30, 30))
    out["printed document"] = doc
    shot = Image.new("RGB", (1280, 800), (250, 250, 252))
    d = ImageDraw.Draw(shot)
    d.rectangle((0, 0, 1280, 60), fill=(40, 60, 120))
    d.rectangle((100, 120, 600, 500), fill=(220, 40, 90))
    out["app screenshot"] = shot
    from sklearn.datasets import load_sample_images
    for i, im in enumerate(load_sample_images().images):
        out[f"everyday photo {i + 1}"] = Image.fromarray(im)
    noise = Image.fromarray(rng.integers(0, 256, (400, 400, 3), dtype=np.uint8))
    out["random noise"] = noise
    return out


def simulate_eyepiece(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """A rough imitation of a phone photo through a microscope eyepiece (round field on black)."""
    side = 1000
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    r = int(side * rng.uniform(0.38, 0.47))
    field_img = img.convert("RGB").resize((2 * r, 2 * r), Image.Resampling.BICUBIC)
    mask = Image.new("L", (2 * r, 2 * r), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, 2 * r, 2 * r), fill=255)
    canvas.paste(field_img, (side // 2 - r, side // 2 - r), mask.filter(ImageFilter.GaussianBlur(12)))
    arr = np.asarray(canvas, np.float32) * rng.uniform(0.85, 1.15, 3)
    arr += rng.normal(0, rng.uniform(2, 6), arr.shape)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(float(rng.uniform(0.6, 1.6))))
    b = io.BytesIO()
    out.save(b, "JPEG", quality=int(rng.integers(55, 85)))
    return Image.open(b).convert("RGB")


def simulated_phone_photos():
    """Run the full app on simulated eyepiece photos of held-out images.

    PBC: 150 test images, scored both clean and as simulated photos.
    BCCD: test-half fields whose white cells are all in the classifier's test half too.
    """
    import re
    sys.path.insert(0, str(ROOT / "src"))
    from bloodsmear.inference import CellClassifier
    clf = CellClassifier()
    rng = np.random.default_rng(11)
    man = pd.read_csv(ROOT / "data" / "manifest.csv")
    test = man[man["split"] == "test"].sample(150, random_state=3)

    def first_label(an):
        if an.mode == "refused":
            return "refused"
        cells = an.field_result.cells if an.mode == "field" else [an.single]
        cells = [c for c in cells if c.label]
        return cells[0].label if cells else None

    clean = photo = refused = classified = 0
    for r in test.itertuples():
        img = Image.open(ROOT / "data" / "raw" / r.path)
        clean += int(first_label(clf.analyze(img)) == r._4)
        lab = first_label(clf.analyze(simulate_eyepiece(img, rng)))
        refused += lab == "refused"
        if lab not in (None, "refused"):
            classified += 1
            photo += int(lab == r._4)

    ext = pd.read_csv(ROOT / "data" / "manifest_external.csv")
    split_of = {}
    for r in ext[ext["lab"] == "bccd"].itertuples():
        split_of.setdefault(re.search(r"BloodImage_(\d+)", r.path).group(1), set()).add(r.split)
    fields = [f for f in json.loads((ROOT / "data" / "fields.json").read_text())
              if f["source"] == "bccd" and f["split"] == "test" and f["wbc_boxes"]
              and split_of.get(re.search(r"BloodImage_(\d+)", f["path"]).group(1)) == {"test"}][:80]
    outcome = {"refused": 0, "white_cell_classified": 0, "finder_found_nothing": 0, "all_cut_outs_rejected": 0}
    for f in fields:
        an = clf.analyze(simulate_eyepiece(Image.open(ROOT / "data" / "raw" / f["path"]), rng))
        if an.mode == "refused":
            outcome["refused"] += 1
        elif first_label(an):
            outcome["white_cell_classified"] += 1
        elif an.mode == "single" or not an.field_result.detections:
            outcome["finder_found_nothing"] += 1
        else:
            outcome["all_cut_outs_rejected"] += 1
    n = len(fields)
    return {"pbc_single_cell_photos": {"n": len(test), "refused": refused / len(test),
                                       "a_white_cell_classified": classified / len(test),
                                       "correct_when_classified": photo / max(classified, 1),
                                       "correct_on_the_same_images_unaltered": clean / len(test)},
            "bccd_field_photos": {"n": n, **{k: v / n for k, v in outcome.items()},
                                  "a_white_cell_classified": outcome["white_cell_classified"] / n},
            "note": "Simulated: round field on black, color cast, noise, blur and JPEG. Not real phone photos. "
                    "BCCD fields are limited to those whose cells the classifier never trained on."}


def classifier_arrays(size):
    """PBC and external cells in standard framing at the model's input size."""
    return (np.load(P / f"pbc_{size}c.npz", allow_pickle=True)["X"],
            np.load(P / f"external_{size}c.npz", allow_pickle=True)["X"])


def calibrate(model, holdout, Xpc, yp, sp):
    """Temperature, review level and unfamiliar-image level, all set on validation data.

    Validation is the PBC validation split plus a quarter of each external lab's
    training half; a held-out lab contributes nothing.
    """
    (Xet, yet, _), (Xev, yev, _) = external_splits(["bccd", "jtsc", "cvblog"], holdout, model.input_shape[1], suffix="c")
    _, F_tr, feat_model = logits_and_features(model, np.concatenate([Xpc[sp == "train"], Xet]))
    y_tr = np.concatenate([yp[sp == "train"], yet])
    L_va, F_va, _ = logits_and_features(model, np.concatenate([Xpc[sp == "val"], Xev]))
    y_va = np.concatenate([yp[sp == "val"], yev])
    T = fit_temperature(L_va, y_va)
    means = np.stack([F_tr[y_tr == c].mean(0) for c in range(len(CLASSES))])
    precision = LedoitWolf().fit(F_tr - means[y_tr]).precision_
    cal = {"model": model, "feat_model": feat_model, "T": T, "means": means, "precision": precision,
           "conf_threshold": float(np.quantile(softmax(L_va / T).max(1), REVIEW_BUDGET))}
    cal["d_va"] = distance(cal, F_va)
    cal["ood_threshold"] = float(np.percentile(cal["d_va"], 99))
    return cal


def distance(cal, F):
    diff = F[:, None, :] - cal["means"][None]
    return np.sqrt(np.maximum(np.einsum("nck,kl,ncl->nc", diff, cal["precision"], diff).min(1), 0))


def score(cal, X, y):
    L, F, _ = logits_and_features(cal["model"], X)
    Pr = softmax(L / cal["T"])
    pred = Pr.argmax(1)
    conf, d = Pr.max(1), distance(cal, F)
    ok = (conf >= cal["conf_threshold"]) & (d <= cal["ood_threshold"])
    out = lab_metrics(y, pred)
    out["macro_f1"] = float(macro_f1(y, pred))
    out["confusion_matrix"] = confusion_matrix(y, pred, labels=range(len(CLASSES))).tolist()
    out["status_line"] = {"held_back": int((~ok).sum()), "errors": int((pred != y).sum()),
                          "errors_held_back": int(((pred != y) & ~ok).sum()),
                          "accuracy_on_confident": float((pred[ok] == y[ok]).mean()) if ok.any() else None,
                          "share_confident": float(ok.mean()),
                          "flagged_unfamiliar": float((d > cal["ood_threshold"]).mean())}
    out["by_confidence"] = [{"min_confidence": t, "share": float((conf >= t).mean()),
                             "accuracy": float((pred[conf >= t] == y[conf >= t]).mean()) if (conf >= t).any() else None}
                            for t in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99)]
    return out, Pr, pred, conf, d


def main():
    tf.keras.utils.set_random_seed(SEED)
    pbc = np.load(P / "pbc_64.npz", allow_pickle=True)
    ext = np.load(P / "external_64.npz", allow_pickle=True)
    gate_data = np.load(P / "gate_64.npz", allow_pickle=True)
    Xp, yp, sp = pbc["X"], pbc["y"].astype(int), pbc["split"]
    Xe, ye, le, se = ext["X"], ext["y"].astype(int), ext["lab"], ext["split"]
    final_name = os.environ.get("FINAL", "v2_final")       # e.g. FINAL=mn_final for the MobileNet runs
    lolo_prefix = os.environ.get("LOLO", "v2_lolo")

    # 1. Unseen labs ------------------------------------------------------------------------
    v1 = pd.read_csv(METRICS / "v1" / "v1_on_external_labs.csv")
    unseen = {}
    for lab in ["bccd", "jtsc"]:
        model = tf.keras.models.load_model(CAND / f"{lolo_prefix}_{lab}.keras")
        Xpc, Xec = classifier_arrays(model.input_shape[1])
        m = le == lab
        cal = calibrate(model, lab, Xpc, yp, sp)
        unseen[lab] = {"v2_never_saw_this_lab": score(cal, Xec[m], ye[m])[0]}
        first = CAND / f"v2a_lolo_{lab}.keras"  # first attempt: from scratch, no standard framing, 64 px
        if first.exists():
            L, _, _ = logits_and_features(tf.keras.models.load_model(first), Xe[m])
            unseen[lab]["v2_first_attempt_no_framing"] = lab_metrics(ye[m], L.argmax(1))
        g = v1[v1["lab"] == lab]
        unseen[lab]["v1"] = lab_metrics(g["cls"].map(CLASSES.index).to_numpy(), g["pred"].map(CLASSES.index).to_numpy())
        print(lab, "v1 acc", round(unseen[lab]["v1"]["accuracy"], 3), "-> v2 unseen acc",
              round(unseen[lab]["v2_never_saw_this_lab"]["accuracy"], 3), flush=True)
        del model, cal

    # 2. Final classifier ------------------------------------------------------------------------
    model = tf.keras.models.load_model(CAND / f"{final_name}.keras")
    size = int(model.input_shape[1])
    Xpc, Xec = classifier_arrays(size)
    cal = calibrate(model, None, Xpc, yp, sp)
    T, conf_threshold, ood_threshold = cal["T"], cal["conf_threshold"], cal["ood_threshold"]
    means, precision, d_va, feat_model = cal["means"], cal["precision"], cal["d_va"], cal["feat_model"]

    def dist(F):
        return distance(cal, F)

    te = sp == "test"
    pbc_res, P_te, pred_te, conf_te, d_te = score(cal, Xpc[te], yp[te])
    pbc_res["macro_f1_ci"] = bootstrap_ci(yp[te], pred_te, macro_f1)
    pbc_res["balanced_accuracy_ci"] = bootstrap_ci(yp[te], pred_te, balanced_accuracy_score)
    pbc_res["ece"] = ece(P_te, yp[te])[0]
    pbc_res["reliability"] = ece(P_te, yp[te])[1]
    pbc_res["confusion_matrix"] = confusion_matrix(yp[te], pred_te, labels=range(len(CLASSES))).tolist()
    manifest = pd.read_csv(ROOT / "data" / "manifest.csv").set_index("file")
    sub = manifest.loc[pbc["files"][te], "subtype"].to_numpy()
    pbc_res["subtypes"] = [{"subtype": s, "n": int((sub == s).sum()),
                            "recall": float((pred_te[sub == s] == yp[te][sub == s]).mean())} for s in sorted(set(sub))]
    final = {"pbc_test": pbc_res, "external_test_halves": {}}
    pooled_y, pooled_p = [], []
    for lab in ["bccd", "jtsc", "cvblog"]:
        m = (le == lab) & (se == "test")
        r, _, pr, _, _ = score(cal, Xec[m], ye[m])
        final["external_test_halves"][lab] = r
        pooled_y.append(ye[m])
        pooled_p.append(pr)
    py, pp = np.concatenate(pooled_y), np.concatenate(pooled_p)
    final["external_pooled"] = lab_metrics(py, pp)
    print("PBC test acc", round(pbc_res["accuracy"], 4), "| external pooled acc", round(final["external_pooled"]["accuracy"], 3),
          flush=True)

    # 3. The whole app: gatekeeper then classifier ---------------------------------------------------
    gate = tf.keras.models.load_model(CAND / "gate_final.keras")
    gate_res = json.loads((CAND / "gate_final_results.json").read_text())
    tau = gate_res["threshold_p_white"]

    def gate_probs(A):
        return np.concatenate([softmax(gate(A[i:i + 256].astype("float32") / 255.0, training=False).numpy())
                               for i in range(0, len(A), 256)])

    def app_outcome(A, y=None, A_cls=None):
        """A: images as uploaded (for the gatekeeper); A_cls: the same cells in standard framing."""
        g = gate_probs(A)
        passes = g[:, 0] >= tau
        out = {"n": int(len(A)), "refused_by_gatekeeper": float((~passes).mean())}
        if y is not None and passes.any():
            L, F, _ = logits_and_features(model, A_cls[passes])
            Pr = softmax(L / T)
            pred = Pr.argmax(1)
            conf_ok = (Pr.max(1) >= conf_threshold) & (dist(F) <= ood_threshold)
            out["accuracy_when_shown"] = float((pred == y[passes]).mean())
            out["shown_as_confident"] = float(conf_ok.mean() * passes.mean())
            out["accuracy_when_confident"] = float((pred[conf_ok] == y[passes][conf_ok]).mean()) if conf_ok.any() else None
        return out

    pipeline = {"pbc_test_cells": app_outcome(Xp[te], yp[te], Xpc[te])}
    for lab in ["bccd", "jtsc", "cvblog"]:
        m = (le == lab) & (se == "test")
        pipeline[f"{lab}_test_cells"] = app_outcome(Xe[m], ye[m], Xec[m])
    pipeline["smear_without_white_cell"] = app_outcome(gate_data["smear_test"])
    for k in gate_data.files:
        if k.startswith("negtest_"):
            pipeline[k.replace("negtest_", "")] = app_outcome(gate_data[k])
    reports = user_report_images()
    arrs = np.stack([resize_uint8(im, 64) for im in reports.values()])
    gp = gate_probs(arrs)
    pipeline["user_report_cases"] = [{"image": k, "p_white_cell": float(p[0]), "gate_says": ["white cell", "smear, no white cell",
                                      "not a blood smear"][int(p.argmax())], "refused": bool(p[0] < tau)}
                                     for k, p in zip(reports, gp)]
    lolo_gate = json.loads((CAND / "gate_lolo_bccd_results.json").read_text())

    # 4. Export ---------------------------------------------------------------------------------
    import onnxruntime as ort
    import tf2onnx
    spec = (tf.TensorSpec((None, size, size, 3), tf.float32, name="image"),)
    gate_spec = (tf.TensorSpec((None, 64, 64, 3), tf.float32, name="image"),)
    feat_model.output_names = ["logits", "features"]
    tf2onnx.convert.from_keras(feat_model, input_signature=spec, opset=17, output_path=str(MODELS / "bloodcell_cnn.onnx"))
    gate.output_names = ["gate_logits"]
    tf2onnx.convert.from_keras(gate, input_signature=gate_spec, opset=17, output_path=str(MODELS / "gate.onnx"))
    xs = Xpc[te][:64].astype("float32") / 255.0
    s1 = ort.InferenceSession(str(MODELS / "bloodcell_cnn.onnx"), providers=["CPUExecutionProvider"])
    s2 = ort.InferenceSession(str(MODELS / "gate.onnx"), providers=["CPUExecutionProvider"])
    o1 = s1.run(None, {"image": xs})[0]
    xg = Xp[te][:64].astype("float32") / 255.0
    o2 = s2.run(None, {"image": xg})[0]
    parity = {"classifier_max_logit_diff": float(np.abs(o1 - feat_model(xs, training=False)[0].numpy()).max()),
              "classifier_same_top_class": float((o1.argmax(1) == feat_model(xs, training=False)[0].numpy().argmax(1)).mean()),
              "gate_max_logit_diff": float(np.abs(o2 - gate(xg, training=False).numpy()).max())}

    def timeit(fn, reps=30):
        fn()
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t) / reps * 1000

    speed = {"classifier_ms": round(timeit(lambda: s1.run(None, {"image": xs[:1]})), 2),
             "gate_ms": round(timeit(lambda: s2.run(None, {"image": xg[:1]})), 2),
             "classifier_mb": round((MODELS / "bloodcell_cnn.onnx").stat().st_size / 1e6, 2),
             "gate_mb": round((MODELS / "gate.onnx").stat().st_size / 1e6, 2)}

    W, b = model.get_layer("logits").get_weights()
    np.savez_compressed(MODELS / "model_aux.npz", cam_weights=W.astype("float32"), cam_bias=b.astype("float32"),
                        ood_means=means.astype("float32"), ood_precision=precision.astype("float32"),
                        val_distances=np.sort(d_va).astype("float32"))
    train_info = json.loads((CAND / f"{final_name}_train.json").read_text())
    meta = {
        "version": 2, "classes": CLASSES, "input_size": size, "framing": "canonical", "temperature": T,
        "architecture": train_info.get("arch", "cnn"),
        "confidence_threshold": conf_threshold, "review_budget": REVIEW_BUDGET, "ood_threshold": ood_threshold,
        "gate": {"classes": ["white_cell", "smear_no_wbc", "not_smear"], "threshold_p_white": tau, "input_size": 64},
        "training": {k: train_info.get(k) for k in ("labs", "aug", "ext_repeat", "n_external_train", "epochs_run", "init",
                                                "train_minutes", "train_energy_wh")},
        "test_summary": {"pbc_accuracy": pbc_res["accuracy"], "pbc_macro_f1": pbc_res["macro_f1"],
                         "external_pooled_accuracy": final["external_pooled"]["accuracy"]},
        "versions": {"tensorflow": tf.__version__, "onnxruntime": ort.__version__, "python": platform.python_version()},
    }
    (MODELS / "model_meta.json").write_text(json.dumps(meta, indent=2))
    phone = simulated_phone_photos()  # uses the exported ONNX models through the app's own code
    report = {"unseen_labs": unseen, "final": final, "app_pipeline": pipeline, "gate_validation": gate_res,
              "simulated_phone_photos": phone,
              "gate_unseen_lab_bccd": lolo_gate, "export": {"parity": parity, "speed": speed},
              "choices": {"temperature": T, "confidence_threshold": conf_threshold, "ood_threshold": ood_threshold,
                          "gate_threshold": tau}}
    (METRICS / "v2_results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"unseen": {k: {kk: round(vv["accuracy"], 3) for kk, vv in v.items()} for k, v in unseen.items()},
                      "pbc": round(pbc_res["accuracy"], 4),
                      "external": {k: round(v["accuracy"], 3) for k, v in final["external_test_halves"].items()},
                      "pipeline": {k: v for k, v in pipeline.items() if k != "user_report_cases"},
                      "user_reports": pipeline["user_report_cases"], "phone": phone, "parity": parity, "speed": speed},
                     indent=1))


if __name__ == "__main__":
    main()
