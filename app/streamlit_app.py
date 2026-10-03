"""Blood Cell Morphology Assistant: Streamlit front end.

Run locally from the project root:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import io
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bloodsmear.config import CELL_INFO, CLASSES, LEUKOCYTES, METRICS_DIR, REFERENCE_RANGES, display_name  # noqa: E402
from bloodsmear.differential import summarize  # noqa: E402
from bloodsmear.explain import overlay  # noqa: E402
from bloodsmear.inference import CellClassifier, Prediction  # noqa: E402
import ui  # noqa: E402

st.set_page_config(page_title="Blood Cell Morphology Assistant", page_icon="🔬", layout="wide",
                   initial_sidebar_state="collapsed")
ui.inject_css()

SAMPLES = ROOT / "data" / "samples"
DEMO = ROOT / "data" / "demo_batch"
METRICS = METRICS_DIR
FILE_TYPES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]


# -- cached resources ------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the model")
def load_classifier() -> CellClassifier:
    return CellClassifier()


@st.cache_data
def load_json(name: str) -> dict:
    return json.loads((METRICS / name).read_text())


@st.cache_data
def sample_catalog() -> pd.DataFrame:
    df = pd.read_csv(SAMPLES / "samples.csv")
    df["group"] = "Test-set cells"
    df["label"] = [f"{display_name(c)} ({f.rsplit('_', 1)[1].split('.')[0]})" for c, f in zip(df["class"], df["file"])]
    other = pd.read_csv(SAMPLES / "other" / "other.csv")
    other["file"] = "other/" + other["file"]
    other["group"] = "Images the app should question"
    return pd.concat([df, other], ignore_index=True)


@st.cache_data(show_spinner="Classifying the demo batch")
def classify_demo() -> tuple[list[dict], pd.DataFrame]:
    files = sorted(DEMO.glob("*.jpg"))
    preds = load_classifier().predict_many([Image.open(f) for f in files])
    key = pd.read_csv(DEMO / "demo_batch_labels.csv").set_index("file")["class"]
    return [pred_row(f.name, p) for f, p in zip(files, preds)], key


def pred_row(name: str, p: Prediction) -> dict:
    return {"Image": ui.thumb_uri(p.image_uint8), "File": name, "Predicted": display_name(p.label),
            "key": p.label, "Confidence": p.confidence, "Status": STATUS[p.status]["short"], "status": p.status}


clf = load_classifier()
meta = clf.meta

STATUS = {
    "confident": {"short": "Confident", "title": "Confident result", "icon": ":material/check_circle:",
                  "box": st.success,
                  "text": "The model is confident, and the image looks like the cells it was trained on."},
    "review": {"short": "Check yourself", "title": "Check this cell yourself", "icon": ":material/visibility:",
               "box": st.warning,
               "text": f"Confidence is below {clf.conf_threshold:.0%}, the level that sends the least confident 2% "
                       "of validation cells to a person. Look at this cell yourself before using the answer."},
    "unfamiliar": {"short": "Unfamiliar image", "title": "This image does not look like the training images",
                   "icon": ":material/help:", "box": st.error,
                   "text": "It may come from a different stain, microscope or magnification, or it may not show "
                           "a single white cell. Treat the answer below as unreliable."},
}

if "history" not in st.session_state:
    st.session_state.history = []
if "seen_tips" not in st.session_state:
    st.session_state.seen_tips = False


# -- header --------------------------------------------------------------------------------
st.title("Blood cell morphology assistant")
st.markdown(
    '<p class="lede">Classifies one white cell or nucleated red cell from a stained peripheral blood smear, '
    'shows which part of the image drove the answer, and says when it is unsure.</p>',
    unsafe_allow_html=True,
)
st.caption(":material/science: Research and teaching prototype. Not a medical device, and not validated "
           "for patient care. See **About and limits**.")

tab_one, tab_diff, tab_hist, tab_perf, tab_about = st.tabs(
    ["Classify a cell", "Differential count", "Session history", "Model performance", "About and limits"])


# -- tab 1: classify one cell ------------------------------------------------------------------
def read_upload(file) -> Image.Image | None:
    try:
        img = Image.open(io.BytesIO(file.getvalue()))
        img.load()
        return img
    except (UnidentifiedImageError, OSError):
        st.error(f"{file.name} could not be read as an image. Upload a JPG, PNG, BMP or TIFF file.",
                 icon=":material/broken_image:")
        return None


def report_text(p: Prediction, source: str) -> str:
    lines = [
        "Blood Cell Morphology Assistant (research prototype, not for clinical use)",
        f"Date: {datetime.now():%Y-%m-%d %H:%M}",
        f"Image: {source}",
        f"Result: {display_name(p.label)}",
        f"Confidence (calibrated): {p.confidence:.1%}",
        f"Status: {STATUS[p.status]['title']}",
        "Other possibilities: " + ", ".join(f"{display_name(c)} {v:.1%}" for c, v in p.top[1:]),
        f"Image typicality: more unusual than {p.unfamiliarity:.0f}% of validation images",
    ]
    lines += [f"Image check: {i.message}" for i in p.issues]
    return "\n".join(lines) + "\n"


def show_result(p: Prediction, source: str):
    s = STATUS[p.status]
    s["box"](f"**{s['title']}.** {s['text']}", icon=s["icon"])

    c1, c2 = st.columns(2, gap="small")
    c1.image(ui.upscale(p.image_uint8), caption=f"What the model saw: the center square at {clf.size} x {clf.size} pixels",
             use_container_width=True)
    c2.image(ui.upscale(overlay(p.image_uint8, p.cam)),
             caption="Brighter cyan means that area pushed the model towards its answer (an 8 x 8 map, so it is coarse)",
             use_container_width=True)

    info = CELL_INFO[p.label]
    sub = ("Unreliable answer. The image is outside what the model has seen, so even a high confidence means little."
           if p.unfamiliar else "Model answer from seven cell types")
    st.markdown(f'<div class="readout">{info.name}</div><div class="readout-sub">{sub}</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Confidence (unreliable here)" if p.unfamiliar else "Confidence", ui.conf(p.confidence),
              help="Checked against test accuracy (calibration error about 0.5%). Most cells score above 95%; few "
                   "fall in the middle, so mid-range values are less reliable. It means nothing for unfamiliar images.")
    typicality = "Typical" if p.unfamiliarity < 90 else ("Less typical" if not p.unfamiliar else "Unfamiliar")
    m2.metric("Image typicality", typicality,
              help=f"More unusual than {p.unfamiliarity:.0f}% of validation images, measured on the model's "
                   "internal features. Above the 99th percentile the app warns you.")
    m3.metric("Typical adult range", info.adult_range or "Not expected",
              help="Share of white cells in a healthy adult. Ranges differ between laboratories.")

    st.markdown("**Other possibilities**")
    for c, v in p.top:
        st.progress(min(max(v, 0.0), 1.0), text=f"{display_name(c)}  {ui.conf(v, 1)}")

    with st.container(border=True):
        st.markdown(f"**What a {info.name.split(' (')[0].lower()} looks like.** {info.looks_like}")
        st.markdown(f"**Why it matters.** {info.clinical_note}")

    for issue in p.issues:
        st.warning(issue.message, icon=":material/photo_camera:")

    st.download_button("Download this result", report_text(p, source), file_name="cell_result.txt",
                       icon=":material/download:")


def add_history(p: Prediction, source: str):
    key = (source, p.label, round(p.confidence, 4))
    if st.session_state.history and st.session_state.history[-1]["_key"] == key:
        return
    st.session_state.history.append({
        "_key": key, "Time": datetime.now().strftime("%H:%M:%S"), "Image": ui.thumb_uri(p.image_uint8),
        "Source": source, "Result": display_name(p.label), "Confidence": p.confidence,
        "Status": STATUS[p.status]["short"],
    })


with tab_one:
    left, right = st.columns([0.9, 1.5], gap="large")
    img, source = None, ""
    with left:
        with st.container(border=True):
            st.subheader("Choose a cell image")
            mode = st.segmented_control("Image source", ["Sample cell", "Upload"], default="Sample cell",
                                        label_visibility="collapsed", key="mode_one")
            if mode == "Upload":
                up = st.file_uploader("Upload one cell", type=FILE_TYPES, key="one_upload",
                                      help="One white cell, centerd, from a stained smear. The model uses the center square.")
                if up is not None:
                    img, source = read_upload(up), up.name
            else:
                cat = sample_catalog()
                group = st.radio("Sample set", list(cat["group"].unique()), horizontal=True,
                                 label_visibility="collapsed")
                sub = cat[cat["group"] == group]
                choice = st.selectbox("Sample", sub["label"].tolist(), key=f"pick_{group}")
                row = sub[sub["label"] == choice].iloc[0]
                img, source = Image.open(SAMPLES / row["file"]), f"Sample: {choice}"
                st.image(img, width=150)
                if group == "Test-set cells":
                    st.caption("From the held-out test set, so the model never trained on it. "
                               f"Expert label: {display_name(row['class'])}.")
                else:
                    st.caption(row["note"])

        with st.expander("How to get a good result", expanded=not st.session_state.seen_tips):
            st.markdown(
                "1. Crop around **one** cell and keep it near the center. The model only looks at the center square.\n"
                "2. Use a color image of a Romanowsky-stained smear (Wright, Giemsa or May-Grünwald-Giemsa) "
                "at oil immersion.\n"
                "3. Read the status line first. If it says to check the cell yourself, do that before using the answer.")
        st.session_state.seen_tips = True

    with right:
        if img is None:
            with st.container(border=True):
                st.markdown("#### No image yet")
                st.markdown("Pick a sample cell on the left or upload your own image. The result, a heatmap and "
                            "notes on the cell type appear here.")
        else:
            from bloodsmear.quality import check_image
            blockers = [i for i in check_image(img) if i.level == "block"]
            if blockers:
                for b in blockers:
                    st.error(b.message, icon=":material/crop:")
            else:
                pred = clf.predict(img)
                add_history(pred, source)
                show_result(pred, source)


# -- tab 2: differential count ---------------------------------------------------------------------
with tab_diff:
    st.subheader("Differential count from single-cell images")
    st.markdown(
        '<p class="lede">Upload one image per cell, or try the demo batch. Cells the model is unsure about, '
        'or that look unlike its training images, are held back for you to review instead of being counted.</p>',
        unsafe_allow_html=True)
    st.caption("A manual differential counts at least 100 consecutive white cells on a smear. These numbers only "
               "mean something if you upload every cell you see, not selected examples. Ranges are typical adult values; "
               "use your laboratory's own. Flags compare percentages, not absolute counts.")

    dmode = st.segmented_control("Cells to count", ["Demo batch of 120 cells", "Upload cells"],
                                 default="Demo batch of 120 cells", label_visibility="collapsed", key="mode_diff")
    rows, answer_key = [], None
    if dmode == "Upload cells":
        ups = st.file_uploader("Upload cell images", type=FILE_TYPES, accept_multiple_files=True, key="diff_upload")
        if ups:
            images, names = [], []
            for f in ups:
                im = read_upload(f)
                if im is not None and min(im.size) >= 64:
                    images.append(im)
                    names.append(f.name)
                elif im is not None:
                    st.warning(f"{f.name} was skipped because it is smaller than 64 pixels on one side.")
            if images:
                with st.spinner(f"Classifying {len(images)} cells"):
                    preds = clf.predict_many(images)
                rows = [pred_row(n, p) for n, p in zip(names, preds)]
        else:
            with st.container(border=True):
                st.markdown("#### No cells uploaded")
                st.markdown("Drop in several single-cell images at once. JPG, PNG, BMP and TIFF all work.")
    else:
        rows, answer_key = classify_demo()
        st.caption("The demo batch is 120 test-set cells in proportions that resemble a mildly left-shifted adult "
                   "differential with a few nucleated red cells. It is not a real patient.")

    if rows:
        dfc = pd.DataFrame(rows)
        dif = summarize(dfc["key"].tolist(), dfc["status"].tolist())
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Images", len(dfc))
        k2.metric("White cells counted", dif.counted_wbc)
        k3.metric("Held for review", dif.held_for_review)
        k4.metric("NRBC per 100 WBC", "None seen" if not dif.nrbc_per_100_wbc else f"{dif.nrbc_per_100_wbc:g}",
                  help="Nucleated red cells (erythroblasts) are reported per 100 white cells, not as part of the 100%.")

        if dif.flags:
            st.warning("\n".join(f"- {f}" for f in dif.flags), icon=":material/flag:")

        table = dif.table.copy()
        table["key"] = LEUKOCYTES
        st.markdown("**Differential.** Blue bars are this batch; gray bands are the typical adult range.")
        if dif.counted_wbc:
            st.altair_chart(ui.differential_chart(table, REFERENCE_RANGES), use_container_width=True)
        else:
            st.info("No white cells were counted with confidence, so there is nothing to chart.")
        st.dataframe(table.drop(columns=["key"]), hide_index=True, use_container_width=True,
                     column_config={"Percent of white cells": st.column_config.NumberColumn("Percent", format="%.1f%%"),
                                    "Typical adult range": st.column_config.TextColumn("Adult range")})

        if answer_key is not None:
            agree = sum(r["key"] == answer_key[r["File"]] for r in rows)
            st.caption(f"Answer key for the demo: the model agreed with the expert label on {agree} of {len(rows)} cells, "
                       "including the ones held for review.")

        st.markdown("**Every cell**")
        show = dfc[["Image", "File", "Predicted", "Confidence", "Status"]]
        if answer_key is not None:
            show = show.assign(**{"Expert label": [display_name(answer_key[f]) for f in dfc["File"]]})
        order = {"Unfamiliar image": 0, "Check yourself": 1, "Confident": 2}
        show = show.sort_values("Status", key=lambda s: s.map(order), kind="stable")
        st.dataframe(show, hide_index=True, use_container_width=True, height=380, column_config={
            "Image": st.column_config.ImageColumn("Cell", width="small"),
            "Confidence": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        })
        st.download_button("Download per-cell results (CSV)", show.drop(columns=["Image"]).to_csv(index=False),
                           file_name="differential_cells.csv", icon=":material/download:")


# -- tab 3: history -------------------------------------------------------------------------------
with tab_hist:
    st.subheader("Cells classified in this session")
    hist = st.session_state.history
    if not hist:
        st.markdown("Nothing yet. Each cell you classify on the first tab is added here, so you can compare "
                    "results side by side. The list clears when you close the page.")
    else:
        h = pd.DataFrame(hist).drop(columns=["_key"])
        a, b, c = st.columns(3)
        a.metric("Cells", len(h))
        b.metric("Confident", int((h["Status"] == "Confident").sum()))
        c.metric("Needing a look", int((h["Status"] != "Confident").sum()))
        st.dataframe(h.iloc[::-1], hide_index=True, use_container_width=True, column_config={
            "Image": st.column_config.ImageColumn("Cell", width="small"),
            "Confidence": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        })
        c1, c2 = st.columns([1, 4])
        c1.download_button("Download CSV", h.drop(columns=["Image"]).to_csv(index=False), file_name="session_history.csv",
                           icon=":material/download:")
        if c2.button("Clear history", icon=":material/delete:"):
            st.session_state.history = []
            st.rerun()


# -- tab 4: performance -----------------------------------------------------------------------------
with tab_perf:
    test = load_json("test_metrics.json")
    comp = load_json("model_comparison.json")
    ext = load_json("external_checks.json")
    exp = load_json("export_checks.json")
    val = load_json("validation_choices.json")

    st.subheader("How the model performs")
    st.markdown(f'<p class="lede">All numbers below come from {test["n"]:,} test images that were split off '
                'before training and used only for final scoring.</p>', unsafe_allow_html=True)

    def ci(k):
        lo, hi = test[f"{k}_ci"]
        return f"95% CI {lo:.1%} to {hi:.1%}"

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Accuracy", ui.pct(test["accuracy"], 1), help=ci("accuracy"))
    t2.metric("Balanced accuracy", ui.pct(test["balanced_accuracy"], 1), help=ci("balanced_accuracy"))
    t3.metric("Macro-F1", ui.pct(test["macro_f1"], 1), help=ci("macro_f1"))
    t4.metric("Calibration error", ui.pct(test["ece_calibrated"], 1),
              help=f"Expected calibration error: the average gap between stated confidence and accuracy. It was "
                   f"{test['ece_uncalibrated']:.2%} before temperature scaling, so the model was already well calibrated "
                   "and scaling changed little.")
    st.caption("Bootstrap 95% confidence intervals: " + "; ".join(
        f"{n} {test[k + '_ci'][0]:.1%} to {test[k + '_ci'][1]:.1%}" for n, k in
        [("accuracy", "accuracy"), ("balanced accuracy", "balanced_accuracy"), ("macro-F1", "macro_f1")]))

    sel = test["selective"]
    with st.container(border=True):
        st.markdown(f"**What the status line catches.** It held back {sel['held_back']} of {test['n']:,} test cells "
                    f"({sel['held_by_confidence']} for low confidence, {sel['held_by_unfamiliar']} as unfamiliar; some fail both) and caught "
                    f"{sel['errors_held_back']} of the {sel['errors']} mistakes. Cells marked Confident were right "
                    f"{sel['accuracy_on_accepted']:.1%} of the time. **It does not catch everything:** "
                    f"{sel['errors_marked_confident']} mistakes were still marked Confident, "
                    f"{sel['errors_marked_confident_above_90']} of them above 90% confidence.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Recall by cell type**")
        pc = pd.DataFrame(test["per_class"])
        pc["name"] = [display_name(c) for c in pc["class"]]
        st.altair_chart(ui.per_class_chart(pc), use_container_width=True)
    with c2:
        st.markdown("**Where the mistakes go** (rows are the expert label)")
        short = {"immature_granulocyte": "Immature gran.", "erythroblast": "Erythroblast"}
        names = [short.get(c, display_name(c)) for c in CLASSES]
        st.altair_chart(ui.confusion_chart(np.array(test["confusion_matrix"]), names), use_container_width=True)

    st.markdown("**Hardest cells, by subtype**")
    subs = pd.DataFrame(test["subtypes"]).sort_values("recall")
    subs["most_common_error"] = subs["most_common_error"].map(lambda c: display_name(c) if isinstance(c, str) else "None")
    st.dataframe(subs.rename(columns={"subtype": "Subtype", "n": "Test cells", "recall": "Correct",
                                      "most_common_error": "Most common wrong answer"}),
                 hide_index=True, use_container_width=True, height=36 * (len(subs) + 1) + 3,
                 column_config={"Correct": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})

    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.markdown("**Does the confidence mean what it says?** Dots near the gray diagonal are well calibrated. "
                    "Dot size is the number of test cells; almost all sit in the top bin.")
        st.altair_chart(ui.reliability_chart(test["reliability"]), use_container_width=True)
    with c4:
        st.markdown("**Answer fewer cells, get more right.** Each dot is a confidence cut-off.")
        st.altair_chart(ui.coverage_chart(sel["curve"], sel["confidence_threshold"]), use_container_width=True)

    st.markdown("**Models compared on the validation set**")
    cmp = pd.DataFrame(comp)
    cmp = cmp.rename(columns={"name": "Model", "val_macro_f1": "Validation macro-F1", "val_accuracy": "Validation accuracy",
                              "params": "Parameters", "train_minutes": "Training minutes (2 CPU cores)",
                              "train_energy_wh": "Training energy (Wh)"})
    cols = [c for c in ["Model", "Validation macro-F1", "Validation accuracy", "Parameters",
                        "Training minutes (2 CPU cores)", "Training energy (Wh)"] if c in cmp]
    st.dataframe(cmp[cols], hide_index=True, use_container_width=True, column_config={
        "Validation macro-F1": st.column_config.NumberColumn(format="%.3f"),
        "Validation accuracy": st.column_config.NumberColumn(format="%.3f"),
        "Parameters": st.column_config.NumberColumn(format="%d")})
    reason = meta["selection"]["reason"]
    st.caption(reason[0].upper() + reason[1:] + ".")

    st.markdown("**Images from somewhere else**")
    ch = ext["checks"]
    other = pd.DataFrame([
        {"Images": "Test cells from the training lab (PBC)", "Count": ch["pbc_test_in_distribution"]["n"],
         "Flagged as unfamiliar": ch["pbc_test_in_distribution"]["flagged_unfamiliar"]},
        {"Images": "White cells from another lab and camera (BCCD)", "Count": ch["bccd_white_cells"]["n"],
         "Flagged as unfamiliar": ch["bccd_white_cells"]["flagged_unfamiliar"]},
        {"Images": "Red cells only, no white cell (BCCD)", "Count": ch["bccd_red_cells_only"]["n"],
         "Flagged as unfamiliar": ch["bccd_red_cells_only"]["flagged_unfamiliar"]},
        {"Images": "Random noise", "Count": ch["random_noise"]["n"], "Flagged as unfamiliar": ch["random_noise"]["flagged_unfamiliar"]},
        {"Images": "Blank fields", "Count": ch["blank_fields"]["n"], "Flagged as unfamiliar": ch["blank_fields"]["flagged_unfamiliar"]},
    ])
    st.dataframe(other, hide_index=True, use_container_width=True, column_config={
        "Flagged as unfamiliar": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
    st.caption("BCCD has no white cell subtype labels, so accuracy there cannot be measured. The check shows only "
               "whether the app notices that the images are different. See About and limits.")

    sp = exp["speed"]
    st.markdown("**Speed and size**")
    s1, s2, s3 = st.columns(3)
    s1.metric("Time per cell (ONNX Runtime)", f"{sp['onnx_ms_per_image']:.1f} ms",
              delta=f"{sp['onnx_ms_per_image'] - sp['keras_ms_per_image']:+.1f} ms vs TensorFlow", delta_color="inverse")
    s2.metric("32 cells at once", f"{sp['onnx_ms_per_32_batch']:.0f} ms")
    s3.metric("Model file", f"{sp['onnx_file_mb']:.1f} MB")
    st.caption(f"The app runs the exported ONNX model, so it does not need TensorFlow. On 64 test images the export "
               f"gave the same top class every time (largest logit difference {exp['parity']['max_abs_logit_diff']:.1e}).")


# -- tab 5: about ----------------------------------------------------------------------------------
with tab_about:
    a1, a2 = st.columns([1.2, 1], gap="large")
    with a1:
        st.subheader("What this is")
        st.markdown(
            "A teaching and research prototype that classifies single cells from Romanowsky-stained peripheral blood "
            "smears into seven types: basophil, eosinophil, erythroblast, immature granulocyte, lymphocyte, monocyte "
            "and neutrophil. I built it to learn how image models behave on the kind of slides I read at the bench, "
            "and where they fail.")
        st.subheader("How it works")
        st.markdown(
            f"1. The image is cropped to its center square and resized to {meta['input_size']} x {meta['input_size']} pixels, "
            "exactly as in training.\n"
            "2. A small convolutional network, trained from scratch, scores the seven cell types. Temperature scaling "
            f"(T = {meta['temperature']:.2f}) was fitted on validation images; the model was already well calibrated, so it "
            "changed little.\n"
            "3. The heatmap is a class activation map: the network's last feature maps weighted by the evidence for the "
            "chosen class.\n"
            "4. Two checks decide the status line. Confidence must reach "
            f"{meta['confidence_threshold']:.0%}, the level that sends the least confident 2% of validation cells for review. "
            "The image's features must also sit within the range of validation images (a Mahalanobis distance below the "
            "99th percentile).")
        st.subheader("Data")
        st.markdown(
            "Training images come from the PBC dataset: normal peripheral blood cells photographed with a CellaVision DM96 "
            "at the Hospital Clinic of Barcelona (Acevedo et al., *Data in Brief*, 2020; CC BY 4.0), in the 13,344-image "
            "version published with cell boxes by medmabcf on GitHub. Platelets are not included in that version. "
            "The BCCD images used for the other-lab check are from the BCCD dataset (MIT licence).")
    with a2:
        st.subheader("Limits you should know")
        st.markdown(
            "- **One lab, one scanner.** Every training image came from the same hospital and the same automated microscope. "
            "Images from other stains, cameras or magnifications, including phone photos through an eyepiece, will often be "
            "wrong. The unfamiliar-image check catches many of these, not all.\n"
            "- **Normal donors.** The source cells come from people without infection, blood disease or cancer. Blasts, "
            "atypical or reactive lymphocytes, malaria parasites, sickle cells and other abnormal findings are not classes "
            "here, and the model will force them into one of the seven.\n"
            "- **One cell per image.** It does not find cells on a whole smear. Crop them first.\n"
            "- **Neighboring stages overlap.** The largest group of test errors (18 of 44) is neutrophils, mostly band forms, "
            "read as immature granulocytes: the band and metamyelocyte boundary.\n"
            "- **Confident mistakes still happen.** On test cells, 19 of 44 errors passed the status line as Confident.\n"
            "- **Reference ranges are generic.** The differential compares against typical adult ranges from mostly Western "
            "populations. Many healthy people of West African ancestry have lower neutrophil counts (Duffy-null associated "
            "neutrophil count). Use local ranges.\n"
            "- **No patient split.** The dataset has no patient identifiers, so cells from one person can sit in both "
            "training and test sets. Exact duplicates were kept in one split, but results on new patients may be lower.")
        with st.container(border=True):
            st.markdown("**Disclaimer.** This is not a medical device. It has not been reviewed by any regulator or tested "
                        "in a clinical laboratory. Do not use it to make decisions about patients, and do not upload "
                        "images with patient names or identifiers.")
        st.caption("Onipayede John Kwaku. Code under the MIT licence.")
