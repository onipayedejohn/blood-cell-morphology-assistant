"""Blood Cell Morphology Assistant (version 2): Streamlit front end.

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
from bloodsmear.inference import Analysis, CellClassifier, Prediction  # noqa: E402
from bloodsmear.quality import check_image  # noqa: E402
import ui  # noqa: E402

st.set_page_config(page_title="Blood Cell Morphology Assistant", page_icon="🔬", layout="wide",
                   initial_sidebar_state="collapsed")
ui.inject_css()

SAMPLES = ROOT / "data" / "samples"
DEMO = ROOT / "data" / "demo_batch"
METRICS = METRICS_DIR
FILE_TYPES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
MODES = {"Auto": "auto", "One cell": "single", "Whole field": "field"}


# -- cached resources ------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the models")
def load_classifier() -> CellClassifier:
    return CellClassifier()


@st.cache_data
def load_json(name: str) -> dict | None:
    p = METRICS / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def sample_catalog() -> pd.DataFrame:
    df = pd.read_csv(SAMPLES / "samples.csv")
    df["group"] = "Single cells (training lab, test set)"
    df["label"] = [f"{display_name(c)} ({f.rsplit('_', 1)[1].split('.')[0]})" for c, f in zip(df["class"], df["file"])]
    df["note"] = [f"From the held-out PBC test set, so the model never trained on it. Expert label: {display_name(c)}."
                  for c in df["class"]]
    other = pd.read_csv(SAMPLES / "other" / "other.csv").fillna("")
    other["file"] = "other/" + other["file"]
    order = ["Single cells (training lab, test set)", "Single cells from another lab", "Whole microscope fields",
             "Images that are not blood cells"]
    out = pd.concat([df, other], ignore_index=True)
    out["group"] = pd.Categorical(out["group"], categories=order, ordered=True)
    return out.sort_values("group", kind="stable")


@st.cache_data(show_spinner="Classifying the demo batch")
def classify_demo() -> tuple[list[dict], pd.Series]:
    files = sorted(DEMO.glob("*.jpg"))
    preds = load_classifier().predict_many([Image.open(f) for f in files])
    key = pd.read_csv(DEMO / "demo_batch_labels.csv").set_index("file")["class"]
    return [pred_row(f.name, p) for f, p in zip(files, preds)], key


clf = load_classifier()
meta = clf.meta

STATUS = {
    "confident": {"short": "Confident", "title": "Confident result", "icon": ":material/check_circle:", "box": st.success,
                  "text": "Confidence is above the review level, and the cell's features sit within the range of the "
                          "training cells. On a lab the model has not been adapted to, even confident answers are often "
                          "wrong."},
    "review": {"short": "Check yourself", "title": "Check this cell yourself", "icon": ":material/visibility:",
               "box": st.warning,
               "text": f"Confidence is below {clf.conf_threshold:.0%}, the level that sends the least confident 2% of "
                       "validation cells to a person. Look at this cell yourself before using the answer."},
    "unfamiliar": {"short": "Unusual cell", "title": "This cell looks unlike the training cells", "icon": ":material/help:",
                   "box": st.warning,
                   "text": "It is a blood cell, but its features sit outside the range of the training images. It may be "
                           "an abnormal cell, or an unusual stain or magnification. Treat the answer as a suggestion."},
    "not_blood": {"short": "Not a blood cell", "title": "This does not look like a stained blood cell",
                  "icon": ":material/block:", "box": st.error, "text": ""},
    "no_white_cell": {"short": "No white cell", "title": "No white cell here", "icon": ":material/search_off:",
                      "box": st.warning, "text": ""},
}

for k, v in {"history": [], "seen_tips": False, "force": None, "what": "Auto"}.items():
    st.session_state.setdefault(k, v)


def lab_caveat() -> str:
    """One line on how far to trust an answer for images from a new lab."""
    adapted = meta.get("adapted")
    if adapted and adapted.get("after"):
        return (f":material/tune: Adapted to {adapted['name']} with {adapted['local_train']} local cells: "
                f"{adapted['after']['accuracy']:.0%} right on {adapted['local_test']} held-back local cells.")
    v2 = load_json("v2_results.json")
    if not v2:
        return ""
    accs = [v["v2_never_saw_this_lab"]["accuracy"] for v in v2["unseen_labs"].values()]
    return (f":material/info: On labs it never saw, this model was right only {min(accs):.0%} to {max(accs):.0%} of "
            "the time, even when confident. Unless it has been adapted to your lab (see About), treat the answer "
            "as a suggestion.")


def confidence_help() -> str:
    text = "On the training lab most cells score above 95%. On new labs scores are lower, and a high score is no guarantee"
    v2 = load_json("v2_results.json")
    hi = [b["accuracy"] for v in (v2 or {}).get("unseen_labs", {}).values()
          for b in v["v2_never_saw_this_lab"].get("by_confidence", []) if b["min_confidence"] == 0.9 and b["accuracy"]]
    if hi:
        text += f": answers above 90% were right only {min(hi):.0%} to {max(hi):.0%} of the time on two unseen labs"
    return text + "."


def name_of(label: str | None) -> str:
    return display_name(label) if label else "No result"


def pred_row(name: str, p: Prediction) -> dict:
    return {"Image": ui.thumb_uri(p.image_uint8), "File": name, "Predicted": name_of(p.label), "key": p.label,
            "Confidence": p.confidence if p.label else None, "Status": STATUS[p.status]["short"], "status": p.status}


def add_history(p: Prediction, source: str):
    key = (source, p.label, round(p.confidence, 4))
    if any(h["_key"] == key for h in st.session_state.history[-40:]):
        return
    st.session_state.history.append({
        "_key": key, "Time": datetime.now().strftime("%H:%M:%S"), "Image": ui.thumb_uri(p.image_uint8),
        "Source": source, "Result": name_of(p.label), "Confidence": p.confidence if p.label else None,
        "Status": STATUS[p.status]["short"]})


def read_upload(file) -> Image.Image | None:
    try:
        img = Image.open(io.BytesIO(file.getvalue()))
        img.load()
        return img
    except (UnidentifiedImageError, OSError):
        st.error(f"{file.name} could not be read as an image. Upload a JPG, PNG, BMP or TIFF file.",
                 icon=":material/broken_image:")
        return None


# -- header --------------------------------------------------------------------------------
st.title("Blood cell morphology assistant")
st.markdown(
    '<p class="lede">Finds and classifies white cells in images of stained peripheral blood smears. It refuses images '
    'that are not blood smears, shows which part of each cell drove the answer, and says when it is unsure.</p>',
    unsafe_allow_html=True)
st.caption(":material/science: Research and teaching prototype. Not a medical device, and not validated for patient "
           "care. See **About and limits**.")

tab_one, tab_diff, tab_hist, tab_perf, tab_about = st.tabs(
    ["Analyze an image", "Differential count", "Session history", "Model performance", "About and limits"])


# -- tab 1 ---------------------------------------------------------------------------------
def report_text(p: Prediction, source: str) -> str:
    lines = ["Blood Cell Morphology Assistant (research prototype, not for clinical use)",
             f"Date: {datetime.now():%Y-%m-%d %H:%M}", f"Image: {source}", f"Result: {name_of(p.label)}",
             f"Confidence: {p.confidence:.1%}", f"Status: {STATUS[p.status]['title']}",
             "Other possibilities: " + ", ".join(f"{display_name(c)} {v:.1%}" for c, v in p.top[1:]),
             f"Blood cell check (gatekeeper): {p.gate.get('white_cell', 1):.0%} white cell"]
    lines += [f"Image check: {i.message}" for i in p.issues]
    return "\n".join(lines) + "\n"


def refusal(gate: dict[str, float], key: str, crop: bool = False):
    p_white = gate.get("white_cell", 0)
    STATUS["not_blood"]["box"](
        f"**This does not look like a stained blood smear, so no cell type is given.** A separate check, trained on "
        f"blood cells from four sources and on thousands of other images (photos, paper, documents, screens), puts the "
        f"chance that this is a white blood cell at {p_white:.0%}.", icon=":material/block:")
    with st.expander("Sure this is a blood smear?"):
        st.markdown("- Use a color image of a Romanowsky-stained smear (Giemsa, Leishman, Wright or Field's).\n"
                    "- Focus on one area at oil immersion, and crop so a white cell sits near the center.\n"
                    "- Avoid photos of the whole slide, the microscope or the screen.")
        if st.button("Classify it anyway", key=f"force_{key}", icon=":material/warning:"):
            st.session_state.force = key
            st.rerun()
        st.caption("Forcing a result on an image the check refused gives an answer that means nothing for images "
                   "that are not blood cells.")


def show_result(p: Prediction, source: str, forced: bool = False):
    s = STATUS[p.status]
    if forced:
        st.error("**Forced result.** The blood cell check refused this image. The answer below is not reliable.",
                 icon=":material/warning:")
    else:
        s["box"](f"**{s['title']}.** {s['text']}", icon=s["icon"])
    c1, c2 = st.columns(2, gap="small")
    c1.image(ui.upscale(p.image_uint8), caption=f"What the model saw: recentered on the nucleus, background color balanced, {clf.size} x {clf.size} pixels",
             width="stretch")
    c2.image(ui.upscale(overlay(p.image_uint8, p.cam)),
             caption="Brighter cyan means that area pushed the model towards its answer (an 8 x 8 map, so it is coarse)",
             width="stretch")
    info = CELL_INFO[p.label or p.top[0][0]]
    st.markdown(f'<div class="readout">{info.name}</div><div class="readout-sub">Model answer from seven cell types</div>',
                unsafe_allow_html=True)
    st.caption(lab_caveat())
    m1, m2, m3 = st.columns(3)
    m1.metric("Confidence", ui.conf(p.confidence), help=confidence_help())
    typ = "Typical" if p.unfamiliarity < 90 else ("Less typical" if not p.unfamiliar else "Unusual")
    m2.metric("Cell typicality", typ, help=f"More unusual than {p.unfamiliarity:.0f}% of validation cells from all "
                                           "training labs. Above the 99th percentile the app warns you.")
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
                       icon=":material/download:", key=f"dl_{source}")


def show_field(an: Analysis, source: str):
    fr = an.field_result
    cells = fr.cells
    if not cells:
        st.warning("**No white cells were found in this image.** The finder looks for dark blue-violet nuclei. Try a "
                   "sharper photo, more light, or crop closer to a white cell.", icon=":material/search_off:")
        return
    statuses = [c.status for c in cells]
    n_conf = statuses.count("confident")
    n_look = sum(s in ("review", "unfamiliar") for s in statuses)
    n_not = sum(s in ("not_blood", "no_white_cell") for s in statuses)
    n_cells = len(cells) - n_not
    if n_cells == 0:
        st.warning(f"**No white cell found.** The finder picked up {len(cells)} dark spot{'s' if len(cells) != 1 else ''}, "
                   "but the blood cell check rejected each one. This may be a field with red cells only, or a photo "
                   "that is too blurred or dark.", icon=":material/search_off:")
    else:
        st.success(f"**Found {n_cells} white cell{'s' if n_cells != 1 else ''}.** Each one was cut out, checked and "
                   "classified on its own." + (f" {n_not} other spot{'s were' if n_not != 1 else ' was'} rejected by "
                                               "the blood cell check." if n_not else ""), icon=":material/grid_view:")
    a, b, c = st.columns(3)
    a.metric("Confident", n_conf)
    b.metric("Need a look", n_look)
    c.metric("Not a white cell", n_not, help="Spots the finder picked up that the blood cell check rejected.")
    st.image(ui.annotate_field(fr.image, [d.display_box() for d in fr.detections], statuses), width="stretch",
             caption="Numbers match the cells below. Blue: confident. Amber: needs a look. Gray: not a white cell.")
    for row in range(0, len(cells), 4):
        cols = st.columns(4)
        for k, (col, p) in enumerate(zip(cols, cells[row:row + 4]), start=row + 1):
            with col:
                st.image(ui.upscale(p.image_uint8, 200), width="stretch")
                st.markdown(f"**{k}. {name_of(p.label)}**" + (f"  {ui.conf(p.confidence)}" if p.label else ""))
                st.caption(STATUS[p.status]["short"])
            add_history(p, f"{source}, cell {k}")
    counted = [p for p in cells if p.label]
    if counted:
        dif = summarize([p.label for p in counted], [p.status for p in counted])
        with st.expander(f"Differential for this field ({dif.counted_wbc} white cells counted)"):
            st.dataframe(dif.table, hide_index=True, width="stretch")
            st.caption("One field rarely holds enough cells for a differential. Use the Differential count tab with "
                       "several fields.")


with tab_one:
    left, right = st.columns([0.9, 1.5], gap="large")
    img, source = None, ""
    with left:
        with st.container(border=True):
            st.subheader("Choose an image")
            src_mode = st.segmented_control("Image source", ["Sample", "Upload"], default="Sample",
                                            label_visibility="collapsed", key="mode_one")
            if src_mode == "Upload":
                up = st.file_uploader("Upload an image", type=FILE_TYPES, key="one_upload",
                                      help="One cropped white cell, or a whole microscope field.")
                if up is not None:
                    img, source = read_upload(up), up.name
            else:
                cat = sample_catalog()
                group = st.selectbox("Sample set", list(cat["group"].cat.categories), key="sample_group")
                sub = cat[cat["group"] == group]
                choice = st.selectbox("Sample", sub["label"].tolist(), key=f"pick_{group}")
                row = sub[sub["label"] == choice].iloc[0]
                img, source = Image.open(SAMPLES / row["file"]), f"Sample: {choice}"
                st.image(img, width=170)
                st.caption(row["note"])
            st.segmented_control("What is in the image?", list(MODES), key="what",
                                 help="Auto decides from the image. Choose Whole field for a photo with several cells.")
        with st.expander("How to get a good result", expanded=not st.session_state.seen_tips):
            st.markdown(
                "1. Use a color photo of a Romanowsky-stained smear (Giemsa, Leishman, Wright or Field's) at oil immersion.\n"
                "2. Upload either one cell cropped near the center, or a whole field. The app finds the white cells.\n"
                "3. Read the status first. If it says to check the cell yourself, do that before using the answer.")
        st.session_state.seen_tips = True

    with right:
        if img is None:
            with st.container(border=True):
                st.markdown("#### No image yet")
                st.markdown("Pick a sample or upload your own image. Results, a heatmap and notes on the cell type "
                            "appear here.")
        else:
            blockers = [i for i in check_image(img) if i.level == "block"]
            mode = MODES.get(st.session_state.what or "Auto", "auto")
            key = f"{source}|{mode}"
            forced = st.session_state.force == key
            if blockers:
                for bl in blockers:
                    st.error(bl.message, icon=":material/crop:")
            else:
                with st.spinner("Checking and classifying"):
                    an = clf.analyze(img, mode=mode, force=forced)
                if an.trimmed:
                    st.caption("The dark border around the microscope field was trimmed before analysis.")
                if an.mode == "refused":
                    refusal(an.whole_image_gate, key)
                elif an.mode == "field":
                    show_field(an, source)
                else:
                    p = an.single
                    if p.status == "not_blood":
                        refusal(p.gate, key)
                    elif p.status == "no_white_cell":
                        st.warning("**No white cell in the middle of this image.** If it is a blood smear, search the "
                                   "whole image instead, or crop around a white cell.", icon=":material/search_off:")
                        st.button("Search the whole image", icon=":material/grid_view:",
                                  on_click=lambda: st.session_state.update(what="Whole field"))
                    else:
                        add_history(p, source)
                        show_result(p, source, forced=forced and p.gate.get("white_cell", 1) < clf.gate_threshold)


# -- tab 2: differential count -------------------------------------------------------------------
with tab_diff:
    st.subheader("Differential count")
    st.markdown('<p class="lede">Upload several images, single cells or whole fields, or try the demo batch. Cells the '
                'model is unsure about are held back for you to review instead of being counted.</p>',
                unsafe_allow_html=True)
    st.caption("A manual differential counts at least 100 consecutive white cells. These numbers only mean something if "
               "you upload every cell or field you look at, not selected examples. Ranges are typical adult values; use "
               "your laboratory's own. Flags compare percentages, not absolute counts.")
    dmode = st.segmented_control("Cells to count", ["Demo batch of 120 cells", "Upload images"],
                                 default="Demo batch of 120 cells", label_visibility="collapsed", key="mode_diff")
    rows, answer_key, refused = [], None, 0
    if dmode == "Upload images":
        ups = st.file_uploader("Upload images", type=FILE_TYPES, accept_multiple_files=True, key="diff_upload")
        if ups:
            with st.spinner(f"Analyzing {len(ups)} images"):
                for f in ups:
                    im = read_upload(f)
                    if im is None or min(im.size) < 64:
                        continue
                    an = clf.analyze(im)
                    if an.mode == "refused":
                        refused += 1
                    elif an.mode == "field":
                        rows += [pred_row(f"{f.name} #{k}", p) for k, p in enumerate(an.field_result.cells, 1)]
                    else:
                        rows.append(pred_row(f.name, an.single))
        else:
            with st.container(border=True):
                st.markdown("#### No images uploaded")
                st.markdown("Drop in several images at once: single cells, whole fields, or both.")
    else:
        rows, answer_key = classify_demo()
        st.caption("The demo batch is 120 test-set cells in proportions that resemble a mildly left-shifted adult "
                   "differential with a few nucleated red cells. It is not a real patient.")

    if refused:
        st.warning(f"{refused} image(s) did not look like blood smears and were left out.", icon=":material/block:")
    if rows:
        dfc = pd.DataFrame(rows)
        dif = summarize(dfc["key"].tolist(), dfc["status"].tolist())
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Cells found", len(dfc))
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
            st.altair_chart(ui.differential_chart(table, REFERENCE_RANGES), width="stretch")
        st.dataframe(table.drop(columns=["key"]), hide_index=True, width="stretch",
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
        order = {"Not a blood cell": 0, "No white cell": 0, "Unusual cell": 1, "Check yourself": 2, "Confident": 3}
        show = show.sort_values("Status", key=lambda s: s.map(order), kind="stable")
        st.dataframe(show, hide_index=True, width="stretch", height=380, column_config={
            "Image": st.column_config.ImageColumn("Cell", width="small"),
            "Confidence": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
        st.download_button("Download per-cell results (CSV)", show.drop(columns=["Image"]).to_csv(index=False),
                           file_name="differential_cells.csv", icon=":material/download:")


# -- tab 3: history -------------------------------------------------------------------------------
with tab_hist:
    st.subheader("Cells classified in this session")
    hist = st.session_state.history
    if not hist:
        st.markdown("Nothing yet. Each cell you analyze on the first tab is added here, so you can compare results side "
                    "by side. The list clears when you close the page.")
    else:
        h = pd.DataFrame(hist).drop(columns=["_key"])
        a, b, c = st.columns(3)
        a.metric("Cells", len(h))
        b.metric("Confident", int((h["Status"] == "Confident").sum()))
        c.metric("Needing a look or refused", int((h["Status"] != "Confident").sum()))
        st.dataframe(h.iloc[::-1], hide_index=True, width="stretch", column_config={
            "Image": st.column_config.ImageColumn("Cell", width="small"),
            "Confidence": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
        c1, c2 = st.columns([1, 4])
        c1.download_button("Download CSV", h.drop(columns=["Image"]).to_csv(index=False), file_name="session_history.csv",
                           icon=":material/download:")
        if c2.button("Clear history", icon=":material/delete:"):
            st.session_state.history = []
            st.rerun()


# -- tab 4: performance -----------------------------------------------------------------------------
with tab_perf:
    v2 = load_json("v2_results.json")
    finder = load_json("finder_results.json")
    st.subheader("How the model performs")
    if v2 is None:
        st.info("Performance results are not available in this copy of the app.")
    else:
        fin, uns, pipe = v2["final"], v2["unseen_labs"], v2["app_pipeline"]
        st.markdown('<p class="lede">Version 2 was trained on cells from four sources, sees every cell in a standard '
                    'framing and color balance, and was trained with augmentation that imitates other stains and cameras. '
                    'Version 1 was trained on one lab and was right 0% to 30% of the time on three other sources.</p>',
                    unsafe_allow_html=True)

        st.markdown("**On labs the model never saw.** Each row comes from a model trained with that lab left out completely.")
        lab_names = {"bccd": "BCCD", "jtsc": "Jiangxi Tecom (China)"}
        st.dataframe(pd.DataFrame([{
            "Lab": lab_names[k], "Cells": v["v1"]["n"], "Version 1 accuracy": v["v1"]["accuracy"],
            "Version 2 accuracy": v["v2_never_saw_this_lab"]["accuracy"],
            "Version 2 balanced accuracy": v["v2_never_saw_this_lab"]["balanced_accuracy"]} for k, v in uns.items()]),
            hide_index=True, width="stretch", column_config={
                c: st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)
                for c in ["Version 1 accuracy", "Version 2 accuracy", "Version 2 balanced accuracy"]})
        st.caption("These are the honest numbers for a new hospital: better than version 1, but far below the training "
                   "lab. Images from your own laboratory are likely to behave like this until the model is adapted.")
        conf_rows = []
        for k, v in uns.items():
            for b in v["v2_never_saw_this_lab"].get("by_confidence", []):
                if b["min_confidence"] in (0.0, 0.9, 0.99) and b["accuracy"] is not None:
                    conf_rows.append({"Lab": lab_names[k], "Cells with confidence at least": f"{b['min_confidence']:.0%}",
                                      "Share of cells": b["share"], "Accuracy": b["accuracy"]})
        if conf_rows:
            with st.expander("Does high confidence help on a new lab? Not enough to rely on."):
                st.caption("Answers the status line marked Confident were right only " + " and ".join(
                    f"{v['v2_never_saw_this_lab']['status_line']['accuracy_on_confident']:.0%} ({lab_names[k]})"
                    for k, v in uns.items()) + " of the time on these unseen labs.")
                st.dataframe(pd.DataFrame(conf_rows), hide_index=True, width="stretch", column_config={
                    c: st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)
                    for c in ["Share of cells", "Accuracy"]})
        adapt = load_json("local_adaptation.json")
        if adapt:
            st.markdown("**Adapting to a new lab** with `scripts/finetune_local.py` (about 3 minutes on 2 CPU cores).")
            st.dataframe(pd.DataFrame([{
                "Lab": lab_names.get(k, k), "Local cells trained on": v["local_train"],
                "Held-back cells": v["local_test"], "Accuracy before": v["before"]["accuracy"],
                "Accuracy after": v["after"]["accuracy"], "Balanced accuracy after": v["after"]["balanced_accuracy"]}
                for k, v in adapt["labs"].items()]),
                hide_index=True, width="stretch", column_config={
                    c: st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)
                    for c in ["Accuracy before", "Accuracy after", "Balanced accuracy after"]})
            st.caption("Each run started from the model that had never seen that lab. Small test sets, so read these "
                       "as rough. Rare types can stay weak or even get worse: see the recall by type in "
                       "reports/metrics/local_adaptation.json.")

        st.markdown("**The final model on held-out test cells**")
        t1, t2, t3 = st.columns(3)
        t1.metric("Training lab (PBC) accuracy", ui.pct(fin["pbc_test"]["accuracy"], 1),
                  help=f"{fin['pbc_test']['n']:,} test cells. Macro-F1 {fin['pbc_test']['macro_f1']:.1%}.")
        t2.metric("Other sources, adapted", ui.pct(fin["external_pooled"]["accuracy"], 1),
                  help=f"{fin['external_pooled']['n']} test cells from BCCD, Jiangxi Tecom and the CellaVision blog. "
                       "Other cells from these sources were in training, so this is accuracy after adaptation, not on a new lab.")
        t3.metric("Calibration error (PBC)", ui.pct(fin["pbc_test"]["ece"], 1))
        names = {"bccd": "BCCD", "jtsc": "Jiangxi Tecom", "cvblog": "CellaVision blog"}
        st.dataframe(pd.DataFrame([{"Test cells": names[k], "Cells": v["n"], "Accuracy": v["accuracy"],
                                    "Balanced accuracy": v["balanced_accuracy"]}
                                   for k, v in fin["external_test_halves"].items()]),
                     hide_index=True, width="stretch", column_config={
                         c: st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)
                         for c in ["Accuracy", "Balanced accuracy"]})

        st.markdown("**Refusing images that are not blood cells**")
        nice = {"pbc_test_cells": "White cells, training lab (should pass)", "bccd_test_cells": "White cells, BCCD (should pass)",
                "jtsc_test_cells": "White cells, Jiangxi Tecom (should pass)",
                "cvblog_test_cells": "White cells, CellaVision blog (should pass)",
                "smear_without_white_cell": "Smear patches with no white cell (should be refused)",
                "cifar_unseen_photos": "Everyday photos never seen in training (should be refused)",
                "synthetic_new_seed": "Paper, documents, screens, noise (should be refused)",
                "pattern_type_never_trained": "A pattern type never used in training (should be refused)",
                "scikit_image_photos": "Other photos, text pages and a histology image (should be refused)"}
        st.dataframe(pd.DataFrame([{"Images": nice[k], "Count": v["n"], "Refused": v["refused_by_gatekeeper"]}
                                   for k, v in pipe.items() if k in nice]),
                     hide_index=True, width="stretch",
                     column_config={"Refused": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
        rep = pd.DataFrame(pipe["user_report_cases"])
        st.caption("Reported cases: " + "; ".join(f"{r.image}: {'refused' if r.refused else 'NOT refused'}"
                                                   for r in rep.itertuples()) + ".")

        if finder:
            st.markdown("**Finding white cells in whole fields** (held-out images with expert boxes)")
            ft = finder["test"]
            st.dataframe(pd.DataFrame([{"Source": {"bccd": "BCCD", "draaslan": "draaslan", "pbc": "PBC"}[k],
                                        "Images": v["images"], "White cells": v["white_cells"], "Found": v["recall"],
                                        "False alarms per image": round(v["false_alarms_per_image"], 2)}
                                       for k, v in ft.items()]),
                         hide_index=True, width="stretch",
                         column_config={"Found": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})

        c1, c2 = st.columns(2, gap="large")
        pc = pd.DataFrame(fin["pbc_test"]["per_class"])
        pc["name"] = [display_name(c) for c in pc["class"]]
        with c1:
            st.markdown("**Recall by cell type, training lab test set**")
            st.altair_chart(ui.per_class_chart(pc), width="stretch")
        with c2:
            st.markdown("**Where the mistakes go** (rows are the expert label)")
            short = {"immature_granulocyte": "Immature gran.", "erythroblast": "Erythroblast"}
            st.altair_chart(ui.confusion_chart(np.array(fin["pbc_test"]["confusion_matrix"]),
                                               [short.get(c, display_name(c)) for c in CLASSES]), width="stretch")
        sl = fin["pbc_test"]["status_line"]
        st.caption(f"Status line on the training lab's test set: {sl['held_back']} cells held back, catching "
                   f"{sl['errors_held_back']} of {sl['errors']} mistakes; cells marked Confident were right "
                   f"{sl['accuracy_on_confident']:.1%} of the time.")
        sp = v2["export"]["speed"]
        s1, s2, s3 = st.columns(3)
        s1.metric("Blood cell check per image", f"{sp['gate_ms']:.1f} ms")
        s2.metric("Classifier per cell", f"{sp['classifier_ms']:.1f} ms")
        s3.metric("Both model files", f"{sp['classifier_mb'] + sp['gate_mb']:.1f} MB")


# -- tab 5: about ----------------------------------------------------------------------------------
with tab_about:
    a1, a2 = st.columns([1.2, 1], gap="large")
    with a1:
        st.subheader("What this is")
        st.markdown(
            "A teaching and research prototype that finds white cells in images of Romanowsky-stained peripheral blood "
            "smears and classifies each one into seven types: basophil, eosinophil, erythroblast, immature granulocyte, "
            "lymphocyte, monocyte and neutrophil. I built it to learn how image models behave on the slides I read at "
            "the bench, and where they fail.")
        st.subheader("How it works")
        st.markdown(
            "1. **Blood cell check.** A small model trained on white cells, smear backgrounds and thousands of other "
            "images decides whether the image is a stained blood smear at all. If not, no cell type is given.\n"
            "2. **Find the cells.** For a whole field, a color-based finder locates the dark blue-violet nuclei and cuts "
            "out each white cell. Each cut-out goes through the blood cell check again.\n"
            "3. **Standardize.** Each cell is recentered on its nucleus at a fixed scale, and the background color is "
            "balanced to the same neutral gray, so tight crops, loose crops and whole-field cut-outs reach the model in the "
            "same framing.\n"
            "4. **Classify.** A convolutional network trained on cells from four sources scores the seven types. The "
            "heatmap shows where the evidence for the answer sits.\n"
            f"5. **Status.** A cell is marked Confident when its confidence reaches {clf.conf_threshold:.0%} (the level "
            "that sends the least confident 2% of validation cells for review) and its features sit within the range of "
            "the training cells.")
        st.subheader("Data")
        st.markdown(
            "- **PBC** (Acevedo et al., *Data in Brief*, 2020; CC BY 4.0): 13,344 cells from a CellaVision DM96 in Barcelona.\n"
            "- **BCCD** (MIT), with subtype-labeled crops from apple2373/BCCD: 351 cells.\n"
            "- **Zheng et al. (2018)** (GPL-3.0): 300 cells from Jiangxi Tecom, China, and 100 from the CellaVision blog.\n"
            "- Not-blood images for the check: CIFAR-10 photos and generated paper, documents and screens.")
    with a2:
        st.subheader("Limits you should know")
        st.markdown(
            "- **Not tested on your laboratory.** On two labs it never saw, it was right only about a third to a half of "
            "the time, and confidence did not fix that. Expect mistakes on your own slides until it is adapted to them.\n"
            "- **Phone photos are untested.** None of the training images were phone photos through an eyepiece. "
            "The app was only checked on simulated ones.\n"
            "- **Normal cells only.** Blasts, atypical lymphocytes, malaria parasites and abnormal red cells are not "
            "classes. The model will force them into one of the seven, though some get marked as unusual.\n"
            "- **The blood cell check can be wrong.** It may refuse a real smear photo, or, rarely, pass something else.\n"
            "- **Reference ranges are generic.** Many healthy people of West African ancestry have lower neutrophil counts "
            "(Duffy-null associated neutrophil count). Use local ranges.")
        st.subheader("Adapt it to your lab")
        st.markdown(
            "The fix for a new lab is some of its own labeled cells. Sort single-cell crops into one folder per "
            "cell type and run `python scripts/finetune_local.py --images data/local --name mylab`. It holds back 30% of "
            "your cells for testing, uses a fifth of the rest to pick the best epoch, trains on the remainder, and reports "
            "accuracy on the held-back cells before and after. In a test on two labs the model had never seen, training "
            "on 84 and 99 local cells (about 3 minutes on 2 CPU cores) raised accuracy on held-back cells from 30% to "
            "86% and from 42% to 81%, though some cell types stayed weak. Steps are in the README.")
        with st.container(border=True):
            st.markdown("**Disclaimer.** This is not a medical device. It has not been reviewed by any regulator or tested "
                        "in a clinical laboratory. Do not use it to make decisions about patients, and do not upload "
                        "images with patient names or identifiers.")
        st.caption("Code under the MIT licence.")
