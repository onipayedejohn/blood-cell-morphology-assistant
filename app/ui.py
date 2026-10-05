"""Small styling layer and chart builders.

The app leans on Streamlit's own components. This file only tightens the type
scale, styles the result readout, and builds the Altair charts with one blue
hue (checked for color-blind separation) so the performance tab reads as a set.
"""

from __future__ import annotations

import base64
import io

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

BLUE = "#2a78d6"        # bars and lines
BLUE_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
REFERENCE = "#d9dce8"   # reference ranges and the ideal-calibration diagonal
INK = "#1c1d2b"
MUTED = "#5b5e70"

CSS = """
<style>
.block-container { max-width: 1160px; padding-top: 2rem; padding-bottom: 4rem; }
h1 { font-size: 2rem !important; font-weight: 700 !important; letter-spacing: -0.015em; margin-bottom: .1rem !important; }
h2 { font-size: 1.35rem !important; font-weight: 650 !important; }
h3 { font-size: 1.1rem !important; font-weight: 650 !important; }
.lede { color: #4a4d60; font-size: 1.02rem; max-width: 68ch; margin: .2rem 0 .9rem; line-height: 1.5; }
.readout { font-size: 2rem; font-weight: 700; line-height: 1.1; color: #1c1d2b; margin: .1rem 0 .15rem; letter-spacing: -0.01em; }
.readout-sub { color: #4a4d60; font-size: .95rem; margin-bottom: .6rem; }
[data-testid="stMetricValue"] > div { font-size: 1.7rem; }
.fine { color: #5b5e70; font-size: .86rem; line-height: 1.45; }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
[data-testid="stImage"] img { border-radius: .5rem; }
.stTabs [data-baseweb="tab-list"] { gap: .25rem; overflow-x: auto; }
footer { visibility: hidden; }
@media (max-width: 640px) {
  .block-container { padding: 1.1rem .9rem 3rem; }
  h1 { font-size: 1.55rem !important; }
  .readout { font-size: 1.6rem; }
}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def pct(x: float, digits: int = 0) -> str:
    return f"{100 * x:.{digits}f}%"


def conf(x: float, digits: int = 0) -> str:
    """Never show a calibrated probability as a flat 100%."""
    if x > 0.995:
        return ">99%" if digits == 0 else ">99.5%"
    if x < 0.005 and digits == 0:
        return "<1%"
    return f"{100 * x:.{digits}f}%"


def thumb_uri(arr_or_img, size: int = 72) -> str:
    img = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    img = img.copy()
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def upscale(arr: np.ndarray, side: int = 360) -> Image.Image:
    return Image.fromarray(arr).resize((side, side), Image.Resampling.LANCZOS)


# -- whole-field annotation ----------------------------------------------------------------
BOX_COLORS = {"confident": (42, 120, 214), "review": (214, 140, 0), "unfamiliar": (214, 140, 0),
              "not_blood": (120, 124, 140), "no_white_cell": (120, 124, 140)}


def annotate_field(img: Image.Image, boxes, statuses, max_side: int = 900) -> Image.Image:
    """Draw a numbered square around each white cell found. Numbers match the list below the image."""
    from PIL import ImageDraw, ImageFont
    im = img.convert("RGB").copy()
    # shrink large photos; enlarge small ones so the numbers stay sharp when the image is stretched
    scale = max_side / max(im.size) if max(im.size) > max_side or max(im.size) < 600 else 1.0
    if scale != 1.0:
        im = im.resize((int(im.width * scale), int(im.height * scale)), Image.Resampling.LANCZOS)
    d = ImageDraw.Draw(im)
    lw = max(2, int(round(max(im.size) / 300)))
    try:
        font = ImageFont.load_default(size=max(14, int(max(im.size) / 40)))
    except TypeError:
        font = ImageFont.load_default()
    for k, (b, st_) in enumerate(zip(boxes, statuses), start=1):
        x0, y0, x1, y1 = [int(v * scale) for v in b]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(im.width - 1, x1), min(im.height - 1, y1)
        col = BOX_COLORS.get(st_, (42, 120, 214))
        d.rectangle((x0, y0, x1, y1), outline=col, width=lw)
        label = str(k)
        tw, th = d.textbbox((0, 0), label, font=font)[2:]
        d.rectangle((x0, y0, x0 + tw + 10, y0 + th + 8), fill=col)
        d.text((x0 + 5, y0 + 3), label, fill="white", font=font)
    return im


# -- charts ---------------------------------------------------------------------------------

def _base(chart: alt.Chart) -> alt.Chart:
    return chart.configure_view(strokeWidth=0).configure_axis(
        labelColor=MUTED, titleColor=MUTED, gridColor="#e7e9f1", domainColor="#c9ccd9", labelFontSize=12,
        titleFontSize=12, titleFontWeight="normal",
    ).configure(font="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif")


def per_class_chart(df: pd.DataFrame) -> alt.Chart:
    """Recall per class as horizontal bars, sorted, with precision in the tooltip."""
    bars = alt.Chart(df).mark_bar(color=BLUE, cornerRadiusEnd=4, height=18).encode(
        x=alt.X("recall:Q", title="Recall on the test set", scale=alt.Scale(domain=[0, 1.08]),
                axis=alt.Axis(format="%", values=[0, 0.25, 0.5, 0.75, 1])),
        y=alt.Y("name:N", sort="-x", title=None, axis=alt.Axis(labelLimit=180)),
        tooltip=[alt.Tooltip("name:N", title="Cell type"), alt.Tooltip("recall:Q", format=".1%"),
                 alt.Tooltip("precision:Q", format=".1%"), alt.Tooltip("support:Q", title="Test cells")],
    )
    labels = bars.mark_text(align="left", dx=4, color=INK, fontSize=12).encode(text=alt.Text("recall:Q", format=".1%"))
    return _base((bars + labels).properties(height=260))


def confusion_chart(cm: np.ndarray, names: list[str]) -> alt.Chart:
    rows = []
    for i, t in enumerate(names):
        total = cm[i].sum()
        for j, p in enumerate(names):
            rows.append({"True": t, "Predicted": p, "Cells": int(cm[i, j]), "Share": cm[i, j] / total if total else 0})
    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(
        x=alt.X("Predicted:N", sort=names, axis=alt.Axis(labelAngle=-40, labelOverlap=False, labelLimit=140)),
        y=alt.Y("True:N", sort=names, title="Expert label", axis=alt.Axis(labelLimit=140)),
    )
    rect = base.mark_rect(stroke="#f6f7fb", strokeWidth=2).encode(
        color=alt.Color("Share:Q", scale=alt.Scale(range=BLUE_RAMP, domain=[0, 1]), legend=None),
        tooltip=["True", "Predicted", "Cells", alt.Tooltip("Share:Q", format=".1%", title="Share of true class")],
    )
    text = base.mark_text(fontSize=12).encode(
        text=alt.condition("datum.Cells > 0", alt.Text("Cells:Q"), alt.value("")),
        color=alt.condition("datum.Share > 0.5", alt.value("white"), alt.value(INK)),
    )
    return _base((rect + text).properties(height=340))


def reliability_chart(rows: list[dict]) -> alt.Chart:
    df = pd.DataFrame(rows)
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color=REFERENCE, strokeWidth=2).encode(x="x:Q", y="y:Q")
    enc = dict(
        x=alt.X("mean_confidence:Q", title="Confidence the app reports", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        y=alt.Y("accuracy:Q", title="How often it was right", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("mean_confidence:Q", format=".1%", title="Mean confidence"),
                 alt.Tooltip("accuracy:Q", format=".1%", title="Accuracy"), alt.Tooltip("n:Q", title="Test cells")],
    )
    line = alt.Chart(df).mark_line(color=BLUE, strokeWidth=1.5, opacity=0.6).encode(**enc)
    dots = alt.Chart(df).mark_circle(color=BLUE, opacity=0.9, stroke="#f6f7fb", strokeWidth=2).encode(
        size=alt.Size("n:Q", scale=alt.Scale(type="sqrt", range=[40, 900]), legend=None), **enc)
    return _base((diag + line + dots).properties(height=280))


def coverage_chart(curve: list[dict], threshold: float) -> alt.Chart:
    df = pd.DataFrame([c for c in curve if c["accuracy"] is not None])
    base = alt.Chart(df).encode(x=alt.X("coverage:Q", title="Share of cells answered", axis=alt.Axis(format="%"),
                                        scale=alt.Scale(domain=[df.coverage.min() - 0.02, 1])))
    line = base.mark_line(color=BLUE, strokeWidth=2, point=alt.OverlayMarkDef(size=60, filled=True, color=BLUE)).encode(
        y=alt.Y("accuracy:Q", title="Accuracy on answered cells", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[df.accuracy.min() - 0.005, 1])),
        tooltip=[alt.Tooltip("threshold:Q", title="Confidence at least", format=".0%"),
                 alt.Tooltip("coverage:Q", format=".1%", title="Answered"),
                 alt.Tooltip("accuracy:Q", format=".2%", title="Accuracy")],
    )
    return _base(line.properties(height=260))


def differential_chart(df: pd.DataFrame, ranges: dict[str, tuple[float, float]]) -> alt.Chart:
    """Observed percentage per white cell type, over a gray band for the typical adult range."""
    d = df.copy()
    d["lo"] = [ranges[k][0] for k in d["key"]]
    d["hi"] = [max(ranges[k][1], 0.4) for k in d["key"]]
    y = alt.Y("Cell type:N", sort=list(d["Cell type"]), title=None, axis=alt.Axis(labelLimit=200))
    band = alt.Chart(d).mark_bar(color=REFERENCE, height=22).encode(
        x=alt.X("lo:Q", title="Percent of white cells"), x2="hi:Q", y=y,
        tooltip=[alt.Tooltip("Cell type:N"), alt.Tooltip("Typical adult range:N")])
    bars = alt.Chart(d).mark_bar(color=BLUE, height=10, cornerRadiusEnd=3).encode(
        x="Percent of white cells:Q", y=y,
        tooltip=[alt.Tooltip("Cell type:N"), alt.Tooltip("Count:Q"),
                 alt.Tooltip("Percent of white cells:Q", format=".1f"), alt.Tooltip("Typical adult range:N"),
                 alt.Tooltip("Flag:N")])
    text = bars.mark_text(align="left", dx=5, color=INK, fontSize=12).encode(
        text=alt.Text("Percent of white cells:Q", format=".0f"))
    return _base((band + bars + text).properties(height=240))
