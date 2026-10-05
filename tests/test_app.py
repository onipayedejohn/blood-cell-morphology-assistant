"""Run the Streamlit script headless and check every tab renders without errors."""

from pathlib import Path

import pytest

from bloodsmear.config import MODEL_DIR

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (MODEL_DIR / "bloodcell_cnn.onnx").exists(), reason="trained model not present")


def test_app_renders_without_exceptions():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 5
    assert any("confidence" in m.label.lower() for m in at.metric)


def test_black_sheet_is_refused_without_a_cell_type():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120)
    at.run()
    at.selectbox(key="sample_group").set_value("Images that are not blood cells").run()
    at.selectbox(key="pick_Images that are not blood cells").set_value("Photo of a black A4 sheet").run()
    assert not at.exception
    assert any("does not look like a stained blood smear" in e.value for e in at.error)
    assert not any(m.label.startswith("Confidence") for m in at.metric)


def test_whole_field_sample_lists_cells():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120)
    at.run()
    at.selectbox(key="sample_group").set_value("Whole microscope fields").run()
    assert not at.exception
    assert any("white cell" in s.value and s.value.startswith("**Found") for s in at.success)
