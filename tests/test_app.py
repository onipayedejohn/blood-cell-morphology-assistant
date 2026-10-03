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


def test_questionable_sample_shows_a_warning():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120)
    at.run()
    at.radio[0].set_value("Images the app should question").run()
    at.selectbox[0].set_value("Random noise").run()
    assert not at.exception
    assert at.warning or at.error
