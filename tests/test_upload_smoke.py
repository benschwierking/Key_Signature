"""Exercise actual upload decoding and rendered prediction ordering."""

import shutil
import subprocess

import numpy as np
import pytest


@pytest.mark.parametrize("extension", ["wav", "mp3", "mp4"])
def test_upload_formats_produce_finite_features(tmp_path, extension):
    import soundfile as sf
    from keyclf.app import prepare_input

    sr = 22050
    time = np.arange(sr * 3) / sr
    chord = sum(np.sin(2 * np.pi * frequency * time) for frequency in (261.6256, 329.6276, 391.9954)) / 4
    wave = tmp_path / "chord.wav"
    sf.write(wave, chord, sr)
    source = wave
    if extension != "wav":
        if not shutil.which("ffmpeg"):
            pytest.fail("ffmpeg is required for MP3/MP4 upload smoke tests")
        source = tmp_path / f"chord.{extension}"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(wave), str(source)], check=True, capture_output=True)
    display, vector = prepare_input(source, {})
    assert display.shape == vector.shape == (36,)
    assert np.isfinite(vector).all() and np.isclose(vector[24:36].sum(), 1)


def test_streamlit_top_three_order_and_relative_key_highlight():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_string("""
import numpy as np
from keyclf.app import render_result
features = np.r_[np.eye(12)[0], np.eye(12)[0], np.ones(12) / 12]
probability = np.zeros(24)
probability[[0, 21, 6]] = [0.6, 0.3, 0.1]
render_result(features, probability)
""").run(timeout=20)
    assert not app.exception
    rows = [item.value for item in app.markdown if item.value.startswith(("1.", "2.", "3."))]
    assert len(rows) == 3
    assert "C major" in rows[0] and "60.0%" in rows[0]
    assert "A minor" in rows[1] and "related key" in rows[1]
    assert "F# major" in rows[2] and "related key" not in rows[2]
