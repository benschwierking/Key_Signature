"""Shared pytest plumbing for the keyclf acceptance harness.

Missing-implementation policy (documented in HARNESS_README.md):
* Tests that need ``keyclf`` request it through the ``features`` / ``train_mod``
  fixtures (or :func:`require_impl`). If the ``keyclf`` module itself is not
  importable the test is SKIPPED with a reason starting "IMPLEMENTATION MISSING".
* Set ``KEYCLF_REQUIRE_IMPL=1`` (CI does) to turn those skips into FAILURES, so
  the acceptance build is red until the implementation lands.
* If ``keyclf`` exists but one of ITS imports is missing, that is a real error
  and is raised, never skipped.
Harness self-tests (key relations, K-S baseline, metrics, criteria, eval smoke
run) need no implementation and must pass today.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(os.environ.get("KEYCLF_REPO_ROOT", Path(__file__).resolve().parents[1]))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REQUIRE_IMPL = os.environ.get("KEYCLF_REQUIRE_IMPL", "").lower() in ("1", "true", "yes")
SR = 22050


def pytest_configure(config):
    config.addinivalue_line("markers", "impl: needs the keyclf implementation (spec §11)")
    config.addinivalue_line("markers", "artifacts: needs data/features cache + models/model.joblib (spec §12)")
    config.addinivalue_line("markers", "harness: harness self-test; must pass without keyclf")
    config.addinivalue_line("markers", "interface_heuristic: checks a spec requirement via naming heuristics")


def _missing(msg: str):
    if REQUIRE_IMPL:
        pytest.fail(msg + " [KEYCLF_REQUIRE_IMPL=1 turns this skip into a failure]", pytrace=False)
    pytest.skip(msg)


def require_impl(modname: str):
    """Import a keyclf module or skip/fail with an explicit IMPLEMENTATION MISSING reason."""
    try:
        return importlib.import_module(modname)
    except ModuleNotFoundError as e:
        if e.name and (modname == e.name or modname.startswith(e.name + ".")):
            _missing(f"IMPLEMENTATION MISSING: cannot import {modname} ({e}); spec §11 requires it")
        raise


def require_artifacts():
    """Return (feature_file, model_path) or skip/fail with an explicit ARTIFACTS MISSING reason."""
    from eval.dataset import HarnessInputError, find_feature_file

    try:
        feat = find_feature_file(REPO_ROOT / "data" / "features")
    except HarnessInputError as e:
        _missing(f"ARTIFACTS MISSING: {e} (spec §12 cached feature file)")
    model = REPO_ROOT / "models" / "model.joblib"
    if not model.exists():
        _missing(f"ARTIFACTS MISSING: {model} (spec §12 exported model)")
    return feat, model


@pytest.fixture
def features():
    """``keyclf.features`` (spec §7, §11)."""
    return require_impl("keyclf.features")


@pytest.fixture
def train_mod():
    """``keyclf.train`` (spec §11, §16 task 7)."""
    return require_impl("keyclf.train")


# ------------------------------------------------------------ synthetic audio
def midi_to_hz(m: float) -> float:
    """Equal-temperament frequency, A4 = MIDI 69 = 440 Hz."""
    return 440.0 * 2 ** ((m - 69) / 12)


def tone(freq: float, dur: float, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    """Pure sine with 10 ms fades (avoids clicks that smear chroma)."""
    t = np.arange(int(dur * sr)) / sr
    env = np.minimum(1.0, np.minimum(t / 0.01, (dur - t) / 0.01))
    return (amp * np.sin(2 * np.pi * freq * t) * env).astype(np.float32)


def sequence(midis, note_dur: float, sr: int = SR) -> np.ndarray:
    """Concatenate pure tones for a list of MIDI notes."""
    return np.concatenate([tone(midi_to_hz(m), note_dur, sr) for m in midis])


def silence(dur: float, sr: int = SR) -> np.ndarray:
    """Digital silence."""
    return np.zeros(int(dur * sr), dtype=np.float32)


def write_wav(path: Path, y: np.ndarray, sr: int = SR) -> Path:
    """Write float32 mono WAV."""
    from scipy.io import wavfile

    wavfile.write(path, sr, np.asarray(y, dtype=np.float32))
    return path


@pytest.fixture
def wav(tmp_path):
    """Factory: wav(name, samples) -> path of a temp WAV file."""
    def _make(name: str, y: np.ndarray, sr: int = SR) -> Path:
        return write_wav(tmp_path / f"{name}.wav", y, sr)
    return _make
