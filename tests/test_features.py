"""Unit tests for ``keyclf.features`` on synthetic tones and scales.

Spec §9 line 81: "Features are chroma-based (first bin, last bin, 12-bin track
histogram) computed with librosa, verified by unit tests on synthetic tones."
Spec §7 lines 53-63 define the features and the module interface:
  load_audio(path), compute_chroma(y, sr), first_last_bins(chroma),
  chroma_histogram(chroma), extract_features(path) -> 36-dim vector.
Spec §16 task 3: "Unit-test with synthetic tones and scales."

Interface assumptions (spec names the functions but not their return types):
  * load_audio(path) -> (y, sr): ``compute_chroma(y, sr)`` takes exactly those.
  * compute_chroma returns librosa's (12, n_frames) layout.
  * first_last_bins returns (first, last), each a 12-dim one-hot (§7 items 1-2).
  * extract_features vector order = §7 list order: first(12) | last(12) | hist(12).
All audio here is generated with numpy (no dataset audio, spec §6/§12).
"""
from __future__ import annotations

import inspect
import re
import shutil
import subprocess

import numpy as np
import pytest

from .conftest import SR, midi_to_hz, sequence, silence, tone

pytestmark = pytest.mark.impl

A, C, E, B = 9, 0, 4, 11
C_MAJOR_SCALE = [60, 62, 64, 65, 67, 69, 71, 72]
DIATONIC = {"C major": ([60, 62, 64, 65, 67, 69, 71, 72], {0, 2, 4, 5, 7, 9, 11}),
            "Eb major": ([63, 65, 67, 68, 70, 72, 74, 75], {3, 5, 7, 8, 10, 0, 2}),
            "A natural minor": ([57, 59, 60, 62, 64, 65, 67, 69], {9, 11, 0, 2, 4, 5, 7})}


def _onehot_at(v, idx):
    v = np.asarray(v, dtype=float).reshape(-1)
    assert v.shape == (12,), f"expected 12-dim one-hot, got shape {v.shape}"
    assert set(np.unique(v)) <= {0.0, 1.0} and v.sum() == 1, f"not one-hot: {v}"
    assert int(v.argmax()) == idx, f"one-hot at {int(v.argmax())}, expected pitch class {idx}"


# ---------------------------------------------------------------- interface
def test_module_exposes_spec_functions(features):
    # Spec §7 line 62: module responsibilities.
    for name in ("load_audio", "compute_chroma", "first_last_bins", "chroma_histogram", "extract_features"):
        assert callable(getattr(features, name, None)), f"keyclf.features.{name} missing (spec §7 line 62)"


def test_features_computed_with_librosa_chroma(features):
    # Spec §7 line 54 / §9 line 81: chroma from librosa.feature.chroma_cqt (or chroma_stft).
    src = inspect.getsource(features)
    assert "librosa" in src, "spec §9 line 81: features must be computed with librosa"
    assert re.search(r"chroma_(cqt|stft)", src), "spec §7 line 54: use librosa.feature.chroma_cqt/chroma_stft"


@pytest.mark.interface_heuristic
def test_chroma_type_hop_and_edge_frames_configurable(features):
    # Spec §7 line 63: "Configurable chroma type, hop length, and how many edge frames
    # count as 'first'/'last'." Heuristic: look for matching parameter / constant / field names.
    names = set()
    for _, obj in inspect.getmembers(features):
        if inspect.isfunction(obj) or inspect.isclass(obj):
            try:
                names |= set(inspect.signature(obj).parameters)
            except (TypeError, ValueError):
                pass
            names |= set(getattr(obj, "__dataclass_fields__", {}) or {})
    names |= {n for n in dir(features) if not n.startswith("__")}
    low = {n.lower() for n in names}
    checks = {
        "chroma type": r"(chroma_?(type|kind|method|fn|func)|^kind$|^method$)",
        "hop length": r"hop",
        "edge frames": r"(edge|n_?frames|n_?first|n_?last|first_frames|last_frames|window)",
    }
    missing = [k for k, rx in checks.items() if not any(re.search(rx, n) for n in low)]
    assert not missing, (f"spec §7 line 63: no configurable {missing} found among parameter/constant names "
                         f"(heuristic; see HARNESS_README.md)")


# ---------------------------------------------------------------- load_audio
def test_load_audio_wav_returns_mono_float_and_rate(features, wav):
    # Spec §7 line 59: "load audio (ffmpeg via librosa)".
    p = wav("a440", tone(440.0, 2.0))
    y, sr = features.load_audio(str(p))
    y = np.asarray(y)
    assert y.ndim == 1 and np.issubdtype(y.dtype, np.floating)
    assert int(sr) > 0
    assert abs(len(y) / sr - 2.0) < 0.1, "loaded duration should match the 2 s synthetic tone"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed (spec §14 system dep)")
def test_load_audio_mp3_via_ffmpeg(features, wav, tmp_path):
    # Spec §8 inputs MP3/MP4/WAV; §14 "System: ffmpeg for MP3/MP4 decoding."
    src = wav("a440", tone(440.0, 2.0))
    mp3 = tmp_path / "a440.mp3"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), str(mp3)], check=True)
    y, sr = features.load_audio(str(mp3))
    assert np.asarray(y).ndim == 1 and abs(len(y) / sr - 2.0) < 0.2


# ---------------------------------------------------------------- compute_chroma
def test_compute_chroma_shape_and_values(features):
    # Spec §7 line 54: 12 pitch-class bins (C, C#, ..., B).
    c = np.asarray(features.compute_chroma(tone(440.0, 2.0), SR))
    assert c.ndim == 2 and c.shape[0] == 12 and c.shape[1] > 0, f"expected (12, n_frames), got {c.shape}"
    assert np.all(np.isfinite(c)) and np.all(c >= 0)


def test_a440_sine_puts_chroma_mass_in_A(features):
    # Spec §9 line 81 "verified by unit tests on synthetic tones": A440 -> pitch class A (index 9).
    c = np.asarray(features.compute_chroma(tone(440.0, 2.0), SR))
    assert int(c.sum(axis=1).argmax()) == A


@pytest.mark.parametrize("midi", range(60, 72), ids=lambda m: f"midi{m}")
def test_every_pitch_class_sine_maps_to_its_bin(features, midi):
    # Spec §7 line 54: octaves folded into C..B; each pure tone C4..B4 lands in its own bin.
    c = np.asarray(features.compute_chroma(tone(midi_to_hz(midi), 1.5), SR))
    assert int(c.sum(axis=1).argmax()) == midi % 12


def test_octaves_fold_to_same_pitch_class(features):
    # Spec §1 / §7: "all octaves folded together".
    for midi in (45, 57, 69, 81):  # A2, A3, A4, A5
        c = np.asarray(features.compute_chroma(tone(midi_to_hz(midi), 1.5), SR))
        assert int(c.sum(axis=1).argmax()) == A, f"A at MIDI {midi} not folded to pitch class A"


# ---------------------------------------------------------------- chroma_histogram
def test_chroma_histogram_on_known_matrix(features):
    # Spec §7 item 3: "chroma energy summed across all frames ... normalized to sum to 1; 12 dims".
    rng = np.random.default_rng(0)
    chroma = rng.random((12, 50))
    h = np.asarray(features.chroma_histogram(chroma), dtype=float).reshape(-1)
    assert h.shape == (12,)
    np.testing.assert_allclose(h, chroma.sum(axis=1) / chroma.sum(), rtol=1e-6)


@pytest.mark.parametrize("name", list(DIATONIC))
def test_scale_histogram_concentrated_on_scale_tones(features, name):
    # Spec §16 task 3 "synthetic tones and scales": the 7 largest bins are exactly the scale's pitch classes.
    midis, pcs = DIATONIC[name]
    h = np.asarray(features.chroma_histogram(features.compute_chroma(sequence(midis, 0.5), SR))).reshape(-1)
    assert np.isclose(h.sum(), 1.0, atol=1e-6) and np.all(h >= 0)
    assert set(np.argsort(h)[-7:].tolist()) == pcs, f"top-7 bins {sorted(np.argsort(h)[-7:])} != {sorted(pcs)}"


# ---------------------------------------------------------------- first_last_bins
def test_first_last_bins_on_known_matrix(features):
    # Spec §7 items 1-2: dominant pitch class (argmax) over first / last frames; one-hot, 12 dims.
    seg = 300  # long segments so any reasonable edge-frame default stays inside them
    chroma = np.full((12, 3 * seg), 0.05)
    chroma[2, :seg] = 1.0          # D at the start
    chroma[5, seg:2 * seg] = 1.0   # F in the middle
    chroma[7, 2 * seg:] = 1.0      # G at the end
    first, last = features.first_last_bins(chroma)
    _onehot_at(first, 2)
    _onehot_at(last, 7)


def test_first_last_bins_from_tone_sequence(features):
    # E ... C ... B: first bin E, last bin B (spec §7 items 1-2).
    y = np.concatenate([tone(midi_to_hz(64), 4.0), tone(midi_to_hz(60), 4.0), tone(midi_to_hz(71), 4.0)])
    first, last = features.first_last_bins(features.compute_chroma(y, SR))
    _onehot_at(first, E)
    _onehot_at(last, B)


# ---------------------------------------------------------------- extract_features
def _check_vector(v):
    v = np.asarray(v, dtype=float)
    assert v.ndim == 1 and v.shape[0] == 36, f"spec §7 line 62: 36-dim vector, got shape {v.shape}"
    assert np.all(np.isfinite(v))
    hist = v[24:36]
    assert np.all(hist >= 0) and np.isclose(hist.sum(), 1.0, atol=1e-6), "spec §7 item 3: histogram sums to 1"
    return v


def test_extract_features_a440(features, wav):
    # Spec §7 line 62: extract_features(path) -> 36-dim vector; A440 -> first = last = hist argmax = A.
    v = _check_vector(features.extract_features(str(wav("a440", tone(440.0, 3.0)))))
    _onehot_at(v[0:12], A)
    _onehot_at(v[12:24], A)
    assert int(v[24:36].argmax()) == A


def test_extract_features_c_major_scale(features, wav):
    # C major scale C4..C5: first bin C, last bin C, histogram top-7 = C D E F G A B.
    v = _check_vector(features.extract_features(str(wav("cmaj", sequence(C_MAJOR_SCALE, 0.6)))))
    _onehot_at(v[0:12], C)
    _onehot_at(v[12:24], C)
    assert set(np.argsort(v[24:36])[-7:].tolist()) == {0, 2, 4, 5, 7, 9, 11}


def test_extract_features_ignores_leading_and_trailing_silence(features, wav):
    # Spec §7 line 55-56 "confidently non-silent frames"; line 59 "trim silence".
    # Silent frames would otherwise argmax to bin 0 (C); E start / B end proves they were skipped.
    y = np.concatenate([silence(1.5), tone(midi_to_hz(64), 4.0), tone(midi_to_hz(67), 4.0),
                        tone(midi_to_hz(71), 4.0), silence(1.5)])
    v = _check_vector(features.extract_features(str(wav("padded", y))))
    _onehot_at(v[0:12], E)
    _onehot_at(v[12:24], B)


def test_extract_features_deterministic(features, wav):
    # Spec §10 line 86 reproducibility: same file -> identical vector.
    p = str(wav("cmaj", sequence(C_MAJOR_SCALE, 0.4)))
    np.testing.assert_array_equal(np.asarray(features.extract_features(p)), np.asarray(features.extract_features(p)))


# ---------------------------------------------------------------- code quality
def test_public_functions_have_docstrings(features):
    # Spec §10 line 89: "readable with docstrings on public functions".
    missing = [n for n, f in inspect.getmembers(features, inspect.isfunction)
               if not n.startswith("_") and f.__module__ == features.__name__ and not inspect.getdoc(f)]
    assert not missing, f"public functions without docstrings: {missing}"
