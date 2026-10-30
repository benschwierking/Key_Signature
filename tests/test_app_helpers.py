"""Unit tests for app probability mapping without launching Streamlit."""

from __future__ import annotations

import numpy as np
import pytest

from keyclf.keys import KEY_NAMES, is_related, parse_key


class _Estimator:
    classes_ = np.array([0, 9, 21])

    def predict_proba(self, _sample):
        return np.array([[0.1, 0.2, 0.7]])


def test_app_probability_mapping_and_relative_highlight_relation():
    from keyclf.app import probability_24

    result = probability_24(_Estimator(), np.zeros(36), {"classes": list(KEY_NAMES)})
    assert result.shape == (24,) and result.argmax() == 21 and np.isclose(result.sum(), 1.0)
    assert is_related(parse_key("A minor"), parse_key("C major"))
    assert not is_related(parse_key("A minor"), parse_key("C minor"))


def test_research_input_preserves_the_display_vector_and_model_order(monkeypatch):
    from keyclf import research_features
    from keyclf.app import prepare_input

    full = np.arange(len(research_features.BLOCKS) * 12, dtype=float)
    monkeypatch.setattr(research_features, "extract_research_features", lambda _path: (full, {}))
    display, selected = prepare_input("unused.wav", {"research_version": "v2.1", "feature_names": ["bass_3", "hist_0"]})
    assert np.array_equal(display, full[:36])
    assert np.array_equal(selected, [51, 24])


def test_histogram_only_research_uses_the_fast_v1_extractor(monkeypatch):
    from keyclf import app, research_features

    monkeypatch.setattr(app, "extract_features", lambda _path, _config: np.arange(36))
    monkeypatch.setattr(research_features, "extract_research_features", lambda _path: pytest.fail("unused rich extractor called"))
    display, selected = app.prepare_input("unused.wav", {"research_version": "v2.1", "feature_names": ["hist_0", "hist_11"]})
    assert np.array_equal(display, np.arange(36))
    assert np.array_equal(selected, [24, 35])
