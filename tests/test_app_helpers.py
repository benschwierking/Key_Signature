"""Unit tests for app probability mapping without launching Streamlit."""

from __future__ import annotations

import numpy as np

from keyclf.app import probability_24
from keyclf.keys import KEY_NAMES, is_related, parse_key


class _Estimator:
    classes_ = np.array([0, 9, 21])

    def predict_proba(self, _sample):
        return np.array([[0.1, 0.2, 0.7]])


def test_app_probability_mapping_and_relative_highlight_relation():
    result = probability_24(_Estimator(), np.zeros(36), {"classes": list(KEY_NAMES)})
    assert result.shape == (24,) and result.argmax() == 21 and np.isclose(result.sum(), 1.0)
    assert is_related(parse_key("A minor"), parse_key("C major"))
    assert not is_related(parse_key("A minor"), parse_key("C minor"))
