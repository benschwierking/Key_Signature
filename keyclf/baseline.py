"""Krumhansl–Schmuckler profile-matching baseline."""

from __future__ import annotations

import numpy as np

KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def key_templates() -> np.ndarray:
    """Return 24 tonic-rotated major/minor K-S templates."""
    return np.vstack([np.roll(KK_MAJOR, tonic) for tonic in range(12)] +
                     [np.roll(KK_MINOR, tonic) for tonic in range(12)])


def ks_scores(histogram: np.ndarray) -> np.ndarray:
    """Return Pearson correlations between histograms and all 24 templates."""
    h = np.atleast_2d(np.asarray(histogram, dtype=float))
    templates = key_templates()
    h_centered = h - h.mean(axis=1, keepdims=True)
    t_centered = templates - templates.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(h_centered, axis=1, keepdims=True) * np.linalg.norm(t_centered, axis=1)
    scores = h_centered @ t_centered.T
    scores = np.divide(scores, denominator, out=np.zeros_like(scores), where=denominator > 0)
    return scores[0] if np.asarray(histogram).ndim == 1 else scores


def ks_top_keys(histogram: np.ndarray, n: int = 3) -> np.ndarray:
    """Return the highest-scoring K-S key indices in descending order."""
    scores = ks_scores(histogram)
    if scores.ndim != 1:
        raise ValueError("ks_top_keys expects one 12-bin histogram")
    return np.argsort(scores)[::-1][:n]
