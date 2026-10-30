"""Plain Krumhansl-Schmuckler (K-S) template-matching baseline.

Spec §9: "A plain K-S template-matching baseline is scored on the same test
split for comparison."  Spec §16 task 5: "classic major/minor profiles".

The harness carries its own reference K-S so the comparison does not depend on
the implementation under test. Input is the 12-bin pitch-class chroma
histogram (spec §7 feature 3).
"""
from __future__ import annotations

import numpy as np

from .keys import N_KEYS

# Classic Krumhansl & Kessler (1982) probe-tone profiles, tonic at index 0.
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def key_templates() -> np.ndarray:
    """(24, 12) template matrix in canonical key order (major C..B, minor C..B).

    Template for tonic t is the profile rotated so its tonic weight sits on bin t.
    """
    rows = [np.roll(KK_MAJOR, t) for t in range(12)] + [np.roll(KK_MINOR, t) for t in range(12)]
    return np.vstack(rows)


def ks_scores(histograms: np.ndarray) -> np.ndarray:
    """Pearson correlation of each histogram with each of the 24 key templates.

    Parameters
    ----------
    histograms : array (n, 12) or (12,)
        Pitch-class distributions (any non-negative scale).

    Returns
    -------
    array (n, 24) of correlations in [-1, 1]; rows with zero variance give 0.
    """
    h = np.atleast_2d(np.asarray(histograms, dtype=float))
    if h.shape[1] != 12:
        raise ValueError(f"expected 12-bin histograms, got shape {h.shape}")
    t = key_templates()
    hc = h - h.mean(axis=1, keepdims=True)
    tc = t - t.mean(axis=1, keepdims=True)
    num = hc @ tc.T
    den = np.linalg.norm(hc, axis=1, keepdims=True) * np.linalg.norm(tc, axis=1)[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, num / den, 0.0)
    return r


def ks_predict(histograms: np.ndarray) -> np.ndarray:
    """Argmax-correlation key index (canonical order) for each histogram."""
    return ks_scores(histograms).argmax(axis=1)


def ks_proba(histograms: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax of correlations / temperature -> (n, 24) pseudo-probabilities.

    K-S has no native probabilities, but spec §10 asks for log loss for the
    baseline too. The harness uses this softmax mapping; the temperature is fit
    on non-test data (see :func:`fit_temperature`), never on the test split.
    Ranking (accuracy, top-3) is unaffected by the temperature.
    """
    z = ks_scores(histograms) / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(histograms: np.ndarray, y_true: np.ndarray,
                    grid: np.ndarray | None = None) -> float:
    """Pick the softmax temperature minimising log loss on the given (non-test) data."""
    from .metrics import log_loss_24

    if grid is None:
        grid = np.logspace(-3, 1, 81)
    if len(y_true) == 0:
        return 1.0
    losses = [log_loss_24(np.asarray(y_true), ks_proba(histograms, T)) for T in grid]
    return float(grid[int(np.argmin(losses))])


assert key_templates().shape == (N_KEYS, 12)
