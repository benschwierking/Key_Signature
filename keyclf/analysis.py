"""Interpretive learned-profile analysis for non-linear final models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss

from .baseline import KK_MAJOR, KK_MINOR
from .keys import mode_of, tonic_of


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Return a finite Pearson correlation for two 12-element profiles."""
    x, y = np.asarray(a, float), np.asarray(b, float)
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _class_probability(estimator, X: np.ndarray, key: int) -> np.ndarray:
    classes = np.asarray(estimator.classes_, dtype=int)
    where = np.flatnonzero(classes == key)
    if not where.size:
        return np.zeros(len(X), dtype=float)
    return np.asarray(estimator.predict_proba(X), dtype=float)[:, int(where[0])]


def random_forest_profiles(
    estimator,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    seed: int = 42,
    repeats: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-key histogram importance and return tonic-aligned mode means.

    Each feature's importance is the increase in one-vs-rest validation log
    loss when that histogram column is permuted. This supplies class-specific
    interpretive profiles when a random forest, which has no ``coef_``, wins.
    """
    X = np.asarray(X_validation, dtype=float)
    y = np.asarray(y_validation, dtype=int)
    rng = np.random.default_rng(seed)
    by_key: list[tuple[int, np.ndarray]] = []
    for key in range(24):
        truth = (y == key).astype(int)
        base_probability = np.clip(_class_probability(estimator, X, key), 1e-9, 1 - 1e-9)
        baseline = log_loss(truth, np.column_stack((1 - base_probability, base_probability)), labels=[0, 1])
        importance = np.zeros(12, dtype=float)
        for offset, column in enumerate(range(24, 36)):
            losses = []
            for _ in range(repeats):
                shuffled = X.copy()
                shuffled[:, column] = rng.permutation(shuffled[:, column])
                probability = np.clip(_class_probability(estimator, shuffled, key), 1e-9, 1 - 1e-9)
                losses.append(log_loss(truth, np.column_stack((1 - probability, probability)), labels=[0, 1]))
            importance[offset] = float(np.mean(losses) - baseline)
        by_key.append((key, np.roll(importance, -tonic_of(key))))
    major = np.mean([profile for key, profile in by_key if mode_of(key) == "major"], axis=0)
    minor = np.mean([profile for key, profile in by_key if mode_of(key) == "minor"], axis=0)
    return major, minor


def save_profile_analysis(
    estimator,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    json_path: str | Path,
    plot_path: str | Path,
    seed: int = 42,
) -> dict:
    """Write random-forest learned-versus-K-S JSON and PNG artifacts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    major, minor = random_forest_profiles(estimator, X_validation, y_validation, seed=seed)
    corr_major, corr_minor = pearson(major, KK_MAJOR), pearson(minor, KK_MINOR)
    degrees = ["1", "b2", "2", "b3", "3", "4", "#4", "5", "b6", "6", "b7", "7"]

    def zscore(values):
        values = np.asarray(values, float)
        return (values - values.mean()) / (values.std() or 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, learned, classic, title, correlation in (
        (axes[0], major, KK_MAJOR, "Major", corr_major),
        (axes[1], minor, KK_MINOR, "Minor", corr_minor),
    ):
        ax.plot(degrees, zscore(classic), "o-", label="K-S profile")
        ax.plot(degrees, zscore(learned), "s--", label="RF permutation importance")
        ax.set_title(f"{title}: Pearson r = {correlation:.3f}")
        ax.set_ylabel("z-score")
        ax.legend()
    fig.tight_layout()
    plot = Path(plot_path)
    plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot, dpi=120)
    plt.close(fig)
    payload = {
        "source": "class-specific one-vs-rest validation permutation importance",
        "learned_major": major.tolist(),
        "learned_minor": minor.tolist(),
        "corr_major": corr_major,
        "corr_minor": corr_minor,
        "plot": str(plot),
    }
    target = Path(json_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
