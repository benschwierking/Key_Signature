"""Learned-vs-K-S profile analysis (spec §9 bullet 6, §16 task 9).

Spec §9: "Learned feature weights for the histogram are extracted per key,
rotated to a common tonic, and plotted against the K-S major and minor
profiles, with the correlation reported."

The spec states NO correlation threshold. The harness only checks that the plot
and the correlation are produced and reported; it never gates on the value.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .dataset import class_columns, final_estimator
from .keys import mode_of, tonic_of
from .ks_baseline import KK_MAJOR, KK_MINOR

HIST_SLICE = slice(24, 36)  # spec §7 order: first(12) | last(12) | histogram(12)


def rotate_to_tonic(weights_12: np.ndarray, tonic_pc: int) -> np.ndarray:
    """Rotate pitch-class weights so the key's tonic lands on index 0."""
    return np.roll(np.asarray(weights_12, dtype=float), -int(tonic_pc))


def learned_profiles(coef: np.ndarray, key_cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average tonic-rotated histogram weights into learned (major, minor) profiles.

    coef      : (n_classes, 36) linear weights (e.g. LogisticRegression.coef_)
    key_cols  : canonical key index of each coef row
    """
    coef = np.asarray(coef, dtype=float)
    maj, mnr = [], []
    for row, k in zip(coef, key_cols):
        r = rotate_to_tonic(row[HIST_SLICE], tonic_of(int(k)))
        (maj if mode_of(int(k)) == "major" else mnr).append(r)
    if not maj or not mnr:
        raise ValueError("model must have at least one major and one minor class")
    return np.mean(maj, axis=0), np.mean(mnr, axis=0)


def pearson(a, b) -> float:
    """Pearson correlation of two equal-length vectors."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1])


def analyse_model(est, meta: dict | None, plot_path: str | Path) -> dict:
    """Extract learned profiles from a linear model, plot vs K-S, return correlations.

    Raises ValueError if the final estimator has no per-class ``coef_`` (e.g. a
    random forest); the caller then falls back to implementation-provided artifacts.
    """
    fe = final_estimator(est)
    coef = getattr(fe, "coef_", None)
    if coef is None or np.ndim(coef) != 2 or np.shape(coef)[1] != 36:
        raise ValueError("final estimator has no (n_classes, 36) coef_; cannot extract per-key weights")
    cols = class_columns(est, meta)
    if coef.shape[0] != len(cols):
        raise ValueError("coef_ rows do not match classes_ (binary model?)")
    maj, mnr = learned_profiles(coef, cols)
    r_maj, r_min = pearson(maj, KK_MAJOR), pearson(mnr, KK_MINOR)
    plot_profiles(maj, mnr, r_maj, r_min, plot_path)
    return {
        "source": "harness (coef_ of final estimator, histogram slice 24:36)",
        "learned_major": maj.tolist(),
        "learned_minor": mnr.tolist(),
        "corr_major": r_maj,
        "corr_minor": r_min,
        "plot": str(plot_path),
    }


def plot_profiles(maj, mnr, r_maj, r_min, path) -> None:
    """Save a two-panel plot of z-scored learned vs K-S profiles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = lambda v: (np.asarray(v) - np.mean(v)) / (np.std(v) or 1.0)  # noqa: E731
    degrees = ["1", "b2", "2", "b3", "3", "4", "#4", "5", "b6", "6", "b7", "7"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, learned, ks, title, r in ((axes[0], maj, KK_MAJOR, "Major", r_maj),
                                      (axes[1], mnr, KK_MINOR, "Minor", r_min)):
        x = np.arange(12)
        ax.plot(x, z(ks), "o-", label="K-S profile")
        ax.plot(x, z(learned), "s--", label="learned (rotated to tonic)")
        ax.set_xticks(x, degrees)
        ax.set_title(f"{title}: Pearson r = {r:.3f}")
        ax.set_ylabel("z-score")
        ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
