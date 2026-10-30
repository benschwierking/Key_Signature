"""SYNTHETIC TEST DATA — NOT GiantSteps, NOT the keyclf implementation.

Generates a fake feature cache + fake model.joblib so the evaluation pipeline
(eval/evaluate.py) can be smoke-tested end to end before ``keyclf`` exists.
Every track id is prefixed ``SYNTHETIC-`` and the feature file is named
``SYNTHETIC_features.csv`` so it can never be mistaken for real results.

Also provides ``synthetic_extract_features`` — a minimal stand-in extractor used
ONLY to exercise the latency-timing code path in the smoke test.

CLI:  python -m tests.fixtures.synthetic_fixture OUT_DIR [--n-tracks 480] [--shuffle-labels]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from eval.keys import KEY_NAMES, mode_of, tonic_of
from eval.ks_baseline import KK_MAJOR, KK_MINOR

FEATURE_COLS = ([f"first_{i}" for i in range(12)] + [f"last_{i}" for i in range(12)]
                + [f"hist_{i}" for i in range(12)])


def make_features(n_tracks: int = 480, seed: int = 0, shuffle_labels: bool = False) -> pd.DataFrame:
    """Synthetic 36-dim vectors in spec §7 order with K-S-shaped histograms."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_tracks):
        k = int(rng.integers(24))
        t = tonic_of(k)
        prof = np.roll(KK_MAJOR if mode_of(k) == "major" else KK_MINOR, t)
        hist = np.clip(prof + rng.normal(0, 0.6, 12), 0.01, None)
        hist /= hist.sum()
        first = np.zeros(12); first[t if rng.random() < 0.8 else rng.integers(12)] = 1
        last = np.zeros(12); last[t if rng.random() < 0.8 else rng.integers(12)] = 1
        split = ("train", "train", "train", "val", "test")[i % 5]  # by track: one row per track
        rows.append({"track_id": f"SYNTHETIC-{i:04d}", "key": KEY_NAMES[k], "split": split,
                     **dict(zip(FEATURE_COLS, np.concatenate([first, last, hist])))})
    df = pd.DataFrame(rows)
    if shuffle_labels:
        df["key"] = rng.permutation(df["key"].to_numpy())
    return df


def train_model(df: pd.DataFrame, seed: int = 0):
    """Fit a StandardScaler+LogisticRegression on the synthetic train rows."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tr = df[df["split"] == "train"]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    model.fit(tr[FEATURE_COLS].to_numpy(), tr["key"].to_numpy())
    return model


def build(out_dir: str | Path, n_tracks: int = 480, seed: int = 0, shuffle_labels: bool = False):
    """Write data/features/SYNTHETIC_features.csv and models/model.joblib under out_dir."""
    import joblib

    out = Path(out_dir)
    (out / "data" / "features").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(parents=True, exist_ok=True)
    df = make_features(n_tracks, seed, shuffle_labels)
    feat = out / "data" / "features" / "SYNTHETIC_features.csv"
    df.to_csv(feat, index=False)
    model_path = out / "models" / "model.joblib"
    joblib.dump(train_model(df, seed), model_path)
    return feat, model_path


def synthetic_extract_features(path) -> np.ndarray:
    """STAND-IN extractor (not keyclf): librosa chroma -> first/last one-hot + histogram."""
    import librosa

    y, sr = librosa.load(path, sr=22050, mono=True)
    y, _ = librosa.effects.trim(y)
    # The synthetic probe is generated at exact equal-tempered frequencies, so
    # automatic tuning estimation is unnecessary and unstable on some numba
    # builds used by local macOS/arm64 development environments.
    c = librosa.feature.chroma_cqt(y=y, sr=sr, tuning=0.0)
    first = np.eye(12)[c[:, :10].sum(1).argmax()]
    last = np.eye(12)[c[:, -10:].sum(1).argmax()]
    h = c.sum(1)
    return np.concatenate([first, last, h / h.sum()])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir")
    ap.add_argument("--n-tracks", type=int, default=480)
    ap.add_argument("--shuffle-labels", action="store_true")
    a = ap.parse_args()
    print(*build(a.out_dir, a.n_tracks, shuffle_labels=a.shuffle_labels), sep="\n")
