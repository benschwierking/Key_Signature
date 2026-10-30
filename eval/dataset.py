"""Loading the cached feature file (spec §12), the model artifact, and the split.

Interface assumptions (the spec does not fix these; documented in HARNESS_README.md):
* Feature cache: exactly one Parquet or CSV file under ``data/features/``
  (spec §12 "a cached feature file (Parquet/CSV) under data/features/").
* It holds one row per track (or per clip, if HF 30 s clips are used, spec §6),
  a key-label column, a track-id column, and 36 numeric feature columns in the
  spec §7 order: first-bin one-hot (12) | last-bin one-hot (12) | histogram (12).
* The test split is taken from (first match wins): a ``split`` column in the
  feature file, an explicit ``--split-file``, or split info stored in the model
  artifact when it is a dict. The harness never invents split ratios.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .keys import N_KEYS, parse_key

N_FEATURES = 36  # Spec §7: extract_features(path) -> 36-dim vector
LABEL_COL_CANDIDATES = ("key", "label", "key_label", "y")
TRACK_COL_CANDIDATES = ("track_id", "track", "id", "filename", "file")
SPLIT_COL_CANDIDATES = ("split", "subset", "fold")
SPLIT_ALIASES = {
    "train": "train", "training": "train",
    "val": "val", "valid": "val", "validation": "val", "dev": "val",
    "test": "test", "testing": "test", "heldout": "test", "held_out": "test", "held-out": "test",
}


class HarnessInputError(RuntimeError):
    """Raised when inputs needed for evaluation are missing or malformed."""


@dataclass
class FeatureTable:
    """Feature matrix plus metadata resolved from the cached feature file."""

    X: np.ndarray            # (n, 36)
    y: np.ndarray            # (n,) canonical key indices
    tracks: np.ndarray       # (n,) track ids (str)
    split: np.ndarray | None # (n,) normalised split names or None
    feature_names: list[str]
    frame: pd.DataFrame


def find_feature_file(features_dir: str | Path = "data/features") -> Path:
    """Return the single Parquet/CSV feature cache under ``features_dir`` (spec §12)."""
    d = Path(features_dir)
    if d.is_file():
        return d
    if not d.is_dir():
        raise HarnessInputError(f"feature directory not found: {d} (spec §12 data/features/)")
    files = sorted([*d.glob("*.parquet"), *d.glob("*.csv")])
    if len(files) != 1:
        raise HarnessInputError(
            f"expected exactly one .parquet/.csv in {d}, found {[f.name for f in files]}; "
            "pass --features explicitly")
    return files[0]


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a Parquet or CSV file into a DataFrame."""
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _pick(cols, candidates, explicit, what, required=True):
    if explicit:
        if explicit not in cols:
            raise HarnessInputError(f"{what} column {explicit!r} not in feature file")
        return explicit
    for c in candidates:
        if c in cols:
            return c
    if required:
        raise HarnessInputError(f"no {what} column found (tried {candidates}); pass it explicitly")
    return None


def load_features(path: str | Path, label_col: str | None = None, track_col: str | None = None,
                  split_col: str | None = None, feature_cols: list[str] | None = None) -> FeatureTable:
    """Load the feature cache and resolve label / track / split / 36 feature columns."""
    df = read_table(path)
    cols = list(df.columns)
    lc = _pick(cols, LABEL_COL_CANDIDATES, label_col, "label")
    tc = _pick(cols, TRACK_COL_CANDIDATES, track_col, "track-id")
    sc = _pick(cols, SPLIT_COL_CANDIDATES, split_col, "split", required=False)
    if feature_cols:
        fcols = list(feature_cols)
    else:
        meta = {lc, tc, sc}
        fcols = [c for c in cols if c not in meta and pd.api.types.is_numeric_dtype(df[c])]
    if len(fcols) != N_FEATURES:
        raise HarnessInputError(
            f"expected {N_FEATURES} feature columns (spec §7), found {len(fcols)}: {fcols[:6]}...; "
            "pass --feature-cols")
    split = None
    if sc is not None:
        split = np.array([normalise_split(s) for s in df[sc]])
    return FeatureTable(
        X=df[fcols].to_numpy(dtype=float),
        y=np.array([parse_key(v) for v in df[lc]], dtype=int),
        tracks=df[tc].astype(str).to_numpy(),
        split=split,
        feature_names=fcols,
        frame=df,
    )


def normalise_split(name) -> str:
    """Map split aliases ('validation', 'held_out', ...) to train / val / test."""
    s = str(name).strip().lower()
    if s not in SPLIT_ALIASES:
        raise HarnessInputError(f"unknown split name {name!r}")
    return SPLIT_ALIASES[s]


def _split_from_mapping(mapping: dict, tracks: np.ndarray) -> np.ndarray:
    lookup: dict[str, str] = {}
    for name, ids in mapping.items():
        norm = normalise_split(name)
        for t in ids:
            t = str(t)
            if t in lookup and lookup[t] != norm:
                raise HarnessInputError(f"track {t} listed in both {lookup[t]} and {norm} (leakage)")
            lookup[t] = norm
    missing = [t for t in tracks if t not in lookup]
    out = np.array([lookup.get(t, "unassigned") for t in tracks])
    if missing and not any(v == "test" for v in out):
        raise HarnessInputError("split mapping assigns no rows of the feature file to 'test'")
    return out


def resolve_split(table: FeatureTable, split_file: str | Path | None = None,
                  artifact_meta: dict | None = None) -> tuple[np.ndarray, str]:
    """Return per-row split names and a description of where they came from."""
    if table.split is not None:
        return table.split, "feature-file split column"
    if split_file:
        p = Path(split_file)
        if p.suffix == ".json":
            mapping = json.loads(p.read_text())
        else:
            sdf = read_table(p)
            tc = _pick(list(sdf.columns), TRACK_COL_CANDIDATES, None, "track-id")
            sc = _pick(list(sdf.columns), SPLIT_COL_CANDIDATES, None, "split")
            mapping = {}
            for t, s in zip(sdf[tc].astype(str), sdf[sc]):
                mapping.setdefault(s, []).append(t)
        return _split_from_mapping(mapping, table.tracks), f"split file {p}"
    meta = artifact_meta or {}
    if isinstance(meta.get("splits"), dict):
        return _split_from_mapping(meta["splits"], table.tracks), "model artifact 'splits'"
    if "test_track_ids" in meta:
        ids = {str(t) for t in meta["test_track_ids"]}
        return np.array(["test" if t in ids else "nontest" for t in table.tracks]), \
            "model artifact 'test_track_ids'"
    raise HarnessInputError(
        "cannot determine the held-out test split (spec §9/§10): add a 'split' column to the "
        "feature cache, pass --split-file, or store 'splits' in the model artifact")


def leakage(tracks: np.ndarray, split: np.ndarray) -> dict[str, list[str]]:
    """Spec §10 "split by track, no leakage": tracks that appear in more than one split."""
    seen: dict[str, set[str]] = {}
    for t, s in zip(tracks, split):
        seen.setdefault(str(t), set()).add(str(s))
    return {t: sorted(s) for t, s in seen.items() if len(s) > 1}


# ---------------------------------------------------------------- model artifact
def load_model(path: str | Path):
    """Load ``models/model.joblib`` (spec §12). Returns (estimator, metadata dict).

    Accepts a bare scikit-learn estimator/pipeline, or a dict holding one under
    'model' / 'estimator' / 'pipeline' plus optional metadata (e.g. 'splits').
    """
    import joblib

    p = Path(path)
    if not p.exists():
        raise HarnessInputError(f"model artifact not found: {p} (spec §12 models/model.joblib)")
    obj = joblib.load(p)
    if isinstance(obj, dict):
        for k in ("model", "estimator", "pipeline", "clf"):
            if k in obj:
                meta = {kk: vv for kk, vv in obj.items() if kk != k}
                return obj[k], meta
        raise HarnessInputError(f"model dict has no 'model' key: {sorted(obj)}")
    return obj, {}


def final_estimator(est):
    """Last step of a Pipeline (or the estimator itself)."""
    steps = getattr(est, "steps", None)
    return steps[-1][1] if steps else est


def class_columns(est, meta: dict | None = None) -> np.ndarray:
    """Canonical key index of each column of ``est.predict_proba``."""
    classes = getattr(est, "classes_", None)
    if classes is None:
        classes = getattr(final_estimator(est), "classes_", None)
    if classes is None:
        raise HarnessInputError("model has no classes_; cannot map probabilities to keys")
    names = (meta or {}).get("classes") or (meta or {}).get("label_names")
    if names is not None and all(isinstance(c, (int, np.integer)) for c in classes):
        classes = [names[int(c)] for c in classes]
    return np.array([parse_key(c) for c in classes], dtype=int)


def predict_proba_24(est, X: np.ndarray, feature_names: list[str] | None = None,
                     meta: dict | None = None) -> np.ndarray:
    """``predict_proba`` remapped onto the 24 canonical key columns."""
    Xin = X
    if feature_names is not None and hasattr(est, "feature_names_in_"):
        Xin = pd.DataFrame(np.atleast_2d(X), columns=feature_names)[list(est.feature_names_in_)]
    p = np.asarray(est.predict_proba(np.atleast_2d(Xin) if Xin is X else Xin), dtype=float)
    cols = class_columns(est, meta)
    out = np.zeros((p.shape[0], N_KEYS))
    for j, k in enumerate(cols):
        out[:, k] += p[:, j]
    return out
