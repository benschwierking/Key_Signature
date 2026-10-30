"""Deterministic model selection, refitting, and artifact export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, top_k_accuracy_score

from .analysis import save_profile_analysis
from .features import DEFAULT_CONFIG, feature_names
from .keys import KEY_NAMES, parse_key

RANDOM_SEED = 42


def find_feature_file(directory: str | Path = "data/features") -> Path:
    """Return the sole Parquet/CSV cache in a directory."""
    root = Path(directory)
    files = sorted([*root.glob("*.parquet"), *root.glob("*.csv")]) if root.is_dir() else []
    if len(files) != 1:
        raise RuntimeError(f"expected exactly one feature cache under {root}, found {files}")
    return files[0]


def load_training_table(path: str | Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load the canonical cache into metadata, features, labels, and splits."""
    source = Path(path)
    frame = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    required = {"track_id", "key", "split"}
    if not required <= set(frame.columns):
        raise RuntimeError(f"feature cache is missing columns {sorted(required - set(frame.columns))}")
    names = feature_names()
    if not set(names) <= set(frame.columns):
        raise RuntimeError("feature cache does not contain the canonical 36 feature columns")
    X = frame[names].to_numpy(dtype=float)
    y = np.array([parse_key(value) for value in frame["key"]], dtype=int)
    split = frame["split"].astype(str).str.lower().replace({"validation": "val"}).to_numpy()
    if not np.isfinite(X).all() or X.shape[1] != 36:
        raise RuntimeError(f"invalid feature matrix shape/content: {X.shape}")
    return frame, X, y, split, names


def probabilities_24(estimator, X: np.ndarray) -> np.ndarray:
    """Map an estimator's probability columns into canonical 24-key order."""
    probability = np.asarray(estimator.predict_proba(X), dtype=float)
    result = np.zeros((len(X), 24), dtype=float)
    for column, key in enumerate(np.asarray(estimator.classes_, dtype=int)):
        result[:, int(key)] = probability[:, column]
    return result


def validation_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Compute the three ordered model-selection metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, probability.argmax(axis=1))),
        "top3_accuracy": float(top_k_accuracy_score(y_true, probability, k=3, labels=np.arange(24))),
        "log_loss": float(log_loss(y_true, probability, labels=np.arange(24))),
    }


def candidate_estimators(seed: int = RANDOM_SEED):
    """Yield every fixed logistic-regression and random-forest candidate."""
    for class_weight in (None, "balanced"):
        for c_value in (0.01, 0.1, 1.0, 10.0):
            config = {"kind": "logistic", "C": c_value, "class_weight": class_weight}
            yield config, LogisticRegression(
                C=c_value,
                class_weight=class_weight,
                solver="lbfgs",
                max_iter=5000,
                random_state=seed,
            )
    for minimum_leaf in (1, 2, 4):
        for max_features in ("sqrt", 0.5):
            config = {"kind": "random_forest", "n_estimators": 500,
                      "min_samples_leaf": minimum_leaf, "max_features": max_features,
                      "class_weight": "balanced_subsample"}
            yield config, RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=minimum_leaf,
                max_features=max_features,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            )


def _rank(result: dict[str, Any]) -> tuple[float, float, float, int]:
    metrics = result["validation"]
    return (metrics["accuracy"], metrics["top3_accuracy"], -metrics["log_loss"],
            int(result["config"]["kind"] == "logistic"))


def make_estimator(config: dict[str, Any], seed: int = RANDOM_SEED):
    """Recreate one selected estimator configuration for final fitting."""
    if config["kind"] == "logistic":
        return LogisticRegression(
            C=float(config["C"]), class_weight=config["class_weight"], solver="lbfgs",
            max_iter=5000, random_state=seed,
        )
    return RandomForestClassifier(
        n_estimators=int(config["n_estimators"]),
        min_samples_leaf=int(config["min_samples_leaf"]),
        max_features=config["max_features"],
        class_weight=config["class_weight"], random_state=seed, n_jobs=-1,
    )


def train_model(
    feature_path: str | Path,
    model_path: str | Path = "models/model.joblib",
    report_path: str | Path = "reports/training_metrics.json",
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Select on validation, refit on train+validation, and save the model."""
    frame, X, y, split, names = load_training_table(feature_path)
    train_mask, val_mask, test_mask = split == "train", split == "val", split == "test"
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise RuntimeError("feature cache must contain non-empty train, val, and test splits")
    selections: list[dict[str, Any]] = []
    for config, estimator in candidate_estimators(seed):
        estimator.fit(X[train_mask], y[train_mask])
        metrics = validation_metrics(y[val_mask], probabilities_24(estimator, X[val_mask]))
        selections.append({"config": config, "validation": metrics})
    winner = max(selections, key=_rank)
    final_estimator = make_estimator(winner["config"], seed)
    fit_mask = train_mask | val_mask
    final_estimator.fit(X[fit_mask], y[fit_mask])
    splits = {name: frame.loc[split == name, "track_id"].astype(str).tolist()
              for name in ("train", "val", "test")}
    artifact = {
        "model": final_estimator,
        "classes": list(KEY_NAMES),
        "feature_names": names,
        "feature_config": {
            "chroma_type": DEFAULT_CONFIG.chroma_type,
            "hop_length": DEFAULT_CONFIG.hop_length,
            "edge_frames": DEFAULT_CONFIG.edge_frames,
            "sample_rate": DEFAULT_CONFIG.sample_rate,
            "silence_top_db": DEFAULT_CONFIG.silence_top_db,
        },
        "seed": seed,
        "splits": splits,
        "model_selection": winner,
    }
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, destination)
    report: dict[str, Any] = {
        "seed": seed,
        "features": str(feature_path),
        "split_counts": {name: int((split == name).sum()) for name in ("train", "val", "test")},
        "candidates": selections,
        "selected": winner,
        "refit_rows": int(fit_mask.sum()),
        "test_rows_untouched_during_selection": int(test_mask.sum()),
        "model": str(destination),
    }
    if winner["config"]["kind"] == "random_forest":
        profile_json = destination.parent / "profile_analysis.json"
        profile_plot = destination.parent / "learned_vs_ks_profiles.png"
        profile = save_profile_analysis(final_estimator, X[val_mask], y[val_mask],
                                        profile_json, profile_plot, seed=seed)
        artifact["profile_json"] = str(profile_json)
        artifact["profile_plot"] = str(profile_plot)
        joblib.dump(artifact, destination)
        report["profile_analysis"] = profile
    report_destination = Path(report_path)
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    """Build the deterministic training command-line parser."""
    parser = argparse.ArgumentParser(description="Train the key-signature classifier")
    parser.add_argument("--features", help="Parquet/CSV cache; defaults to the sole data/features file")
    parser.add_argument("--features-dir", default="data/features")
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--report", default="reports/training_metrics.json")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for fixed-seed model selection and export."""
    args = build_parser().parse_args(argv)
    features = Path(args.features) if args.features else find_feature_file(args.features_dir)
    report = train_model(features, args.model, args.report, seed=args.seed)
    print(json.dumps(report["selected"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
