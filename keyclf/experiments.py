"""Opt-in, resumable accuracy experiments with a fixed original test split.

Three mode-stratified folds operate on development tracks only. All transposed
copies stay with their original recording. A single final selection precedes
test inference. The v1 artifact and executable 36-feature contract are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, adjusted_rand_score, classification_report, confusion_matrix,
    log_loss, normalized_mutual_info_score, recall_score, silhouette_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .keys import KEY_NAMES, parse_key
from .research_features import BLOCKS
from .train import probabilities_24

SEED = 42
EXPERIMENT_VERSION = "mode-cv-rotation-v2.2"


def rotate_training(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate every 12-bin block and tonic, retaining the original major/minor mode."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    if X.shape[1] % 12:
        raise ValueError("all research features must be 12-bin pitch-class blocks")
    blocks = X.reshape(len(X), -1, 12)
    return (
        np.concatenate([np.roll(blocks, shift, axis=2).reshape(X.shape) for shift in range(12)]),
        np.concatenate([(y % 12 + shift) % 12 + (y // 12) * 12 for shift in range(12)]),
    )


def summarize(y: np.ndarray, probability: np.ndarray) -> dict:
    """Report exact/top-three accuracy, mode recall, and explicit multiclass recall."""
    prediction = probability.argmax(axis=1)
    present = np.unique(y)
    ranked = np.argsort(probability, axis=1)[:, -3:]
    result = {
        "n": len(y), "accuracy": float(accuracy_score(y, prediction)),
        "top3_accuracy": float(np.mean(np.any(ranked == y[:, None], axis=1))),
        "log_loss": float(log_loss(y, probability, labels=np.arange(24))),
        "macro_recall_present": float(recall_score(y, prediction, labels=present, average="macro", zero_division=0)),
        "macro_recall_all24": float(recall_score(y, prediction, labels=np.arange(24), average="macro", zero_division=0)),
        "weighted_recall": float(recall_score(y, prediction, average="weighted", zero_division=0)),
        "mode_accuracy": float(np.mean(prediction // 12 == y // 12)),
        "confusion_matrix": confusion_matrix(y, prediction, labels=np.arange(24)).tolist(),
        "classification_report": classification_report(y, prediction, labels=np.arange(24), target_names=KEY_NAMES, output_dict=True, zero_division=0),
    }
    for mode, mask in (("major", y < 12), ("minor", y >= 12)):
        result[f"{mode}_n"] = int(mask.sum())
        result[f"{mode}_exact_recall"] = float(np.mean(prediction[mask] == y[mask])) if mask.any() else None
        result[f"{mode}_mode_recall"] = float(np.mean(prediction[mask] // 12 == y[mask] // 12)) if mask.any() else None
    return result


def estimator_for(config: dict):
    """Construct the stated classifier, fitting scaling inside each training fold."""
    if config["algorithm"] == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=config["C"], class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=SEED,
        ))
    return RandomForestClassifier(
        n_estimators=500, min_samples_leaf=config.get("leaf", 1),
        max_features=config.get("max_features", 0.5), class_weight="balanced_subsample",
        random_state=SEED, n_jobs=2,
    )


def candidates(available: list[str]) -> list[dict]:
    """Return a small preregistered matrix of feature ablations and algorithms."""
    groups = {"v1": ["first", "last", "hist"], "histogram": ["hist"]}
    if "harmonic" in available:
        groups.update({
            "harmonic_bass": ["harmonic", "bass"],
            "persistence": ["harmonic", "bass", "sustain"],
            "beat_accent": ["harmonic", "bass", "beat", "accent"],
            "temporal": ["harmonic", "bass", "quarter1", "quarter2", "quarter3", "quarter4"],
            "all_features": list(BLOCKS),
        })
    result = [{"name": "v1_rf_reference", "group": "v1", "blocks": groups["v1"], "algorithm": "forest", "augment": False, "leaf": 1, "max_features": 0.5}]
    for group, blocks in groups.items():
        for c in (0.1, 1.0, 10.0):
            result.append({"name": f"{group}_scaled_lr_C{c}_transpose", "group": group, "blocks": blocks, "algorithm": "logistic", "C": c, "augment": True})
        result.append({"name": f"{group}_rf_transpose", "group": group, "blocks": blocks, "algorithm": "forest", "augment": True, "leaf": 2, "max_features": "sqrt"})
    result.append({"name": "v1_scaled_lr_C1_unaugmented", "group": "v1", "blocks": groups["v1"], "algorithm": "logistic", "C": 1.0, "augment": False})
    return result


def select_columns(frame: pd.DataFrame, blocks: list[str]) -> list[str]:
    """Return features in numeric pitch order, never alphabetical column order."""
    names = [f"{block}_{pc}" for block in blocks for pc in range(12)]
    if not set(names) <= set(frame):
        raise ValueError("feature cache is missing a requested block")
    return names


def audit(frame: pd.DataFrame, y: np.ndarray, split: np.ndarray) -> dict:
    """Record class coverage, duplicate tracks, and train/validation/test proportions."""
    if frame.track_id.duplicated().any():
        raise ValueError("duplicate track IDs in the research cache")
    counts = {name: int((split == name).sum()) for name in ("train", "val", "test")}
    return {
        "n": len(frame), "split_counts": counts,
        "split_percentages": {name: value / len(frame) * 100 for name, value in counts.items()},
        "class_counts": {name: int((y == k).sum()) for k, name in enumerate(KEY_NAMES)},
        "mode_counts": {"major": int((y < 12).sum()), "minor": int((y >= 12).sum())},
        "duplicate_track_ids": 0, "artist_grouping_available": False,
        "test_id_sha256": hashlib.sha256("\n".join(sorted(frame.loc[split == "test", "track_id"].astype(str))).encode()).hexdigest(),
    }


def validate_reference(frame: pd.DataFrame, reference: pd.DataFrame) -> None:
    """Reject missing recordings, changed labels, or changed original split IDs."""
    fields = ["track_id", "key", "split"]
    current = frame[fields].astype(str).sort_values("track_id").reset_index(drop=True)
    expected = reference[fields].astype(str).sort_values("track_id").reset_index(drop=True)
    if not current.equals(expected):
        raise ValueError("research cache must preserve every original recording, label, and split")


def run_experiments(features: Path, output: Path, finalize: bool = False, reference_features: Path | None = None) -> dict:
    """Compare candidates on development folds, optionally finalize one test evaluation."""
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(features).sort_values("track_id").reset_index(drop=True)
    if "harmonic_0" in frame:
        if reference_features is None or not reference_features.exists():
            raise ValueError("research extraction requires --reference with the original complete feature cache")
        validate_reference(frame, pd.read_parquet(reference_features))
    y = np.array([parse_key(label) for label in frame.key])
    split = frame.split.to_numpy()
    if set(split) != {"train", "val", "test"}:
        raise ValueError("fixed train, val, and test split required")
    info = audit(frame, y, split)
    (output / "audit.json").write_text(json.dumps(info, indent=2) + "\n")
    development = np.flatnonzero(split != "test")
    test = np.flatnonzero(split == "test")
    yd = y[development]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    folds = list(cv.split(development, yd // 12))
    fold_ids = [{"training": frame.iloc[development[tr]].track_id.astype(str).tolist(),
                 "validation": frame.iloc[development[va]].track_id.astype(str).tolist()} for tr, va in folds]
    (output / "fold_ids.json").write_text(json.dumps(fold_ids, indent=2) + "\n")
    available = [name for name in BLOCKS if f"{name}_0" in frame]
    results = []
    cache_signature = hashlib.sha256(features.read_bytes()).hexdigest()
    for config in candidates(available):
        checkpoint = output / f"candidate_{config['name']}.json"
        signature = hashlib.sha256(json.dumps({"protocol": EXPERIMENT_VERSION, "config": config, "cache": cache_signature, "seed": SEED}, sort_keys=True).encode()).hexdigest()
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved.get("signature") == signature:
                results.append(saved)
                print(f"resumed {config['name']}", flush=True)
                continue
        columns = select_columns(frame, config["blocks"])
        X = frame[columns].to_numpy(dtype=float)[development]
        oof = np.zeros((len(development), 24))
        fold_metrics = []
        for fold, (tr, va) in enumerate(folds):
            xt, yt = X[tr], yd[tr]
            if config["augment"]:
                xt, yt = rotate_training(xt, yt)
            model = estimator_for(config)
            with threadpool_limits(limits=2):
                model.fit(xt, yt)
                probability = probabilities_24(model, X[va])
            oof[va] = probability
            fold_metrics.append(summarize(yd[va], probability))
        metrics = summarize(yd, oof)
        saved = {"signature": signature, "config": config, "development_cv": metrics,
                 "fold_exact_accuracy": [m["accuracy"] for m in fold_metrics],
                 "test_used_for_selection": False}
        checkpoint.write_text(json.dumps(saved, indent=2) + "\n")
        results.append(saved)
        print(json.dumps({"candidate": config["name"], "accuracy": metrics["accuracy"],
                          "macro_recall": metrics["macro_recall_present"], "major_recall": metrics["major_exact_recall"]}), flush=True)
    # Preregistered ranking: exact accuracy, top three, log loss. Macro recall
    # is reported rather than silently replacing the user's exact-key target.
    winner = max(results, key=lambda r: (r["development_cv"]["accuracy"], r["development_cv"]["top3_accuracy"], -r["development_cv"]["log_loss"]))
    report = {"seed": SEED, "experiment_version": EXPERIMENT_VERSION, "audit": info,
              "environment": {"python": platform.python_version(), **{name: version(name) for name in ("numpy", "scikit-learn", "pandas", "librosa")}},
              "cv": "3 folds stratified by mode, development tracks only",
              "augmentation": "12 pitch-class rotations inside training folds only; mode unchanged",
              "selection_order": ["exact_accuracy", "top3_accuracy", "negative_log_loss"],
              "selected": winner, "candidates": results, "test_evaluated": False}
    if finalize:
        receipt = output / "final_test.json"
        if receipt.exists():
            prior = json.loads(receipt.read_text())
            if prior["selected_name"] != winner["config"]["name"] or prior["cache_signature"] != cache_signature:
                raise RuntimeError("test has already been evaluated for a different selection; preserve the receipt")
            report["final"] = prior
        else:
            config = winner["config"]
            names = select_columns(frame, config["blocks"])
            X = frame[names].to_numpy(dtype=float)
            xt, yt = X[development], y[development]
            if config["augment"]:
                xt, yt = rotate_training(xt, yt)
            model = estimator_for(config)
            with threadpool_limits(limits=2):
                model.fit(xt, yt)
                fit_prob = probabilities_24(model, X[development])
                test_prob = probabilities_24(model, X[test])
            final = {"selected_name": config["name"], "cache_signature": cache_signature,
                     "train_resubstitution": summarize(y[development], fit_prob),
                     "test": summarize(y[test], test_prob), "fit_recordings": len(development),
                     "augmented_training_rows": len(xt), "feature_names": names}
            receipt.write_text(json.dumps(final, indent=2) + "\n")
            pd.DataFrame({"track_id": frame.iloc[test].track_id.to_numpy(), "truth": y[test],
                          "prediction": test_prob.argmax(axis=1)}).to_csv(output / "test_predictions.csv", index=False)
            joblib.dump({"model": model, "feature_names": names, "research_version": "v2.1",
                         "config": config, "classes": list(KEY_NAMES), "seed": SEED,
                         "splits": {s: frame.loc[split == s, "track_id"].astype(str).tolist() for s in ("train", "val", "test")}}, output / "research_model.joblib")
            report["final"] = final
        report["test_evaluated"] = True
    (output / "experiment_results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def export_projections(features: Path, output: Path, blocks: list[str]) -> dict:
    """Export data for editable PCA/t-SNE and exploratory K-means plots."""
    frame = pd.read_parquet(features).sort_values("track_id").reset_index(drop=True)
    names = select_columns(frame, blocks)
    X = frame[names].to_numpy(dtype=float)
    dev = frame.split.to_numpy() != "test"
    scaled = StandardScaler().fit(X[dev])
    Z = scaled.transform(X)
    pca = PCA(n_components=2, random_state=SEED).fit(Z[dev])
    xy = pca.transform(Z)
    km = KMeans(n_clusters=24, random_state=SEED, n_init=20).fit(Z[dev])
    clusters = km.predict(Z)
    y = np.array([parse_key(k) for k in frame.key])
    points = frame[["track_id", "key", "split"]].copy()
    points["mode"] = np.where(y < 12, "major", "minor")
    points["tonic"] = y % 12
    points["pca1"], points["pca2"] = xy[:, 0], xy[:, 1]
    points["cluster"] = clusters
    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, init="pca", learning_rate="auto", max_iter=1000).fit_transform(Z[dev])
    points["tsne1"], points["tsne2"] = np.nan, np.nan
    points.loc[dev, ["tsne1", "tsne2"]] = tsne
    points.to_csv(output / "projection_points.csv", index=False)
    points.to_json(output / "projection_points.json", orient="records")
    metrics = {"pca_explained_variance": pca.explained_variance_ratio_.tolist(),
               "pca_fit": "development only, test transformed afterward", "tsne_fit": "development only",
               "kmeans_clusters": 24, "kmeans_silhouette": float(silhouette_score(Z[dev], clusters[dev])),
               "kmeans_adjusted_rand": float(adjusted_rand_score(y[dev], clusters[dev])),
               "kmeans_normalized_mutual_information": float(normalized_mutual_info_score(y[dev], clusters[dev])),
               "classifier_is_clustering": False, "feature_blocks": blocks}
    (output / "projection_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main(argv: list[str] | None = None) -> int:
    """Run opt-in experiments and save a separate research artifact and report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/research-v2"))
    parser.add_argument("--reference", type=Path, default=Path("data/features/giantsteps_features.parquet"))
    parser.add_argument("--finalize", action="store_true", help="evaluate the CV-selected model once on the fixed test set")
    parser.add_argument("--projections", action="store_true")
    args = parser.parse_args(argv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        report = run_experiments(args.features, args.out, args.finalize, args.reference)
        if args.projections:
            with threadpool_limits(limits=2):
                export_projections(args.features, args.out, report["selected"]["config"]["blocks"])
    print(json.dumps({"selected": report["selected"]["config"], "cv_accuracy": report["selected"]["development_cv"]["accuracy"],
                      "test": report.get("final", {}).get("test", {}).get("accuracy")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
