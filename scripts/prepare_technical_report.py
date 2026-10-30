"""Prepare measured numbers and point coordinates for the technical PowerPoint.

This script writes data only. The presentation builder uses editable native
PowerPoint charts, including scatter plots, rather than rasterizing the data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from threadpoolctl import threadpool_limits

from keyclf.experiments import estimator_for, rotate_training, select_columns, summarize
from keyclf.keys import parse_key
from keyclf.train import probabilities_24


def learning_curve(frame, config):
    """Measure the selected configuration at four original-recording budgets."""
    development = frame[frame.split != "test"].reset_index(drop=True)
    X = development[select_columns(frame, config["blocks"])].to_numpy(dtype=float)
    y = np.array([parse_key(k) for k in development.key])
    folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(X, y // 12))
    curve = []
    for fraction in (0.25, 0.5, 0.75, 1.0):
        scores, sizes = [], []
        for fold, (tr, va) in enumerate(folds):
            if fraction < 1:
                subset, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=fraction, random_state=42 + fold).split(tr, y[tr] // 12))
                tr = tr[subset]
            xt, yt = X[tr], y[tr]
            if config["augment"]:
                xt, yt = rotate_training(xt, yt)
            estimator = estimator_for(config)
            with threadpool_limits(limits=2):
                estimator.fit(xt, yt)
                scores.append(float(np.mean(estimator.predict(X[va]) == y[va])))
            sizes.append(len(tr))
        curve.append({"fraction": fraction, "mean_recordings": float(np.mean(sizes)),
                      "mean_accuracy": float(np.mean(scores)), "std_accuracy": float(np.std(scores)),
                      "fold_accuracies": scores})
        print(json.dumps({"learning_curve": curve[-1]}), flush=True)
    return curve


def main():
    """Combine current experiment evidence with the original benchmark report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--research", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--baseline-features", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--baseline-cv", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.research / "experiment_results.json").read_text())
    old = json.loads(args.baseline_metrics.read_text())
    baseline_frame = pd.read_parquet(args.baseline_features).sort_values("track_id").reset_index(drop=True)
    artifact = joblib.load(args.baseline_model)
    names = artifact["feature_names"]
    y = np.array([parse_key(k) for k in baseline_frame.key])
    mask = baseline_frame.split.to_numpy() != "test"
    p = probabilities_24(artifact["model"], baseline_frame[names].to_numpy(dtype=float))
    baseline_fit = summarize(y[mask], p[mask])
    baseline_test = summarize(y[~mask], p[~mask])
    frame = pd.read_parquet(args.features).sort_values("track_id").reset_index(drop=True)
    baseline_cv = json.loads((args.baseline_cv / "experiment_results.json").read_text())
    reference = next(r for r in baseline_cv["candidates"] if r["config"]["name"] == "v1_rf_reference")
    curve_path = args.research / "learning_curve.json"
    curve = json.loads(curve_path.read_text()) if curve_path.exists() else learning_curve(frame, report["selected"]["config"])
    curve_path.write_text(json.dumps(curve, indent=2) + "\n")
    points = json.loads((args.research / "projection_points.json").read_text())
    projection = json.loads((args.research / "projection_metrics.json").read_text())
    extra_path = Path(__file__).resolve().parents[1] / "reports/research-v2/additional_data_audit.json"
    additional = json.loads(extra_path.read_text()) if extra_path.exists() else {}
    latency_path = args.research / "research_latency.json"
    research_latency = json.loads(latency_path.read_text()) if latency_path.exists() else None
    payload = {
        "date": "2026-10-30", "research": report, "baseline_fit": baseline_fit,
        "baseline_test": baseline_test, "baseline_cv": reference["development_cv"],
        "original_recorded_test": old["model"], "ks_baseline": old["ks_baseline"],
        "baseline_latency": old["latency"], "learning_curve": curve,
        "projection": projection, "points": points,
        "additional_data": additional,
        "research_latency": research_latency,
        "original_feature_count": 36, "research_feature_count": len(report["final"]["feature_names"]),
        "precision_definition": "true positives divided by predicted positives",
        "recall_definition": "true positives divided by actual positives",
        "classification_is_supervised": True,
    }
    (args.research / "technical_report_data.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"baseline_test": baseline_test["accuracy"], "research_test": report["final"]["test"]["accuracy"],
                      "research_macro_recall": report["final"]["test"]["macro_recall_present"]}), flush=True)


if __name__ == "__main__":
    main()
