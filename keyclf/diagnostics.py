"""Bounded, leakage-safe diagnostics for an acceptance shortfall."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .keys import KEY_NAMES


def per_class_errors(metrics: dict) -> pd.DataFrame:
    """Summarize held-out support, accuracy, and most common confusion per key."""
    matrix = np.asarray(metrics["model"]["confusion_matrix"], dtype=int)
    rows = []
    for key, name in enumerate(KEY_NAMES):
        support = int(matrix[key].sum())
        correct = int(matrix[key, key])
        wrong = matrix[key].copy()
        wrong[key] = 0
        confused = int(wrong.argmax()) if wrong.sum() else None
        rows.append(
            {
                "key": name,
                "support": support,
                "correct": correct,
                "accuracy": float(correct / support) if support else None,
                "most_confused_with": KEY_NAMES[confused] if confused is not None else None,
                "confusion_count": int(wrong[confused]) if confused is not None else 0,
            }
        )
    return pd.DataFrame(rows)


def write_diagnostics(
    metrics_path: str | Path = "reports/metrics.json",
    training_path: str | Path = "reports/training_metrics.json",
    output_dir: str | Path = "reports",
    alternate_training_path: str | Path | None = None,
) -> dict:
    """Write per-class CSV and a compact JSON diagnosis without retuning on test."""
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    training = json.loads(Path(training_path).read_text(encoding="utf-8"))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table = per_class_errors(metrics)
    table.to_csv(output / "per_class_errors.csv", index=False)
    ranked = sorted(
        training["candidates"],
        key=lambda row: (
            row["validation"]["accuracy"],
            row["validation"]["top3_accuracy"],
            -row["validation"]["log_loss"],
            row["config"]["kind"] == "logistic",
        ),
        reverse=True,
    )
    lowest_support = table.sort_values(["support", "key"]).head(8).to_dict("records")
    for row in lowest_support:
        for key, value in list(row.items()):
            if pd.isna(value):
                row[key] = None
    payload = {
        "policy": "diagnostic only; held-out test split was not used for selection or changed",
        "test": {
            "accuracy": metrics["model"]["accuracy"],
            "top3_accuracy": metrics["model"]["top3_accuracy"],
            "ks_accuracy": metrics["ks_baseline"]["accuracy"],
            "n": metrics["split"]["test_n"],
        },
        "selected_validation": training["selected"],
        "top_validation_candidates": ranked[:5],
        "lowest_support_keys": lowest_support,
        "per_class_csv": "per_class_errors.csv",
    }
    if alternate_training_path:
        alternate = json.loads(Path(alternate_training_path).read_text(encoding="utf-8"))
        payload["bounded_alternative"] = {
            "name": "librosa chroma_stft",
            "selected_validation": alternate["selected"],
            "test_evaluated": False,
            "decision": "rejected because validation exact accuracy was below the selected CQT pipeline",
        }
    (output / "diagnostics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the bounded acceptance-shortfall report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default="reports/metrics.json")
    parser.add_argument("--training", default="reports/training_metrics.json")
    parser.add_argument("--out", default="reports")
    parser.add_argument("--alternate-training")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            write_diagnostics(
                args.metrics, args.training, args.out, args.alternate_training
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
