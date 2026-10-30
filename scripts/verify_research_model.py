"""Verify a second deterministic refit without selecting or evaluating a model."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from keyclf.experiments import estimator_for, rotate_training
from keyclf.keys import parse_key
from keyclf.train import probabilities_24


def main():
    """Refit the fixed configuration and compare probabilities and split IDs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_parquet(args.features).sort_values("track_id").reset_index(drop=True)
    artifact = joblib.load(args.model)
    X = frame[artifact["feature_names"]].to_numpy(dtype=float)
    y = np.array([parse_key(k) for k in frame.key])
    train = frame.split.to_numpy() != "test"
    splits = {s: frame.loc[frame.split == s, "track_id"].astype(str).tolist() for s in ("train", "val", "test")}
    assert json.dumps(splits, sort_keys=True) == json.dumps(artifact["splits"], sort_keys=True)
    xt, yt = X[train], y[train]
    if artifact["config"]["augment"]:
        xt, yt = rotate_training(xt, yt)
    model = estimator_for(artifact["config"])
    with threadpool_limits(limits=2):
        model.fit(xt, yt)
        original = probabilities_24(artifact["model"], X)
        repeated = probabilities_24(model, X)
    np.testing.assert_allclose(original, repeated, rtol=0, atol=1e-12)
    result = {"passed": True, "split_ids_byte_equivalent": True,
              "maximum_probability_difference": float(np.max(np.abs(original - repeated))),
              "absolute_tolerance": 1e-12, "recordings_compared": len(frame),
              "configuration_reselected": False, "test_metrics_computed": False}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
