"""Spec §10 quality-rubric checks for training (``keyclf.train``) and its artifacts.

Spec §10 lines 86-89:
  - "Results reproducible from a fixed random seed and a single train command."
  - "Train/validation/test split by track, no leakage."
  - "Report (for both the ML model and the K-S baseline): accuracy, top-3 accuracy,
     log loss, "related-key" accuracy (exact or relative major/minor or perfect fifth),
     confusion matrix."
  - "Code passes `pytest` and is readable with docstrings on public functions."
Spec §13: the single train command is ``python -m keyclf.train``.
Spec §12: training reads the cached feature file under data/features/ and writes
models/model.joblib; "training never re-fetches audio".

Interface assumptions (the spec does not name train.py functions):
  * ``python -m keyclf.train`` run with the repo root as working directory reads
    ``data/features/`` and writes ``models/model.joblib`` relative to it.
    Override with KEYCLF_TRAIN_CMD (``{workdir}`` is substituted).
  * The split is discoverable by the harness (see eval/dataset.py resolve_split).
"""
from __future__ import annotations

import inspect
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from eval.dataset import leakage, load_features, load_model, predict_proba_24, resolve_split
from eval.metrics import REPORT_KEYS

from .conftest import REPO_ROOT, require_artifacts, require_impl

pytestmark = pytest.mark.impl
TRAIN_TIMEOUT_S = int(os.environ.get("KEYCLF_TRAIN_TIMEOUT", "1800"))  # harness safety net, not a criterion


def _run_train(workdir: Path) -> Path:
    cmd = os.environ.get("KEYCLF_TRAIN_CMD", f"{shlex.quote(sys.executable)} -m keyclf.train")
    cmd = cmd.replace("{workdir}", str(workdir))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), os.environ.get("PYTHONPATH", "")])}
    proc = subprocess.run(cmd, shell=True, cwd=workdir, env=env, capture_output=True, text=True,
                          timeout=TRAIN_TIMEOUT_S)
    assert proc.returncode == 0, f"train command failed ({cmd}):\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    model = workdir / "models" / "model.joblib"
    assert model.exists(), (f"`{cmd}` did not write models/model.joblib under the working directory "
                            "(spec §12); set KEYCLF_TRAIN_CMD if your CLI takes paths")
    return model


def _sandbox(tmp: Path) -> Path:
    """Copy only data/features (+ annotations if present) — never audio (spec §12)."""
    for sub in ("data/features", "data/annotations"):
        if (REPO_ROOT / sub).exists():
            shutil.copytree(REPO_ROOT / sub, tmp / sub)
    for cfg in ("pyproject.toml", "setup.cfg", "config.yaml", "config.yml", "keyclf.yaml"):
        if (REPO_ROOT / cfg).exists():
            shutil.copy(REPO_ROOT / cfg, tmp / cfg)
    return tmp


def test_train_module_importable(train_mod):
    # Spec §11: keyclf/train.py "train/eval models, save model.joblib + metrics".
    assert train_mod is not None


def test_single_train_command_is_reproducible(train_mod, tmp_path):
    # Spec §10 line 86: "Results reproducible from a fixed random seed and a single train command."
    feat, _ = require_artifacts()
    preds, splits = [], []
    for run in ("run1", "run2"):
        wd = _sandbox(tmp_path / run)
        est, meta = load_model(_run_train(wd))
        table = load_features(wd / "data" / "features" / feat.name)
        preds.append(predict_proba_24(est, table.X, table.feature_names, meta))
        splits.append(resolve_split(table, None, meta)[0])
    np.testing.assert_array_equal(splits[0], splits[1], err_msg="split differs between identical train runs")
    np.testing.assert_allclose(preds[0], preds[1], rtol=0, atol=1e-12,
                               err_msg="model predictions differ between identical train runs")


def test_split_by_track_no_leakage(train_mod):
    # Spec §10 line 87: "Train/validation/test split by track, no leakage."
    feat, model_path = require_artifacts()
    table = load_features(feat)
    _, meta = load_model(model_path)
    split, src = resolve_split(table, None, meta)
    leaks = leakage(table.tracks, split)
    assert not leaks, f"{len(leaks)} tracks appear in more than one split ({src}): {list(leaks.items())[:5]}"
    for s in ("train", "test"):
        assert (split == s).any(), f"split '{s}' is empty ({src})"
    # §10 line 87 names a validation split; §5 picks the final model "on the validation split".
    assert (split == "val").any(), f"no validation split found ({src}); spec §5 compares models on it"


def test_report_has_all_metrics_for_model_and_ks_baseline(train_mod, tmp_path):
    # Spec §10 line 88: both systems report accuracy, top-3, log loss, related-key accuracy, confusion matrix.
    from eval.evaluate import build_parser, run_evaluation

    feat, model_path = require_artifacts()
    args = build_parser().parse_args(["--model", str(model_path), "--features", str(feat),
                                      "--out", str(tmp_path), "--skip-latency"])
    m = run_evaluation(args)
    for system in ("model", "ks_baseline"):
        for key in REPORT_KEYS:
            assert key in m[system], f"{system} report missing {key} (spec §10 line 88)"
        cm = np.array(m[system]["confusion_matrix"])
        assert cm.shape == (24, 24) and cm.sum() == m["split"]["test_n"]
        for key in ("accuracy", "top3_accuracy", "log_loss", "related_key_accuracy"):
            assert np.isfinite(m[system][key])


@pytest.mark.parametrize("modname", ["keyclf.train", "keyclf.features", "keyclf.data"])
def test_public_functions_have_docstrings(modname):
    # Spec §10 line 89: "readable with docstrings on public functions". (app.py excluded: imports streamlit.)
    mod = require_impl(modname)
    missing = [n for n, f in inspect.getmembers(mod, inspect.isfunction)
               if not n.startswith("_") and f.__module__ == mod.__name__ and not inspect.getdoc(f)]
    assert not missing, f"{modname}: public functions without docstrings: {missing}"
