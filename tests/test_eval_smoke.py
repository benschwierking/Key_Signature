"""End-to-end smoke run of eval/evaluate.py on SYNTHETIC fixtures (no keyclf, no dataset audio).

Uses tests/fixtures/synthetic_fixture.py: a fake feature cache + fake LogisticRegression
model + a stand-in extractor for the latency path. Proves the evaluation pipeline,
artifacts and exit codes work before the implementation exists. Results here are
NOT evidence about the real classifier.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from eval.evaluate import main

from .conftest import REPO_ROOT
from .fixtures.synthetic_fixture import build

pytestmark = pytest.mark.harness
EXTRACTOR = "tests.fixtures.synthetic_fixture:synthetic_extract_features"


def _junit(path, skipped=False):
    body = "<skipped/>" if skipped else ""
    path.write_text(f'<testsuites><testsuite><testcase classname="tests.test_features" name="t">{body}'
                    f'</testcase></testsuite></testsuites>')
    return path


def _args(feat, model, out, junit, *extra):
    return ["--features", str(feat), "--model", str(model), "--out", str(out), "--junit", str(junit),
            "--extractor", EXTRACTOR, "--latency-duration", "8", "--latency-runs", "1", *extra]


def test_smoke_all_pass(tmp_path, capsys):
    feat, model = build(tmp_path / "fx")
    out = tmp_path / "reports"
    rc = main(_args(feat, model, out, _junit(tmp_path / "j.xml")))
    text = capsys.readouterr().out
    assert rc == 0, text
    m = json.loads((out / "metrics.json").read_text())
    assert m["all_section9_pass"] is True
    for f in ("metrics.json", "summary.md", "confusion_matrix_model.png", "confusion_matrix_model.csv",
              "confusion_matrix_ks.png", "confusion_matrix_ks.csv", "learned_vs_ks_profiles.png"):
        assert (out / f).stat().st_size > 0, f
    assert m["ks_baseline"]["test_tracks_sha256"] == m["split"]["test_tracks_sha256"]
    assert m["latency"]["status"] == "ok" and "SYNTHETIC" in m["latency"]["audio"]
    assert m["ks_baseline"]["temperature_fit_on"] == "val"
    assert main(["--criteria-only", str(out / "metrics.json")]) == 0


def test_smoke_shuffled_labels_fail_gate(tmp_path, capsys):
    # Same pipeline, labels permuted => accuracy collapses => §9 gate must exit 1.
    feat, model = build(tmp_path / "fx", shuffle_labels=True)
    rc = main(_args(feat, model, tmp_path / "r", _junit(tmp_path / "j.xml")))
    out = capsys.readouterr().out
    assert rc == 1 and "[FAIL] C1-exact-accuracy" in out


def test_missing_extractor_fails_latency_not_crash(tmp_path, capsys):
    # Before keyclf exists the default extractor is unimportable: C3 FAILs with a clear reason.
    feat, model = build(tmp_path / "fx")
    rc = main(["--features", str(feat), "--model", str(model), "--out", str(tmp_path / "r"),
               "--extractor", "keyclf_does_not_exist.features:extract_features"])
    out = capsys.readouterr().out
    assert rc == 1 and "[FAIL] C3-latency" in out and "cannot import" in out
    assert "[FAIL] C5-chroma-features-unit-tested" in out  # no --junit evidence


def test_missing_inputs_exit_2(tmp_path):
    assert main(["--features-dir", str(tmp_path / "nope"), "--model", str(tmp_path / "m.joblib"),
                 "--out", str(tmp_path / "r")]) == 2


def test_cli_module_invocation(tmp_path):
    feat, model = build(tmp_path / "fx", n_tracks=240)
    proc = subprocess.run([sys.executable, "-m", "eval.evaluate", *_args(feat, model, tmp_path / "r",
                                                                          _junit(tmp_path / "j.xml"))],
                          cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode in (0, 1), proc.stderr
    assert "C1-exact-accuracy" in proc.stdout
    proc2 = subprocess.run([sys.executable, "-m", "eval.check_criteria", str(tmp_path / "r" / "metrics.json")],
                           cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc2.returncode == proc.returncode and "spec:" in proc2.stdout
