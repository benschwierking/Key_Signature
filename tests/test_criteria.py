"""Self-tests for the spec §9 criteria gate (eval/criteria.py) on synthetic metrics.

Boundary tests use ONLY the §9 numbers: 80 % exact, 95 % top-3, < 10 s.
Optionally, set KEYCLF_METRICS=path/to/metrics.json (and KEYCLF_JUNIT) to also
gate a real report from pytest; CI gates via ``python -m eval.evaluate --criteria-only``.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from eval import criteria as C

pytestmark = pytest.mark.harness

JUNIT_OK = """<testsuites><testsuite name="pytest">
<testcase classname="tests.test_features" name="test_a440_sine_puts_chroma_mass_in_A"/>
<testcase classname="tests.test_features" name="test_extract_features_a440"/>
<testcase classname="tests.test_train" name="test_x"><skipped/></testcase>
</testsuite></testsuites>"""


@pytest.fixture
def good(tmp_path):
    for f in ("cm.png", "cm.csv", "prof.png"):
        (tmp_path / f).write_bytes(b"x")
    (tmp_path / "junit.xml").write_text(JUNIT_OK)
    cm = np.zeros((24, 24), int); cm[0, 0] = 90; cm[0, 21] = 10
    m = {
        "split": {"test_n": 100, "test_tracks_sha256": "abc", "leakage": {}},
        "model": {"accuracy": 0.80, "top3_accuracy": 0.95, "log_loss": 0.5, "related_key_accuracy": 0.9,
                  "confusion_matrix": cm.tolist(), "n": 100},
        "ks_baseline": {"accuracy": 0.6, "top3_accuracy": 0.8, "log_loss": 1.0, "related_key_accuracy": 0.8,
                        "confusion_matrix": cm.tolist(), "n": 100, "test_tracks_sha256": "abc"},
        "latency": {"status": "ok", "runs_s": [1.0, 9.99], "mean_s": 5.5, "max_s": 9.99, "audio": "x"},
        "profiles": {"plot": "prof.png", "corr_major": 0.7, "corr_minor": 0.6},
        "artifacts": {"confusion_matrix_png": "cm.png", "confusion_matrix_csv": "cm.csv"},
        "feature_tests": {"junit": "junit.xml"},
    }
    return m, tmp_path


def _status(m, base, cid, junit=None):
    return {r.id: r.status for r in C.evaluate_criteria(m, base, junit)}[cid]


def test_all_pass_at_exact_spec_boundaries(good):
    # §9: "≥ 80%" and "at least 95%" are inclusive; latency 9.99 s < 10 s.
    m, base = good
    res = C.evaluate_criteria(m, base)
    assert C.all_pass(res), C.format_results(res)
    assert {r.id for r in res if r.status != "INFO"} == {
        "C1-exact-accuracy", "C2-top3", "C3-latency", "C4-confusion-matrix",
        "C5-chroma-features-unit-tested", "C6-learned-vs-ks-profiles", "C7-ks-baseline-same-split"}


@pytest.mark.parametrize("path,value,cid", [
    (("model", "accuracy"), 0.7999, "C1-exact-accuracy"),          # §9 line 77
    (("model", "top3_accuracy"), 0.9499, "C2-top3"),               # §9 line 78
    (("latency", "max_s"), 10.0, "C3-latency"),                    # §9 line 79: strictly under 10 s
    (("latency", "status"), "error", "C3-latency"),
    (("artifacts", "confusion_matrix_png"), "nope.png", "C4-confusion-matrix"),  # §9 line 80
    (("profiles", "corr_major"), None, "C6-learned-vs-ks-profiles"),            # §9 line 82
    (("profiles", "plot"), "missing.png", "C6-learned-vs-ks-profiles"),
    (("ks_baseline", "test_tracks_sha256"), "other", "C7-ks-baseline-same-split"),  # §9 line 83
    (("ks_baseline", "accuracy"), float("nan"), "C7-ks-baseline-same-split"),
])
def test_each_criterion_fails_independently(good, path, value, cid):
    m, base = good
    m = copy.deepcopy(m)
    m[path[0]][path[1]] = value
    assert _status(m, base, cid) == "FAIL"
    assert not C.all_pass(C.evaluate_criteria(m, base))


def test_profile_correlation_has_no_threshold(good):
    # §9 line 82 states no correlation threshold: even a negative r passes if produced and reported.
    m, base = good
    m["profiles"]["corr_major"] = -0.9
    assert _status(m, base, "C6-learned-vs-ks-profiles") == "PASS"


def test_feature_unit_test_evidence(good, tmp_path):
    # §9 line 81: needs passing tests/test_features.py results; skipped or missing => not verified.
    m, base = good
    skipped = tmp_path / "skipped.xml"
    skipped.write_text(JUNIT_OK.replace('name="test_extract_features_a440"/>',
                                        'name="test_extract_features_a440"><skipped/></testcase>'))
    assert _status(m, base, "C5-chroma-features-unit-tested", skipped) == "FAIL"
    m2 = copy.deepcopy(m); m2["feature_tests"]["junit"] = None
    assert _status(m2, base, "C5-chroma-features-unit-tested") == "FAIL"
    s = C.junit_feature_summary(base / "junit.xml")
    assert (s["tests"], s["passed"], s["skipped"]) == (2, 2, 0)  # test_train rows ignored


def test_format_quotes_spec_text(good):
    m, base = good
    out = C.format_results(C.evaluate_criteria(m, base))
    assert "≥ 80% exact-key accuracy on held-out test split." in out
    assert "Upload-to-result in under 10 seconds." in out and "ALL §9 ACCEPTANCE CRITERIA PASS" in out


@pytest.mark.skipif(not os.environ.get("KEYCLF_METRICS"), reason="set KEYCLF_METRICS to gate a real metrics.json")
def test_real_metrics_json_meets_section9():
    p = Path(os.environ["KEYCLF_METRICS"])
    res = C.evaluate_criteria(json.loads(p.read_text()), p.parent, os.environ.get("KEYCLF_JUNIT"))
    assert C.all_pass(res), C.format_results(res)
