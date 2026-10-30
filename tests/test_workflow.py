"""Self-test: .github/workflows/acceptance.yml parses and wires the spec gates."""
from __future__ import annotations

import pytest

from .conftest import REPO_ROOT

pytestmark = pytest.mark.harness
yaml = pytest.importorskip("yaml")
WF = REPO_ROOT / ".github" / "workflows" / "acceptance.yml"


def _wf():
    return yaml.safe_load(WF.read_text())


def test_workflow_parses_and_has_jobs():
    d = _wf()
    assert {"harness-selftest", "acceptance"} <= set(d["jobs"])
    assert (d.get("on") or d.get(True))  # PyYAML reads the bare `on:` key as True


def test_acceptance_job_wiring():
    d = _wf()
    job = d["jobs"]["acceptance"]
    text = WF.read_text()
    runs = "\n".join(s.get("run", "") for s in job["steps"])
    assert d["env"]["PYTHON_VERSION"] == "3.11"                       # spec §5 Python 3.11+
    assert "ffmpeg" in runs                                           # spec §14
    assert "python -m keyclf.train" in runs                           # spec §10 line 86, §13
    assert "pytest" in runs and "--junitxml" in runs                  # spec §9 line 81, §10 line 89
    assert "eval.evaluate" in runs and "--criteria-only" in runs      # spec §9 gate
    assert job["env"]["KEYCLF_REQUIRE_IMPL"] == "1"
    uploads = [s for s in job["steps"] if str(s.get("uses", "")).startswith("actions/upload-artifact")]
    assert uploads and "confusion_matrix_" in uploads[0]["with"]["path"] and "metrics.json" in uploads[0]["with"]["path"]
    gate = job["steps"][-1]
    assert gate.get("if") == "always()" and "exit $fail" in gate["run"]
    assert "giantsteps/backup" not in text.lower()  # spec §6/§12: CI must not bulk-download audio
