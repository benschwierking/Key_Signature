"""Spec §9 acceptance-criteria checks over ``metrics.json``.

Each criterion quotes the spec text verbatim. Thresholds come only from §9:
  * 80 %  exact-key accuracy  (§9 line 77; also §3)
  * 95 %  top-3               (§9 line 78)
  * 10 s  upload-to-result    (§9 line 79)
No other thresholds are applied. §10 rubric items are reported as INFO lines
(they are enforced by pytest: tests/test_train.py), not by this gate.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

SPEC_PATH = "specs/key-signature-classifier-spec.md"

# Spec §9 thresholds (verbatim numbers, nothing else).
EXACT_ACC_MIN = 0.80   # §9: "≥ 80% exact-key accuracy on held-out test split."
TOP3_MIN = 0.95        # §9: "Correct key in top 3 at least 95% of the time."
LATENCY_MAX_S = 10.0   # §9: "Upload-to-result in under 10 seconds."


@dataclass
class Result:
    """Outcome of one acceptance check."""

    id: str
    spec_ref: str
    spec_text: str
    status: str  # PASS / FAIL / INFO
    detail: str


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _resolve(base: Path, p: str | None) -> Path | None:
    if not p:
        return None
    q = Path(p)
    return q if q.is_absolute() else (base / q)


def junit_feature_summary(junit_path: str | Path | None, module: str = "test_features") -> dict:
    """Count passed/failed/skipped tests from ``module`` in a pytest JUnit XML file."""
    out = {"junit": str(junit_path) if junit_path else None, "tests": 0, "passed": 0,
           "failed": 0, "errors": 0, "skipped": 0}
    if not junit_path or not Path(junit_path).exists():
        out["missing"] = True
        return out
    root = ET.parse(junit_path).getroot()
    for tc in root.iter("testcase"):
        where = (tc.get("classname") or "") + " " + (tc.get("file") or "")
        if module not in where:
            continue
        out["tests"] += 1
        if tc.find("failure") is not None:
            out["failed"] += 1
        elif tc.find("error") is not None:
            out["errors"] += 1
        elif tc.find("skipped") is not None:
            out["skipped"] += 1
        else:
            out["passed"] += 1
    return out


def evaluate_criteria(metrics: dict, base_dir: str | Path = ".",
                      junit_path: str | Path | None = None) -> list[Result]:
    """Return one Result per spec §9 criterion (+ INFO lines for §10)."""
    base = Path(base_dir)
    res: list[Result] = []
    model = metrics.get("model") or {}
    ks = metrics.get("ks_baseline") or {}
    split = metrics.get("split") or {}

    # §9 bullet 1 (line 77)
    acc = model.get("accuracy")
    n = split.get("test_n", 0)
    ok = _finite(acc) and n > 0 and acc >= EXACT_ACC_MIN
    res.append(Result("C1-exact-accuracy", "§9 line 77 (also §3)",
                      "≥ 80% exact-key accuracy on held-out test split.",
                      "PASS" if ok else "FAIL",
                      f"accuracy={acc!r} on n_test={n} (need ≥ {EXACT_ACC_MIN})"))

    # §9 bullet 2 (line 78)
    top3 = model.get("top3_accuracy")
    ok = _finite(top3) and n > 0 and top3 >= TOP3_MIN
    res.append(Result("C2-top3", "§9 line 78", "Correct key in top 3 at least 95% of the time.",
                      "PASS" if ok else "FAIL", f"top3_accuracy={top3!r} (need ≥ {TOP3_MIN})"))

    # §9 bullet 3 (line 79)
    lat = metrics.get("latency") or {}
    mx = lat.get("max_s")
    ok = lat.get("status") == "ok" and _finite(mx) and mx < LATENCY_MAX_S
    detail = (f"max={mx!r}s mean={lat.get('mean_s')!r}s over {len(lat.get('runs_s') or [])} runs "
              f"({lat.get('audio')}) (need < {LATENCY_MAX_S}s)")
    if lat.get("status") != "ok":
        detail = f"latency not measured: {lat.get('error', 'missing')}"
    res.append(Result("C3-latency", "§9 line 79 (§16 task 11)", "Upload-to-result in under 10 seconds.",
                      "PASS" if ok else "FAIL", detail))

    # §9 bullet 4 (line 80)
    arts = metrics.get("artifacts") or {}
    png, csv = _resolve(base, arts.get("confusion_matrix_png")), _resolve(base, arts.get("confusion_matrix_csv"))
    cm = model.get("confusion_matrix")
    cm_ok = isinstance(cm, list) and len(cm) == 24 and sum(map(sum, cm)) == n and n > 0
    ok = bool(png and png.exists() and csv and csv.exists() and cm_ok)
    res.append(Result("C4-confusion-matrix", "§9 line 80", "Confusion matrix generated showing where mistakes land.",
                      "PASS" if ok else "FAIL",
                      f"png={png} exists={bool(png and png.exists())}, csv={csv} exists={bool(csv and csv.exists())}, "
                      f"24x24 counts sum to n_test={cm_ok}"))

    # §9 bullet 5 (line 81)
    jp = junit_path or (metrics.get("feature_tests") or {}).get("junit")
    if jp and not Path(jp).exists():  # relative to metrics.json's directory as a fallback
        jp = _resolve(base, jp)
    js = junit_feature_summary(jp)
    ok = js["tests"] > 0 and js["passed"] == js["tests"]
    res.append(Result("C5-chroma-features-unit-tested", "§9 line 81 (§7, §16 task 3)",
                      "Features are chroma-based (first bin, last bin, 12-bin track histogram) computed "
                      "with librosa, verified by unit tests on synthetic tones.",
                      "PASS" if ok else "FAIL",
                      ("no JUnit XML from tests/test_features.py supplied (--junit)" if js.get("missing") else
                       f"tests/test_features.py: {js['passed']}/{js['tests']} passed, {js['failed']} failed, "
                       f"{js['errors']} errors, {js['skipped']} skipped (skips count as not verified)")))

    # §9 bullet 6 (line 82) — artifact + reported correlation only; NO threshold in spec.
    prof = metrics.get("profiles") or {}
    plot = _resolve(base, prof.get("plot"))
    ok = bool(plot and plot.exists() and _finite(prof.get("corr_major")) and _finite(prof.get("corr_minor")))
    res.append(Result("C6-learned-vs-ks-profiles", "§9 line 82 (§16 task 9)",
                      "Learned feature weights for the histogram are extracted per key, rotated to a common "
                      "tonic, and plotted against the K-S major and minor profiles, with the correlation reported.",
                      "PASS" if ok else "FAIL",
                      f"plot={plot} exists={bool(plot and plot.exists())}, corr_major={prof.get('corr_major')!r}, "
                      f"corr_minor={prof.get('corr_minor')!r} (spec gives no correlation threshold; "
                      f"only presence is checked){' — ' + prof['error'] if prof.get('error') else ''}"))

    # §9 bullet 7 (line 83)
    same = (ks.get("n") == n and n > 0 and ks.get("test_tracks_sha256") == split.get("test_tracks_sha256"))
    ok = _finite(ks.get("accuracy")) and same
    res.append(Result("C7-ks-baseline-same-split", "§9 line 83 (§16 task 5)",
                      "A plain K-S template-matching baseline is scored on the same test split for comparison.",
                      "PASS" if ok else "FAIL",
                      f"ks accuracy={ks.get('accuracy')!r}, n={ks.get('n')} vs model n={n}, same track hash={same}"))

    # §10 rubric — informational here; gated by pytest (tests/test_train.py).
    from .metrics import REPORT_KEYS
    missing = [f"{sysname}.{k}" for sysname, d in (("model", model), ("ks_baseline", ks))
               for k in REPORT_KEYS if k not in d]
    res.append(Result("R-report-complete", "§10 line 88",
                      "Report (for both the ML model and the K-S baseline): accuracy, top-3 accuracy, log loss, "
                      "\"related-key\" accuracy (exact or relative major/minor or perfect fifth), confusion matrix.",
                      "INFO", "complete" if not missing else f"missing: {missing}"))
    leak = split.get("leakage") or {}
    res.append(Result("R-no-leakage", "§10 line 87", "Train/validation/test split by track, no leakage.",
                      "INFO", "no track in more than one split" if not leak else f"{len(leak)} leaking tracks"))
    return res


def all_pass(results: list[Result]) -> bool:
    """True when every §9 criterion (non-INFO) passed."""
    return all(r.status == "PASS" for r in results if r.status != "INFO")


def format_results(results: list[Result]) -> str:
    """Human-readable PASS/FAIL table with the spec text for each criterion."""
    lines = []
    for r in results:
        lines.append(f"[{r.status}] {r.id} ({r.spec_ref})\n    spec: {r.spec_text}\n    got:  {r.detail}")
    n_fail = sum(r.status == "FAIL" for r in results)
    lines.append(f"\n{'ALL §9 ACCEPTANCE CRITERIA PASS' if n_fail == 0 else f'{n_fail} §9 CRITERION(S) FAILED'}")
    return "\n".join(lines)


def results_as_dicts(results: list[Result]) -> list[dict]:
    """JSON-serialisable form of the results."""
    return [asdict(r) for r in results]
