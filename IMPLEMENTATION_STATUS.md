# Implementation status

This is the durable resume ledger for the key-signature classifier.

## Sources of truth

1. Product behavior: [`specs/key-signature-classifier-spec.md`](specs/key-signature-classifier-spec.md)
2. Acceptance contract: [`HARNESS_README.md`](HARNESS_README.md) and [`.github/workflows/acceptance.yml`](.github/workflows/acceptance.yml)
3. Progress only: this file

## Completion state

- Local v1 accepted: **no**
- Public release accepted: **no**
- GitHub target: `benschwierking/Key_Signature` (public)
- Data publication: **blocked pending documented redistribution basis**

## Milestones

| Milestone | State | Evidence |
|---|---|---|
| Canonical repository and packaging | complete | Spec moved; pyproject, README, MIT license, provenance added |
| Harness self-tests | complete | 81 passed, 1 optional skip on Python 3.11 |
| Feature interfaces and synthetic tests | complete | 31/31 required feature tests passed |
| Resumable GiantSteps pipeline | complete | 604/604 CQT tracks, 0 failures, 135 resumed, no audio retained |
| Deterministic model training | complete | 125 tests passed; two isolated train runs reproduced exactly |
| Interpretive analysis and notebook | implemented | LR and RF paths plus exploration notebook added |
| Streamlit application | complete | AppTest imports/runs; regular Streamlit selected after Stlite dependency review |
| Strict C1–C7 acceptance | blocked by measured accuracy | C1 46.15% and C2 69.23% fail; C3–C7 pass |
| Public GitHub source release | pending | Local `gh` token requires reauthentication |

## Current checkpoint

- Date: 2026-10-29 America/Los_Angeles
- Last successful action: bounded STFT alternative rejected on validation (58.24% exact vs CQT 63.74%); its test split was not inspected
- Environment: Python 3.11 virtualenv installed; ffmpeg available
- Committed source baseline: `12e4b28` (`Build resumable key-signature classifier v1`)
- Blockers:
  - local v1: honest held-out accuracy is below C1 and C2; the bounded failure policy is exhausted
  - public source release: GitHub CLI token is invalid and needs interactive reauthentication
  - public data release: feature-cache redistribution rights are not documented
- Generated local artifacts (all intentionally ignored):
  - `data/features/giantsteps_features.parquet` (604 CQT rows)
  - `data/work/` (atomic per-track checkpoints)
  - `models/model.joblib`
  - `reports/pytest-junit.xml`, `reports/metrics.json`, confusion matrices, profile artifacts, and diagnostics

## Measured results

- Fixed split, seed 42: 422 train / 91 validation / 91 held-out test; no track leakage.
- Selected on validation only: 500-tree random forest, `min_samples_leaf=1`,
  `max_features=0.5`, `class_weight=balanced_subsample`.
- CQT validation: exact 0.637363, top-three 0.714286, log loss 2.535430.
- Held-out test: exact 0.461538, top-three 0.692308, 91 tracks.
- Krumhansl-Schmuckler baseline on the same test split: exact 0.406593.
- Cold-path maximum latency: 1.505 seconds (three synthetic 120-second MP3 runs).
- Learned/profile correlation: major 0.807370, minor 0.672247.
- Bounded STFT alternative: validation exact 0.582418 and top-three 0.714286;
  rejected without evaluating its test partition.
- Gate: C3, C4, C5, C6, and C7 pass; C1 and C2 fail. No threshold,
  label, row, or split was changed after observing test performance.

## Operation log

| Operation | Result |
|---|---|
| `.venv/bin/python -m pytest -q` | harness self-test: 81 passed, 1 optional criterion skip |
| `KEYCLF_REQUIRE_IMPL=1 .venv/bin/python -m pytest -q --junitxml reports/pytest-junit.xml` | complete run: 125 passed, 1 optional skip; isolated training predictions and split IDs matched |
| `.venv/bin/python -m keyclf.features --workers 2` | official cache: 604/604 extracted, 0 failed, 135 compatible checkpoints resumed, no audio retained |
| `.venv/bin/python -m keyclf.train` | deterministic CQT selection and train+validation refit completed |
| `.venv/bin/python -m eval.evaluate --junit reports/pytest-junit.xml` | C1/C2 failed honestly; C3-C7 passed |
| `.venv/bin/python -m eval.evaluate --criteria-only reports/metrics.json --junit reports/pytest-junit.xml` | reproduced the same C1/C2 gate failure |
| `.venv/bin/python -m keyclf.diagnostics --alternate-training /tmp/keyclf-stft-training.json` | per-class report written; STFT rejected on validation |
| strict pytest rerun after profile vectorization | 117 passed and 1 skipped before interruption at 14m32s; a fresh collector then blocked in stdlib `binascii` import while macOS `syspolicyd` saturated the host. An evaluator replay likewise blocked reading NumPy during import. Manual training and targeted changed-code tests had already passed; these are recorded as host contention, not passing reruns. |

## Resume invariant

Run the exact next action below. The feature extractor will reuse compatible
checkpoints and never needs raw audio to remain on disk. Do not publish
`data/`, `models/`, or `reports/` unless `DATA_PROVENANCE.md` first records a
valid redistribution basis. The commit containing this ledger is the current
handoff commit (`git rev-parse HEAD`).

## Exact next action

```bash
cd keyclf-harness && gh auth login -h github.com --web
```
