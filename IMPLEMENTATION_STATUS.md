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
| Public GitHub source release | complete | Public `main` pushed to `benschwierking/Key_Signature`; source-only boundary verified |
| Accuracy research requested 2026-10-30 | complete, targets not met | 30 configurations, three development folds, one selected-model fixed-test evaluation; exact accuracy improved to 51.65% |
| Technical PowerPoint requested 2026-10-30 | complete | 19 slides; editable charts/tables; all previews inspected and package/import/layout validation passed |

## Current checkpoint

- Date: 2026-10-30 America/Los_Angeles
- Active operation: final source handoff; requested research batch and PowerPoint complete
- Last successful action: exported and validated `reports/research-v2/presentation/Key_Signature_Technical_Review.pptx` (19 slides, 8 native charts, 8 native tables); visually inspected all slides, including final revised previews
- Runtime recovery: Desktop cache and virtualenv contain macOS `dataless` cloud placeholders. A new local runtime under `/private/tmp/keyclf-research.1I9aUa` avoids stalled imports. The earlier attribution solely to policy-service contention was not established.
- Research checkpoint: 604/604 new feature rows complete, zero failures, two checkpoint resumes, no audio retained. SHA256 `a11c602fab1edb3f7aeff716d518b6db657a2574b2aef2bb4284a2edf45a6f15`.
- Research selection complete: histogram-only random forest, 500 trees, leaf size 2, sqrt feature subset, balanced_subsample and 12 transpositions inside training folds. Development CV exact accuracy 55.36%.
- Research fixed test: exact 51.65% (47/91), top-three 71.43% (65/91), macro recall over represented keys 41.46%, weighted recall 51.65%, major exact recall 33.33% (5/15), minor exact recall 55.26% (42/76), log loss 1.8601. These remain below the original 80%/95% targets.
- Research training resubstitution: 98.83% exact and 99.36% macro recall on 513 original development tracks after fitting 6,156 rotated training rows. This is training fit, not a generalization estimate; the training/test gap still indicates overfitting.
- All original IDs/labels/splits match; the 36 original feature columns reproduced exactly (maximum difference 0.0). New harmonic/beat/persistence/temporal groups did not beat the augmented histogram on development selection.
- Verification: latest complete strict run passed **136 tests with one optional criterion skip** in 19.58 seconds. JUnit: `reports/research-v2/baseline_acceptance/pytest-junit.xml`. Original gate replay reproduced 46.15% exact and 69.23% top-three with C3-C7 passing.
- Development CV: original reference forest 46.98% exact; transposition-augmented histogram forest 55.36%. The selected configuration was fixed before research test inference. The 91-track benchmark was already examined during v1, so a fresh blind test is needed for a stronger generalization claim.
- Reproducibility: a second fixed-configuration refit retained byte-equivalent split IDs and matched all 604 probability vectors within 1.12e-16 (absolute tolerance 1e-12), without reselection or recomputing test metrics. Evidence: `reports/research-v2/reproducibility.json`.
- Research inference: selected histogram extraction/model path, three synthetic 120-second MP3 runs with no warm-up; maximum 1.145 seconds. Model loading excluded, matching the harness timing boundary. The complete 180-feature extraction path was not timed.
- Cluster diagnostics: development-fitted PCA explains 36.90% in two dimensions; exploratory 24-cluster K-means silhouette 0.128, adjusted Rand index 0.156. The classifier is supervised random forest, not K-means. Test points are transformed only after PCA fitting; t-SNE uses development tracks only.
- Additional-data metadata: 1,349 single-key candidates (429 major / 920 minor), no exact ID or MD5 overlaps, 137 ambiguous/unknown entries recorded separately. No additional audio collected.
- Source milestones: `c6e6d62` added the research pipeline; `d7fb278` fixed cold synthetic self-tests. The harness stand-in uses FFT/FFmpeg to avoid timing first-time numba compilation; real C3 still times the actual extractor with the unchanged strict limit and no warm-up.
- Upload verification: synthetic WAV, MP3, MP4 decoding and Streamlit top-three/relative-key rendering passed all four smoke tests. The histogram-only research app fast path has its own regression test.
- GitHub harness self-tests passed at source commit `d7fb278` (run `41124294169`). Full hosted acceptance remains intentionally pending an authorized cache. The commit containing this ledger is the handoff commit (`git rev-parse HEAD`).
- Environment: Python 3.11 virtualenv installed; ffmpeg available
- Committed source milestones: `12e4b28` (v1 build) and `6291ad3` (acceptance diagnostics and handoff)
- Blockers:
  - local v1: honest held-out accuracy is below C1 and C2; the bounded failure policy is exhausted
  - public release acceptance: feature-cache redistribution rights are not documented, and the measured C1/C2 shortfall remains
- Generated local artifacts (all intentionally ignored):
  - `data/features/giantsteps_features.parquet` (604 CQT rows)
  - `data/work/` (atomic per-track checkpoints)
  - `models/model.joblib`
  - `reports/pytest-junit.xml`, `reports/metrics.json`, confusion matrices, profile artifacts, and diagnostics
  - `data/work/research_features.parquet`, `data/work/research-v2/` (604 research checkpoints)
  - `models/research_model.joblib` (available in the upload app separately from v1)
  - `reports/research-v2/` (candidate results, final test receipt, recall, projections, learning curve, latency, reproducibility, additional-data metadata)
  - `reports/research-v2/presentation/Key_Signature_Technical_Review.pptx` (SHA256 `6235a08349d02b6a78597b03d3313ebce335ef91ce4fa6ae7b43ddc0cd903c19`)
  - `reports/research-v2/.deck-build/validation.json`, rendered previews and private draft

## Original v1 measured results

- Fixed split, seed 42: 422 train / 91 validation / 91 held-out test; no track leakage.
- Selected on validation only: 500-tree random forest, `min_samples_leaf=1`,
  `max_features=0.5`, `class_weight=balanced_subsample`.
- CQT validation: exact 0.637363, top-three 0.714286, log loss 2.535430.
- Held-out test: exact 0.461538, top-three 0.692308, 91 tracks.
- Krumhansl-Schmuckler baseline on the same test split: exact 0.406593.
- Latest cold-path maximum latency: 0.868 seconds (three synthetic 120-second MP3 runs; historical first run was 1.505 seconds).
- Latest learned/profile correlation: major 0.843700, minor 0.701450 (fresh Python 3.11/scikit-learn environment; source definitions unchanged).
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
| `gh repo create benschwierking/Key_Signature --public --source=. --remote=origin ...` | public repository created after reauthenticating `benschwierking` |
| `git push --progress -u origin main` | 61 objects / 78.33 KiB pushed; remote `main` established and verified public |
| `python -m keyclf.research_features --workers 2` | 604/604 research tracks, 0 failures, 2 resumed; original 36 features match exactly; no audio retained |
| `python -m keyclf.experiments --features /private/tmp/keyclf-research.1I9aUa/research_features.parquet --reference /private/tmp/keyclf-research.1I9aUa/original_features.parquet --out /private/tmp/keyclf-research.1I9aUa/final_experiments --finalize --projections` | 30 configurations / 90 development fold fits; histogram RF selected; fixed test 47/91 exact, 65/91 top-three |
| `KEYCLF_REQUIRE_IMPL=1 pytest --junitxml reports/pytest-junit.xml` in the isolated acceptance sandbox | 136 passed, 1 optional skip, 0 failures/errors; latest JUnit copied to research report directory |
| `python scripts/verify_research_model.py --features ... --model ... --out reports/research-v2/reproducibility.json` | second refit passed; same split IDs; maximum probability difference 1.11e-16 |
| `python scripts/measure_research_latency.py --model ... --out .../research_latency.json` | actual selected-model path, no warm-up, maximum 1.145 seconds; temporary synthetic audio deleted |
| `scripts/audit_additional_data.py` | 1,349 metadata candidates, including 429 major; 137 ambiguous/unknown entries kept separate; no extra audio collected or used for training |
| `scripts/prepare_technical_report.py` and `scripts/build_technical_deck.mjs` | measured learning curve and 19-slide editable PowerPoint exported; first-party re-import, native chart validation, package and layout checks passed |

## Resume invariant

The requested research batch is complete. The next command opens a local demo,
not another test-driven tuning cycle. Further accuracy work should first verify
independent major-key examples and define a new blind benchmark; this batch did
not train on the additional metadata candidates. The feature extractor reuses
compatible checkpoints and never needs raw audio to remain on disk. The working
temporary Python runtime avoids Desktop cloud placeholders and may need to be
recreated after operating-system temporary-directory cleanup. Do not publish
`data/`, `models/`, or `reports/` unless `DATA_PROVENANCE.md` first records a
valid redistribution basis. The commit containing this ledger is the current
handoff commit (`git rev-parse HEAD`).

## Exact next action

```bash
cd "/Users/ben/Desktop/ML and Big Data/Key_Signature/keyclf-harness" && /private/tmp/keyclf-research.1I9aUa/.venv/bin/streamlit run keyclf/app.py
```
