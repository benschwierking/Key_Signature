# keyclf acceptance harness

This harness is test-only. It checks the `keyclf` package against
`specs/key-signature-classifier-spec.md`, which is its **only** source of acceptance criteria.
It adds no thresholds, metrics or module names beyond what the spec states. Every
check cites a spec section and line, and every PASS/FAIL line printed by `eval/criteria.py`
quotes the spec text it enforces.

Until `keyclf/` exists, the implementation tests **skip** with a reason that starts
with `IMPLEMENTATION MISSING: cannot import keyclf.<module> ...`. Set `KEYCLF_REQUIRE_IMPL=1`
to turn those skips into **failures**. CI sets it.

The judge's synthetic smoke test uses an explicitly separate FFmpeg/NumPy FFT
stand-in to exercise timing and exit codes without first-time numba compilation.
Real acceptance measures `keyclf.features:extract_features` with the unchanged
strict `< 10.0 s` maximum and no warm-up. Synthetic fixture results never count
as evidence about the real classifier.

## Layout
```
tests/conftest.py              markers, REPO_ROOT, require_impl skip/fail policy, synthetic-tone helpers
tests/test_features.py         §7 + §9 line 81 feature unit tests on synthetic tones (marker: impl)
tests/test_train.py            §10 rubric: reproducibility, no leakage, report completeness, docstrings (impl)
tests/test_criteria.py         self-tests of the §9 criteria module (harness)
tests/test_harness_helpers.py  self-tests of key maths, K-S, metrics, dataset/split, profiles (harness)
tests/test_eval_smoke.py       end-to-end eval on SYNTHETIC data, PASS and FAIL paths (harness)
tests/test_workflow.py         acceptance.yml parses and wires every gate (harness)
tests/fixtures/synthetic_fixture.py   SYNTHETIC fake feature file + model (NOT GiantSteps)
eval/criteria.py               §9 criteria C1–C7 as code; the only place thresholds live
eval/evaluate.py               metrics, plots, metrics.json, summary.md, --criteria-only, exit code
eval/check_criteria.py         thin wrapper: python -m eval.check_criteria reports/metrics.json
eval/{keys,ks_baseline,metrics,dataset,profiles}.py   helpers
.github/workflows/acceptance.yml
requirements-harness.txt
```

## Criterion → enforcing check

| Spec | Criterion (quoted) | Enforced by | Pass rule |
|---|---|---|---|
| §9 line 77 (§3) | ≥ 80% exact-key accuracy on held-out test split | `criteria.py` **C1-exact-accuracy**, computed by `evaluate.py` on the `test` split | `exact_accuracy >= 0.80` (0.80 itself passes) |
| §9 line 78 | Correct key in top 3 at least 95% of the time | **C2-top3** | `top3_accuracy >= 0.95` |
| §9 line 79 (§16 task 11) | Upload-to-result in under 10 seconds | **C3-latency**: times `extract_features(file)` + `predict_proba`, no warm-up, `--latency-runs` runs | `max < 10.0 s` (strict; the mean is reported for §17) |
| §9 line 80 | Confusion matrix generated showing where mistakes land | **C4-confusion-matrix**: `reports/confusion_matrix_model.{png,csv}` (24×24, sums to n_test) | both files exist and are well-formed |
| §9 line 81 (§7, §16 task 3) | Features are chroma-based (first bin, last bin, 12-bin histogram) computed with librosa, verified by unit tests on synthetic tones | `tests/test_features.py` (31 tests) + **C5** reads the pytest JUnit XML | every `test_features` test ran and passed; **a skip counts as not verified (FAIL)** |
| §9 line 82 (§16 task 9) | Learned histogram weights per key, rotated to common tonic, plotted against K-S major/minor, correlation reported | **C6-learned-vs-ks-profiles**: `eval/profiles.py` (LR `coef_[:,24:36]`, `np.roll(w,-tonic)`, mean per mode) → `learned_vs_ks_profiles.png` + Pearson r | plot exists and both correlations are finite. **There is no threshold on r, because the spec gives none.** |
| §9 line 83 (§16 task 5) | A plain K-S template-matching baseline is scored on the same test split | **C7-ks-baseline-same-split**: `eval/ks_baseline.py` (Krumhansl-Kessler, Pearson) | K-S scored, same n and same sha256 of test track ids as the model |
| §10 line 86 | Reproducible from a fixed seed and a single train command | `test_train.py::test_single_train_command_is_reproducible` (runs `python -m keyclf.train` twice in sandboxes) | identical predictions and splits |
| §10 line 87 | Split by track, no leakage | `test_train.py::test_split_by_track_no_leakage`; INFO line **R-no-leakage** | no track id in more than one split; train/val/test non-empty |
| §10 line 88 | Report for model and K-S: accuracy, top-3, log loss, related-key accuracy, confusion matrix | `test_train.py::test_report_has_all_metrics_for_model_and_ks_baseline`; INFO **R-report-complete**; written to `metrics.json`/`summary.md` | all five present for both models |
| §10 line 89 | Code passes `pytest`, docstrings on public functions | CI pytest step (its result gates the build); `test_*::test_public_functions_have_docstrings` | pytest exit 0 |
| §14 | ffmpeg for MP3 decoding | `test_load_audio_mp3_via_ffmpeg`; the CI step installs ffmpeg | — |

The §9 gate is C1–C7. `evaluate.py` and `--criteria-only` exit with **1** if any of them fails,
with **2** on an input or configuration error, and with **0** only if all pass. R-* lines are informational in the
criteria output, but the §10 items are enforced as pytest tests, and CI fails when pytest fails.

## Run locally
```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements-harness.txt      # needs system ffmpeg (apt install ffmpeg / brew install ffmpeg)
pip install -e .                             # once keyclf has packaging (spec §13)

pytest -m harness                            # harness self-tests; these pass with no keyclf
pytest                                       # impl tests skip with "IMPLEMENTATION MISSING" until keyclf exists
KEYCLF_REQUIRE_IMPL=1 pytest --junitxml reports/pytest-junit.xml   # strict mode, as in CI

# full acceptance run, after `python -m keyclf.features` and `python -m keyclf.train`:
python -m eval.evaluate --junit reports/pytest-junit.xml          # writes reports/*, exit 0/1/2
python -m eval.evaluate --criteria-only reports/metrics.json --junit reports/pytest-junit.xml
```
Useful flags: `--features`, `--label-col/--track-col/--split-col`, `--split-file`,
`--latency-audio PATH` or `--latency-url URL` (one real track, deleted afterwards), `--latency-runs`,
`--skip-latency` (C3 then FAILs as "not measured"), `--profile-json/--profile-plot` (RF models), `--extractor mod:func`.

Synthetic demo, **clearly fake and not GiantSteps**:
`python -m tests.fixtures.synthetic_fixture /tmp/fake && python -m eval.evaluate --model /tmp/fake/models/model.joblib --features /tmp/fake/data/features/SYNTHETIC_features.csv --extractor tests.fixtures.synthetic_fixture:synthetic_extract_features --out /tmp/fake/reports`
This run exits **1**. C1–C4, C6 and C7 pass, but C5 FAILS because there is no JUnit evidence that the real `keyclf` feature tests passed, which is the intended behaviour.

## CI data policy (spec §6, §12)
CI never downloads the roughly 850MB audio set. The `acceptance` job uses the first of these that applies:
1. A feature cache committed at `data/features/*.parquet|*.csv`, which the spec says is "gitignored as needed".
2. Repository variable `KEYCLF_FEATURES_URL` (optionally `KEYCLF_FEATURES_FILENAME`), pointing at the small cached feature file.
3. Neither of these → `::error::` and the job fails. It never falls back to fetching audio.

For the initial public source release, the real-data `acceptance` job is skipped
on ordinary pushes unless `KEYCLF_FEATURES_URL` is configured or repository
variable `KEYCLF_FEATURES_COMMITTED=1` confirms that an authorized cache has
been committed. `workflow_dispatch` always attempts the strict job and fails if
its inputs are absent. The `harness-selftest` job still runs on every push; this
temporary source-only mode is recorded in `DATA_PROVENANCE.md` and must not be
represented as real-data acceptance.

The model is retrained with `python -m keyclf.train` (§13), because `models/` may be gitignored (§11).
A manual `workflow_dispatch` run with `retrain=false` uses a committed `models/model.joblib`. For latency,
the job synthesizes a labeled SYNTHETIC 120 s MP3, or fetches a single track if `vars.KEYCLF_LATENCY_URL` is set.
Artifacts uploaded: `reports/` (metrics.json, summary.md, confusion matrices PNG and CSV, profile plot, JUnit).

## Interface assumptions (from §7 and §11 names; anything else is flagged)
- `keyclf.features`: `load_audio(path) -> (y, sr)` (mono float); `compute_chroma(y, sr) -> (12, T)`;
  `first_last_bins(chroma) -> (first, last)`, each either a pitch-class int or a 12-dim one-hot (both accepted);
  `chroma_histogram(chroma) -> 12` summing to 1; `extract_features(path) -> 36`. Order follows the §7 list:
  `[first one-hot 12 | last one-hot 12 | histogram 12]`.
- `keyclf.train`: `python -m keyclf.train` run from the repo root writes `models/model.joblib` relative to the cwd.
  If a different CLI is needed, set `KEYCLF_TRAIN_CMD` (`{workdir}` is substituted).
- `model.joblib` is a scikit-learn estimator or pipeline with `predict_proba`, or a dict holding one under
  `model|estimator|pipeline|clf`, optionally with `splits` or `test_track_ids`.
- Feature file: one parquet or csv in `data/features/`, with 36 numeric feature columns plus label, track and split columns
  (auto-detected; can be overridden). Labels are key strings such as "Eb minor", "c:min" or "Bbm"; enharmonics fold to 24 classes; integer labels are taken as canonical 0–11 major, 12–23 minor.
- Split source, in order: split column → `--split-file` → model meta. The harness never invents a split.

## Ambiguities and how they were handled (no new criteria added)
- **Related key** (§10, "relative major/minor or perfect fifth"): a perfect fifth is ±7 semitones in the same mode (both directions), and the parallel key is excluded.
- **Latency**: "under 10 seconds" is gated on the **max** over runs (strict `<`), with the mean reported per §17. The probe is a 120 s clip (about the length of a GiantSteps preview). This is only a measurement setting.
- **K-S log loss**: K-S produces correlations, not probabilities, so it uses softmax(r/T) with T fit on **val** (train if there is no val), never on test.
- **C6 correlation**: the spec asks only that it is "reported", so C6 checks that the artifact exists and that r is finite. A negative r still passes. For RF models (§16 task 9) the implementation supplies `--profile-json/--profile-plot`.
- **C5** relies on the unit tests themselves, so a skipped feature test counts as a FAIL rather than a pass.
- **§10 report**: the spec doesn't name a metrics file for `train.py`, so completeness is checked on the harness's own evaluation of the trained model.
- `test_chroma_type_hop_and_edge_frames_configurable` is a naming heuristic (marker `interface_heuristic`). Deselect it with `-m "not interface_heuristic"` if the names differ.
- `app.py` is left out of the docstring test because importing it requires streamlit. The §8 demo is not in §9 and is not tested.
