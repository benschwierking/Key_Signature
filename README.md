# Key Signature Classifier

A learning-focused classifier for the 24 major/minor keys. It extracts three
chroma-derived components from a track—first pitch class, last pitch class, and
a track-wide 12-bin histogram—then compares a learned model with the classic
Krumhansl–Schmuckler profile-matching baseline.

The product requirements live in
[`specs/key-signature-classifier-spec.md`](specs/key-signature-classifier-spec.md).
The executable acceptance contract is documented in
[`HARNESS_README.md`](HARNESS_README.md), and resumable implementation state is
recorded in [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md).

## Install

Python 3.11 and ffmpeg are required.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -m harness
```

## Build the feature cache

The command sparse-clones annotations only, streams previews one at a time,
deletes every temporary audio file, and resumes from per-track checkpoints:

```bash
python -m keyclf.features
```

Useful development flags include `--limit 24`, `--workers 2`, `--chroma-type
stft`, and `--rebuild`. The only canonical cache is
`data/features/giantsteps_features.parquet`.

## Train and evaluate

```bash
python -m keyclf.train
KEYCLF_REQUIRE_IMPL=1 pytest --junitxml reports/pytest-junit.xml
python -m eval.evaluate --junit reports/pytest-junit.xml
python -m eval.evaluate --criteria-only reports/metrics.json \
  --junit reports/pytest-junit.xml
```

Training uses the fixed split stored in the feature cache, performs model
selection on validation data, refits on train plus validation, and never uses
the test split for selection.

## Run the app

```bash
streamlit run keyclf/app.py
```

Upload an MP3, MP4, or WAV to see the chroma histogram, first and last pitch
classes, predicted key, confidence, and related-key-aware top three.

The v1 deployment target is regular Streamlit. The browser-only Stlite attempt
and dependency limitation are recorded in [`DEPLOYMENT.md`](DEPLOYMENT.md).

## Accuracy research

The opt-in research path preserves the v1 artifact and original test IDs. It
adds harmonic/bass pitch summaries, beat accents, pitch persistence, and
section-level distributions. Persistence is a duration proxy, and beat accents
do not identify bar downbeats or transcribe individual notes.

```bash
python -m keyclf.research_features --workers 2
python -m keyclf.experiments --features data/work/research_features.parquet \
  --out reports/research-v2
```

The experiment runner uses three mode-stratified development folds and applies
12 transpositions only inside each training fold. It ranks configurations by
exact accuracy, top-three accuracy, and log loss. Add `--finalize --projections`
only after fixing the experiment matrix to evaluate the selected model once
on the original test partition and export PCA, t-SNE, and exploratory K-means
coordinates. It records exact recall by key, macro recall, weighted recall,
and major/minor mode performance. The final test receipt prevents silently
evaluating a different selection in the same output directory.

Copy the resulting `research_model.joblib` into `models/` to enable the
research classifier in the upload app. The app keeps the original histogram
for display and selects the model's feature columns in the recorded order.

`scripts/prepare_technical_report.py` prepares the measured learning curve and
presentation data. `scripts/build_technical_deck.mjs` creates an editable
PowerPoint using the Codex presentation runtime. Generated caches, models,
point coordinates, and slides remain under ignored work/report directories.

## Data policy

No raw audio is stored. Dataset and derived artifact rights are separate from
the MIT-licensed code; see [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md).
