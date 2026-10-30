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

## Data policy

No raw audio is stored. Dataset and derived artifact rights are separate from
the MIT-licensed code; see [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md).
