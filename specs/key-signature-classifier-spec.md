# Spec: Song Key-Signature Classifier

## 1. Summary
A personal learning project that classifies a song's key signature (24 classes: 12 major + 12 minor) from audio. Features are expressed as chroma computed with librosa: the first chroma bin, the last chroma bin, and a 12-bin pitch-class chroma histogram across the whole track (all octaves folded together).

**Framing:** this project is essentially re-deriving the Krumhansl-Schmuckler (K-S) key-finding algorithm with machine learning. K-S correlates a pitch-class distribution against fixed, psychologically derived major/minor templates; here the model learns those weights from data. Comparing the learned weights with the classic K-S profiles is a core interpretive analysis, not just an accuracy benchmark. A scikit-learn model is trained offline and served through a Streamlit Lite demo app.

## 2. Target customer
Ben, as a learning project and demo tool. No external users.

## 3. Success metrics
- 80% exact-key accuracy on a held-out test split.
- Show the next-closest matches, treating relative major/minor (e.g. C major and A minor) as related.

## 4. Tiers, monetization, auth, sales wedge
Not applicable (personal project). No accounts, no payments.

## 5. Tech stack
- Python 3.11+
- `librosa` for audio loading, pitch/chroma extraction, frequency-to-note conversion
- `scikit-learn` for modeling
- `joblib` for model export
- `music21` for label sanity checks (key-profile cross-check)
- `matplotlib` or `altair` for histograms and confusion matrix
- `streamlit` (Streamlit Lite / stlite for the browser demo)
- `pytest` for tests

### Model choice (OPEN DECISION — resolve during implementation)
- **Default:** multinomial `LogisticRegression` (trained on cross-entropy loss, honoring the original requirement; gives calibrated probabilities for top-3).
- **Optional baseline:** `RandomForestClassifier`, evaluated with cross-entropy (log loss) as a metric.
- Pick the final model after comparing both on the validation split.

### Architecture constraint
Streamlit Lite runs entirely in the browser and cannot do training or heavy audio processing well. Therefore:
- Training and dataset feature extraction run **offline** in Python.
- The exported model (`model.joblib`) plus a lightweight feature-extraction path is loaded by the app.
- If librosa cannot run acceptably in Streamlit Lite, fall back to regular Streamlit (server-side Python) for inference. Document whichever path is used.

## 6. Data
- **Primary:** GiantSteps Key dataset (public, annotated for key estimation, electronic dance music only).
- **No bulk audio download.** The full audio set is roughly 850MB; do not store it locally. Instead:
  - Clone only the annotations/metadata from the [GiantSteps Key repo](https://github.com/GiantSteps/giantsteps-key-dataset) (about 400KB).
  - Pair each key label with the track's public streaming URL on the JKU backup mirror: `https://www.cp.jku.at/datasets/giantsteps/backup/<filename>.mp3` (e.g. `10089.LOFI.mp3`; filenames come from the repo's `md5/` folder). This is the primary host in the repo's own `audio_dl.sh`. Beatport's sample server (`https://geo-samples.beatport.com/lofi/<filename>.mp3`) is only a fallback and was returning 404 for every track spot-checked on 2026-10-29, so do not rely on it.
  - Stream each track into memory, compute chroma, keep only the 36-dim feature vector, and discard the audio. Note: `librosa.load` does not accept URLs directly, so fetch the bytes (e.g. `requests` into `io.BytesIO`, or a temp file deleted immediately) and decode from there.
  - Target local footprint: under 1MB of annotations plus a small cached feature file.
  - Handle dead or slow URLs: retry with backoff, log failures, and report how many tracks could not be fetched.
- **Optional conveniences:**
  - `mirdata`'s GiantSteps Key loader for pulling individual tracks or subsets with annotations attached.
  - The Hugging Face version (`m-a-p/GS`), sliced into 30-second clips, as an even lighter option (note that 30-second clips change what "first" and "last" chroma bin mean; document it if used).
- **Possible later add-on:** Million Song Dataset. It ships precomputed features/metadata, not full audio, so it does not fit the chroma pipeline directly.
- **Label noise risk:** key labels can be wrong; songs modulate or borrow chords. Mitigation: score each track with the K-S template baseline (or music21's key analysis) and flag tracks where it strongly disagrees with the label. Report how many are flagged; optionally train with and without them.

## 7. Features (chroma-based, `features.py`)
Replaces raw note-level histograms. All features come from `librosa.feature.chroma_cqt` (or `chroma_stft`), which folds every octave into 12 pitch-class bins (C, C#, ..., B).
1. **First chroma bin** - dominant pitch class (argmax) over the first confidently non-silent frames; one-hot, 12 dims.
2. **Last chroma bin** - dominant pitch class over the last confidently non-silent frames; one-hot, 12 dims.
3. **Pitch-class chroma histogram** - chroma energy summed across all frames of the track, normalized to sum to 1; 12 dims.

Pipeline: load audio (ffmpeg via librosa) -> trim silence -> compute chroma -> derive first/last bins and the track-wide histogram -> cache features -> visualize distributions.

Module responsibilities:
- `load_audio(path)`, `compute_chroma(y, sr)`, `first_last_bins(chroma)`, `chroma_histogram(chroma)`, `extract_features(path) -> 36-dim vector`.
- Configurable chroma type, hop length, and how many edge frames count as "first"/"last".

## 8. Demo app (Streamlit)
Inputs:
- **File upload** (MP3/MP4/WAV) — core feature.
- **Microphone input** — STRETCH GOAL. Browser capture may work, but librosa on live audio in-browser is likely too slow/unsupported; may require regular Streamlit with a Python backend.

Outputs after a song is processed:
- Note (pitch-class) histogram chart.
- Detected first and last chroma bins (pitch classes).
- Predicted key with confidence score.
- Top-3 predicted keys, with relative major/minor highlighted.

## 9. Acceptance criteria (v1 done when all true)
- [ ] ≥ 80% exact-key accuracy on held-out test split.
- [ ] Correct key in top 3 at least 95% of the time.
- [ ] Upload-to-result in under 10 seconds.
- [ ] Confusion matrix generated showing where mistakes land.
- [ ] Features are chroma-based (first bin, last bin, 12-bin track histogram) computed with librosa, verified by unit tests on synthetic tones.
- [ ] Learned feature weights for the histogram are extracted per key, rotated to a common tonic, and plotted against the K-S major and minor profiles, with the correlation reported.
- [ ] A plain K-S template-matching baseline is scored on the same test split for comparison.

## 10. Quality rubric
- Results reproducible from a fixed random seed and a single train command.
- Train/validation/test split by track, no leakage.
- Report (for both the ML model and the K-S baseline): accuracy, top-3 accuracy, log loss, "related-key" accuracy (exact or relative major/minor or perfect fifth), confusion matrix.
- Code passes `pytest` and is readable with docstrings on public functions.

## 11. Design rules (code organization)
Small Python package, not notebook-only:
```
keyclf/
  data.py        # clone annotations, build label->URL index, stream audio into memory, label checks
  features.py    # audio -> chroma -> first/last chroma bin + 12-bin chroma histogram
  train.py       # train/eval models, save model.joblib + metrics
  app.py         # Streamlit demo
tests/
  test_features.py
  test_train.py
notebooks/
  explore_histograms.ipynb
models/          # exported model.joblib (gitignored if large)
data/            # raw + processed (gitignored)
```

## 12. Storage
- No raw audio stored. Only cloned annotations (~400KB) under `data/annotations/` and a cached feature file (Parquet/CSV) under `data/features/`, both gitignored as needed.
- Feature cache means each track is streamed once; training never re-fetches audio.
- Exported model in `models/model.joblib`.

## 13. Onboarding
README with: install (`pip install -e .`), clone annotations, `python -m keyclf.features`, `python -m keyclf.train`, `streamlit run keyclf/app.py`.

## 14. Dependencies
librosa, requests, scikit-learn, joblib, optional mirdata and datasets (Hugging Face), music21, numpy, pandas, matplotlib/altair, streamlit (or stlite), pytest. System: ffmpeg for MP3/MP4 decoding.

## 15. Non-goals
- **Chord detection** (explicit non-goal).
- Everything else (mid-song key changes, tempo detection, out-of-dataset accuracy guarantees) is out of scope by default and may be noisy; not actively rejected.

## 16. Task list
1. Set up repo skeleton, packaging, and test runner.
2. Clone the GiantSteps Key annotations only; write `data.py` to parse labels and build a label-to-streaming-URL index (`https://www.cp.jku.at/datasets/giantsteps/backup/<filename>.mp3`, with Beatport only as a fallback). Spot-check that URLs are live and confirm license/usage terms.
3. Implement `features.py` with librosa chroma: first chroma bin, last chroma bin, 12-bin track histogram. Unit-test with synthetic tones and scales.
4. Build exploration notebook showing average chroma histograms per key.
5. Implement a K-S template-matching baseline (classic major/minor profiles); use it both as an accuracy baseline and as the music21-style label cross-check to flag suspicious labels.
6. Stream every track into memory, compute chroma, cache only the feature vectors, and log any tracks that fail to fetch. Optionally try `mirdata` or Hugging Face `m-a-p/GS` for subsets.
7. Implement `train.py`: split, train logistic regression (default) and random forest (baseline), compute metrics, save model and confusion matrix.
8. Pick final model (resolve open decision) and record why.
9. Interpretive analysis: pull the histogram weights from logistic regression (or feature importances from the random forest), rotate each key's weights to a common tonic, average into learned major and minor profiles, plot against the K-S profiles, and report correlation.
10. Build `app.py`: upload, chroma histogram, first/last bins, prediction + confidence, top-3 with relative keys highlighted.
11. Measure upload-to-result latency; confirm under 10 s.
12. Try Streamlit Lite deployment; fall back to regular Streamlit if needed.
13. Stretch: microphone input.
14. Write README and verify all acceptance criteria.

## 17. Stats to track
Dataset size, tracks that failed to stream, flagged-label count, per-key sample counts, accuracy, top-3 accuracy, related-key accuracy, log loss, mean inference latency, K-S baseline accuracy, and correlation between learned and K-S profiles.
