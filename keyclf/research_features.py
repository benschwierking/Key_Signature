"""Versioned harmonic, bass, persistence, and beat-accent research features.

The v1 36-number interface stays in features.py. Every research block has
twelve pitch classes so transposition can rotate all blocks consistently.
Beat accents are estimated from onsets, not claimed to identify bar downbeats.
Sustain is a persistence proxy, not a polyphonic note-duration transcription.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy.ndimage import minimum_filter1d

from .data import annotation_revision, build_track_index, download_preview, sparse_clone_annotations
from .features import compute_chroma, first_last_bins, load_audio
from .keys import key_name

VERSION = "harmonic-beat-v2.1"
BLOCKS = (
    "first", "last", "hist", "harmonic", "bass", "beat", "accent", "sustain",
    "harmonic_first", "harmonic_last", "quarter1", "quarter2", "quarter3", "quarter4", "variation",
)


def normalize(values: np.ndarray) -> np.ndarray:
    """Return a finite nonnegative pitch distribution, including silent input."""
    vector = np.maximum(np.nan_to_num(np.asarray(values, dtype=float)), 0)
    return vector / vector.sum() if vector.sum() > 0 else np.ones(12) / 12


def aggregate_research_blocks(
    chroma: np.ndarray, harmonic: np.ndarray, bass: np.ndarray,
    beat_frames: np.ndarray, onset_strength: np.ndarray,
) -> dict[str, np.ndarray]:
    """Aggregate pitch-time matrices without labels or estimated key inputs."""
    first, last = first_last_bins(chroma)
    h = np.asarray(harmonic, dtype=float)
    boundaries = np.unique(np.clip(np.r_[0, beat_frames, h.shape[1]], 0, h.shape[1])).astype(int)
    beat_vectors, accents = [], []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if stop > start:
            beat_vectors.append(normalize(h[:, start:stop].mean(axis=1)))
            values = onset_strength[start:min(stop, len(onset_strength))]
            accents.append(float(values.max()) if len(values) else 0.0)
    beats = np.asarray(beat_vectors or [normalize(h.sum(axis=1))])
    weights = np.asarray(accents or [1.0])
    weights = 1 + weights / max(float(weights.max()), 1e-8)
    blocks = {
        "first": first, "last": last, "hist": normalize(chroma.sum(axis=1)),
        "harmonic": normalize(h.sum(axis=1)), "bass": normalize(bass.sum(axis=1)),
        "beat": normalize(beats.mean(axis=0)),
        "accent": normalize(np.average(beats, axis=0, weights=weights)),
        "sustain": normalize(minimum_filter1d(h, size=5, axis=1, mode="nearest").sum(axis=1)),
        "harmonic_first": normalize(h[:, :20].sum(axis=1)),
        "harmonic_last": normalize(h[:, -20:].sum(axis=1)),
        "variation": normalize(h.std(axis=1)),
    }
    for i, part in enumerate(np.array_split(h, 4, axis=1), 1):
        blocks[f"quarter{i}"] = normalize(part.sum(axis=1))
    return {name: blocks[name] for name in BLOCKS}


def extract_research_features(path: str | Path) -> tuple[np.ndarray, dict]:
    """Extract 180 pitch-class features plus diagnostic beat metadata."""
    y, sr = load_audio(path)
    y, _ = librosa.effects.trim(y, top_db=60)
    if not y.size:
        y = np.zeros(4096)
    chroma = compute_chroma(y, sr)
    harmonic_audio, percussive = librosa.effects.hpss(y)
    cqt = np.abs(librosa.cqt(harmonic_audio, sr=sr, hop_length=512, fmin=librosa.note_to_hz("C1"), n_bins=84, tuning=0.0))
    harmonic = cqt.reshape(7, 12, -1).sum(axis=0)
    harmonic /= np.maximum(harmonic.max(axis=0, keepdims=True), 1e-10)
    bass = cqt[:36].reshape(3, 12, -1).sum(axis=0)
    bass /= np.maximum(bass.max(axis=0, keepdims=True), 1e-10)
    onset = librosa.onset.onset_strength(y=percussive, sr=sr, hop_length=512)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=512, trim=False)
    blocks = aggregate_research_blocks(chroma, harmonic, bass, beats, onset)
    return np.concatenate(list(blocks.values())), {
        "tempo_bpm": float(np.asarray(tempo).reshape(-1)[0]), "beat_count": len(beats),
        "duration_s": len(y) / sr, "bar_downbeats_estimated": False,
        "note_events_transcribed": False,
    }


def build_research_cache(annotations: Path, work: Path, output: Path, workers: int = 2, limit: int | None = None) -> dict:
    """Stream and checkpoint each track while retaining the original split IDs."""
    tracks = build_track_index(sparse_clone_annotations(annotations))
    if limit:
        tracks = tracks[:limit]
    work.mkdir(parents=True, exist_ok=True)

    def process(track):
        target = work / f"{track.track_id}.json"
        if target.exists():
            prior = json.loads(target.read_text())
            if (prior.get("version") == VERSION and prior.get("split") == track.split
                    and prior.get("key") == key_name(track.label) and prior.get("md5") == track.md5):
                return prior, True
        with tempfile.TemporaryDirectory(prefix="keyclf-research-audio-") as temporary:
            audio = Path(temporary) / track.filename
            source = download_preview(track, audio)
            features, diagnostics = extract_research_features(audio)
        payload = {"track_id": track.track_id, "key": key_name(track.label), "split": track.split,
                   "version": VERSION, "features": features.tolist(), "diagnostics": diagnostics,
                   "source_url": source, "md5": track.md5}
        temp = target.with_suffix(".json.tmp")
        temp.write_text(json.dumps(payload) + "\n")
        os.replace(temp, target)
        return payload, False

    completed, failures, resumed = [], [], 0
    def safe_process(track):
        try:
            return process(track), None
        except Exception as error:
            return None, {"track_id": track.track_id, "error": repr(error)}

    workers = max(1, workers)
    pool = ThreadPoolExecutor(max_workers=workers)
    iterator = iter(tracks)
    pending = set()
    interrupted = False

    def submit_next():
        try:
            pending.add(pool.submit(safe_process, next(iterator)))
        except StopIteration:
            pass

    try:
        for _ in range(2 * workers):
            submit_next()
        processed = 0
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.remove(future)
                result, failure = future.result()
                processed += 1
                if failure:
                    failures.append(failure)
                else:
                    payload, reused = result
                    completed.append(payload)
                    resumed += reused
                if processed % 10 == 0 or processed == len(tracks):
                    print(json.dumps({"processed": processed, "requested": len(tracks), "failed": len(failures), "resumed": resumed}), flush=True)
                submit_next()
    except KeyboardInterrupt:
        interrupted = True
        for future in pending:
            future.cancel()
        raise
    finally:
        pool.shutdown(wait=not interrupted, cancel_futures=True)
    rows = []
    for payload in sorted(completed, key=lambda p: p["track_id"]):
        row = {k: payload[k] for k in ("track_id", "key", "split")}
        row.update(zip((f"{block}_{pc}" for block in BLOCKS for pc in range(12)), payload["features"]))
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".parquet.tmp")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    os.replace(temporary, output)
    summary = {"version": VERSION, "annotation_revision": annotation_revision(annotations),
               "completed": len(rows), "requested": len(tracks), "failed": len(failures),
               "failures": failures, "resumed": resumed, "output": str(output), "raw_audio_retained": False,
               "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    (work / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run the opt-in research feature extraction outside the canonical v1 cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/giantsteps-key-dataset"))
    parser.add_argument("--work", type=Path, default=Path("data/work/research-v2"))
    parser.add_argument("--output", type=Path, default=Path("data/work/research_features.parquet"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    result = build_research_cache(args.annotations, args.work, args.output, args.workers, args.limit)
    print(json.dumps(result, indent=2))
    return int(result["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
