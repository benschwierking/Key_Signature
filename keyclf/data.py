"""GiantSteps metadata, streaming, checkpoint, and feature-cache pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .baseline import ks_top_keys
from .features import FeatureConfig, config_signature, extract_features, feature_names
from .keys import key_name, parse_key

ANNOTATIONS_REPO = "https://github.com/GiantSteps/giantsteps-key-dataset.git"
PRIMARY_AUDIO_BASE = "https://www.cp.jku.at/datasets/giantsteps/backup/"
FALLBACK_AUDIO_BASE = "https://geo-samples.beatport.com/lofi/"
RANDOM_SEED = 42


@dataclass(frozen=True)
class Track:
    """One annotated GiantSteps track and its immutable split assignment."""

    track_id: str
    filename: str
    label: int
    split: str
    md5: str | None = None


def sparse_clone_annotations(destination: str | Path) -> Path:
    """Sparse-clone only GiantSteps annotations, hashes, and source metadata."""
    dest = Path(destination)
    if (dest / "annotations" / "key").is_dir() and (dest / "md5").is_dir():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and any(dest.iterdir()):
        raise RuntimeError(f"annotation destination exists but is incomplete: {dest}")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", ANNOTATIONS_REPO, str(dest)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(dest), "sparse-checkout", "set", "--no-cone",
         "annotations/key", "md5", "README", "audio_dl.sh"],
        check=True,
    )
    return dest


def annotation_revision(annotations_dir: str | Path) -> str:
    """Return the checked-out annotation commit, or ``unknown`` when unavailable."""
    proc = subprocess.run(
        ["git", "-C", str(annotations_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def assign_stratified_splits(
    ids_and_labels: list[tuple[str, int]],
    seed: int = RANDOM_SEED,
) -> dict[str, str]:
    """Assign deterministic approximately 70/15/15 splits within every class."""
    grouped: dict[int, list[str]] = {}
    for track_id, label in ids_and_labels:
        grouped.setdefault(int(label), []).append(str(track_id))
    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    for label in sorted(grouped):
        members = np.array(sorted(grouped[label]), dtype=object)
        rng.shuffle(members)
        n = len(members)
        n_test = max(1, int(round(n * 0.15))) if n >= 3 else int(n == 2)
        n_val = max(1, int(round(n * 0.15))) if n >= 3 else 0
        while n - n_test - n_val < 1:
            if n_val:
                n_val -= 1
            elif n_test:
                n_test -= 1
            else:
                break
        for i, track_id in enumerate(members.tolist()):
            assignment[str(track_id)] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return assignment


def build_track_index(annotations_dir: str | Path, seed: int = RANDOM_SEED) -> list[Track]:
    """Parse GiantSteps key and MD5 files into a stable split-aware index."""
    root = Path(annotations_dir)
    key_files = sorted((root / "annotations" / "key").glob("*.key"))
    if not key_files:
        raise RuntimeError(f"no GiantSteps key annotations found under {root}")
    parsed: list[tuple[str, str, int, str | None]] = []
    for key_file in key_files:
        base = key_file.name.removesuffix(".key")
        filename = f"{base}.mp3"
        track_id = base.removesuffix(".LOFI")
        raw_label = key_file.read_text(encoding="utf-8").strip()
        label = parse_key(raw_label)
        music21_crosscheck(raw_label, label)
        md5_path = root / "md5" / f"{base}.md5"
        expected = md5_path.read_text(encoding="utf-8").strip().split()[0].lower() if md5_path.exists() else None
        parsed.append((track_id, filename, label, expected))
    splits = assign_stratified_splits([(track_id, label) for track_id, _, label, _ in parsed], seed)
    return [Track(track_id, filename, label, splits[track_id], expected)
            for track_id, filename, label, expected in parsed]


def music21_crosscheck(label: str, parsed: int) -> None:
    """Raise when music21 disagrees with this project's canonical label parser."""
    from music21 import key as music21_key

    del label  # the canonical spelling avoids music21's full-label parser ambiguity
    tonic, mode = key_name(parsed).split()
    analysed = music21_key.Key(tonic, mode)
    expected_tonic = int(analysed.tonic.pitchClass)
    expected_mode = str(analysed.mode)
    actual_tonic = int(parsed) % 12
    actual_mode = "major" if int(parsed) < 12 else "minor"
    if (actual_tonic, actual_mode) != (expected_tonic, expected_mode):
        raise ValueError(
            f"label parser disagrees with music21 for {label!r}: "
            f"{actual_tonic}/{actual_mode} vs {expected_tonic}/{expected_mode}"
        )


def file_md5(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a file's hexadecimal MD5 for upstream integrity verification."""
    digest = hashlib.md5()  # noqa: S324 - required to verify upstream-provided MD5
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_preview(
    track: Track,
    destination: str | Path,
    retries: int = 3,
    session: requests.Session | None = None,
) -> str:
    """Download and verify one preview, returning the successful source URL."""
    target = Path(destination)
    client = session or requests.Session()
    errors: list[str] = []
    for base in (PRIMARY_AUDIO_BASE, FALLBACK_AUDIO_BASE):
        url = base + track.filename
        for attempt in range(retries):
            try:
                with client.get(url, stream=True, timeout=(10, 60)) as response:
                    response.raise_for_status()
                    with target.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                if track.md5 and file_md5(target) != track.md5:
                    raise RuntimeError("MD5 mismatch")
                return url
            except Exception as exc:  # noqa: BLE001 - retry all transport/integrity failures
                target.unlink(missing_ok=True)
                errors.append(f"{url} attempt {attempt + 1}: {exc}")
                if attempt + 1 < retries:
                    time.sleep(2**attempt)
    raise RuntimeError("; ".join(errors))


def checkpoint_path(work_dir: str | Path, track_id: str) -> Path:
    """Return the atomic JSON checkpoint path for a track."""
    return Path(work_dir) / "tracks" / f"{track_id}.json"


def read_checkpoint(
    work_dir: str | Path,
    track: Track,
    signature: str,
) -> dict[str, Any] | None:
    """Load a compatible completed track checkpoint, if one exists."""
    path = checkpoint_path(work_dir, track.track_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    vector = payload.get("features")
    if (payload.get("status") != "complete" or payload.get("config_signature") != signature or
            payload.get("track_id") != track.track_id or payload.get("label") != track.label or
            not isinstance(vector, list) or len(vector) != 36):
        return None
    return payload


def write_checkpoint(work_dir: str | Path, payload: dict[str, Any]) -> Path:
    """Atomically persist a completed per-track extraction checkpoint."""
    path = checkpoint_path(work_dir, str(payload["track_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def extract_track(track: Track, work_dir: str | Path, config: FeatureConfig) -> dict[str, Any]:
    """Download, extract, delete audio, and checkpoint one track."""
    signature = config_signature(config)
    existing = read_checkpoint(work_dir, track, signature)
    if existing is not None:
        existing["resumed"] = True
        return existing
    with tempfile.TemporaryDirectory(prefix=f"keyclf-audio-{track.track_id}-") as temporary:
        audio = Path(temporary) / track.filename
        source_url = download_preview(track, audio)
        vector = extract_features(audio, config=config)
    payload: dict[str, Any] = {
        "status": "complete",
        "track_id": track.track_id,
        "filename": track.filename,
        "label": track.label,
        "key": key_name(track.label),
        "split": track.split,
        "md5": track.md5,
        "source_url": source_url,
        "config": asdict(config),
        "config_signature": signature,
        "features": vector.tolist(),
        "ks_top3": ks_top_keys(vector[24:36], 3).astype(int).tolist(),
        "ks_flagged": bool(track.label not in ks_top_keys(vector[24:36], 3)),
        "resumed": False,
    }
    write_checkpoint(work_dir, payload)
    return payload


def assemble_feature_cache(checkpoints: list[dict[str, Any]], output: str | Path) -> Path:
    """Assemble checkpoints into the harness-compatible single Parquet cache."""
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    other = [p for pattern in ("*.parquet", "*.csv") for p in target.parent.glob(pattern) if p != target]
    if other:
        raise RuntimeError(f"refusing ambiguous cache directory; remove other feature files: {other}")
    names = feature_names()
    rows: list[dict[str, Any]] = []
    for payload in sorted(checkpoints, key=lambda item: str(item["track_id"])):
        row: dict[str, Any] = {
            "track_id": str(payload["track_id"]),
            "key": str(payload["key"]),
            "split": str(payload["split"]),
        }
        row.update(dict(zip(names, map(float, payload["features"]))))
        rows.append(row)
    frame = pd.DataFrame(rows, columns=["track_id", "key", "split", *names])
    temporary = target.with_name(f".{target.name}.tmp")
    if target.suffix == ".csv":
        frame.to_csv(temporary, index=False)
    else:
        frame.to_parquet(temporary, index=False)
    os.replace(temporary, target)
    return target


def build_feature_cache(
    annotations_dir: str | Path = "data/annotations/giantsteps-key-dataset",
    work_dir: str | Path = "data/work",
    output: str | Path = "data/features/giantsteps_features.parquet",
    config: FeatureConfig = FeatureConfig(),
    workers: int = 2,
    limit: int | None = None,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Build or resume the complete feature cache without retaining raw audio."""
    annotations = sparse_clone_annotations(annotations_dir)
    work = Path(work_dir)
    if rebuild:
        for path in (work / "tracks").glob("*.json"):
            path.unlink()
    tracks = build_track_index(annotations)
    if limit is not None:
        tracks = tracks[:max(0, int(limit))]
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    worker_count = max(1, int(workers))
    pool = ThreadPoolExecutor(max_workers=worker_count)
    interrupted = False
    iterator = iter(tracks)
    pending: dict[Any, Track] = {}

    def submit_next() -> bool:
        try:
            track = next(iterator)
        except StopIteration:
            return False
        pending[pool.submit(extract_track, track, work, config)] = track
        return True

    try:
        for _ in range(worker_count * 2):
            if not submit_next():
                break
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                track = pending.pop(future)
                try:
                    completed.append(future.result())
                except Exception as exc:  # noqa: BLE001 - log failure and continue the resumable run
                    failures.append({"track_id": track.track_id, "filename": track.filename,
                                     "error": repr(exc)})
                submit_next()
    except KeyboardInterrupt:
        interrupted = True
        for future in pending:
            future.cancel()
        raise
    finally:
        pool.shutdown(wait=not interrupted, cancel_futures=True)
    cache = assemble_feature_cache(completed, output) if completed else None
    summary: dict[str, Any] = {
        "annotation_revision": annotation_revision(annotations),
        "config": asdict(config),
        "config_signature": config_signature(config),
        "requested": len(tracks),
        "completed": len(completed),
        "resumed": sum(bool(item.get("resumed")) for item in completed),
        "failed": len(failures),
        "failures": failures,
        "ks_flagged": sum(bool(item.get("ks_flagged")) for item in completed),
        "cache": str(cache) if cache else None,
        "raw_audio_retained": False,
    }
    work.mkdir(parents=True, exist_ok=True)
    report = work / "extraction_report.json"
    report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
