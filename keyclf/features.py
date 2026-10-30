"""Audio-to-chroma feature extraction and resumable cache CLI.

The public functions intentionally match specification §7 and the acceptance
harness: first-bin one-hot (12), last-bin one-hot (12), then a normalized
track-wide chroma histogram (12).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for deterministic librosa chroma extraction."""

    chroma_type: str = "cqt"
    hop_length: int = 512
    edge_frames: int = 20
    sample_rate: int = 22050
    silence_top_db: float = 60.0

    def __post_init__(self) -> None:
        if self.chroma_type not in {"cqt", "stft"}:
            raise ValueError("chroma_type must be 'cqt' or 'stft'")
        if self.hop_length <= 0 or self.edge_frames <= 0 or self.sample_rate <= 0:
            raise ValueError("hop_length, edge_frames, and sample_rate must be positive")


DEFAULT_CONFIG = FeatureConfig()
CHROMA_TYPE = DEFAULT_CONFIG.chroma_type
HOP_LENGTH = DEFAULT_CONFIG.hop_length
EDGE_FRAMES = DEFAULT_CONFIG.edge_frames


def config_signature(config: FeatureConfig = DEFAULT_CONFIG) -> str:
    """Return a stable short hash for checkpoint compatibility checks."""
    raw = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def load_audio(path: str | Path, sample_rate: int = DEFAULT_CONFIG.sample_rate) -> tuple[np.ndarray, int]:
    """Load an audio file as mono floating-point samples at ``sample_rate``.

    Librosa handles WAV and common MP3 builds directly. If its backend rejects
    a container such as MP4, ffmpeg decodes a mono float stream as a fallback.
    """
    source = str(path)
    try:
        y, sr = librosa.load(source, sr=sample_rate, mono=True)
        return np.asarray(y, dtype=np.float32), int(sr)
    except Exception as original:
        if not shutil.which("ffmpeg"):
            raise RuntimeError(f"cannot decode {source}; ffmpeg is unavailable") from original
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", source, "-f", "f32le", "-ac", "1", "-ar", str(sample_rate), "-"],
            check=False,
            capture_output=True,
        )
        if proc.returncode != 0:
            message = proc.stderr.decode(errors="replace").strip()
            raise RuntimeError(f"ffmpeg could not decode {source}: {message}") from original
        return np.frombuffer(proc.stdout, dtype="<f4").copy(), int(sample_rate)


def compute_chroma(
    y: np.ndarray,
    sr: int,
    chroma_type: str = DEFAULT_CONFIG.chroma_type,
    hop_length: int = DEFAULT_CONFIG.hop_length,
) -> np.ndarray:
    """Compute a non-negative librosa chromagram with shape ``(12, frames)``."""
    samples = np.asarray(y, dtype=float).reshape(-1)
    if samples.size == 0:
        return np.zeros((12, 1), dtype=float)
    if chroma_type == "cqt":
        # Fix tuning at concert pitch. Automatic tuning estimation is noisy for
        # short synthetic tones and has triggered a native numba crash on some
        # macOS/arm64 builds; the dataset annotations assume equal temperament.
        chroma = librosa.feature.chroma_cqt(
            y=samples, sr=int(sr), hop_length=int(hop_length), tuning=0.0
        )
    elif chroma_type == "stft":
        chroma = librosa.feature.chroma_stft(y=samples, sr=int(sr), hop_length=int(hop_length))
    else:
        raise ValueError("chroma_type must be 'cqt' or 'stft'")
    result = np.nan_to_num(np.asarray(chroma, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(result, 0.0)


def first_last_bins(
    chroma: np.ndarray,
    edge_frames: int = DEFAULT_CONFIG.edge_frames,
) -> tuple[np.ndarray, np.ndarray]:
    """Return first and last dominant pitch classes as 12-dimensional one-hots."""
    matrix = np.asarray(chroma, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != 12 or matrix.shape[1] == 0:
        raise ValueError(f"expected chroma shape (12, frames), got {matrix.shape}")
    width = max(1, min(int(edge_frames), matrix.shape[1]))
    first_index = int(matrix[:, :width].sum(axis=1).argmax())
    last_index = int(matrix[:, -width:].sum(axis=1).argmax())
    first = np.zeros(12, dtype=float)
    last = np.zeros(12, dtype=float)
    first[first_index] = 1.0
    last[last_index] = 1.0
    return first, last


def chroma_histogram(chroma: np.ndarray) -> np.ndarray:
    """Sum chroma energy over time and normalize the 12 bins to one."""
    matrix = np.asarray(chroma, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != 12:
        raise ValueError(f"expected chroma shape (12, frames), got {matrix.shape}")
    histogram = np.maximum(matrix, 0.0).sum(axis=1)
    total = float(histogram.sum())
    return histogram / total if total > 0 else np.full(12, 1.0 / 12.0)


def extract_features(path: str | Path, config: FeatureConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Extract ``[first one-hot | last one-hot | histogram]`` from an audio file."""
    y, sr = load_audio(path, sample_rate=config.sample_rate)
    trimmed, _ = librosa.effects.trim(y, top_db=config.silence_top_db)
    if trimmed.size:
        y = trimmed
    chroma = compute_chroma(y, sr, chroma_type=config.chroma_type, hop_length=config.hop_length)
    first, last = first_last_bins(chroma, edge_frames=config.edge_frames)
    vector = np.concatenate((first, last, chroma_histogram(chroma))).astype(float)
    if vector.shape != (36,) or not np.isfinite(vector).all():
        raise RuntimeError(f"invalid feature vector produced for {path}: {vector.shape}")
    return vector


def feature_names() -> list[str]:
    """Return the canonical ordered names of the 36 feature columns."""
    return ([f"first_{i}" for i in range(12)] + [f"last_{i}" for i in range(12)] +
            [f"hist_{i}" for i in range(12)])


def build_parser() -> argparse.ArgumentParser:
    """Build the feature-cache command-line parser."""
    parser = argparse.ArgumentParser(description="Build or resume the GiantSteps chroma feature cache")
    parser.add_argument("--annotations-dir", default="data/annotations/giantsteps-key-dataset")
    parser.add_argument("--work-dir", default="data/work")
    parser.add_argument("--output", default="data/features/giantsteps_features.parquet")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--chroma-type", choices=["cqt", "stft"], default=DEFAULT_CONFIG.chroma_type)
    parser.add_argument("--hop-length", type=int, default=DEFAULT_CONFIG.hop_length)
    parser.add_argument("--edge-frames", type=int, default=DEFAULT_CONFIG.edge_frames)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for resumable dataset feature extraction."""
    from .data import build_feature_cache

    args = build_parser().parse_args(argv)
    config = FeatureConfig(chroma_type=args.chroma_type, hop_length=args.hop_length,
                           edge_frames=args.edge_frames)
    result = build_feature_cache(
        annotations_dir=Path(args.annotations_dir),
        work_dir=Path(args.work_dir),
        output=Path(args.output),
        config=config,
        workers=args.workers,
        limit=args.limit,
        rebuild=args.rebuild,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["completed"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
