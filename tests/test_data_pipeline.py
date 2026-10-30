"""Local-only tests for resumable data extraction and cache assembly."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from keyclf import data
from keyclf.features import FeatureConfig, config_signature, feature_names


def test_annotation_crosscheck_compares_the_raw_label():
    data.music21_crosscheck("Ab major", 8)
    data.music21_crosscheck("a", 21)
    with pytest.raises(ValueError, match="disagrees"):
        data.music21_crosscheck("Ab major", 20)


def test_stratified_split_is_deterministic_and_has_all_partitions():
    rows = [(f"{key}-{i}", key) for key in range(24) for i in range(10)]
    first = data.assign_stratified_splits(rows, seed=42)
    second = data.assign_stratified_splits(rows, seed=42)
    assert first == second
    for key in range(24):
        assert {first[f"{key}-{i}"] for i in range(10)} == {"train", "val", "test"}


def test_checkpoint_roundtrip_and_incompatible_config(tmp_path):
    track = data.Track("one", "one.LOFI.mp3", 0, "train", None)
    config = FeatureConfig()
    payload = {
        "status": "complete", "track_id": "one", "label": 0, "split": "train",
        "key": "C major", "config_signature": config_signature(config),
        "features": np.r_[np.eye(12)[0], np.eye(12)[0], np.ones(12) / 12].tolist(),
    }
    data.write_checkpoint(tmp_path, payload)
    assert data.read_checkpoint(tmp_path, track, config_signature(config)) is not None
    assert data.read_checkpoint(tmp_path, track, config_signature(FeatureConfig(chroma_type="stft"))) is None


def test_extract_track_resumes_without_redownload_and_deletes_audio(tmp_path, monkeypatch):
    track = data.Track("one", "one.LOFI.mp3", 0, "train", None)
    calls: list[Path] = []

    def fake_download(_track, destination):
        path = Path(destination)
        calls.append(path)
        path.write_bytes(b"temporary audio")
        return "https://example.test/one.mp3"

    monkeypatch.setattr(data, "download_preview", fake_download)
    monkeypatch.setattr(data, "extract_features", lambda _path, config: np.r_[
        np.eye(12)[0], np.eye(12)[0], np.ones(12) / 12,
    ])
    first = data.extract_track(track, tmp_path, FeatureConfig())
    second = data.extract_track(track, tmp_path, FeatureConfig())
    assert first["resumed"] is False and second["resumed"] is True
    assert len(calls) == 1 and not calls[0].parent.exists()


class _Response:
    def __init__(self, content: bytes):
        self.content = content

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        del chunk_size
        yield self.content


class _Session:
    def __init__(self):
        self.urls: list[str] = []

    def get(self, url, **_kwargs):
        self.urls.append(url)
        return _Response(b"bad" if "cp.jku.at" in url else b"good")


def test_download_falls_back_after_checksum_mismatch(tmp_path):
    expected = hashlib.md5(b"good").hexdigest()
    track = data.Track("one", "one.LOFI.mp3", 0, "train", expected)
    session = _Session()
    target = tmp_path / track.filename
    source = data.download_preview(track, target, retries=1, session=session)
    assert "beatport" in source and target.read_bytes() == b"good"
    assert len(session.urls) == 2


def test_assembled_cache_has_exact_harness_schema(tmp_path):
    vector = np.r_[np.eye(12)[0], np.eye(12)[11], np.ones(12) / 12]
    checkpoints = [{"track_id": "a", "key": "C major", "split": "test", "features": vector.tolist()}]
    output = data.assemble_feature_cache(checkpoints, tmp_path / "features.parquet")
    frame = pd.read_parquet(output)
    assert frame.columns.tolist() == ["track_id", "key", "split", *feature_names()]
    assert frame[feature_names()].shape == (1, 36)
