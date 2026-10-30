"""Check transposition and musical feature semantics used in research."""

import numpy as np
import pandas as pd
import pytest

from keyclf.experiments import rotate_training, summarize, validate_reference
from keyclf.research_features import BLOCKS, aggregate_research_blocks


def test_rotation_moves_all_pitch_blocks_and_preserves_mode():
    X = np.tile(np.eye(12)[0], 3)[None, :]
    augmented, labels = rotate_training(X, np.array([12]))
    assert augmented.shape == (12, 36)
    assert np.array_equal(labels, np.arange(12, 24))
    for shift in range(12):
        assert np.array_equal(augmented[shift].reshape(3, 12).argmax(axis=1), [shift] * 3)


def test_persistent_pitch_and_section_boundaries_have_expected_semantics():
    chroma = np.zeros((12, 80))
    chroma[0, :40] = 1
    chroma[7, 40:] = 1
    chroma[5, 15] = 0.8
    blocks = aggregate_research_blocks(chroma, chroma, chroma, np.arange(0, 80, 10), np.ones(80))
    assert tuple(blocks) == BLOCKS
    assert blocks["quarter1"].argmax() == 0 and blocks["quarter4"].argmax() == 7
    assert blocks["sustain"][5] == 0
    for value in blocks.values():
        assert value.shape == (12,) and np.isfinite(value).all() and np.isclose(value.sum(), 1)


def test_recall_names_distinguish_mode_from_exact_key():
    y = np.array([0, 12, 13])
    probability = np.eye(24)[[1, 12, 12]]
    metrics = summarize(y, probability)
    assert metrics["accuracy"] == metrics["weighted_recall"] == 1 / 3
    assert metrics["major_exact_recall"] == 0
    assert metrics["major_mode_recall"] == 1
    assert metrics["minor_exact_recall"] == 0.5


def test_partial_extraction_or_changed_test_assignments_are_rejected():
    reference = pd.DataFrame({"track_id": ["a", "b"], "key": ["C major", "A minor"], "split": ["train", "test"]})
    validate_reference(reference.iloc[::-1], reference)
    with pytest.raises(ValueError, match="every original recording"):
        validate_reference(reference.iloc[:1], reference)
    changed = reference.copy()
    changed.loc[1, "split"] = "train"
    with pytest.raises(ValueError, match="every original recording"):
        validate_reference(changed, reference)
