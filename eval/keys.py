"""Key-label conventions and key relations used by the harness.

Spec §1: "24 classes: 12 major + 12 minor".
Spec §3 / §10: "related-key" accuracy = exact OR relative major/minor OR perfect fifth.

Canonical class index (harness convention, documented in HARNESS_README.md):
    0..11  -> C major .. B major   (tonic pitch class = index)
    12..23 -> C minor .. B minor   (tonic pitch class = index - 12)
Pitch classes follow spec §7's bin order "C, C#, ..., B" (librosa chroma order).
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np

# Spec §7: 12 pitch-class bins "(C, C#, ..., B)" — librosa chroma ordering.
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MODES = ("major", "minor")
N_KEYS = 24  # Spec §1: 24 classes (12 major + 12 minor)

KEY_NAMES = [f"{pc} {m}" for m in MODES for pc in PITCH_CLASSES]

_NOTE_BASE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_LABEL_RE = re.compile(
    r"^\s*([A-Ga-g])\s*([#♯b♭]*)\s*[:_\-\s]?\s*"
    r"(major|maj|minor|min|m|M|)?\s*$"
)


def key_index(tonic_pc: int, mode: str) -> int:
    """Return the canonical class index for a tonic pitch class (0-11) and mode."""
    if mode not in MODES:
        raise ValueError(f"mode must be 'major' or 'minor', got {mode!r}")
    return (tonic_pc % 12) + (12 if mode == "minor" else 0)


def tonic_of(idx: int) -> int:
    """Tonic pitch class (0-11) of canonical key index ``idx``."""
    _check(idx)
    return idx % 12


def mode_of(idx: int) -> str:
    """Mode ('major' / 'minor') of canonical key index ``idx``."""
    _check(idx)
    return "major" if idx < 12 else "minor"


def key_name(idx: int) -> str:
    """Human-readable name for canonical key index ``idx`` (e.g. 'A minor')."""
    _check(idx)
    return KEY_NAMES[idx]


def _check(idx: int) -> None:
    if not (0 <= int(idx) < N_KEYS):
        raise ValueError(f"key index out of range 0..23: {idx}")


def parse_key(label) -> int:
    """Parse a key label into a canonical index.

    Accepts strings such as 'C major', 'A minor', 'Eb minor', 'F# maj', 'c:min',
    'Bbm', 'D♭ major' (enharmonics fold together) and integers 0..23, which are
    taken to already follow the canonical convention (major 0-11, minor 12-23).
    """
    if isinstance(label, (int, np.integer)):
        _check(int(label))
        return int(label)
    s = str(label).strip()
    if s.isdigit():
        return parse_key(int(s))
    m = _LABEL_RE.match(s)
    if not m:
        raise ValueError(f"unparseable key label: {label!r}")
    letter, accidentals, mode = m.groups()
    pc = _NOTE_BASE[letter.upper()]
    for a in accidentals:
        pc += 1 if a in "#♯" else -1
    mode = (mode or "").lower() if mode != "M" else "major"
    if mode in ("minor", "min", "m"):
        mode_name = "minor"
    elif mode in ("major", "maj", ""):
        # A bare lowercase tonic ('a') is a common minor shorthand; uppercase/bare = major.
        mode_name = "minor" if (mode == "" and letter.islower()) else "major"
    else:  # pragma: no cover - regex prevents this
        raise ValueError(f"unparseable mode in {label!r}")
    return key_index(pc % 12, mode_name)


def parse_keys(labels: Iterable) -> np.ndarray:
    """Vectorised :func:`parse_key`."""
    return np.array([parse_key(x) for x in labels], dtype=int)


def relative_key(idx: int) -> int:
    """Relative major/minor (spec §3: "C major and A minor" are related).

    Major tonic t  -> minor tonic t+9 (a minor third below).
    Minor tonic t  -> major tonic t+3 (a minor third above).
    """
    t = tonic_of(idx)
    if mode_of(idx) == "major":
        return key_index((t + 9) % 12, "minor")
    return key_index((t + 3) % 12, "major")


def fifth_keys(idx: int) -> tuple[int, int]:
    """Keys a perfect fifth away, same mode: (fifth above, fifth below).

    Spec §10 says "perfect fifth" without a direction; the harness accepts both
    neighbours on the circle of fifths (tonic ±7 semitones, same mode). See
    HARNESS_README.md "Ambiguities".
    """
    t, m = tonic_of(idx), mode_of(idx)
    return key_index((t + 7) % 12, m), key_index((t + 5) % 12, m)


def related_keys(idx: int) -> set[int]:
    """Set of keys that count as correct for spec §10 "related-key" accuracy.

    exact OR relative major/minor OR perfect fifth.
    """
    return {int(idx), relative_key(idx), *fifth_keys(idx)}


def is_related(true_idx: int, pred_idx: int) -> bool:
    """True if ``pred_idx`` is exact / relative / perfect-fifth of ``true_idx``."""
    return int(pred_idx) in related_keys(int(true_idx))
