"""Canonical key-label parsing and key-relation helpers."""

from __future__ import annotations

import re

NOTE_TO_PC = {
    "C": 0, "B#": 0,
    "C#": 1, "DB": 1,
    "D": 2,
    "D#": 3, "EB": 3,
    "E": 4, "FB": 4,
    "F": 5, "E#": 5,
    "F#": 6, "GB": 6,
    "G": 7,
    "G#": 8, "AB": 8,
    "A": 9,
    "A#": 10, "BB": 10,
    "B": 11, "CB": 11,
}
PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
KEY_NAMES = tuple(f"{p} major" for p in PITCH_NAMES) + tuple(f"{p} minor" for p in PITCH_NAMES)


def parse_key(label: str | int) -> int:
    """Return the canonical 0–23 class for a human-readable key label."""
    if isinstance(label, int):
        if 0 <= label < 24:
            return label
        raise ValueError(f"key index outside 0..23: {label}")
    text = str(label).strip().replace("♭", "b").replace("♯", "#")
    if text.isdigit():
        return parse_key(int(text))
    text = re.sub(r"\s+", " ", text)
    match = re.fullmatch(r"([A-Ga-g])([#b]?)(?:\s*:?\s*(major|minor|maj|min|M|m))?", text)
    if not match:
        raise ValueError(f"unrecognized key label: {label!r}")
    note, accidental, mode = match.groups()
    token = (note.upper() + accidental).upper()
    if token not in NOTE_TO_PC:
        raise ValueError(f"unrecognized tonic: {note}{accidental}")
    if mode in ("minor", "min", "m") or (mode is None and note.islower()):
        offset = 12
    else:
        offset = 0
    return NOTE_TO_PC[token] + offset


def key_name(key: int) -> str:
    """Return the canonical display name for a 0–23 key class."""
    return KEY_NAMES[int(key)]


def tonic_of(key: int) -> int:
    """Return a key class's pitch-class tonic."""
    return int(key) % 12


def mode_of(key: int) -> str:
    """Return ``major`` or ``minor`` for a key class."""
    return "major" if int(key) < 12 else "minor"


def relative_key(key: int) -> int:
    """Return the relative major or minor key class."""
    k = int(key)
    return ((k % 12 + 9) % 12) + 12 if k < 12 else (k % 12 + 3) % 12


def fifth_neighbors(key: int) -> tuple[int, int]:
    """Return the same-mode keys a perfect fifth above and below."""
    k = int(key)
    offset = 0 if k < 12 else 12
    tonic = k % 12
    return offset + (tonic + 7) % 12, offset + (tonic - 7) % 12


def is_related(reference: int, candidate: int) -> bool:
    """Whether two classes are exact, relative, or same-mode fifth neighbors."""
    return int(candidate) in {int(reference), relative_key(reference), *fifth_neighbors(reference)}
