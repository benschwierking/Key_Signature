"""Find independent major-key candidates in additional annotation metadata."""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from keyclf.data import build_track_index, music21_crosscheck
from keyclf.keys import key_name, parse_key


def main():
    """Write a deduplicated acquisition manifest without downloading audio."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--additional", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    original = build_track_index(args.original)
    ids = {track.track_id for track in original}
    hashes = {track.md5 for track in original if track.md5}
    additional, candidates, rejected = [], [], []
    for source in sorted((args.additional / "annotations" / "key").glob("*.key")):
        raw = source.read_text().strip()
        # MTG includes tab-separated fields beyond the label. Preserve those
        # fields as uninterpreted metadata rather than claiming a confidence scale.
        fields = raw.split("\t")
        label = fields[0].strip()
        try:
            parsed = parse_key(label)
            music21_crosscheck(label, parsed)
        except ValueError as error:
            rejected.append({"file": source.name, "annotation": raw, "error": str(error)})
            continue
        base = source.name.removesuffix(".key")
        identity = base.removesuffix(".LOFI")
        checksum = args.additional / "md5" / f"{base}.md5"
        md5 = checksum.read_text().strip().split()[0].lower() if checksum.exists() else None
        row = {"track_id": identity, "key": key_name(parsed), "mode": "major" if parsed < 12 else "minor",
               "md5": md5, "priority": "major coverage" if parsed < 12 else "later coverage",
               "url": f"https://www.cp.jku.at/datasets/giantsteps/mtg_key_backup/{base}.mp3",
               "annotation_extra_fields": json.dumps(fields[1:]),
               "rights_reviewed": False, "audio_downloaded": False}
        additional.append(row)
        if identity not in ids and (not md5 or md5 not in hashes):
            candidates.append(row)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = candidates
    pd.DataFrame(rows).to_csv(args.out / "additional_data_manifest.csv", index=False)
    summary = {"additional_annotations": len(additional), "rejected_annotations": rejected,
               "exact_id_or_md5_overlap": len(additional) - len(candidates),
               "independent_candidates": len(candidates), "mode_counts": dict(Counter(r["mode"] for r in rows)),
               "key_counts": dict(Counter(r["key"] for r in rows)),
               "audio_downloaded": False, "rights_reviewed": False,
               "duplicate_check_limits": "exact recording IDs and audio MD5 only; near duplicates and artist grouping require further review"}
    (args.out / "additional_data_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
