"""Acceptance evaluation for the key-signature classifier.

Spec references: §9 (acceptance criteria), §10 (report rubric), §12 (storage:
``models/model.joblib`` and the cached feature file under ``data/features/``).

Usage (from the repo root):
    python -m eval.evaluate                                   # evaluate + gate
    python -m eval.evaluate --junit reports/pytest-junit.xml  # include §9 unit-test evidence
    python -m eval.evaluate --criteria-only reports/metrics.json [--junit ...]

Exit codes: 0 = every §9 criterion PASS, 1 = at least one §9 criterion FAIL,
2 = inputs missing/malformed (cannot evaluate).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

from . import criteria as crit
from .dataset import (HarnessInputError, find_feature_file, leakage, load_features, load_model,
                      predict_proba_24, resolve_split)
from .keys import KEY_NAMES
from .ks_baseline import fit_temperature, ks_proba
from .metrics import full_report

DEFAULT_EXTRACTOR = "keyclf.features:extract_features"  # spec §7 module responsibilities
# Default synthetic clip length for the latency check. GiantSteps Key audio are
# ~2-minute previews (spec §6 dataset); this is a measurement setting, not a criterion.
DEFAULT_LATENCY_DURATION_S = 120.0


# ------------------------------------------------------------------ helpers
def tracks_hash(tracks) -> str:
    """Order-independent SHA-256 of a set of track ids (proves 'same test split', §9)."""
    h = hashlib.sha256()
    for t in sorted(map(str, tracks)):
        h.update(t.encode() + b"\n")
    return h.hexdigest()


def save_confusion(cm: np.ndarray, png: Path, csv: Path, title: str) -> None:
    """Write a 24x24 confusion matrix as CSV and PNG (spec §9 line 80, §10 line 88)."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pd.DataFrame(cm, index=[f"true: {k}" for k in KEY_NAMES],
                 columns=[f"pred: {k}" for k in KEY_NAMES]).to_csv(csv)
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(24), KEY_NAMES, rotation=90, fontsize=7)
    ax.set_yticks(range(24), KEY_NAMES, fontsize=7)
    ax.set_xlabel("predicted key")
    ax.set_ylabel("true key")
    ax.set_title(title)
    for i in range(24):
        for j in range(24):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)


def load_extractor(spec: str):
    """Import 'module:function' (default keyclf.features:extract_features)."""
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn or "extract_features")


def synth_song(path: Path, duration_s: float, sr: int = 22050, fmt: str = "mp3") -> Path:
    """Write a SYNTHETIC test clip (I-IV-V-I in C major, harmonic tones) for latency timing.

    Not dataset audio: spec §6/§12 forbid storing GiantSteps audio, so the
    default latency probe is generated on the fly and deleted afterwards.
    If ffmpeg is available the clip is encoded to MP3 so the ffmpeg decode path
    used for real uploads (spec §8 MP3/MP4/WAV, §14 ffmpeg) is exercised.
    """
    from scipy.io import wavfile

    chords = [[60, 64, 67], [65, 69, 72], [67, 71, 74], [60, 64, 67]]
    seg = 2.0
    t = np.arange(int(seg * sr)) / sr
    env = np.minimum(1.0, np.minimum(t / 0.02, (seg - t) / 0.05))
    blocks = []
    for i in range(int(np.ceil(duration_s / seg))):
        y = np.zeros_like(t)
        for m in chords[i % 4]:
            f = 440.0 * 2 ** ((m - 69) / 12)
            for h, a in ((1, 1.0), (2, 0.4), (3, 0.2)):
                y += a * np.sin(2 * np.pi * f * h * t)
        blocks.append(y * env)
    y = np.concatenate(blocks)[: int(duration_s * sr)]
    y = (0.3 * y / np.abs(y).max()).astype(np.float32)
    wav = path.with_suffix(".wav")
    wavfile.write(wav, sr, y)
    if fmt == "mp3" and shutil.which("ffmpeg"):
        mp3 = path.with_suffix(".mp3")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-b:a", "128k", str(mp3)],
                       check=True)
        wav.unlink()
        return mp3
    return wav


def measure_latency(est, meta, feature_names, extractor_spec: str, audio_paths: list[str],
                    urls: list[str], runs: int, duration_s: float, fmt: str) -> dict:
    """Time upload-to-result = extract_features(path) + predict_proba (spec §9 line 79).

    Every run is timed (no warm-up) so the first, cold call counts too.
    The gate uses the MAX over runs; mean is reported as well (spec §17).
    """
    try:
        extract = load_extractor(extractor_spec)
    except Exception as e:  # noqa: BLE001 - surface any import failure as a criterion failure
        return {"status": "error", "error": f"cannot import {extractor_spec}: {e!r}",
                "extractor": extractor_spec}
    tmp = Path(tempfile.mkdtemp(prefix="keyclf-latency-"))
    probes: list[tuple[str, str]] = []
    try:
        for p in audio_paths:
            probes.append((p, f"file {p}"))
        for i, u in enumerate(urls):
            dst = tmp / f"url{i}{Path(u).suffix or '.mp3'}"
            urllib.request.urlretrieve(u, dst)  # temp file deleted below (spec §6: discard audio)
            probes.append((str(dst), f"url {u}"))
        if not probes:
            clip = synth_song(tmp / "synthetic_latency_probe", duration_s, fmt=fmt)
            probes.append((str(clip), f"SYNTHETIC {duration_s:.0f}s {clip.suffix[1:]} clip"))
        times, desc = [], []
        for path, d in probes:
            for _ in range(runs):
                t0 = time.perf_counter()
                v = np.asarray(extract(path), dtype=float).reshape(-1)
                proba = predict_proba_24(est, v.reshape(1, -1), feature_names, meta)
                times.append(time.perf_counter() - t0)
                if v.shape[0] != 36:
                    return {"status": "error", "error": f"extractor returned {v.shape[0]} dims, spec §7 says 36"}
                _ = proba.argmax()
            desc.append(d)
        return {"status": "ok", "runs_s": times, "mean_s": float(np.mean(times)),
                "max_s": float(np.max(times)), "first_s": float(times[0]),
                "audio": "; ".join(desc), "extractor": extractor_spec}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": repr(e), "extractor": extractor_spec}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ main flow
def run_evaluation(args) -> dict:
    """Compute every §9/§10 metric and write artifacts into ``args.out``."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    feat_path = Path(args.features) if args.features else find_feature_file(args.features_dir)
    table = load_features(feat_path, args.label_col, args.track_col, args.split_col,
                          args.feature_cols.split(",") if args.feature_cols else None)
    est, meta = load_model(args.model)
    split, split_src = resolve_split(table, args.split_file, meta)
    test = split == "test"
    if not test.any():
        raise HarnessInputError("held-out test split is empty")
    nontest_fit = (split == "val") if (split == "val").any() else (split == "train")

    leak = leakage(table.tracks, split)
    counts = {str(s): int((split == s).sum()) for s in sorted(set(split))}
    th = tracks_hash(table.tracks[test])

    # ML model (§9 lines 77-78, §10 line 88)
    p_model = predict_proba_24(est, table.X[test], table.feature_names, meta)
    rep_model = full_report(table.y[test], p_model)
    rep_model.update(n=int(test.sum()), test_tracks_sha256=th)

    # K-S baseline on the same rows (§9 line 83, §10 line 88). Histogram = cols 24:36 (§7 order).
    hist = table.X[:, 24:36]
    temp = fit_temperature(hist[nontest_fit], table.y[nontest_fit]) if nontest_fit.any() else 1.0
    rep_ks = full_report(table.y[test], ks_proba(hist[test], temp))
    rep_ks.update(n=int(test.sum()), test_tracks_sha256=th, softmax_temperature=temp,
                  temperature_fit_on=("val" if (split == "val").any() else "train") if nontest_fit.any() else "none")

    arts = {}
    for name, rep, title in (("model", rep_model, "ML model"), ("ks", rep_ks, "K-S baseline")):
        png, csv = out / f"confusion_matrix_{name}.png", out / f"confusion_matrix_{name}.csv"
        save_confusion(np.array(rep["confusion_matrix"]), png, csv, f"{title} — held-out test split")
        arts[f"confusion_matrix{'' if name == 'model' else '_ks'}_png"] = png.name
        arts[f"confusion_matrix{'' if name == 'model' else '_ks'}_csv"] = csv.name

    # Learned vs K-S profiles (§9 line 82): compute from coef_ when possible, else
    # accept implementation-produced artifacts. No correlation threshold exists.
    profile_json = args.profile_json or meta.get("profile_json")
    profile_plot = args.profile_plot or meta.get("profile_plot")
    if profile_json:
        prof = json.loads(Path(profile_json).read_text())
        prof.setdefault("source", f"implementation artifact {profile_json}")
        if profile_plot:
            dst = out / Path(profile_plot).name
            if Path(profile_plot).resolve() != dst.resolve():
                shutil.copy(profile_plot, dst)
            prof["plot"] = dst.name
    else:
        try:
            from .profiles import analyse_model
            prof = analyse_model(est, meta, out / "learned_vs_ks_profiles.png")
            prof["plot"] = "learned_vs_ks_profiles.png"
        except Exception as e:  # noqa: BLE001
            prof = {"error": f"{e}; pass --profile-json/--profile-plot from the implementation"}

    latency = {"status": "skipped", "error": "--skip-latency given"} if args.skip_latency else measure_latency(est, meta, table.feature_names, args.extractor, args.latency_audio or [],
                              args.latency_url or [], args.latency_runs, args.latency_duration,
                              args.latency_format)

    metrics = {
        "spec": crit.SPEC_PATH,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "inputs": {"model": str(args.model), "features": str(feat_path), "split_source": split_src},
        "split": {"counts": counts, "test_n": int(test.sum()), "test_tracks_sha256": th, "leakage": leak},
        "model": rep_model,
        "ks_baseline": rep_ks,
        "latency": latency,
        "profiles": prof,
        "artifacts": arts,
        "feature_tests": {"junit": str(args.junit) if args.junit else None},
    }
    return metrics


def write_summary(metrics: dict, results, path: Path) -> None:
    """Human-readable markdown summary next to metrics.json."""
    m, k = metrics["model"], metrics["ks_baseline"]
    lat = metrics["latency"]
    prof = metrics.get("profiles", {})
    lines = [
        "# keyclf acceptance report", "",
        f"Spec: `{metrics['spec']}` · generated {metrics['generated_at']}", "",
        f"Inputs: model `{metrics['inputs']['model']}`, features `{metrics['inputs']['features']}`, "
        f"split from {metrics['inputs']['split_source']}; split counts {metrics['split']['counts']}", "",
        "## §10 report (held-out test split)", "",
        "| metric | ML model | K-S baseline |", "|---|---|---|",
        *[f"| {name} | {m[key]:.4f} | {k[key]:.4f} |" for name, key in (
            ("exact accuracy", "accuracy"), ("top-3 accuracy", "top3_accuracy"),
            ("log loss", "log_loss"), ("related-key accuracy", "related_key_accuracy"))],
        "", "Confusion matrices: `confusion_matrix_model.{png,csv}`, `confusion_matrix_ks.{png,csv}`.",
        f"K-S log loss uses softmax(correlation / T), T={k.get('softmax_temperature'):.4g} fit on "
        f"{k.get('temperature_fit_on')} split (never test).", "",
        "## Latency (upload-to-result)", "",
        (f"mean {lat['mean_s']:.3f}s, max {lat['max_s']:.3f}s, first {lat['first_s']:.3f}s — {lat['audio']}"
         if lat.get("status") == "ok" else f"not measured: {lat.get('error')}"), "",
        "## Learned vs K-S profiles", "",
        (f"Pearson r major = {prof['corr_major']:.3f}, minor = {prof['corr_minor']:.3f} (plot `{prof['plot']}`); "
         "spec states no threshold." if "corr_major" in prof else f"not produced: {prof.get('error')}"), "",
        "## §9 acceptance criteria", "", "| status | criterion | spec | detail |", "|---|---|---|---|",
        *[f"| {r.status} | {r.spec_text} | {r.spec_ref} | {r.detail} |" for r in results],
    ]
    path.write_text("\n".join(lines) + "\n")


def criteria_only(metrics_path: Path, junit: str | None) -> int:
    """Re-check §9 criteria from an existing metrics.json; print PASS/FAIL per criterion."""
    metrics = json.loads(metrics_path.read_text())
    results = crit.evaluate_criteria(metrics, metrics_path.parent, junit)
    print(crit.format_results(results))
    return 0 if crit.all_pass(results) else 1


def build_parser() -> argparse.ArgumentParser:
    """CLI definition."""
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--model", default="models/model.joblib", help="spec §12 model artifact")
    a.add_argument("--features", help="feature cache file (default: the single file in --features-dir)")
    a.add_argument("--features-dir", default="data/features", help="spec §12 feature cache directory")
    a.add_argument("--label-col"), a.add_argument("--track-col"), a.add_argument("--split-col")
    a.add_argument("--feature-cols", help="comma-separated 36 feature columns in spec §7 order")
    a.add_argument("--split-file", help="JSON {split: [track ids]} or CSV track_id,split")
    a.add_argument("--out", default="reports", help="output directory for metrics.json etc.")
    a.add_argument("--extractor", default=DEFAULT_EXTRACTOR, help="module:function for latency timing")
    a.add_argument("--latency-audio", action="append", help="audio file(s) to time (repeatable)")
    a.add_argument("--latency-url", action="append",
                   help="audio URL(s) streamed to a temp file, timed, then deleted (spec §6)")
    a.add_argument("--latency-runs", type=int, default=3)
    a.add_argument("--latency-duration", type=float, default=DEFAULT_LATENCY_DURATION_S,
                   help="length of the synthetic probe clip when no audio/URL is given")
    a.add_argument("--latency-format", choices=["mp3", "wav"], default="mp3")
    a.add_argument("--skip-latency", action="store_true",
                   help="do not time extraction (criterion C3 then FAILS; for tests/debugging only)")
    a.add_argument("--profile-json", help="implementation-produced {corr_major, corr_minor, ...}")
    a.add_argument("--profile-plot", help="implementation-produced learned-vs-K-S plot")
    a.add_argument("--junit", help="pytest JUnit XML containing tests/test_features.py results (§9 line 81)")
    a.add_argument("--criteria-only", metavar="METRICS_JSON", help="only re-check §9 criteria")
    return a


def main(argv=None) -> int:
    """Entry point; returns the process exit code."""
    args = build_parser().parse_args(argv)
    if args.criteria_only:
        return criteria_only(Path(args.criteria_only), args.junit)
    try:
        metrics = run_evaluation(args)
    except HarnessInputError as e:
        print(f"ERROR (cannot evaluate): {e}", file=sys.stderr)
        return 2
    out = Path(args.out)
    results = crit.evaluate_criteria(metrics, out, args.junit)
    metrics["criteria"] = crit.results_as_dicts(results)
    metrics["all_section9_pass"] = crit.all_pass(results)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_summary(metrics, results, out / "summary.md")
    print(crit.format_results(results))
    print(f"\nwrote {out / 'metrics.json'} and {out / 'summary.md'}")
    return 0 if metrics["all_section9_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
