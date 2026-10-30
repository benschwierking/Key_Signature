"""Time the actual research extractor and selected model without a warm-up."""

import argparse
import json
import tempfile
import time
from pathlib import Path

import joblib

from eval.evaluate import synth_song
from keyclf.app import prepare_input, probability_24


def main():
    """Measure three actual inference calls on a temporary 120-second MP3."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    artifact = joblib.load(args.model)
    times = []
    with tempfile.TemporaryDirectory(prefix="keyclf-research-latency-") as temporary:
        audio = synth_song(Path(temporary) / "probe", 120)
        for _ in range(3):
            start = time.perf_counter()
            _, vector = prepare_input(audio, artifact)
            probability = probability_24(artifact["model"], vector, artifact)
            times.append(time.perf_counter() - start)
            assert probability.shape == (24,)
    report = {"status": "ok", "audio": "synthetic 120-second MP3", "runs_s": times,
              "first_s": times[0], "max_s": max(times), "mean_s": sum(times) / len(times),
              "warmup_calls": 0, "model_loading_timed": False,
              "raw_audio_retained": False, "below_ten_seconds": max(times) < 10}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
