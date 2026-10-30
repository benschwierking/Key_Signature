"""Thin wrapper: ``python -m eval.check_criteria reports/metrics.json [--junit X]``.

Equivalent to ``python -m eval.evaluate --criteria-only reports/metrics.json``.
Prints PASS/FAIL per spec §9 criterion with the spec text; exits 1 on any FAIL.
"""
import argparse
import sys
from pathlib import Path

from .evaluate import criteria_only

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("metrics", nargs="?", default="reports/metrics.json")
    p.add_argument("--junit")
    a = p.parse_args()
    sys.exit(criteria_only(Path(a.metrics), a.junit))
