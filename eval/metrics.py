"""Metric computations for spec §9 / §10 (24-class key labels, canonical order).

All functions take integer canonical key indices (see ``eval.keys``) and, where
needed, an (n, 24) probability matrix whose columns are in canonical order.
"""
from __future__ import annotations

import numpy as np

from .keys import N_KEYS, related_keys

EPS = 1e-15


def accuracy(y_true, y_pred) -> float:
    """Exact-key accuracy (spec §9 bullet 1, §10)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(y_true == y_pred)) if len(y_true) else float("nan")


def top_k_accuracy(y_true, proba, k: int = 3) -> float:
    """Fraction of rows whose true key is among the k most probable (spec §9 bullet 2: top 3)."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    # stable tie-breaking: argsort on -p with mergesort keeps lower index first
    topk = np.argsort(-proba, axis=1, kind="mergesort")[:, :k]
    return float(np.mean([y in row for y, row in zip(y_true, topk)]))


def log_loss_24(y_true, proba) -> float:
    """Cross-entropy over the full 24-class space (spec §10 "log loss").

    Rows are renormalised; probabilities are clipped to [EPS, 1] so classes the
    model never saw in training (probability 0) give a large but finite loss.
    """
    y_true = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    p = p / np.clip(p.sum(axis=1, keepdims=True), EPS, None)
    pt = np.clip(p[np.arange(len(y_true)), y_true], EPS, 1.0)
    return float(-np.mean(np.log(pt)))


def related_key_accuracy(y_true, y_pred) -> float:
    """Spec §10: prediction is exact OR relative major/minor OR perfect fifth."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean([int(p) in related_keys(int(t)) for t, p in zip(y_true, y_pred)]))


def confusion_matrix_24(y_true, y_pred) -> np.ndarray:
    """24x24 counts, rows = true key, cols = predicted key (canonical order)."""
    cm = np.zeros((N_KEYS, N_KEYS), dtype=int)
    for t, p in zip(np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)):
        cm[t, p] += 1
    return cm


def full_report(y_true, proba) -> dict:
    """All spec §10 report metrics for one system from (n, 24) probabilities."""
    proba = np.asarray(proba, dtype=float)
    y_pred = proba.argmax(axis=1)
    return {
        "n": int(len(y_true)),
        "accuracy": accuracy(y_true, y_pred),
        "top3_accuracy": top_k_accuracy(y_true, proba, 3),
        "log_loss": log_loss_24(y_true, proba),
        "related_key_accuracy": related_key_accuracy(y_true, y_pred),
        "confusion_matrix": confusion_matrix_24(y_true, y_pred).tolist(),
    }


REPORT_KEYS = ("accuracy", "top3_accuracy", "log_loss", "related_key_accuracy", "confusion_matrix")
