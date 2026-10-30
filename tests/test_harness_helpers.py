"""Harness self-tests: key relations, K-S baseline, metrics, dataset/split helpers.

These need NO keyclf implementation and must pass today. They protect the
judge itself: a wrong relative-key or K-S implementation would silently
mis-score spec §9 / §10.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from eval import keys as K
from eval.dataset import (HarnessInputError, find_feature_file, leakage, load_features,
                          normalise_split, predict_proba_24, resolve_split)
from eval.ks_baseline import KK_MAJOR, KK_MINOR, fit_temperature, key_templates, ks_predict, ks_proba, ks_scores
from eval.metrics import (accuracy, confusion_matrix_24, full_report, log_loss_24, related_key_accuracy,
                          top_k_accuracy)
from eval.profiles import learned_profiles, pearson, rotate_to_tonic

pytestmark = pytest.mark.harness


# ------------------------------------------------------------- labels (§1: 24 classes)
def test_24_canonical_classes():
    # Spec §1: "24 classes: 12 major + 12 minor".
    assert K.N_KEYS == 24 and len(set(K.KEY_NAMES)) == 24
    assert K.KEY_NAMES[0] == "C major" and K.KEY_NAMES[9] == "A major" and K.KEY_NAMES[21] == "A minor"


@pytest.mark.parametrize("label,expected", [
    ("C major", "C major"), ("A minor", "A minor"), ("Eb minor", "D# minor"), ("D# minor", "D# minor"),
    ("Bb major", "A# major"), ("F# maj", "F# major"), ("c:min", "C minor"), ("Bbm", "A# minor"),
    ("D♭ major", "C# major"), ("Cb major", "B major"), ("E# minor", "F minor"), ("G", "G major"),
    ("a", "A minor"), ("  ab   minor ", "G# minor"), ("CM", "C major"), (21, "A minor"), ("0", "C major"),
])
def test_parse_key(label, expected):
    # GiantSteps labels such as "Eb minor" must fold enharmonically into the 24 classes.
    assert K.key_name(K.parse_key(label)) == expected


@pytest.mark.parametrize("bad", ["H major", "C lydian", "", "24", 24, -1, "major"])
def test_parse_key_rejects_garbage(bad):
    with pytest.raises(ValueError):
        K.parse_key(bad)


# ------------------------------------------------------------- relations (§3, §10)
@pytest.mark.parametrize("key,rel", [
    ("C major", "A minor"),   # spec §3's own example
    ("A minor", "C major"), ("G major", "E minor"), ("F major", "D minor"), ("Eb major", "C minor"),
    ("F# minor", "A major"), ("C# minor", "E major"), ("B major", "G# minor"), ("D minor", "F major"),
])
def test_relative_major_minor(key, rel):
    # Spec §3: "relative major/minor (e.g. C major and A minor) as related".
    assert K.relative_key(K.parse_key(key)) == K.parse_key(rel)


def test_relative_is_an_involution_and_switches_mode():
    for k in range(24):
        r = K.relative_key(k)
        assert K.relative_key(r) == k and K.mode_of(r) != K.mode_of(k)


@pytest.mark.parametrize("key,above,below", [
    ("C major", "G major", "F major"), ("A minor", "E minor", "D minor"),
    ("F# major", "C# major", "B major"), ("Bb minor", "F minor", "Eb minor"), ("B major", "F# major", "E major"),
])
def test_perfect_fifth_neighbours(key, above, below):
    # Spec §10: "perfect fifth" — same mode, tonic a fifth above or below (direction unspecified -> both).
    up, down = K.fifth_keys(K.parse_key(key))
    assert (up, down) == (K.parse_key(above), K.parse_key(below))


def test_related_set_is_exact_relative_and_fifths_only():
    # Spec §10: related = exact OR relative major/minor OR perfect fifth — and nothing else.
    c = K.parse_key("C major")
    assert {K.key_name(k) for k in K.related_keys(c)} == {"C major", "A minor", "G major", "F major"}
    assert not K.is_related(c, K.parse_key("C minor"))   # parallel minor is NOT in the spec's definition
    assert not K.is_related(c, K.parse_key("D major"))   # whole step
    assert not K.is_related(c, K.parse_key("E minor"))   # relative of the fifth is not listed
    assert all(len(K.related_keys(k)) == 4 for k in range(24))


# ------------------------------------------------------------- K-S baseline (§9 line 83)
def test_ks_templates_are_rotations():
    t = key_templates()
    assert t.shape == (24, 12)
    np.testing.assert_array_equal(t[0], KK_MAJOR)
    np.testing.assert_array_equal(t[12], KK_MINOR)
    assert t[7].argmax() == 7 and t[21].argmax() == 9  # G major tonic on G, A minor tonic on A


def test_ks_recovers_every_key_from_its_own_profile():
    # Feeding each key's own (rotated) profile must return that key (perfect template match, r = 1).
    pred = ks_predict(key_templates())
    np.testing.assert_array_equal(pred, np.arange(24))
    np.testing.assert_allclose(ks_scores(key_templates()).diagonal(), 1.0)


def test_ks_on_diatonic_histograms():
    # A flat C-major-scale histogram should be classified as C major or its relative A minor.
    h = np.zeros(12); h[[0, 2, 4, 5, 7, 9, 11]] = 1
    h[0] += 1; h[7] += 0.5  # tonic + dominant emphasis
    assert K.key_name(int(ks_predict(h)[0])) == "C major"
    h2 = np.zeros(12); h2[[9, 11, 0, 2, 4, 5, 7]] = 1; h2[9] += 1; h2[4] += 0.5
    assert K.key_name(int(ks_predict(h2)[0])) == "A minor"


def test_ks_scale_invariant_and_silence_safe():
    rng = np.random.default_rng(1)
    h = rng.random(12)
    np.testing.assert_allclose(ks_scores(h), ks_scores(h * 37.0))
    assert np.all(ks_scores(np.zeros(12)) == 0)
    p = ks_proba(np.zeros((1, 12)))
    np.testing.assert_allclose(p, 1 / 24)


def test_ks_proba_rows_sum_to_one_and_preserve_ranking():
    h = np.random.default_rng(2).random((5, 12))
    for T in (0.01, 1.0):
        p = ks_proba(h, T)
        np.testing.assert_allclose(p.sum(1), 1.0)
        np.testing.assert_array_equal(p.argmax(1), ks_predict(h))


def test_fit_temperature_prefers_sharp_when_templates_are_right():
    h = key_templates()
    y = np.arange(24)
    assert fit_temperature(h, y) < 0.1


# ------------------------------------------------------------- metrics (§9, §10)
def test_accuracy_and_topk():
    y = np.array([0, 1, 2, 3])
    p = np.zeros((4, 24))
    p[0, 0] = 0.9; p[1, 5] = 0.5; p[1, 1] = 0.3; p[2, 3] = 0.5; p[2, 4] = 0.3; p[2, 5] = 0.15; p[2, 2] = 0.05
    p[3, 3] = 1.0
    assert accuracy(y, p.argmax(1)) == 0.5
    assert top_k_accuracy(y, p, 3) == 0.75   # row 2's true key ranks 4th
    assert top_k_accuracy(y, p, 1) == 0.5


def test_log_loss_matches_sklearn_and_handles_unseen_classes():
    from sklearn.metrics import log_loss

    rng = np.random.default_rng(3)
    p = rng.random((30, 24)); p /= p.sum(1, keepdims=True)
    y = rng.integers(0, 24, 30)
    assert np.isclose(log_loss_24(y, p), log_loss(y, p, labels=np.arange(24)))
    q = np.zeros((1, 24)); q[0, 0] = 1.0
    assert np.isfinite(log_loss_24([5], q)) and log_loss_24([5], q) > 30  # clipped, finite
    assert np.isclose(log_loss_24([0], q), 0.0)


def test_related_key_accuracy():
    cm, am, gm, fm, dm = (K.parse_key(x) for x in ("C major", "A minor", "G major", "F major", "D major"))
    cmin = K.parse_key("C minor")
    y_true = [cm, cm, cm, cm, cm, cm]
    y_pred = [cm, am, gm, fm, dm, cmin]
    assert related_key_accuracy(y_true, y_pred) == pytest.approx(4 / 6)


def test_confusion_matrix_24():
    cm = confusion_matrix_24([0, 0, 21], [0, 9, 0])
    assert cm.shape == (24, 24) and cm.sum() == 3 and cm[0, 0] == 1 and cm[0, 9] == 1 and cm[21, 0] == 1


def test_full_report_keys():
    p = np.eye(24)[[0, 1, 2]]
    r = full_report([0, 1, 3], p)
    assert set(r) >= {"accuracy", "top3_accuracy", "log_loss", "related_key_accuracy", "confusion_matrix"}
    assert r["accuracy"] == pytest.approx(2 / 3)


# ------------------------------------------------------------- dataset / split (§10, §12)
def _table(tmp_path, rows, name="feats.csv"):
    df = pd.DataFrame(rows)
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def _rows(n, split=True):
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        r = {"track_id": f"t{i // 2}", "key": K.KEY_NAMES[i % 24]}
        if split:
            r["split"] = ["train", "validation", "test"][(i // 2) % 3]
        r.update({f"f{j}": float(v) for j, v in enumerate(rng.random(36))})
        out.append(r)
    return out


def test_load_features_and_split_column(tmp_path):
    t = load_features(_table(tmp_path, _rows(12)))
    assert t.X.shape == (12, 36) and t.split is not None and set(t.split) == {"train", "val", "test"}
    split, src = resolve_split(t)
    assert "column" in src and not leakage(t.tracks, split)


def test_leakage_detected():
    tracks = np.array(["a", "a", "b"])
    assert leakage(tracks, np.array(["train", "test", "test"])) == {"a": ["test", "train"]}
    assert leakage(tracks, np.array(["train", "train", "test"])) == {}


def test_split_from_file_and_artifact(tmp_path):
    t = load_features(_table(tmp_path, _rows(12, split=False)))
    sf = tmp_path / "split.json"
    sf.write_text(json.dumps({"train": ["t0", "t1", "t2"], "val": ["t3"], "test": ["t4", "t5"]}))
    split, _ = resolve_split(t, sf)
    assert list(split[-4:]) == ["test"] * 4
    split2, src = resolve_split(t, None, {"test_track_ids": ["t5"]})
    assert (split2 == "test").sum() == 2 and "artifact" in src
    with pytest.raises(HarnessInputError):
        resolve_split(t)  # never invents a split
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"train": ["t0"], "test": ["t0"]}))
    with pytest.raises(HarnessInputError, match="leakage"):
        resolve_split(t, bad)


def test_wrong_feature_count_rejected(tmp_path):
    rows = _rows(4)
    for r in rows:
        r.pop("f35")
    with pytest.raises(HarnessInputError, match="36"):
        load_features(_table(tmp_path, rows))


def test_find_feature_file(tmp_path):
    d = tmp_path / "data" / "features"
    d.mkdir(parents=True)
    with pytest.raises(HarnessInputError):
        find_feature_file(d)
    (d / "f.parquet").write_bytes(b"")
    assert find_feature_file(d).name == "f.parquet"
    (d / "g.csv").write_text("")
    with pytest.raises(HarnessInputError):
        find_feature_file(d)


def test_normalise_split_aliases():
    assert [normalise_split(s) for s in ("Train", "valid", "VALIDATION", "held_out", "test")] == \
        ["train", "val", "val", "test", "test"]


def test_predict_proba_24_maps_string_classes_to_canonical():
    from sklearn.linear_model import LogisticRegression

    X = np.random.default_rng(0).random((40, 36))
    y = np.array(["Eb minor", "C major", "A minor", "G major"] * 10)
    X[:, 0] += (y == "C major") * 5
    est = LogisticRegression(max_iter=500).fit(X, y)
    p = predict_proba_24(est, X)
    assert p.shape == (40, 24)
    np.testing.assert_allclose(p.sum(1), 1.0)
    assert set(np.nonzero(p.sum(0))[0]) == {K.parse_key(k) for k in set(y)}
    assert p[y == "C major"].argmax(1).tolist() == [0] * 10


# ------------------------------------------------------------- profiles (§9 line 82)
def test_rotate_to_tonic():
    w = np.arange(12)
    assert rotate_to_tonic(w, 7)[0] == 7 and rotate_to_tonic(w, 7)[5] == 0


def test_learned_profiles_recover_ks_shape():
    # If a model's histogram weights were exactly the rotated K-S templates, the learned
    # profiles must correlate perfectly with K-S (r = 1). No threshold is applied anywhere else.
    coef = np.zeros((24, 36))
    coef[:, 24:] = key_templates()
    maj, mnr = learned_profiles(coef, np.arange(24))
    assert pearson(maj, KK_MAJOR) == pytest.approx(1.0) and pearson(mnr, KK_MINOR) == pytest.approx(1.0)
