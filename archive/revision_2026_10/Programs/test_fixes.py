"""
test_fixes.py — regression tests for P1.2, P1.3, P1.5, P1.6, P1.7, P1.8.
Run:  pytest -v test_fixes.py
"""
import os
import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

import image_processing_pipeline as P


def road(**kw):
    base = dict(name="Test Rd", length=500.0, width=7.0, lanes=3,
                current_cars=60, capacity=100, avg_speed=30,
                congestion_times="7-9 AM", restrictions=[], special_events=[],
                has_crosswalk=True, has_bikelane=False, condition="Good")
    base.update(kw)
    return base


# ── P1.3: detections change the ranking ──────────────────────
def test_detection_changes_top_ranking():
    r = road()
    _, _, without = P.analyze_traffic(r, [])
    _, reasons, with_turn = P.analyze_traffic(r, [("Left Turn Arrow", 0.9)])
    assert without != with_turn
    assert "Add dedicated turn lanes" in with_turn
    assert any("Detected" in x for x in reasons)


def test_detection_multiplier_values():
    rules = P.triggered_detection_rules([("Left Turn Arrow", .9), ("Right Turn Arrow", .8)])
    assert len(rules) == 1                                   # fires once, not x1.21
    assert P.detection_multiplier("Add turn arrows", rules) == pytest.approx(1.1)
    rules = P.triggered_detection_rules([("Stop Line", .7)])
    assert P.detection_multiplier("Add pedestrian signals", rules) == pytest.approx(1.05)
    rules = P.triggered_detection_rules([("Double Solid Yellow Line", .7)])
    assert P.detection_multiplier("Convert shoulder to travel lane", rules) == pytest.approx(0.9)
    assert P.detection_multiplier("Add roundabout", rules) == 1.0


def test_detection_is_soft_not_a_gate():
    # a detection must never remove a candidate that restrictions allow
    r = road()
    _, _, a = P.analyze_traffic(r, [])
    _, _, b = P.analyze_traffic(r, [("Stop Line", .9), ("Double Solid Yellow Line", .9)])
    assert len(a) == len(b)


# ── P1.5: congestion index is within [0, 1] and classes unchanged ──
def test_congestion_normalised_range_and_thresholds():
    # avg_speed=1 (not 0: the original code treats 0 as "missing", see notes)
    worst = road(current_cars=200, capacity=100, avg_speed=1,
                 special_events=["a", "b", "c"])
    level, reasons, _ = P.analyze_traffic(worst, [])
    assert level == "Severely Congested"
    score = float(reasons[0].split(":")[1].split("/")[0])
    assert 0.99 <= score <= 1.0                       # worst case now tops out at 1.0
    assert P.CONGESTED_THRESHOLD == pytest.approx(0.3333, abs=1e-3)
    assert P.SEVERE_THRESHOLD == pytest.approx(0.5833, abs=1e-3)


def test_congestion_classification_matches_original_thresholds():
    # raw 0.45 -> Congested under the old 0.4/0.7 rule; must stay Congested
    r = road(current_cars=90, capacity=100, avg_speed=50,
             congestion_times=None, special_events=[])        # raw = 0.45
    assert P.analyze_traffic(r, [])[0] == "Congested"


# ── P1.6: exact match vs per-label ───────────────────────────
def test_exact_match_is_stricter_than_per_label():
    y_true = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 1, 0]], "float32")
    y_pred = np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0]], "float32")
    em, pl = P.ExactMatchAccuracy(threshold=0.5), tf.keras.metrics.BinaryAccuracy(threshold=0.5)
    em.update_state(y_true, y_pred); pl.update_state(y_true, y_pred)
    assert em.result().numpy() == pytest.approx(1 / 3)
    assert pl.result().numpy() == pytest.approx(13 / 15)
    em.reset_state()
    assert em.result().numpy() == 0.0


def test_metric_names():
    names = [m.name for m in P.make_metrics()]
    assert names == ["auc", "per_label_acc", "exact_match_accuracy"]


# ── P1.2: flip swaps the arrow labels ────────────────────────
def test_flip_permutation_swaps_only_arrows():
    perm = P._flip_label_permutation()
    l, r = P.CLASSES.index("Left Turn Arrow"), P.CLASSES.index("Right Turn Arrow")
    assert perm[l] == r and perm[r] == l
    assert all(perm[i] == i for i in range(P.NUM_CLASSES) if i not in (l, r))
    label = np.zeros(P.NUM_CLASSES, "float32"); label[l] = 1
    swapped = label[perm]
    assert swapped[r] == 1 and swapped[l] == 0


def test_naive_mode_is_only_place_randomflip_lives():
    old = P.FLIP_MODE
    try:
        P.FLIP_MODE = "label_swap"
        assert not any("flip" in l.name for l in P.build_augmentation().layers)
        P.FLIP_MODE = "naive"
        assert any("flip" in l.name for l in P.build_augmentation().layers)
    finally:
        P.FLIP_MODE = old


# ── P1.7 / P1.8: loader handles PNG + flags missing annotations ──
@pytest.fixture
def data_dirs(tmp_path, monkeypatch):
    img_d, ann_d = tmp_path / "img", tmp_path / "ann"
    img_d.mkdir(); ann_d.mkdir()
    monkeypatch.setattr(P, "IMAGE_DIR", str(img_d) + "/")
    monkeypatch.setattr(P, "ANNOTATIONS_DIR", str(ann_d) + "/")
    monkeypatch.setattr(P, "MISSING_ANN_LOG", str(tmp_path / "missing.txt"))
    xml = "<annotation><object><name>Stop Line</name></object></annotation>"
    for i in range(10):
        ext = "png" if i % 2 else "jpg"
        Image.new("RGB", (64, 48), (i * 20, 50, 50)).save(img_d / f"im{i}.{ext}")
        if i not in (3, 7):                                   # two images lack XML
            (ann_d / f"im{i}.xml").write_text(xml)
    return tmp_path


def test_missing_annotations_excluded_and_logged(data_dirs):
    paths, labels = P.collect_pairs()
    assert len(paths) == 8 and labels.shape == (8, P.NUM_CLASSES)
    assert (data_dirs / "missing.txt").read_text().split() == ["im3.png", "im7.png"]


def test_missing_annotations_can_be_kept(data_dirs, monkeypatch):
    monkeypatch.setattr(P, "EXCLUDE_UNANNOTATED", False)
    paths, _ = P.collect_pairs()
    assert len(paths) == 10


def test_png_and_jpg_both_decode(data_dirs):
    train_ds, val_ds = P.build_datasets()
    n = sum(int(x.shape[0]) for x, _ in train_ds) + sum(int(x.shape[0]) for x, _ in val_ds)
    assert n == 8                                             # PNGs included, no decode error
    x, _ = next(iter(train_ds))
    assert x.shape[1:] == (224, 224, 3)
