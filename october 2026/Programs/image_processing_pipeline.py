"""
=============================================================
  AI Traffic Optimization System — Multi-Label Edition
  Road-marking detection (EfficientNetV2S, sigmoid multi-label)
  + rule-based traffic-solution ranking.
=============================================================
  USAGE  (run from the folder containing this file)

    python image_processing_pipeline.py --mode cv        # honest performance estimate
    python image_processing_pipeline.py --mode ablate    # flip / fine-tune / resolution ablations
    python image_processing_pipeline.py --mode train     # final model on ALL data
    python image_processing_pipeline.py --mode analyze   # detection + road survey

  EVALUATION PROTOCOL (what the paper should describe)
    • 5-fold iterative-stratified cross-validation (multi-label aware),
      every image is predicted exactly once, by a model that never saw it.
    • Fixed training length — NO early stopping or checkpoint selection on
      the evaluation fold (that would leak the test data into model choice).
    • Metrics: per-class ROC-AUC and average precision with bootstrap 95% CIs,
      macro averages, exact-match accuracy, micro-F1.
    • Baselines evaluated on the SAME folds: class-prevalence predictor,
      HOG + logistic regression, and a frozen-EfficientNetV2S-features +
      logistic-regression linear probe (the honest "is training the head
      worth it?" comparison).
    • Optional repeated CV (--repeats R, different fold assignments) so the
      result does not hinge on one lucky/unlucky split.
    • Near-duplicate audit: images of the same scene landing in both train and
      eval folds inflate every metric; the audit counts such pairs and
      --auto-group / --groups-csv keep them in the same fold.
    • Paired bootstrap tests of the CNN against every baseline / ablation.
    • The deployed model is retrained on all data with the same recipe;
      its expected performance is the CV estimate, not a held-out number.
    • Everything is written to ./results (config, OOF predictions, tables).
=============================================================
"""

import os
import sys
import gc
import json
import random
import argparse
import platform
import collections
import contextlib
import csv
import re
import time
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import xml.etree.ElementTree as ET
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — no display needed
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Suppress noisy TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

# ─────────────────────────────────────────────────────────────
# SECTION 1 — CONFIGURATION  (single source of truth)
# ─────────────────────────────────────────────────────────────

# ── 1a: Class list — ORDER MATTERS.
# The index here maps 1-to-1 with positions in the label vector.
# Never reorder this list after you start training, or all your
# saved labels will silently point to the wrong class.
CLASSES = [
    "Left Turn Arrow",
    "Right Turn Arrow", 
    "Stop Line",
    "Solid White Line",
    "Double Solid Yellow Line",
]
NUM_CLASSES = len(CLASSES)  # 5

# Alternate spellings found in the XML files → canonical class name.
# Straight-or-left arrows are merged into "Left Turn Arrow" (7 objects);
# state this decision in the paper.  "Stright Arrow" (straight only) is
# deliberately NOT mapped: it is not a turn marking.
CLASS_ALIASES = {
    "Stopline":                  "Stop Line",          # 33 objects
    "Stright, Left Arrow":       "Left Turn Arrow",    # 4 objects
    "Straight, Left Turn Arrow": "Left Turn Arrow",    # 3 objects
}

# ── 1b: Directory paths  (override with --data-root / env vars, no editing needed)
DATA_ROOT       = os.environ.get(
    "TRAFFIC_DATA_ROOT", r"C:\Users\fmanw\Desktop\old laptop\traffic\Real Model\data")
IMAGE_DIR       = os.path.join(DATA_ROOT, "Images")
ANNOTATIONS_DIR = os.path.join(DATA_ROOT, "Annotations")
TEST_IMAGES_DIR = os.path.join(DATA_ROOT, "Test Images")
OUTPUT_DIR      = os.path.join(DATA_ROOT, "Modified_Images")
MODEL_PATH      = os.environ.get(
    "TRAFFIC_MODEL_PATH", r"C:\Users\fmanw\Desktop\old laptop\traffic\trained_model_multilabel.keras")
RESULTS_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# ── 1c: Training hyperparameters
IMG_SIZE         = (224, 224)   # EfficientNetV2S native size
BATCH_SIZE       = 32
EPOCHS           = 20           # FIXED training length (no early stopping on eval data)
FINE_TUNE_LAYERS = 0            # 0 = Phase-2 OFF (recommended at this data size);
                                #     >0 = unfreeze that many top backbone layers
FINE_TUNE_EPOCHS = 10           # Phase-2 length when enabled
SEED             = 42
N_FOLDS          = 5
N_BOOT           = 1000         # bootstrap resamples for confidence intervals
PRETRAINED_WEIGHTS = "imagenet" # None = random init (smoke tests only)
REPEATS          = 1            # repeated CV: independent fold assignments (3 is a good paper value)
DUP_THRESHOLD    = 0.97         # tiny-image correlation above which two images count as near-duplicates
AUTO_GROUP       = False        # True: keep near-duplicate clusters inside ONE fold
GROUPS_CSV       = None         # optional CSV (image,group): e.g. same intersection/site => same group
RESUME           = False        # reuse saved out-of-fold CSVs (same settings!) instead of retraining
USE_PROBE        = True         # include the frozen-feature linear-probe baseline
LR_HEAD          = 1e-3         # Phase-1 learning rate
LR_FINE          = 1e-5         # Phase-2 learning rate (very low)
THRESHOLD        = 0.35         # Sigmoid score → positive prediction
VAL_SPLIT        = 0.2          # 80/20 train/val split
AUTOTUNE         = tf.data.AUTOTUNE

# ── 1c-bis: Experiment switches (P1.2, P1.8)
# FLIP_MODE controls horizontal-flip augmentation:
#   "label_swap" : flip the image AND swap Left/Right Turn Arrow labels (correct)
#   "none"       : no horizontal flip at all
#   "naive"      : ORIGINAL BEHAVIOUR — flips the image, leaves labels alone (bug).
#                  Kept only so the before/after AUC comparison is reproducible.
FLIP_MODE           = "label_swap"
# Images with no annotation file / unparseable XML are excluded from training
# instead of being silently labelled "no markings present".
EXCLUDE_UNANNOTATED = True
MISSING_ANN_LOG     = "missing_annotations.txt"

# ── 1c-ter: Congestion index normalisation (P1.5)
# Raw index = 0.5 (load) + 0.3 (speed) + 0.1 (peak times) + 0.3 (events, capped
# at 3) = 1.2 maximum.  We divide by 1.2 so the displayed score is truly 0–1,
# and rescale the thresholds so every classification stays identical.
CONGESTION_RAW_MAX    = 1.2
CONGESTED_THRESHOLD   = 0.4 / CONGESTION_RAW_MAX   # ≈ 0.333
SEVERE_THRESHOLD      = 0.7 / CONGESTION_RAW_MAX   # ≈ 0.583

# ── 1d: Restrictions & Solutions — PRESERVED EXACTLY from original
RESTRICTIONS = [
    "no new lanes", "no widening", "no construction",
    "no road elevation changes", "no underpasses", "no overpasses",
    "no intersection modifications", "no median changes",
    "no curb modifications", "no sidewalk changes",
    "no bus lanes", "no bike lanes", "no HOV lanes",
    "no turning lanes", "no reversible lanes", "no shoulder conversions",
    "no new road markings", "no lane marking changes",
    "no crosswalk modifications", "no arrow modifications",
    "no stop line changes", "no yield markings",
    "no new traffic signals", "no signal timing changes",
    "no stop sign additions", "no yield sign additions",
    "no speed limit changes", "no school zone modifications",
    "no pedestrian signals", "no flashing beacons",
    "no smart traffic systems", "no surveillance cameras",
    "no vehicle detection systems", "no dynamic signage",
    "no license plate readers", "no traffic sensors",
    "no congestion pricing", "no parking changes",
    "no loading zone changes", "no transit priority",
    "no truck restrictions", "no delivery zones",
    "no tree removal", "no landscaping changes",
    "no drainage modifications", "no lighting changes",
    "no sound barrier changes", "no wildlife crossings",
]

SOLUTIONS = {
    "lane_additions": [
        {"name": "Add general purpose lane",  "impact": 0.8, "cost": 0.8, "safety": 0.6, "type": "physical"},
        {"name": "Add auxiliary lane",         "impact": 0.7, "cost": 0.7, "safety": 0.7, "type": "physical"},
        {"name": "Add climbing lane",          "impact": 0.6, "cost": 0.6, "safety": 0.8, "type": "physical"},
    ],
    "lane_conversions": [
        {"name": "Convert to HOV lane",           "impact": 0.7, "cost": 0.5, "safety": 0.7, "type": "physical"},
        {"name": "Convert to bus lane",           "impact": 0.6, "cost": 0.4, "safety": 0.8, "type": "physical"},
        {"name": "Convert to bike lane",          "impact": 0.5, "cost": 0.3, "safety": 0.9, "type": "physical"},
        {"name": "Convert shoulder to travel lane","impact": 0.6, "cost": 0.4, "safety": 0.5, "type": "physical"},
    ],
    "intersection_improvements": [
        {"name": "Add roundabout",               "impact": 0.9, "cost": 0.9, "safety": 0.9, "type": "physical"},
        {"name": "Add dedicated turn lanes",     "impact": 0.8, "cost": 0.7, "safety": 0.8, "type": "physical"},
        {"name": "Add channelization islands",   "impact": 0.7, "cost": 0.6, "safety": 0.8, "type": "physical"},
        {"name": "Add pedestrian refuge island", "impact": 0.5, "cost": 0.5, "safety": 0.9, "type": "physical"},
    ],
    "grade_separations": [
        {"name": "Add pedestrian overpass", "impact": 0.4, "cost": 0.7, "safety": 0.9, "type": "physical"},
        {"name": "Add pedestrian underpass","impact": 0.4, "cost": 0.8, "safety": 0.9, "type": "physical"},
        {"name": "Add vehicle overpass",    "impact": 0.9, "cost": 0.95,"safety": 0.8, "type": "physical"},
    ],
    "signal_optimizations": [
        {"name": "Optimize signal timing",    "impact": 0.7, "cost": 0.3, "safety": 0.7, "type": "operational"},
        {"name": "Add adaptive signals",      "impact": 0.8, "cost": 0.6, "safety": 0.8, "type": "operational"},
        {"name": "Add pedestrian signals",    "impact": 0.5, "cost": 0.4, "safety": 0.9, "type": "operational"},
        {"name": "Coordinate signal systems", "impact": 0.8, "cost": 0.5, "safety": 0.8, "type": "operational"},
    ],
    "marking_improvements": [
        {"name": "Refresh lane markings",    "impact": 0.5, "cost": 0.2, "safety": 0.7, "type": "operational"},
        {"name": "Add turn arrows",          "impact": 0.6, "cost": 0.3, "safety": 0.7, "type": "operational"},
        {"name": "Add bike lane markings",   "impact": 0.5, "cost": 0.3, "safety": 0.8, "type": "operational"},
        {"name": "Add crosswalk markings",   "impact": 0.4, "cost": 0.2, "safety": 0.9, "type": "operational"},
    ],
    "smart_systems": [
        {"name": "Add traffic cameras",        "impact": 0.6, "cost": 0.5, "safety": 0.7, "type": "operational"},
        {"name": "Implement smart parking",    "impact": 0.5, "cost": 0.6, "safety": 0.6, "type": "operational"},
        {"name": "Add variable message signs", "impact": 0.5, "cost": 0.5, "safety": 0.7, "type": "operational"},
        {"name": "Install vehicle detection",  "impact": 0.6, "cost": 0.4, "safety": 0.8, "type": "operational"},
    ],
    "policy_changes": [
        {"name": "Implement congestion pricing","impact": 0.7, "cost": 0.3, "safety": 0.7, "type": "policy"},
        {"name": "Add transit priority",        "impact": 0.6, "cost": 0.4, "safety": 0.8, "type": "policy"},
        {"name": "Create delivery zones",       "impact": 0.5, "cost": 0.2, "safety": 0.7, "type": "policy"},
        {"name": "Adjust parking regulations",  "impact": 0.4, "cost": 0.1, "safety": 0.6, "type": "policy"},
    ],
}

# NOTE: no directories are created at import time any more.  The previous
# version created folders (and could silently generate synthetic training
# data) as a side effect of importing the file.


# ─────────────────────────────────────────────────────────────
# SECTION 2 — ANNOTATION PARSER  (PRESERVED from original)
# Reads Pascal VOC XML files and converts them to the multi-hot
# binary label vector this model actually needs.
# ─────────────────────────────────────────────────────────────

def parse_annotation(xml_path: str, ignored=None):
    """
    Parse a Pascal VOC XML annotation file and return a
    multi-hot label vector of shape (NUM_CLASSES,).

    Multi-hot means each index is independently 0 or 1:
      [1, 0, 1, 0, ..., 1]   ← three markings present
    A single image can and should have multiple 1s.

    Args:
        xml_path : full path to the .xml annotation file
        ignored  : optional collections.Counter; object names that are not
                   one of CLASSES (after alias resolution) are tallied here,
                   so the dataset audit can report what is being ignored.

    Returns:
        np.ndarray of dtype float32, shape (NUM_CLASSES,) — or None if the
        annotation is MISSING or UNPARSEABLE.  (P1.8: previously this case
        returned an all-zero vector, which silently trained the model to say
        "no markings" on every un-annotated image.)  A file that parses fine
        but contains no recognised class is a legitimate all-zero label.
    """
    if not os.path.exists(xml_path):
        return None

    label_vector = np.zeros(NUM_CLASSES, dtype=np.float32)
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            class_name = (obj.find("name").text or "").strip()
            class_name = CLASS_ALIASES.get(class_name, class_name)
            if class_name in CLASSES:
                label_vector[CLASSES.index(class_name)] = 1.0
            elif ignored is not None:
                ignored[class_name] += 1
    except Exception as e:
        print(f"[WARN] Failed to parse {xml_path}: {e}")
        return None

    return label_vector


# ─────────────────────────────────────────────────────────────
# SECTION 3 — SYNTHETIC DATA GENERATOR  (PRESERVED from original)
# Creates PIL-drawn sample images + matching XML files when no
# real training data exists yet.
# ─────────────────────────────────────────────────────────────

def generate_sample_data(num_samples: int = 50):
    """
    Generate synthetic training images and Pascal VOC XML
    annotations so the pipeline can run without real data.
    Images are simple; the goal is to validate the pipeline,
    not to produce a production-quality model.
    """
    print(f"\n[INFO] Generating {num_samples} synthetic training samples...")
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

    for i in range(num_samples):
        img  = Image.new("RGB", (224, 224), color=(100, 150, 100))
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        # Pick 1–4 random classes for this image
        elements   = random.sample(CLASSES, random.randint(1, 4))
        ann_chunks = []

        for element in elements:
            if "Arrow" in element:
                x, y  = random.randint(50, 150), random.randint(50, 150)
                glyph = ("→" if "Right" in element
                         else "←" if "Left" in element
                         else "↗" if "+" in element
                         else "↑")
                draw.text((x, y), glyph, fill=(255, 255, 0))
                ann_chunks.append(
                    f"<object><name>{element}</name>"
                    f"<bndbox><xmin>{x}</xmin><ymin>{y}</ymin>"
                    f"<xmax>{x+20}</xmax><ymax>{y+20}</ymax>"
                    f"</bndbox></object>"
                )
            elif "Line" in element or "Markings" in element:
                y     = random.randint(80, 150)
                color = (255,255,255) if "White" in element else (255,255,0)
                if "Dashed" in element:
                    for xd in range(20, 200, 15):
                        draw.line([(xd, y), (xd+8, y)], fill=color, width=2)
                else:
                    draw.line([(20, y), (200, y)], fill=color, width=3)
                ann_chunks.append(
                    f"<object><name>{element}</name>"
                    f"<bndbox><xmin>20</xmin><ymin>{y-5}</ymin>"
                    f"<xmax>200</xmax><ymax>{y+5}</ymax>"
                    f"</bndbox></object>"
                )
            elif element == "Crosswalk Markings":
                for yw in range(100, 120, 5):
                    draw.line([(50, yw), (150, yw)], fill=(255,255,255), width=2)
                ann_chunks.append(
                    f"<object><name>{element}</name>"
                    f"<bndbox><xmin>50</xmin><ymin>100</ymin>"
                    f"<xmax>150</xmax><ymax>120</ymax>"
                    f"</bndbox></object>"
                )

        img.save(os.path.join(IMAGE_DIR, f"sample_{i}.jpg"))

        with open(os.path.join(ANNOTATIONS_DIR, f"sample_{i}.xml"), "w") as f:
            f.write(
                f'<annotation><filename>sample_{i}.jpg</filename>'
                f'{"".join(ann_chunks)}</annotation>'
            )

    print(f"[INFO] Synthetic data written to {IMAGE_DIR}")


# ─────────────────────────────────────────────────────────────
# SECTION 4 — tf.data PIPELINE  (REBUILT for multi-label)
#
# WHY we can't use image_dataset_from_directory here:
#   That utility infers labels from sub-folder names, producing
#   a single integer or one-hot vector per image.  Our images
#   live in ONE flat folder; their labels come from separate XML
#   files and can be any combination of 20 classes at once.
#   We have to build the dataset manually from (path, label) pairs.
#
# PIPELINE ORDER (critical — do NOT reorder):
#   collect pairs → split → shuffle → map(load+decode) →
#   cache → shuffle again → batch → prefetch
# ─────────────────────────────────────────────────────────────

LAST_AUDIT = {}      # filled by collect_pairs(); saved into results/config.json


def collect_pairs():
    """
    Scan IMAGE_DIR for supported images, parse each matching XML to build a
    multi-hot vector, and return:
      image_paths : list of file paths (str)
      label_matrix: np.ndarray (N, NUM_CLASSES), float32

    Prints a dataset audit (reportable in the paper): images scanned, missing
    annotations, per-class IMAGE counts, and which object names are ignored.
    Raises FileNotFoundError instead of silently fabricating data.
    """
    for d in (IMAGE_DIR, ANNOTATIONS_DIR):
        if not os.path.isdir(d):
            raise FileNotFoundError(
                f"Folder not found: {d}\n"
                f"Fix DATA_ROOT at the top of the file, pass --data-root, or set "
                f"the TRAFFIC_DATA_ROOT environment variable.")

    image_paths  = []
    label_rows   = []
    extensions   = {".jpg", ".jpeg", ".png"}
    unannotated  = []        # image filenames with no usable XML
    all_zero     = 0         # annotated, but no recognised class (valid)
    total_images = 0
    ignored      = collections.Counter()

    for fname in sorted(os.listdir(IMAGE_DIR)):           # sorted → reproducible
        ext = os.path.splitext(fname)[1].lower()
        if ext not in extensions:
            continue

        img_path = os.path.join(IMAGE_DIR, fname)
        xml_path = os.path.join(ANNOTATIONS_DIR, os.path.splitext(fname)[0] + ".xml")

        total_images += 1
        label_vector = parse_annotation(xml_path, ignored)  # multi-hot, or None
        if label_vector is None:
            unannotated.append(fname)
            if EXCLUDE_UNANNOTATED:
                continue                                    # P1.8: don't train on it
            label_vector = np.zeros(NUM_CLASSES, dtype=np.float32)
        elif not label_vector.any():
            all_zero += 1
        image_paths.append(img_path)
        label_rows.append(label_vector)

    if not image_paths:
        raise FileNotFoundError(
            f"No usable annotated images in {IMAGE_DIR} / {ANNOTATIONS_DIR}. "
            f"(Use --synthetic only for a plumbing smoke test.)")

    label_matrix = np.array(label_rows, dtype=np.float32) # (N, NUM_CLASSES)
    n = len(label_matrix)

    # ── Dataset audit — report these numbers in the paper.
    print(f"[INFO] Annotation audit: {total_images} images scanned | "
          f"{len(unannotated)} missing/unparseable XML | "
          f"{all_zero} annotated with no recognised class")
    if unannotated:
        with open(MISSING_ANN_LOG, "w") as f:
            f.write("\n".join(unannotated) + "\n")
        action = "EXCLUDED from training" if EXCLUDE_UNANNOTATED else "kept with all-zero labels"
        print(f"[WARN] {len(unannotated)} image(s) {action}; list → {MISSING_ANN_LOG}")
    print(f"[INFO] Using {n} images.  Per-class image counts (positives / total):")
    for i, c in enumerate(CLASSES):
        print(f"         {c:28s} {int(label_matrix[:, i].sum()):3d} / {n}")
    if ignored:
        top = ", ".join(f"{k} ({v})" for k, v in ignored.most_common())
        print(f"[INFO] Object names ignored (not in CLASSES): {top}")
    weak = [c for i, c in enumerate(CLASSES) if label_matrix[:, i].sum() < N_FOLDS]
    if weak:
        print(f"[WARN] Fewer than {N_FOLDS} positive images for: {', '.join(weak)} "
              f"— per-class metrics for these are not reliable.")

    LAST_AUDIT.clear()
    LAST_AUDIT.update({
        "images_scanned": total_images, "images_used": n,
        "missing_or_unparseable_xml": len(unannotated),
        "annotated_no_recognised_class": all_zero,
        "per_class_image_counts": {c: int(label_matrix[:, i].sum()) for i, c in enumerate(CLASSES)},
        "ignored_object_names": dict(ignored),
        "class_aliases": CLASS_ALIASES,
    })
    return image_paths, label_matrix


def multilabel_stratified_folds(Y, k=None, seed=None, weights=None):
    """
    Iterative stratification for multi-label data (Sechidis et al., 2011),
    simplified.  Rarest labels are distributed first, each item going to the
    fold that still 'wants' the most examples of that label, so every fold
    gets a proportional share of every class.  A plain shuffled split can put
    all positives of a rare class into one fold.

    `weights` (optional, per row) lets a row stand for several images; used by
    group_stratified_folds so a cluster of near-duplicates moves as one unit.
    k / seed are resolved at CALL time, so --folds / --seed take effect.

    Returns an int array fold_of[i] in [0, k).
    """
    k = N_FOLDS if k is None else k
    seed = SEED if seed is None else seed
    rng = np.random.RandomState(seed)
    n, c = Y.shape
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    fold_of = -np.ones(n, dtype=int)
    desired_total = np.full(k, w.sum() / k, dtype=float)
    desired_label = np.tile((Y * w[:, None]).sum(0) / k, (k, 1))      # (k, c)
    unassigned = set(range(n))

    while unassigned:
        idx = np.array(sorted(unassigned))
        counts = Y[idx].sum(0)
        if not (counts > 0).any():
            break
        lab = int(np.argmin(np.where(counts > 0, counts, np.inf)))   # rarest remaining label
        cand = idx[Y[idx, lab] == 1]
        rng.shuffle(cand)
        for i in cand:
            d = desired_label[:, lab]
            best = np.flatnonzero(d == d.max())
            if len(best) > 1:                                # tie → fold needing more images
                t = desired_total[best]
                best = best[t == t.max()]
            f = int(rng.choice(best))
            fold_of[i] = f
            unassigned.discard(int(i))
            desired_total[f] -= w[i]
            desired_label[f] -= Y[i] * w[i]

    rest = sorted(unassigned)                                # rows with no positive label
    rng.shuffle(rest)
    for i in rest:
        f = int(np.argmax(desired_total))
        fold_of[i] = f
        desired_total[f] -= w[i]
    return fold_of


def group_stratified_folds(Y, groups, k=None, seed=None):
    """
    Multi-label stratified folds where every GROUP (same scene / site /
    near-duplicate cluster) lands in exactly one fold.  A group's label vector
    is the union of its members' labels and it is weighted by its size.
    """
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    Yg = np.zeros((len(uniq), Y.shape[1]))
    np.maximum.at(Yg, inv, Y)
    sizes = np.bincount(inv).astype(float)
    fold_g = multilabel_stratified_folds(Yg, k=k, seed=seed, weights=sizes)
    return fold_g[inv]


def fold_splits(fold_of, k=None):
    """[(train_idx, val_idx), ...] for each of the k folds."""
    k = N_FOLDS if k is None else k
    return [(np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)) for f in range(k)]


def _flip_label_permutation():
    """
    Index permutation that swaps 'Left Turn Arrow' <-> 'Right Turn Arrow'
    and leaves every other class in place.  Derived from CLASSES by NAME so
    it stays correct if the class list is ever extended.
    """
    perm = list(range(NUM_CLASSES))
    l = CLASSES.index("Left Turn Arrow")
    r = CLASSES.index("Right Turn Arrow")
    perm[l], perm[r] = r, l
    return perm


def make_datasets(image_paths, label_matrix, train_idx, val_idx, seed=None):
    """
    Build tf.data datasets for explicit index sets.

    The validation set is NOT shuffled, so model.predict(val_ds) rows line up
    with val_idx (needed for out-of-fold predictions).  Pass an empty val_idx
    to train on everything (returns val_ds = None).
    """
    def load_and_preprocess(path, label):
        raw = tf.io.read_file(path)
        # P1.7: dispatch on extension (PNG bytes break decode_jpeg).
        is_png = tf.strings.regex_full_match(tf.strings.lower(path), r".*\.png")
        image  = tf.cond(
            is_png,
            lambda: tf.image.decode_png(raw, channels=3),
            lambda: tf.image.decode_jpeg(raw, channels=3),   # .jpg / .jpeg
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMG_SIZE)
        # Pixels stay in [0,255]; EfficientNetV2 preprocess_input is in the model.
        return image, label

    # P1.2: label-aware horizontal flip, TRAIN ONLY, drawn fresh every epoch
    # (applied after .cache()).  Mirroring a left arrow makes a right arrow,
    # so the two label slots are swapped together with the pixels.
    flip_perm = _flip_label_permutation()

    def maybe_flip(image, label):
        if FLIP_MODE != "label_swap":
            return image, label
        do_flip = tf.random.uniform([]) < 0.5
        image = tf.cond(do_flip, lambda: tf.image.flip_left_right(image), lambda: image)
        label = tf.cond(do_flip, lambda: tf.gather(label, flip_perm), lambda: label)
        return image, label

    def make(idx, training):
        paths  = [str(image_paths[i]) for i in idx]
        labels = np.asarray(label_matrix[idx], dtype=np.float32)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE).cache()
        if training:
            ds = ds.map(maybe_flip, num_parallel_calls=AUTOTUNE)
            ds = ds.shuffle(buffer_size=max(len(idx), 1), seed=seed)
        return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    seed = SEED if seed is None else seed
    train_ds = make(train_idx, True)
    val_ds   = make(val_idx, False) if len(val_idx) else None
    return train_ds, val_ds


def build_datasets():
    """
    Single stratified hold-out split (fold 0 of a k-fold with k = 1/VAL_SPLIT).
    Convenience for quick experiments; for reportable numbers use --mode cv.
    """
    image_paths, label_matrix = collect_pairs()
    k = max(2, int(round(1.0 / VAL_SPLIT)))
    fold_of = multilabel_stratified_folds(label_matrix, k=k, seed=SEED)
    train_idx, val_idx = fold_splits(fold_of, k)[0]
    print(f"[INFO] Train: {len(train_idx)} images | Val: {len(val_idx)} images")
    return make_datasets(image_paths, label_matrix, train_idx, val_idx)


# ─────────────────────────────────────────────────────────────
# SECTION 5 — AUGMENTATION PIPELINE  (inside model graph)
#
# Augmentation lives INSIDE the model rather than in .map().
# This means:
#   • It only activates during model.fit(), never model.predict().
#   • It runs on the GPU when one is available.
#   • No conditional logic needed in the dataset pipeline itself.
# ─────────────────────────────────────────────────────────────

def build_augmentation():
    """
    Keras preprocessing layers tuned for top-down traffic camera footage:
      RandomFlip          — NOT here by default any more: it lives in tf.data
                            (maybe_flip) so arrow labels can be swapped. Only
                            re-added when FLIP_MODE == "naive" (ablation).
      RandomRotation      — camera not perfectly level
      RandomZoom          — varying camera height / mounting distance
      RandomContrast      — glare, shadow, time-of-day variation
      RandomBrightness    — dawn vs. noon vs. overcast
      RandomTranslation   — marking at edge of frame
    """
    aug_layers = []
    if FLIP_MODE == "naive":                # reproduces the original bug
        aug_layers.append(layers.RandomFlip("horizontal"))
    aug_layers += [
        layers.RandomRotation(0.05),        # ±18°
        layers.RandomZoom(0.10),            # ±10%
        layers.RandomContrast(0.20),        # ±20% contrast
        layers.RandomBrightness(0.20),      # ±20% brightness
        layers.RandomTranslation(0.10, 0.10),
    ]
    return keras.Sequential(aug_layers, name="augmentation")


# ─────────────────────────────────────────────────────────────
# SECTION 6 — MODEL ARCHITECTURE  (REBUILT for multi-label)
#
# WHAT CHANGED vs. single-label:
#   output activation : softmax → sigmoid
#     softmax forces probabilities to sum to 1 across classes,
#     which is correct ONLY when exactly one class is true.
#     sigmoid gives each class an independent probability in
#     [0,1], which is exactly what multi-label needs.
#
#   loss : categorical_crossentropy → binary_crossentropy
#     binary_crossentropy computes a separate log-loss for each
#     of the 20 output neurons and averages them.  It treats
#     every class as its own binary (present / not present) task.
#
#   metrics : accuracy (misleading for multi-label) →
#             per-class AUC + subset accuracy (exact match)
#
# BASE MODEL: EfficientNetV2S
#   Chosen over MobileNetV2 because road marking detection needs
#   fine-grained texture discrimination (dashed vs. solid lines,
#   single vs. double, white vs. yellow).  EfficientNetV2S is
#   ~8% more accurate on ImageNet at only 2× the params, and
#   still fast enough for a school-project training run.
# ─────────────────────────────────────────────────────────────

@keras.utils.register_keras_serializable(package="traffic")
class ExactMatchAccuracy(keras.metrics.Metric):
    """
    TRUE subset accuracy (P1.6): an image counts as correct only if EVERY
    label in its multi-hot vector is predicted correctly.

    keras.metrics.BinaryAccuracy is per-label (it averages over all
    N x NUM_CLASSES cells), so with sparse labels it is dominated by the
    easy negatives and looks far higher than exact-match accuracy.
    """

    def __init__(self, threshold=THRESHOLD, name="exact_match_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.matches = self.add_weight(name="matches", shape=(), initializer="zeros")
        self.count   = self.add_weight(name="count",   shape=(), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred >= self.threshold, tf.float32)   # same rule as inference
        match  = tf.reduce_all(tf.equal(y_true, y_pred), axis=1)  # per-image exact match
        self.matches.assign_add(tf.reduce_sum(tf.cast(match, tf.float32)))
        self.count.assign_add(tf.cast(tf.shape(match)[0], tf.float32))

    def result(self):
        return self.matches / tf.maximum(self.count, 1.0)

    def reset_state(self):
        self.matches.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        return {**super().get_config(), "threshold": self.threshold}


def make_metrics():
    """Metrics shared by both training phases. Report the last two side by side."""
    return [
        keras.metrics.AUC(name="auc", multi_label=True),
        keras.metrics.BinaryAccuracy(name="per_label_acc", threshold=THRESHOLD),
        ExactMatchAccuracy(threshold=THRESHOLD, name="exact_match_accuracy"),
    ]


def build_model(augmentation, prior=None):
    """
    Multi-label EfficientNetV2S with two-phase training support.

    Args:
        prior : optional array (NUM_CLASSES,) of training-set class prevalences.
                The output bias is initialised to logit(prior) so the untrained
                model already predicts the base rate instead of 0.5 for every
                class (standard practice for imbalanced sigmoid outputs; it
                also stops all five classes "firing" before any learning).

    Returns:
        model      — full compiled Keras model
        base_model — reference to the backbone (for fine-tuning)
    """

    # ── 6a: Load EfficientNetV2S backbone, pre-trained on ImageNet.
    # include_top=False strips the ImageNet 1000-class head so we
    # can attach our own 20-class sigmoid head.
    base_model = keras.applications.EfficientNetV2S(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights=PRETRAINED_WEIGHTS,
    )

    # ── 6b: Freeze all backbone weights for Phase 1.
    # We only train our new classification head first.
    # This prevents the tiny, randomly-initialized head from
    # sending large gradients back and wrecking the pre-trained
    # feature extractor before it's had a chance to adapt.
    base_model.trainable = False

    # ── 6c: Build the model with the Functional API.
    inputs = keras.Input(shape=IMG_SIZE + (3,), name="image_input")

    # Augmentation — only active during model.fit()
    x = augmentation(inputs)

    # EfficientNetV2 rescales/normalises INSIDE the backbone (Keras default), so
    # pixels stay in [0,255]; preprocess_input is a pass-through kept for clarity.
    x = keras.applications.efficientnet_v2.preprocess_input(x)

    # Frozen backbone feature extraction
    x = base_model(x, training=False)   # training=False → BatchNorm stays frozen

    # GlobalAveragePooling collapses (7×7×1280) → (1280,)
    # This is more parameter-efficient than Flatten and
    # produces a spatially-averaged feature descriptor.
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # BatchNormalization on the pooled features stabilises training
    # of the new head without touching backbone BatchNorm layers.
    x = layers.BatchNormalization(name="head_bn")(x)

    # Dropout for regularisation — prevents co-adaptation of
    # neurons in the dense head on a small dataset.
    x = layers.Dropout(0.3, name="head_dropout")(x)

    # ── CRITICAL CHANGE: sigmoid, not softmax ──────────────────
    # Each of the 20 neurons independently outputs P(class_i present).
    # A road image can be [1,0,1,0,...,1] — multiple 1s are valid.
    if prior is not None:
        p_ = np.clip(np.asarray(prior, dtype=np.float64), 0.01, 0.99)
        bias_init = keras.initializers.Constant(np.log(p_ / (1.0 - p_)))
    else:
        bias_init = "zeros"
    outputs = layers.Dense(
        NUM_CLASSES,
        activation="sigmoid",           # ← multi-label output
        bias_initializer=bias_init,
        name="multi_label_output"
    )(x)

    model = keras.Model(inputs, outputs, name="MultiLabelRoadMarking_EfficientNetV2S")

    # ── 6d: Compile — Phase 1  ─────────────────────────────────
    # binary_crossentropy: treats each label as an independent
    # binary classification problem, averages loss across all 20.
    # AUC with multi_label=True reports the average ROC AUC
    # across all 20 classes — a much better signal than accuracy
    # for imbalanced multi-label problems.
    model.compile(
        optimizer=keras.optimizers.Adam(LR_HEAD),
        loss="binary_crossentropy",                          # ← multi-label loss
        metrics=make_metrics(),
    )

    return model, base_model


# ─────────────────────────────────────────────────────────────
# SECTION 7 — FINE-TUNING  (REBUILT for EfficientNetV2S)
# ─────────────────────────────────────────────────────────────

def fine_tune(model, base_model, train_ds, verbose=0):
    """
    Phase 2 (OPTIONAL, FINE_TUNE_LAYERS > 0): unfreeze the top N backbone
    layers and continue at a very low learning rate.  On ~60 training images
    this lowered AUC in the first experiments, so it is OFF by default and is
    reported as an ablation instead of being assumed to help.
    """
    base_model.trainable = True
    for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(LR_FINE),
        loss="binary_crossentropy",
        metrics=make_metrics(),
    )
    return model.fit(train_ds, epochs=FINE_TUNE_EPOCHS, verbose=verbose, shuffle=False)


def train_model(train_ds, prior=None, epochs=None, verbose=0):
    """
    Fixed-length training that never looks at evaluation data: no
    EarlyStopping / ModelCheckpoint on a validation fold (selecting the best
    epoch on the images you then report on is a leak that inflates results).
    The LR schedule watches TRAINING loss only.

    Returns (model, history) where history is {metric_name: [per-epoch values]}.
    """
    epochs = EPOCHS if epochs is None else epochs
    model, base_model = build_model(build_augmentation(), prior)
    callbacks = [keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=3, min_lr=1e-6)]
    h1 = model.fit(train_ds, epochs=epochs, callbacks=callbacks, verbose=verbose, shuffle=False)
    history = {k: list(v) for k, v in h1.history.items()}
    if FINE_TUNE_LAYERS > 0:
        h2 = fine_tune(model, base_model, train_ds, verbose=verbose)
        for k, v in h2.history.items():
            history.setdefault(k, []).extend(v)
    return model, history


# ─────────────────────────────────────────────────────────────
# SECTION 8 — INFERENCE HELPER
# Replaces the original predict_traffic_elements() with a
# version that works with the new model and sigmoid threshold.
# ─────────────────────────────────────────────────────────────

def predict_traffic_elements(image_path: str, model):
    """
    Run multi-label inference on a single image.

    Returns:
        detected     : list of (class_name, confidence_float) tuples
                       for all classes with sigmoid score > THRESHOLD
        img_rgb      : numpy array (H, W, 3) for visualisation
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"cv2.imread returned None for {image_path}")

        img_rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        in_h, in_w  = int(model.input_shape[1]), int(model.input_shape[2])
        img_resized = cv2.resize(img_rgb, (in_w, in_h))   # cv2 wants (width, height)
        img_batch   = np.expand_dims(img_resized.astype(np.float32), axis=0)
        # Note: EfficientNetV2 preprocess_input is inside the model graph,
        # so we pass raw [0,255] pixel values directly.

        scores   = model.predict(img_batch, verbose=0)[0]   # shape (20,)
        detected = [
            (CLASSES[i], float(scores[i]))
            for i in range(NUM_CLASSES)
            if scores[i] >= THRESHOLD
        ]
        detected.sort(key=lambda x: x[1], reverse=True)
        return detected, img_rgb

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return [], None


# ─────────────────────────────────────────────────────────────
# SECTION 9 — BUSINESS LOGIC  (PRESERVED EXACTLY from original)
# ─────────────────────────────────────────────────────────────

# ── P1.1 / P1.4: constraint registry ─────────────────────────
# Single source of truth for which restrictions are ENFORCED.  Each predicate
# takes a solution dict and returns True if the restriction forbids it.
# All 48 entries in RESTRICTIONS stay catalogued, but only the keys below are
# offered in the survey UI, so the interface never promises what it can't do.
RESTRICTION_PREDICATES = {
    "no new lanes":             lambda s: "lane" in s["name"].lower() and "add" in s["name"].lower(),
    "no construction":          lambda s: s["type"] == "physical" and s["cost"] > 0.4,
    "no overpasses":            lambda s: "overpass"  in s["name"].lower(),
    "no underpasses":           lambda s: "underpass" in s["name"].lower(),
    "no bike lanes":            lambda s: "bike lane" in s["name"].lower(),
    "no bus lanes":             lambda s: "bus lane"  in s["name"].lower(),
    # P1.1: was `"HOV" in s["name"].lower()` — an uppercase needle inside a
    # lowercased haystack can never match, so the restriction was a no-op.
    "no HOV lanes":             lambda s: "hov"       in s["name"].lower(),
    "no signal timing changes": lambda s: "signal timing" in s["name"].lower(),
    "no pedestrian signals":    lambda s: "pedestrian signal" in s["name"].lower(),
}
assert set(RESTRICTION_PREDICATES) <= set(RESTRICTIONS), "predicate for unknown restriction"
IMPLEMENTED_RESTRICTIONS = [r for r in RESTRICTIONS if r in RESTRICTION_PREDICATES]


def filter_solutions(solutions, restrictions):
    """Return the solutions not forbidden by any active, implemented restriction."""
    active = [r for r in restrictions if r in RESTRICTION_PREDICATES]
    return [s for s in solutions if not any(RESTRICTION_PREDICATES[r](s) for r in active)]


# ── P1.3: detected markings as context modifiers (soft, never hard gates) ──
# Each rule fires at most once (so Left+Right arrows together give x1.1, not
# x1.21) and multiplies the base score of the named solutions.
DETECTION_RULES = [
    {"id": "turn_arrows",
     "any_of": ["Left Turn Arrow", "Right Turn Arrow"],
     "factors": {"Add dedicated turn lanes": 1.1, "Add turn arrows": 1.1},
     "note": "turn-movement context → favour turn-lane / arrow treatments"},
    {"id": "stop_line",
     "any_of": ["Stop Line"],
     "factors": {"Add pedestrian signals": 1.05, "Optimize signal timing": 1.05},
     "note": "intersection context → favour pedestrian signals / signal timing"},
    {"id": "double_solid_yellow",
     "any_of": ["Double Solid Yellow Line"],
     "factors": {"Convert shoulder to travel lane": 0.9},
     "note": "no-passing roadway, narrower margin → penalise shoulder conversion"},
]


def triggered_detection_rules(detected_elements):
    """detected_elements: list of (class_name, confidence) from predict_traffic_elements."""
    names = {n for n, _ in (detected_elements or [])}
    return [r for r in DETECTION_RULES if names & set(r["any_of"])]


def detection_multiplier(solution_name, rules):
    m = 1.0
    for r in rules:
        m *= r["factors"].get(solution_name, 1.0)
    return m


def get_input(prompt, input_type=str, options=None, default=None):
    while True:
        try:
            response = input(prompt).strip()
            if not response and default is not None:
                return default
            if response.upper() == "N/A":
                return None
            if input_type == bool:
                if response.lower() in ("y", "yes"):
                    return True
                elif response.lower() in ("n", "no"):
                    return False
                else:
                    print("Please answer 'y', 'n', or 'N/A'")
                    continue
            if options:
                if response.lower() in [o.lower() for o in options]:
                    return next(o for o in options if o.lower() == response.lower())
                print(f"Please choose from: {', '.join(options)} or 'N/A'")
                continue
            return input_type(response)
        except ValueError:
            print(f"Please enter a valid {input_type.__name__} or 'N/A'")


def get_road_details():
    print("\n=== Road Information Survey ===")
    print("(For each question, answer or type 'N/A' if not applicable)\n")
    print("Available Restrictions (enforced):")
    for i in range(0, len(IMPLEMENTED_RESTRICTIONS), 4):
        row = IMPLEMENTED_RESTRICTIONS[i:i+4]
        print("  ".join(f"{i+j+1}. {item:<30}" for j, item in enumerate(row)))
    print()

    details = {
        "name":            get_input("1. Road name: ", default="Unnamed Road"),
        "length":          get_input("2. Road length (meters): ", float),
        "width":           get_input("3. Road width (meters): ", float),
        "lanes":           get_input("4. Number of lanes: ", int),
        "current_cars":    get_input("5. Current cars on road: ", int),
        "capacity":        get_input("6. Maximum car capacity: ", int),
        "avg_speed":       get_input("7. Average speed (km/h): ", int),
        "congestion_times":get_input("8. Peak congestion times (e.g., '7-9 AM, 4-6 PM'): "),
        "restrictions":    [],
        "special_events":  [],
        "has_crosswalk":   get_input("11. Pedestrian crossing? (y/n/N/A): ", bool),
        "has_bikelane":    get_input("12. Bike lane? (y/n/N/A): ", bool),
        "condition":       get_input("13. Road condition (Good/Fair/Poor/N/A): ", str, ["Good","Fair","Poor"]),
    }

    if get_input("Any special events? (y/n/N/A): ", bool):
        raw = get_input("10. Special events (comma separated): ", str)
        details["special_events"] = [e.strip() for e in raw.split(",") if e.strip()]

    if get_input("9. Any restrictions? (y/n/N/A): ", bool):
        print("\nSelect restrictions by number (comma separated):")
        while True:
            choices = input("Enter restriction numbers: ").strip()
            if choices.upper() == "N/A":
                break
            try:
                selected = [int(c.strip()) for c in choices.split(",") if c.strip()]
                valid    = [i for i in selected if 1 <= i <= len(IMPLEMENTED_RESTRICTIONS)]
                details["restrictions"] = [IMPLEMENTED_RESTRICTIONS[i-1] for i in valid]
                break
            except ValueError:
                print(f"Enter numbers between 1–{len(IMPLEMENTED_RESTRICTIONS)} separated by commas")

    return details


def calculate_solution_score(solution, congestion_level, road_info):
    impact_w = 0.5
    safety_w = 0.3
    cost_w   = 0.2
    if congestion_level == "Severely Congested":
        impact_w, safety_w = 0.6, 0.2
    elif congestion_level == "Congested":
        impact_w, safety_w = 0.55, 0.25

    score = (solution["impact"] * impact_w
             + solution["safety"] * safety_w
             + (1 - solution["cost"]) * cost_w)

    if road_info["has_crosswalk"]  and "pedestrian" in solution["name"].lower(): score *= 1.1
    if road_info["has_bikelane"]   and "bike"       in solution["name"].lower(): score *= 1.1
    if road_info["lanes"] and road_info["lanes"] <= 2 and "lane" in solution["name"].lower(): score *= 1.15
    if road_info["congestion_times"]               and "signal"    in solution["name"].lower(): score *= 1.1
    return score


def analyze_traffic(road_info, detected_elements):
    congestion_score = 0.0

    if road_info["current_cars"] and road_info["capacity"]:
        congestion_score += min(road_info["current_cars"] / road_info["capacity"], 1) * 0.5
    if road_info["avg_speed"]:
        congestion_score += max(0, (50 - road_info["avg_speed"]) / 50) * 0.3
    if road_info["congestion_times"]:
        congestion_score += 0.1
    if road_info["special_events"]:
        congestion_score += 0.1 * min(len(road_info["special_events"]), 3)

    # P1.5: raw index peaks at 1.2; normalise so "/1.0" below is truthful.
    congestion_score = congestion_score / CONGESTION_RAW_MAX
    if congestion_score >= SEVERE_THRESHOLD:
        congestion = "Severely Congested"
    elif congestion_score >= CONGESTED_THRESHOLD:
        congestion = "Congested"
    else:
        congestion = "Not Congested"

    reasons = [f"Congestion score: {congestion_score:.2f}/1.0 → {congestion}"]

    all_solutions = [s for cat in SOLUTIONS.values() for s in cat]

    # P1.1/P1.4: shared registry (also what test_restrictions.py exercises)
    candidates = filter_solutions(all_solutions, road_info["restrictions"])

    # P1.3: detected markings now modify the ranking
    rules = triggered_detection_rules(detected_elements)
    for r in rules:
        reasons.append(f"Detected {' / '.join(r['any_of'])} — {r['note']}")

    viable = []
    for sol in candidates:
        base = calculate_solution_score(sol, congestion, road_info)
        viable.append({
            **sol,
            "base_score": base,
            "score": base * detection_multiplier(sol["name"], rules),
        })

    viable.sort(key=lambda x: x["score"], reverse=True)
    n_top = 5 if congestion == "Severely Congested" else 4 if congestion == "Congested" else 3
    top_solutions = [s["name"] for s in viable[:n_top]]

    return congestion, reasons, top_solutions


def visualize_solution(original_img, road_info, solutions, detected_elements):
    img    = original_img.copy()
    h, w   = img.shape[:2]
    pil    = Image.fromarray(img)
    draw   = ImageDraw.Draw(pil)
    font   = ImageFont.load_default()
    large  = ImageFont.load_default(size=20)

    draw.text((20, 20), road_info["name"],               fill=(0,0,0), font=large)
    draw.text((20, 50), f"Condition: {road_info['condition']}", fill=(0,0,0), font=font)

    if detected_elements:
        draw.text((20, 80), "Detected Markings:", fill=(0,0,0), font=font)
        for i, (name, conf) in enumerate(detected_elements[:4]):
            draw.text((30, 110 + i*25), f"- {name} ({conf*100:.1f}%)", fill=(0,0,0), font=font)

    img = np.array(pil)

    visual_map = {
        "Add general purpose lane":   lambda: (
            cv2.rectangle(img, (0,h-100),(w,h),(200,200,200),-1),
            cv2.putText(img,"NEW LANE",(w//2-40,h-50),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2),
        ),
        "Convert to HOV lane":        lambda: (
            cv2.rectangle(img,(0,h-60),(w,h-30),(0,100,200),-1),
            cv2.putText(img,"HOV LANE",(w//2-40,h-45),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1),
        ),
        "Convert to bus lane":        lambda: (
            cv2.rectangle(img,(0,h-80),(w,h-40),(255,165,0),-1),
            cv2.putText(img,"BUS LANE",(w//2-40,h-60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),1),
        ),
        "Add roundabout":             lambda: (
            cv2.circle(img,(w//2,h//2),50,(0,255,255),-1),
            cv2.putText(img,"ROUNDABOUT",(w//2-60,h//2),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1),
        ),
        "Add pedestrian overpass":    lambda: (
            cv2.rectangle(img,(w//3,h//3),(2*w//3,h//3+20),(255,0,0),-1),
            cv2.line(img,(w//3,h//3+20),(w//3,h),(255,0,0),3),
            cv2.line(img,(2*w//3,h//3+20),(2*w//3,h),(255,0,0),3),
            cv2.putText(img,"OVERPASS",(w//3+10,h//3+15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1),
        ),
        "Optimize signal timing":     lambda: (
            cv2.circle(img,(w-50,50),20,(0,255,0),-1),
            cv2.putText(img,"OPTIMIZED",(w-120,50),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),1),
        ),
        "Add bike lane markings":     lambda: (
            cv2.rectangle(img,(0,h-60),(w,h-30),(0,255,0),-1),
            cv2.putText(img,"BIKE LANE",(w//2-40,h-45),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1),
        ),
    }

    applied = 0
    for sol in solutions:
        if sol in visual_map and applied < 3:
            visual_map[sol]()
            applied += 1

    return img


# ─────────────────────────────────────────────────────────────
# SECTION 10 — EVALUATION  (metrics, bootstrap CIs)
#
# With ~76 images a single 15-image validation split is far too noisy to
# support a claim.  Everything below works on POOLED OUT-OF-FOLD predictions:
# every image is scored once per repeat by a model that never trained on it,
# and uncertainty is quantified by bootstrapping over images.
#
# `P` below may be one (N, C) array or a LIST of them (one per CV repeat);
# metrics are computed per repeat and averaged, and the bootstrap resamples
# images once and averages across repeats inside every resample.
# ─────────────────────────────────────────────────────────────

def _as_list(P):
    return [P] if isinstance(P, np.ndarray) else list(P)


def _defined(y):
    """AUC needs both a positive and a negative example."""
    return 0 < y.sum() < len(y)


def _nanmean0(a):
    """nan-aware mean over axis 0 without RuntimeWarnings (all-NaN column -> NaN)."""
    a = np.asarray(a, dtype=float)
    ok = np.isfinite(a)
    cnt = ok.sum(0)
    tot = np.where(ok, a, 0.0).sum(0)
    out = np.full(a.shape[1:], np.nan)
    np.divide(tot, cnt, out=out, where=cnt > 0)
    return out


def _nanmean(a):
    a = np.asarray(a, dtype=float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def per_class_auc(Y, P, constant=False):
    out = np.full(NUM_CLASSES, np.nan)
    for k in range(NUM_CLASSES):
        if _defined(Y[:, k]):
            out[k] = 0.5 if constant else roc_auc_score(Y[:, k], P[:, k])
    return out


def per_class_ap(Y, P, constant=False):
    """Average precision: far more informative than AUC for rare classes."""
    out = np.full(NUM_CLASSES, np.nan)
    for k in range(NUM_CLASSES):
        if Y[:, k].sum() > 0:
            out[k] = Y[:, k].mean() if constant else average_precision_score(Y[:, k], P[:, k])
    return out


def per_class_auc_rep(Y, Ps, constant=False):
    return _nanmean0([per_class_auc(Y, P, constant) for P in _as_list(Ps)])


def per_class_ap_rep(Y, Ps, constant=False):
    return _nanmean0([per_class_ap(Y, P, constant) for P in _as_list(Ps)])


def bce_loss(Y, P):
    P = np.clip(P, 1e-7, 1 - 1e-7)
    return float(-np.mean(Y * np.log(P) + (1 - Y) * np.log(1 - P)))


def prf_per_class(Y, P, thr):
    """Precision / recall / F1 per class at threshold `thr` (NaN when undefined,
    e.g. precision of a class the model never predicts)."""
    pred, y = P >= thr, Y > 0.5
    tp = (pred & y).sum(0).astype(float)
    fp = (pred & ~y).sum(0).astype(float)
    fn = (~pred & y).sum(0).astype(float)
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), np.nan)
    rec  = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), np.nan)
    f1   = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), np.nan)
    return prec, rec, f1


def arrow_swap_rate(Y, P, thr):
    """Among images with exactly ONE turn arrow, how often is the OPPOSITE arrow
    (and not the true one) predicted?  Directly measures left/right confusion."""
    l, r = CLASSES.index("Left Turn Arrow"), CLASSES.index("Right Turn Arrow")
    pred = P >= thr
    only_l = (Y[:, l] > 0.5) & (Y[:, r] < 0.5)
    only_r = (Y[:, r] > 0.5) & (Y[:, l] < 0.5)
    n = only_l.sum() + only_r.sum()
    if n == 0:
        return float("nan")
    wrong = (only_l & pred[:, r] & ~pred[:, l]) | (only_r & pred[:, l] & ~pred[:, r])
    return float(wrong.sum() / n)


def summarize(Y, P, constant=False, threshold=None):
    """All reported metrics for one model (pooled OOF predictions, averaged over repeats)."""
    thr = THRESHOLD if threshold is None else threshold
    Ps, Yi = _as_list(P), Y.astype(int)
    preds = [(p >= thr).astype(int) for p in Ps]
    prf = [prf_per_class(Y, p, thr) for p in Ps]
    auc, ap = per_class_auc_rep(Y, Ps, constant), per_class_ap_rep(Y, Ps, constant)
    f1 = _nanmean0([x[2] for x in prf])
    return {
        "auc": auc, "ap": ap,
        "prec": _nanmean0([x[0] for x in prf]),
        "rec":  _nanmean0([x[1] for x in prf]),
        "f1":   f1, "macro_f1": _nanmean(f1),
        "macro_auc": _nanmean(auc), "macro_ap": _nanmean(ap),
        "exact_match":   float(np.mean([np.mean(np.all(q == Yi, axis=1)) for q in preds])),
        "per_label_acc": float(np.mean([np.mean(q == Yi) for q in preds])),
        "micro_f1": float(np.mean([f1_score(Yi.ravel(), q.ravel(), zero_division=0) for q in preds])),
        "bce": float(np.mean([bce_loss(Y, p) for p in Ps])),
        "arrow_swap_rate": _nanmean([arrow_swap_rate(Y, p, thr) for p in Ps]),
    }


def bootstrap_auc(Y, preds, constants=(), n_boot=None, seed=None):
    """
    Percentile bootstrap over IMAGES.  `preds` is {name: P or [P per repeat]}.
    Every model is scored on the SAME resamples, so differences between models
    are paired.  Returns {name: array (n_boot, NUM_CLASSES)} of per-class AUCs
    (NaN where a resample lacks positives or negatives).
    """
    n_boot = N_BOOT if n_boot is None else n_boot
    rng = np.random.RandomState(SEED if seed is None else seed)
    n = len(Y)
    preds = {k: _as_list(v) for k, v in preds.items()}
    out = {name: np.full((n_boot, NUM_CLASSES), np.nan) for name in preds}
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        Yb = Y[idx]
        for name, Ps in preds.items():
            out[name][b] = per_class_auc_rep(Yb, [P[idx] for P in Ps], constant=name in constants)
    return out


def _ci(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return float("nan"), float("nan")
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def _f(v, d=3):
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def _fci(v, lo, hi):
    return f"{_f(v)} [{_f(lo, 2)}, {_f(hi, 2)}]"


# ─────────────────────────────────────────────────────────────
# SECTION 11 — BASELINES + DATA-LEAKAGE AUDIT
# (all baselines are evaluated on the SAME folds as the CNN)
# ─────────────────────────────────────────────────────────────

BASE_PREV  = "Baseline: class prevalence"
BASE_HOG   = "Baseline: HOG + logistic regression"
BASE_PROBE = "Baseline: frozen EfficientNetV2S features + logistic regression"


def prevalence_baseline(Y_train, n_eval):
    """
    Predict each class's training prevalence for every image.  A constant
    predictor has ROC-AUC = 0.5 BY CONSTRUCTION, so for AUC the comparison is
    simply "is the CNN above 0.5?".  This baseline matters for the other
    metrics: BCE, exact-match and F1 (what you get from the base rates alone).
    """
    return np.tile(Y_train.mean(0), (n_eval, 1))


def load_gray_images(image_paths, size=(128, 128)):
    out = []
    for pth in image_paths:
        im = cv2.imdecode(np.fromfile(pth, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise ValueError(f"Could not read image: {pth}")
        out.append(cv2.resize(im, size))
    return np.stack(out)


def _hog_numpy(im, cell=16, block=2, bins=9):
    """Plain-NumPy HOG (unsigned gradients, 16px cells, 2x2-cell L2-norm blocks).
    Used when cv2.HOGDescriptor is missing (OpenCV 5.x removed it)."""
    im = im.astype(np.float32)
    gx = np.zeros_like(im); gy = np.zeros_like(im)
    gx[:, 1:-1] = im[:, 2:] - im[:, :-2]
    gy[1:-1, :] = im[2:, :] - im[:-2, :]
    mag = np.hypot(gx, gy)
    ang = (np.rad2deg(np.arctan2(gy, gx)) % 180.0) / (180.0 / bins)     # [0, bins)
    lo = np.floor(ang).astype(int) % bins
    hi = (lo + 1) % bins
    w_hi = ang - np.floor(ang)
    ch, cw = im.shape[0] // cell, im.shape[1] // cell
    hist = np.zeros((ch, cw, bins), dtype=np.float32)
    for i in range(ch):
        for j in range(cw):
            sl = (slice(i * cell, (i + 1) * cell), slice(j * cell, (j + 1) * cell))
            m, l, h, w = mag[sl].ravel(), lo[sl].ravel(), hi[sl].ravel(), w_hi[sl].ravel()
            hist[i, j] = (np.bincount(l, m * (1 - w), bins) + np.bincount(h, m * w, bins))
    feats = []
    for i in range(ch - block + 1):
        for j in range(cw - block + 1):
            v = hist[i:i + block, j:j + block].ravel()
            feats.append(v / np.sqrt((v ** 2).sum() + 1e-6))
    return np.concatenate(feats)


def hog_features(gray_imgs):
    """Classic hand-crafted features: Histogram of Oriented Gradients."""
    if hasattr(cv2, "HOGDescriptor"):
        hog = cv2.HOGDescriptor((128, 128), (32, 32), (16, 16), (16, 16), 9)
        return np.stack([hog.compute(im).ravel() for im in gray_imgs]).astype(np.float32)
    return np.stack([_hog_numpy(im) for im in gray_imgs]).astype(np.float32)


def backbone_features(image_paths, batch=8):
    """
    Frozen, ImageNet-pretrained EfficientNetV2S global-average-pooled features,
    extracted ONCE (no augmentation, no labels used -> no leakage into CV).
    Feeding these to a logistic regression is the "linear probe": it answers
    whether the extra machinery of training the CNN head (augmentation, BCE
    fine-tuning, schedule) buys anything over just using the features.
    """
    backbone = keras.applications.EfficientNetV2S(
        input_shape=IMG_SIZE + (3,), include_top=False,
        weights=PRETRAINED_WEIGHTS, pooling="avg")
    imgs = []
    for pth in image_paths:
        im = cv2.imdecode(np.fromfile(pth, dtype=np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError(f"Could not read image: {pth}")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        imgs.append(cv2.resize(im, (IMG_SIZE[1], IMG_SIZE[0])).astype(np.float32))
    feats = backbone.predict(np.stack(imgs), batch_size=batch, verbose=0)
    del backbone
    keras.backend.clear_session()
    gc.collect()
    return feats.astype(np.float32)


def logreg_fold(X_train, Y_train, X_val, C=0.1):
    """
    One L2 logistic regression per class on standardised features.  C is fixed
    a priori (NOT tuned, same value for every baseline) so no baseline sees
    more of the data than the CNN does.  No class re-weighting, so outputs stay
    roughly calibrated and BCE / threshold metrics are comparable with the CNN.
    """
    P = np.zeros((len(X_val), NUM_CLASSES))
    for k in range(NUM_CLASSES):
        y = Y_train[:, k]
        if y.min() == y.max():                    # class absent/ubiquitous in train fold
            P[:, k] = y.mean()
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=C, max_iter=2000))
        clf.fit(X_train, y)
        P[:, k] = clf.predict_proba(X_val)[:, 1]
    return P


hog_logreg_fold = logreg_fold          # backwards-compatible name


def tiny_features(image_paths, size=32):
    """Zero-mean, unit-norm 32x32 grayscale vectors: correlation = dot product."""
    X = load_gray_images(image_paths, (size, size)).reshape(len(image_paths), -1).astype(np.float64)
    X -= X.mean(1, keepdims=True)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    return X


def duplicate_audit(image_paths, threshold=None):
    """
    Heuristic near-duplicate detector (tiny-image correlation > threshold).
    Returns (pairs (M,2) int array, cluster_id (N,) int array).  If photos of
    the same scene end up on both sides of a CV split, the model is partly
    being tested on its training data and every score is optimistic.
    """
    thr = DUP_THRESHOLD if threshold is None else threshold
    X = tiny_features(image_paths)
    C = X @ X.T
    iu = np.triu_indices(len(X), 1)
    hit = C[iu] > thr
    pairs = np.stack([iu[0][hit], iu[1][hit]], axis=1) if hit.any() else np.zeros((0, 2), dtype=int)
    parent = list(range(len(X)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i, j in pairs:
        parent[find(int(i))] = find(int(j))
    roots = np.array([find(i) for i in range(len(X))])
    _, cluster = np.unique(roots, return_inverse=True)
    return pairs, cluster


def load_groups_csv(path, image_paths):
    """CSV with header `image,group` (basename -> group id).  Every image needs a row."""
    mapping = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            mapping[row["image"].strip()] = row["group"].strip()
    names = [os.path.basename(p) for p in image_paths]
    missing = [n for n in names if n not in mapping]
    if missing:
        raise ValueError(f"{len(missing)} image(s) missing from {path}, e.g. {missing[:3]}")
    return np.array([mapping[n] for n in names])


def resolve_groups(image_paths):
    """Decide how folds are formed and run the leakage audit.  -> (groups|None, info, pairs)"""
    pairs, cluster = duplicate_audit(image_paths)
    sizes = np.bincount(cluster)
    info = {"dup_threshold": DUP_THRESHOLD, "near_duplicate_pairs": int(len(pairs)),
            "images_in_duplicate_clusters": int((sizes[cluster] >= 2).sum()),
            "largest_cluster": int(sizes.max()), "grouping": "none"}
    groups = None
    if GROUPS_CSV:
        groups, info["grouping"] = load_groups_csv(GROUPS_CSV, image_paths), f"csv:{GROUPS_CSV}"
    elif AUTO_GROUP:
        groups, info["grouping"] = cluster, f"auto near-duplicate clusters (r>{DUP_THRESHOLD})"
    print(f"[AUDIT] near-duplicate pairs (r>{DUP_THRESHOLD}): {len(pairs)} | images in such clusters: "
          f"{info['images_in_duplicate_clusters']} | largest cluster: {info['largest_cluster']} | "
          f"fold grouping: {info['grouping']}")
    return groups, info, pairs


def count_leaks(pairs, fold_of):
    """Near-duplicate pairs whose two images sit in DIFFERENT folds (= train/eval leakage)."""
    return int(sum(fold_of[i] != fold_of[j] for i, j in pairs))


# ─────────────────────────────────────────────────────────────
# SECTION 12 — CROSS-VALIDATION RUNNER + REPORTING
# ─────────────────────────────────────────────────────────────

MAIN_NAME = "EfficientNetV2S, head-only, label-swap flip  [MAIN]"
ABLATIONS = {
    "flip_naive":  ("EfficientNetV2S + NAIVE flip (arrow labels not swapped)", {"FLIP_MODE": "naive"}),
    "flip_none":   ("EfficientNetV2S, no horizontal flip",                      {"FLIP_MODE": "none"}),
    "finetune_10": ("EfficientNetV2S + Phase-2 fine-tune (top 10 layers)",      {"FINE_TUNE_LAYERS": 10}),
    "finetune_50": ("EfficientNetV2S + Phase-2 fine-tune (top 50 layers)",      {"FINE_TUNE_LAYERS": 50}),
    "res_384":     ("EfficientNetV2S, head-only, 384x384 input",                {"IMG_SIZE": (384, 384)}),
}
DEFAULT_ABLATIONS = "flip_naive,flip_none,finetune_10,finetune_50"   # res_384 is ~3x slower: opt-in


@contextlib.contextmanager
def config_overrides(**kw):
    """Temporarily set module-level config values (restored on exit)."""
    old = {k: globals()[k] for k in kw}
    globals().update(kw)
    try:
        yield
    finally:
        globals().update(old)


def set_seed(seed):
    keras.utils.set_random_seed(int(seed))        # python, numpy and TF


def _safe(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()[:60]


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def write_config(tag, extra=None):
    """Everything needed to reproduce a run, saved next to its results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg = {
        "tag": tag, "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command_line": sys.argv,
        "classes": CLASSES, "class_aliases": CLASS_ALIASES,
        "hyperparameters": {
            "IMG_SIZE": IMG_SIZE, "BATCH_SIZE": BATCH_SIZE, "EPOCHS": EPOCHS,
            "LR_HEAD": LR_HEAD, "LR_FINE": LR_FINE, "FINE_TUNE_LAYERS": FINE_TUNE_LAYERS,
            "FINE_TUNE_EPOCHS": FINE_TUNE_EPOCHS, "THRESHOLD": THRESHOLD,
            "FLIP_MODE": FLIP_MODE, "SEED": SEED, "N_FOLDS": N_FOLDS, "N_BOOT": N_BOOT,
            "REPEATS": REPEATS, "PRETRAINED_WEIGHTS": PRETRAINED_WEIGHTS,
            "DUP_THRESHOLD": DUP_THRESHOLD, "AUTO_GROUP": AUTO_GROUP, "GROUPS_CSV": GROUPS_CSV,
            "augmentation": "rotation 0.05, zoom 0.10, contrast 0.2, brightness 0.2, translation 0.1 "
                            "(+ label-aware horizontal flip when FLIP_MODE=label_swap)",
            "optimizer": "Adam", "loss": "binary_crossentropy",
        },
        "dataset_audit": dict(LAST_AUDIT),
        "versions": {"python": platform.python_version(), "tensorflow": tf.__version__,
                     "keras": keras.__version__, "numpy": np.__version__,
                     "opencv": cv2.__version__, "platform": platform.platform()},
    }
    if extra:
        cfg.update(extra)
    path = os.path.join(RESULTS_DIR, f"config_{tag}.json")
    with open(path, "w") as f:
        json.dump(_jsonable(cfg), f, indent=2)
    return path


def describe_folds(Y, splits):
    print("\n[CV] Fold composition (images | positives per class) — repeat 1:")
    print("     fold  n_val  " + "  ".join(f"{c[:10]:>10s}" for c in CLASSES))
    for f, (_, va) in enumerate(splits):
        print(f"     {f:>4d}  {len(va):>5d}  " +
              "  ".join(f"{int(Y[va, k].sum()):>10d}" for k in range(NUM_CLASSES)))


def run_cv(image_paths, Y, splits, name, overrides=None, verbose=0, repeat=0):
    """
    Train one fresh model per fold, predict the held-out fold, and return the
    pooled out-of-fold predictions (N, NUM_CLASSES) + per-fold macro AUCs.
    """
    oof = np.full(Y.shape, np.nan)
    fold_auc = []
    with config_overrides(**(overrides or {})):
        print(f"\n[CV] === {name}  (repeat {repeat + 1}/{REPEATS}) ===")
        print(f"[CV] flip={FLIP_MODE} | fine_tune_layers={FINE_TUNE_LAYERS} | "
              f"epochs={EPOCHS} | img={IMG_SIZE[0]} | weights={PRETRAINED_WEIGHTS}")
        for f, (tr, va) in enumerate(splits):
            t0 = time.time()
            seed_f = SEED + 100 * repeat + f
            set_seed(seed_f)
            train_ds, val_ds = make_datasets(image_paths, Y, tr, va, seed=seed_f)
            model, _ = train_model(train_ds, prior=Y[tr].mean(0), verbose=verbose)
            P = model.predict(val_ds, verbose=0)
            oof[va] = P
            fold_auc.append(_nanmean(per_class_auc(Y[va], P)))
            print(f"[CV]   fold {f + 1}/{len(splits)}  train={len(tr)} val={len(va)}  "
                  f"macro-AUC={_f(fold_auc[-1])}  ({time.time() - t0:.0f}s)")
            del model, train_ds, val_ds
            keras.backend.clear_session()
            gc.collect()
    assert np.isfinite(oof).all(), "some images never received an out-of-fold prediction"
    return oof, fold_auc


def save_oof(name, image_paths, Y, P, fold_of, repeat=0):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"oof_{_safe(name)}_rep{repeat + 1}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "fold"] + [f"true_{c}" for c in CLASSES] + [f"pred_{c}" for c in CLASSES])
        for i, pth in enumerate(image_paths):
            w.writerow([os.path.basename(pth), int(fold_of[i])] +
                       [int(v) for v in Y[i]] + [f"{v:.5f}" for v in P[i]])
    return path


def load_oof(name, image_paths, fold_of, repeat=0):
    """Load a previously saved OOF prediction CSV; None unless images AND fold assignment match."""
    path = os.path.join(RESULTS_DIR, f"oof_{_safe(name)}_rep{repeat + 1}.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        by = {r["image"]: r for r in csv.DictReader(f)}
    names = [os.path.basename(p) for p in image_paths]
    try:
        if any(int(by[n]["fold"]) != int(fold_of[i]) for i, n in enumerate(names)):
            return None
        return np.array([[float(by[n][f"pred_{c}"]) for c in CLASSES] for n in names])
    except (KeyError, ValueError):
        return None


def plot_auc_ci(Y, summaries, boot, path):
    """Grouped bars: per-class AUC with 95% bootstrap CI, one bar per model."""
    names = list(summaries)
    x = np.arange(NUM_CLASSES)
    w = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=(max(9, 1.6 * NUM_CLASSES), 4.8))
    for j, name in enumerate(names):
        auc = summaries[name]["auc"]
        lo = np.array([_ci(boot[name][:, k])[0] for k in range(NUM_CLASSES)])
        hi = np.array([_ci(boot[name][:, k])[1] for k in range(NUM_CLASSES)])
        err = np.vstack([np.nan_to_num(auc - lo), np.nan_to_num(hi - auc)])
        ax.bar(x + j * w, np.nan_to_num(auc), w, yerr=err, capsize=2,
               label=(name if len(name) < 46 else name[:43] + "..."))
    ax.axhline(0.5, color="k", linestyle="--", linewidth=0.8)
    ax.set_xticks(x + w * (len(names) - 1) / 2)
    ax.set_xticklabels([c.replace(" ", "\n", 1) for c in CLASSES], fontsize=8)
    ax.set_ylabel("ROC-AUC (out-of-fold)"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6, loc="lower right"); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)


def _tex(s):
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("[MAIN]", "(main)")):
        s = s.replace(a, b)
    return s


def write_latex(path, runs, summ, boot):
    """Paper-ready (booktabs) per-class AUC table with bootstrap CIs."""
    cols = "l" + "c" * (NUM_CLASSES + 1)
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{Out-of-fold ROC-AUC (95\% percentile-bootstrap CI over images).}",
         rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
         "Model & " + " & ".join(_tex(c) for c in CLASSES) + r" & Macro \\", r"\midrule"]
    for n in runs:
        cells = []
        for k in range(NUM_CLASSES):
            lo, hi = _ci(boot[n][:, k])
            cells.append(f"{_f(summ[n]['auc'][k], 2)} [{_f(lo, 2)}, {_f(hi, 2)}]")
        macro = np.array([_nanmean(row) for row in boot[n]])
        lo, hi = _ci(macro)
        cells.append(f"{_f(summ[n]['macro_auc'], 2)} [{_f(lo, 2)}, {_f(hi, 2)}]")
        L.append(_tex(n) + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def build_report(Y, runs, tag, main_name):
    """
    runs: OrderedDict name -> {"P": [oof per repeat], "constant": bool, "fold_auc": [...]}
    Writes results/results_<tag>.md/.json/.tex and an AUC-with-CI figure.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    preds = {n: _as_list(r["P"]) for n, r in runs.items()}
    consts = {n for n, r in runs.items() if r.get("constant")}
    summ = {n: summarize(Y, preds[n], constant=n in consts) for n in runs}
    R = len(next(iter(preds.values())))
    print(f"\n[REPORT] bootstrapping ({N_BOOT} resamples)...")
    boot = bootstrap_auc(Y, preds, consts)

    counts = LAST_AUDIT.get("per_class_image_counts", {})
    dup = LAST_AUDIT.get("duplicate_audit", {})
    lines = [f"# Results — {tag}", "",
             f"{len(Y)} images, {N_FOLDS}-fold iterative-stratified CV x {R} repeat(s), pooled out-of-fold "
             f"predictions, 95% percentile-bootstrap CIs over images ({N_BOOT} resamples).  "
             f"Threshold for exact-match / precision / recall / F1 = {THRESHOLD}.", "",
             "## Table 0 — dataset (images containing each class)", "",
             "| Class | Images | Prevalence |", "|---|---|---|"]
    for i, c in enumerate(CLASSES):
        lines.append(f"| {c} | {counts.get(c, int(Y[:, i].sum()))} | {Y[:, i].mean():.2f} |")
    lines += ["", "## Table A — overall", "",
              "| Model | Macro AUC [95% CI] | Mean fold AUC ± SD | Macro AUC per repeat | Macro AP | Macro-F1 | "
              "Exact-match | Micro-F1 | BCE | Arrow L/R swap rate |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in runs.items():
        m = summ[n]
        macro = np.array([_nanmean(row) for row in boot[n]])
        lo, hi = _ci(macro)
        fa = r.get("fold_auc") or []
        fold_txt = f"{np.nanmean(fa):.3f} ± {np.nanstd(fa):.3f}" if fa else "—"
        rep_txt = (", ".join(_f(_nanmean(per_class_auc(Y, P, n in consts))) for P in preds[n])
                   if R > 1 else "—")
        lines.append(f"| {n} | {_fci(m['macro_auc'], lo, hi)} | {fold_txt} | {rep_txt} | {_f(m['macro_ap'])} | "
                     f"{_f(m['macro_f1'])} | {_f(m['exact_match'])} | {_f(m['micro_f1'])} | "
                     f"{_f(m['bce'])} | {_f(m['arrow_swap_rate'])} |")
    lines += ["", "Prevalence baseline: constant predictor, so AUC = 0.5 by construction and AP = class "
                  "prevalence; its value is as the BCE / exact-match / F1 reference.  The linear-probe "
                  "baseline uses the same frozen ImageNet features as the CNN with a fixed-C logistic "
                  "regression (no tuning).", "",
              "## Table B — per-class ROC-AUC [95% CI]", "",
              "| Model | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * NUM_CLASSES]
    for n in runs:
        cells = []
        for k in range(NUM_CLASSES):
            lo, hi = _ci(boot[n][:, k])
            cells.append(_fci(summ[n]["auc"][k], lo, hi))
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines += ["", "## Table C — per-class average precision (chance level = prevalence)", "",
              "| Model | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * NUM_CLASSES]
    lines.append("| *prevalence (chance)* | " + " | ".join(_f(v) for v in Y.mean(0)) + " |")
    for n in runs:
        lines.append(f"| {n} | " + " | ".join(_f(v) for v in summ[n]["ap"]) + " |")
    lines += ["", f"## Table E — per-class precision / recall / F1 at threshold {THRESHOLD}", "",
              "n/a = undefined (e.g. the model never predicts that class).", "",
              "| Model | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * NUM_CLASSES]
    for n in runs:
        cells = [f"{_f(summ[n]['prec'][k], 2)} / {_f(summ[n]['rec'][k], 2)} / {_f(summ[n]['f1'][k], 2)}"
                 for k in range(NUM_CLASSES)]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")

    # paired differences: main model vs EVERYTHING else (ablations AND baselines)
    others = [n for n in runs if n != main_name]
    if others and main_name in runs:
        arrows = [CLASSES.index("Left Turn Arrow"), CLASSES.index("Right Turn Arrow")]
        lines += ["", "## Table D — paired difference of the main model minus each alternative "
                      "(bootstrap, same resamples)", "",
                  "| Alternative | Δ macro AUC [95% CI] | Δ arrow-class AUC [95% CI] | P(main ≤ alt) |",
                  "|---|---|---|---|"]
        for n in others:
            d_macro = np.array([_nanmean(boot[main_name][b]) - _nanmean(boot[n][b]) for b in range(N_BOOT)])
            d_arrow = np.array([_nanmean(boot[main_name][b, arrows]) - _nanmean(boot[n][b, arrows])
                                for b in range(N_BOOT)])
            fin = d_macro[np.isfinite(d_macro)]
            p_le = float(np.mean(fin <= 0)) if len(fin) else float("nan")
            lines.append(f"| {n} | {_fci(_nanmean(d_macro), *_ci(d_macro))} | "
                         f"{_fci(_nanmean(d_arrow), *_ci(d_arrow))} | {_f(p_le, 3)} |")
        lines += ["", "Positive Δ = main model better.  A CI containing 0 means the data cannot "
                      "distinguish the two.  P(main ≤ alt) is the bootstrap fraction of resamples where the "
                      "alternative matched or beat the main model (an approximate one-sided p-value; "
                      "uncorrected for the number of comparisons — say so in the paper)."]

    # plain-language guard rails
    lines += ["", "## Reading guide / limitations to state in the paper", ""]
    if main_name in runs:
        ok = [c for k, c in enumerate(CLASSES)
              if np.isfinite(summ[main_name]["auc"][k]) and _ci(boot[main_name][:, k])[0] > 0.5]
        bad = [c for c in CLASSES if c not in ok]
        lines.append(f"- Classes whose AUC CI lies entirely above chance (0.5): {', '.join(ok) if ok else 'none'}.")
        if bad:
            lines.append(f"- NOT distinguishable from chance: {', '.join(bad)}.")
        for n in runs:
            if n in (BASE_HOG, BASE_PROBE):
                d = np.array([_nanmean(boot[main_name][b]) - _nanmean(boot[n][b]) for b in range(N_BOOT)])
                lo, hi = _ci(d)
                if np.isfinite(lo) and lo > 0:
                    verdict = "the CNN is significantly BETTER (CI excludes 0)"
                elif np.isfinite(hi) and hi < 0:
                    verdict = "the CNN is significantly WORSE (CI excludes 0)"
                else:
                    verdict = "NOT distinguishable (CI contains 0): the trained CNN has not been shown to beat it"
                lines.append(f"- Main macro AUC {_f(summ[main_name]['macro_auc'])} vs {n.replace('Baseline: ', '')} "
                             f"{_f(summ[n]['macro_auc'])}: {verdict}.")
    if dup:
        lines.append(f"- Near-duplicate audit (r>{dup.get('dup_threshold')}): {dup.get('near_duplicate_pairs')} pairs, "
                     f"{dup.get('images_in_duplicate_clusters')} images in clusters; fold grouping = {dup.get('grouping')}; "
                     f"pairs split across folds in repeat 1 = {dup.get('pairs_split_across_folds_rep1')}.  "
                     f"Splits that separate near-duplicates make all numbers optimistic.")
    lines += [f"- Dataset is small ({len(Y)} images) and image-level (presence/absence), not localisation; "
              f"images are treated as independent for the bootstrap unless grouped.",
              "- Labels are derived from bounding-box XMLs; class-alias merges are listed in the config JSON.",
              "- The THRESHOLD is fixed a priori, not tuned on evaluation data; AUC/AP are threshold-free.",
              "- Scores are uncalibrated sigmoid outputs: do not describe them as probabilities in the paper."]
    text = "\n".join(lines) + "\n"
    md_path = os.path.join(RESULTS_DIR, f"results_{tag}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(RESULTS_DIR, f"results_{tag}.json"), "w") as f:
        json.dump(_jsonable({n: {**summ[n], "fold_auc": runs[n].get("fold_auc")} for n in runs}), f, indent=2)
    write_latex(os.path.join(RESULTS_DIR, f"results_{tag}.tex"), runs, summ, boot)
    plot_auc_ci(Y, summ, boot, os.path.join(RESULTS_DIR, f"auc_per_class_{tag}.png"))
    print("\n" + text)
    print(f"[REPORT] saved → {md_path}  (+ .json, .tex, auc_per_class_{tag}.png)")
    return summ, boot


def run_experiments(configs, tag, with_baselines=True, verbose=0):
    """configs: OrderedDict name -> overrides dict (first entry is the main model)."""
    image_paths, Y = collect_pairs()
    k = N_FOLDS
    groups, dup_info, dup_pairs = resolve_groups(image_paths)
    fold_sets = []
    for r in range(REPEATS):
        seed_r = SEED + 1000 * r
        fold_sets.append(group_stratified_folds(Y, groups, k=k, seed=seed_r) if groups is not None
                         else multilabel_stratified_folds(Y, k=k, seed=seed_r))
    splits_all = [fold_splits(f, k) for f in fold_sets]
    describe_folds(Y, splits_all[0])
    dup_info["pairs_split_across_folds_rep1"] = count_leaks(dup_pairs, fold_sets[0])
    if dup_info["pairs_split_across_folds_rep1"] and groups is None:
        print(f"[WARN] {dup_info['pairs_split_across_folds_rep1']} near-duplicate pair(s) straddle train/eval "
              f"folds → scores are optimistic.  Re-run with --auto-group (or --groups-csv).")
    LAST_AUDIT["duplicate_audit"] = dup_info
    write_config(tag, {"fold_assignment_per_repeat": [
        {os.path.basename(p): int(f) for p, f in zip(image_paths, fo)} for fo in fold_sets]})

    runs = collections.OrderedDict()
    for name, overrides in configs.items():
        Ps, fold_auc = [], []
        for r, splits in enumerate(splits_all):
            P = load_oof(name, image_paths, fold_sets[r], r) if RESUME else None
            if P is not None:
                fa = [_nanmean(per_class_auc(Y[va], P[va])) for _, va in splits]
                print(f"[CV] {name} repeat {r + 1}/{REPEATS}: RESUMED from saved OOF predictions")
            else:
                P, fa = run_cv(image_paths, Y, splits, name, overrides, verbose, repeat=r)
                save_oof(name, image_paths, Y, P, fold_sets[r], repeat=r)
            Ps.append(P); fold_auc += fa
        runs[name] = {"P": Ps, "fold_auc": fold_auc, "constant": False}

    if with_baselines:
        print("\n[CV] === Baselines (same folds) ===")
        X_hog = hog_features(load_gray_images(image_paths))
        X_probe = backbone_features(image_paths) if USE_PROBE else None
        base = collections.OrderedDict([(BASE_PREV, []), (BASE_HOG, [])])
        if USE_PROBE:
            base[BASE_PROBE] = []
        fa_base = {n: [] for n in base}
        for r, splits in enumerate(splits_all):
            Pp, Ph, Pb = np.zeros_like(Y), np.zeros_like(Y), np.zeros_like(Y)
            for f, (tr, va) in enumerate(splits):
                Pp[va] = prevalence_baseline(Y[tr], len(va))
                Ph[va] = logreg_fold(X_hog[tr], Y[tr], X_hog[va])
                fa_base[BASE_HOG].append(_nanmean(per_class_auc(Y[va], Ph[va])))
                if USE_PROBE:
                    Pb[va] = logreg_fold(X_probe[tr], Y[tr], X_probe[va])
                    fa_base[BASE_PROBE].append(_nanmean(per_class_auc(Y[va], Pb[va])))
            base[BASE_PREV].append(Pp); base[BASE_HOG].append(Ph)
            if USE_PROBE:
                base[BASE_PROBE].append(Pb)
            for nm in base:
                save_oof(nm, image_paths, Y, base[nm][-1], fold_sets[r], repeat=r)
        for nm, Ps in base.items():
            runs[nm] = {"P": Ps, "fold_auc": fa_base[nm] or None, "constant": nm == BASE_PREV}

    return build_report(Y, runs, tag, main_name=next(iter(configs)))


# ─────────────────────────────────────────────────────────────
# SECTION 13 — MODES: train / analyze / main
# ─────────────────────────────────────────────────────────────

def set_data_root(root):
    globals().update(
        DATA_ROOT=root,
        IMAGE_DIR=os.path.join(root, "Images"),
        ANNOTATIONS_DIR=os.path.join(root, "Annotations"),
        TEST_IMAGES_DIR=os.path.join(root, "Test Images"),
        OUTPUT_DIR=os.path.join(root, "Modified_Images"),
    )


def mode_train(verbose=2):
    """Train the DEPLOYMENT model on all annotated images with the CV recipe."""
    image_paths, Y = collect_pairs()
    train_ds, _ = make_datasets(image_paths, Y, np.arange(len(Y)), np.array([], dtype=int))
    set_seed(SEED)
    print(f"\n[TRAIN] {len(Y)} images | epochs={EPOCHS} | flip={FLIP_MODE} | "
          f"fine_tune_layers={FINE_TUNE_LAYERS}")
    model, hist = train_model(train_ds, prior=Y.mean(0), verbose=verbose)
    os.makedirs(os.path.dirname(os.path.abspath(MODEL_PATH)), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\n[Done] Model saved → {MODEL_PATH}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, title in zip(axes, ["loss", "auc"], ["Training loss (BCE)", "Training AUC"]):
        if key in hist:
            ax.plot(hist[key])
        ax.set_title(title + "  (training data only — not a performance estimate)")
        ax.set_xlabel("epoch"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    curves = os.path.join(RESULTS_DIR, "training_curves_final_model.png")
    plt.savefig(curves, dpi=150); plt.close(fig)
    write_config("train", {"final_epoch_metrics": {k: v[-1] for k, v in hist.items() if v}})
    print(f"[Done] Curves → {curves}")
    print("[NOTE] The expected performance of this model is the --mode cv estimate; "
          "there is deliberately no validation score here.")


def run_system(model, image_path=None):
    """Detection on one image + road survey + ranked solutions + annotated image."""
    if image_path is None:
        if not os.path.isdir(TEST_IMAGES_DIR):
            print(f"[WARN] Test image folder not found: {TEST_IMAGES_DIR}")
            return
        test_imgs = [f for f in sorted(os.listdir(TEST_IMAGES_DIR))
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not test_imgs:
            print("[WARN] No test images found in TEST_IMAGES_DIR.")
            return
        image_path = os.path.join(TEST_IMAGES_DIR, test_imgs[0])

    road_info = get_road_details()
    print(f"\n[Inference] Processing: {os.path.basename(image_path)}")

    detected, original_img = predict_traffic_elements(image_path, model)
    if original_img is None:
        return

    print("\n=== Road Marking Detection Report ===")
    if detected:
        for i, (name, conf) in enumerate(detected, 1):
            print(f"  {i}. {name}: {conf*100:.1f}%")
    else:
        print("  No markings detected above threshold.")

    congestion, reasons, solutions = analyze_traffic(road_info, detected)
    print(f"\n=== Traffic Analysis ===")
    print(f"  Congestion Level : {congestion}")
    for r in reasons:
        print(f"  {r}")
    print(f"\n  Recommended Solutions:")
    for i, sol in enumerate(solutions, 1):
        print(f"    {i}. {sol}")

    modified = visualize_solution(original_img, road_info, solutions, detected)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{road_info['name'].replace(' ','_')}_{ts}.jpg")
    cv2.imwrite(out_path, cv2.cvtColor(modified, cv2.COLOR_RGB2BGR))
    print(f"\n[Done] Annotated image saved → {out_path}")


def mode_analyze(image_path=None):
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] No trained model at {MODEL_PATH}\n"
              f"        Train one first:  python {os.path.basename(__file__)} --mode train")
        return
    model = keras.models.load_model(MODEL_PATH, compile=False)
    run_system(model, image_path)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Traffic road-marking detection + solution ranking (research pipeline).")
    ap.add_argument("--mode", choices=["cv", "ablate", "train", "analyze"], default="analyze",
                    help="cv: cross-validated evaluation + baselines | ablate: cv for several "
                         "variants | train: final model on all data | analyze: run the system")
    ap.add_argument("--data-root", help="folder containing Images/ Annotations/ Test Images/")
    ap.add_argument("--epochs", type=int, help=f"training epochs (default {EPOCHS})")
    ap.add_argument("--folds", type=int, help=f"CV folds (default {N_FOLDS})")
    ap.add_argument("--seed", type=int, help=f"random seed (default {SEED})")
    ap.add_argument("--finetune", type=int, help="unfreeze top N backbone layers (0 = off)")
    ap.add_argument("--flip-mode", choices=["label_swap", "none", "naive"])
    ap.add_argument("--weights", choices=["imagenet", "none"], help="backbone init (none = smoke tests only)")
    ap.add_argument("--img-size", type=int, help="square input size (default 224)")
    ap.add_argument("--boot", type=int, help=f"bootstrap resamples (default {N_BOOT})")
    ap.add_argument("--ablations", default=DEFAULT_ABLATIONS,
                    help=f"comma list from: {', '.join(ABLATIONS)}  (default: {DEFAULT_ABLATIONS})")
    ap.add_argument("--repeats", type=int, help=f"repeated CV with different fold assignments (default {REPEATS}; 3 recommended for the paper)")
    ap.add_argument("--auto-group", action="store_true",
                    help="keep near-duplicate image clusters inside one CV fold")
    ap.add_argument("--dup-threshold", type=float, help=f"near-duplicate correlation threshold (default {DUP_THRESHOLD})")
    ap.add_argument("--groups-csv", help="CSV 'image,group' (e.g. same intersection => same group); overrides --auto-group")
    ap.add_argument("--resume", action="store_true",
                    help="reuse saved OOF predictions in results/ (only if you changed nothing else)")
    ap.add_argument("--no-probe", action="store_true", help="skip the frozen-feature linear-probe baseline")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--synthetic", type=int, default=0,
                    help="SMOKE TEST ONLY: generate N fake images in an EMPTY data folder")
    ap.add_argument("--image", help="image path for --mode analyze (default: first test image)")
    ap.add_argument("--verbose", type=int, default=0, help="Keras verbosity during CV (0/1/2)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.data_root:
        set_data_root(args.data_root)
    upd = {}
    if args.epochs is not None:    upd["EPOCHS"] = args.epochs
    if args.folds is not None:     upd["N_FOLDS"] = args.folds
    if args.seed is not None:      upd["SEED"] = args.seed
    if args.finetune is not None:  upd["FINE_TUNE_LAYERS"] = args.finetune
    if args.flip_mode:             upd["FLIP_MODE"] = args.flip_mode
    if args.weights:               upd["PRETRAINED_WEIGHTS"] = None if args.weights == "none" else "imagenet"
    if args.img_size:              upd["IMG_SIZE"] = (args.img_size, args.img_size)
    if args.boot is not None:      upd["N_BOOT"] = args.boot
    if args.repeats is not None:   upd["REPEATS"] = max(1, args.repeats)
    if args.auto_group:            upd["AUTO_GROUP"] = True
    if args.dup_threshold:         upd["DUP_THRESHOLD"] = args.dup_threshold
    if args.groups_csv:            upd["GROUPS_CSV"] = args.groups_csv
    if args.no_probe:              upd["USE_PROBE"] = False
    if args.resume:                upd["RESUME"] = True
    globals().update(upd)

    print("=== AI Traffic Optimization System (Multi-Label) ===")
    print(f"[mode={args.mode}] data root: {DATA_ROOT}")

    if args.synthetic:
        if os.path.isdir(IMAGE_DIR) and os.listdir(IMAGE_DIR):
            sys.exit("[ERROR] --synthetic refuses to write into a non-empty Images folder.")
        generate_sample_data(args.synthetic)
        print("[SMOKE TEST] Synthetic data — results are meaningless, never report them.")

    if args.mode == "cv":
        run_experiments(collections.OrderedDict([(MAIN_NAME, {})]), "cv",
                        with_baselines=not args.no_baselines, verbose=args.verbose)
    elif args.mode == "ablate":
        chosen = [a.strip() for a in args.ablations.split(",") if a.strip()]
        unknown = [a for a in chosen if a not in ABLATIONS]
        if unknown:
            sys.exit(f"[ERROR] unknown ablation(s): {unknown}. Choose from {list(ABLATIONS)}")
        configs = collections.OrderedDict([(MAIN_NAME, {})])
        for a in chosen:
            label, ov = ABLATIONS[a]
            configs[label] = ov
        run_experiments(configs, "ablation", with_baselines=not args.no_baselines,
                        verbose=args.verbose)
    elif args.mode == "train":
        mode_train()
    else:
        mode_analyze(args.image)


if __name__ == "__main__":
    main()