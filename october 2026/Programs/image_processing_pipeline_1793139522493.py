"""
=============================================================
  AI Traffic Optimization System — Multi-Label Edition
  Rebuilt data pipeline + model for multi-label road marking
  detection while preserving all original business logic:
      • CLASSES / RESTRICTIONS / SOLUTIONS registry
    • XML annotation parser (Pascal VOC format)
    • analyze_traffic / calculate_solution_score
    • visualize_solution
    • get_road_details survey
=============================================================
  KEY CHANGES vs. the previous single-label version
  ─────────────────────────────────────────────────
  ❌ image_dataset_from_directory  → sub-folder labels don't
     exist; every image can have MULTIPLE markings at once.
  ❌ label_mode="categorical"      → that's one-hot for a
      single class; we need a multi-hot binary vector per image.
  ❌ softmax + categorical_crossentropy → wrong for multi-label.
  ✅ Custom tf.data pipeline that pairs each image with its
      five-class multi-hot vector built from exact XML labels.
  ✅ sigmoid output (one independent probability per class).
  ✅ binary_crossentropy loss (each label treated separately).
  ✅ Per-class AUC as the primary evaluation metric.
=============================================================
"""

import os

# Set this before importing TensorFlow so the runtime sees the setting.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import xml.etree.ElementTree as ET
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — no display needed
import matplotlib.pyplot as plt

# Suppress noisy TF logs
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

# ── 1b: Portable paths; override TRAFFIC_DATA_DIR for another dataset.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("TRAFFIC_DATA_DIR", PROJECT_DIR / "data")).expanduser()
IMAGE_DIR = DATA_DIR / "Images"
ANNOTATIONS_DIR = DATA_DIR / "Annotations"
TEST_IMAGES_DIR = DATA_DIR / "Test Images"
MODEL_PATH = Path(os.environ.get("TRAFFIC_MODEL_PATH", DATA_DIR / "trained_model_multilabel.keras")).expanduser()
OUTPUT_DIR = Path(os.environ.get("TRAFFIC_OUTPUT_DIR", DATA_DIR / "Modified_Images")).expanduser()
DATASET_VERSION = os.environ.get("TRAFFIC_DATASET_VERSION", "unspecified")
ALLOW_SYNTHETIC_DATA = os.environ.get("TRAFFIC_ALLOW_SYNTHETIC_DATA", "").strip().lower() in {
    "1", "true", "yes"
}
SYNTHETIC_DATA_USED = False
CLASS_SCHEMA_VERSION = "road-markings-5class-v1"
SEED = 42

# ── 1c: Training hyperparameters
IMG_SIZE         = (224, 224)   # EfficientNetV2S native size
BATCH_SIZE       = 32
EPOCHS           = 25           # Phase-1: head only
FINE_TUNE_EPOCHS = 15           # Phase-2: top backbone layers
LR_HEAD          = 1e-3         # Phase-1 learning rate
LR_FINE          = 1e-5         # Phase-2 learning rate (very low)
THRESHOLD        = 0.35         # Sigmoid score → positive prediction
VAL_SPLIT        = 0.2          # 80/20 train/val split
AUTOTUNE         = tf.data.AUTOTUNE

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

# Explicit intervention capabilities. Filtering never infers these attributes
# from a solution's display name, preventing "Add bike lane markings" from
# being mistaken for construction of a new bike lane.
SOLUTION_TAGS = {
    "Add general purpose lane": {"creates_lane", "widening", "construction", "physical", "lane_related"},
    "Add auxiliary lane": {"creates_lane", "widening", "construction", "physical", "lane_related"},
    "Add climbing lane": {"creates_lane", "widening", "construction", "physical", "lane_related"},
    "Convert to HOV lane": {"hov_lane", "lane_conversion", "construction", "physical", "lane_related"},
    "Convert to bus lane": {"bus_lane", "lane_conversion", "construction", "physical", "lane_related", "transit_priority"},
    "Convert to bike lane": {"bike_facility", "lane_conversion", "construction", "physical", "lane_related"},
    "Convert shoulder to travel lane": {"shoulder_conversion", "creates_lane", "construction", "physical", "lane_related"},
    "Add roundabout": {"intersection_modification", "construction", "physical"},
    "Add dedicated turn lanes": {"turning_lane", "creates_lane", "intersection_modification", "construction", "physical", "lane_related"},
    "Add channelization islands": {"intersection_modification", "median_change", "curb_modification", "construction", "physical"},
    "Add pedestrian refuge island": {"intersection_modification", "median_change", "curb_modification", "construction", "physical", "pedestrian"},
    "Add pedestrian overpass": {"overpass", "road_elevation_change", "construction", "physical", "pedestrian"},
    "Add pedestrian underpass": {"underpass", "road_elevation_change", "construction", "physical", "pedestrian"},
    "Add vehicle overpass": {"overpass", "road_elevation_change", "construction", "physical"},
    "Optimize signal timing": {"signal_timing", "signal", "operational"},
    "Add adaptive signals": {"new_traffic_signal", "signal_timing", "signal", "smart_traffic_system", "operational"},
    "Add pedestrian signals": {"pedestrian_signal", "new_traffic_signal", "signal", "crosswalk_modification", "pedestrian", "operational"},
    "Coordinate signal systems": {"signal_timing", "signal", "smart_traffic_system", "operational"},
    "Refresh lane markings": {"lane_marking_change", "lane_related", "operational"},
    "Add turn arrows": {"new_road_marking", "lane_marking_change", "arrow_modification", "lane_related", "operational"},
    "Add bike lane markings": {"new_road_marking", "lane_marking_change", "bike_facility", "lane_related", "operational"},
    "Add crosswalk markings": {"new_road_marking", "lane_marking_change", "crosswalk_modification", "pedestrian", "operational"},
    "Add traffic cameras": {"surveillance_camera", "smart_traffic_system", "operational"},
    "Implement smart parking": {"parking_change", "smart_traffic_system", "operational"},
    "Add variable message signs": {"dynamic_signage", "smart_traffic_system", "operational"},
    "Install vehicle detection": {"vehicle_detection_system", "traffic_sensor", "smart_traffic_system", "operational"},
    "Implement congestion pricing": {"congestion_pricing", "policy"},
    "Add transit priority": {"transit_priority", "policy"},
    "Create delivery zones": {"delivery_zone", "loading_zone", "parking_change", "policy"},
    "Adjust parking regulations": {"parking_change", "policy"},
}

# Every user-visible restriction has an explicit capability predicate. Some
# restrictions currently match no item in the 30-option library; they remain
# implemented and will filter future options as soon as those capabilities are
# added. No unsupported options are invented here.
RESTRICTION_TAGS = {
    "no new lanes": {"creates_lane"},
    "no widening": {"widening"},
    "no construction": {"construction"},
    "no road elevation changes": {"road_elevation_change"},
    "no underpasses": {"underpass"},
    "no overpasses": {"overpass"},
    "no intersection modifications": {"intersection_modification"},
    "no median changes": {"median_change"},
    "no curb modifications": {"curb_modification"},
    "no sidewalk changes": {"sidewalk_modification"},
    "no bus lanes": {"bus_lane"},
    "no bike lanes": {"bike_facility"},
    "no hov lanes": {"hov_lane"},
    "no turning lanes": {"turning_lane"},
    "no reversible lanes": {"reversible_lane"},
    "no shoulder conversions": {"shoulder_conversion"},
    "no new road markings": {"new_road_marking"},
    "no lane marking changes": {"lane_marking_change"},
    "no crosswalk modifications": {"crosswalk_modification"},
    "no arrow modifications": {"arrow_modification"},
    "no stop line changes": {"stop_line_change"},
    "no yield markings": {"yield_marking"},
    "no new traffic signals": {"new_traffic_signal"},
    "no signal timing changes": {"signal_timing"},
    "no stop sign additions": {"stop_sign_addition"},
    "no yield sign additions": {"yield_sign_addition"},
    "no speed limit changes": {"speed_limit_change"},
    "no school zone modifications": {"school_zone_modification"},
    "no pedestrian signals": {"pedestrian_signal"},
    "no flashing beacons": {"flashing_beacon"},
    "no smart traffic systems": {"smart_traffic_system"},
    "no surveillance cameras": {"surveillance_camera"},
    "no vehicle detection systems": {"vehicle_detection_system"},
    "no dynamic signage": {"dynamic_signage"},
    "no license plate readers": {"license_plate_reader"},
    "no traffic sensors": {"traffic_sensor"},
    "no congestion pricing": {"congestion_pricing"},
    "no parking changes": {"parking_change"},
    "no loading zone changes": {"loading_zone"},
    "no transit priority": {"transit_priority"},
    "no truck restrictions": {"truck_restriction"},
    "no delivery zones": {"delivery_zone"},
    "no tree removal": {"tree_removal"},
    "no landscaping changes": {"landscaping_change"},
    "no drainage modifications": {"drainage_modification"},
    "no lighting changes": {"lighting_change"},
    "no sound barrier changes": {"sound_barrier_change"},
    "no wildlife crossings": {"wildlife_crossing"},
}
RESTRICTION_ALIASES = {
    "no lane additions": "no new lanes",
    "no lane addition": "no new lanes",
}


def normalize_restriction(restriction):
    """Return the canonical restriction label, supporting common synonyms."""
    normalized = " ".join(str(restriction or "").strip().lower().split())
    return RESTRICTION_ALIASES.get(normalized, normalized)


def blocked_by_restrictions(solution, restrictions):
    """Return the canonical restrictions that exclude this solution."""
    tags = set(solution.get("tags", ()))
    blocked = []
    for restriction in restrictions or ():
        canonical = normalize_restriction(restriction)
        prohibited_tags = RESTRICTION_TAGS.get(canonical)
        if prohibited_tags and tags.intersection(prohibited_tags):
            blocked.append(canonical)
    return blocked

# Create only output locations. Missing input folders should be reported, not
# silently populated with synthetic samples.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# Make each new training run repeatable. This does not make the historical
# training logs reproducible, nor does it substitute for repeated seeds.
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


# ─────────────────────────────────────────────────────────────
# SECTION 2 — ANNOTATION PARSER  (PRESERVED from original)
# Reads Pascal VOC XML files and converts them to the multi-hot
# binary label vector this model actually needs.
# ─────────────────────────────────────────────────────────────

def parse_annotation(xml_path: str):
    """
    Parse a Pascal VOC XML annotation file and return a
    multi-hot label vector of shape (NUM_CLASSES,).

    Multi-hot means each index is independently 0 or 1:
      [1, 0, 1, 0, ..., 1]   ← three markings present
    A single image can and should have multiple 1s.

    Args:
        xml_path : full path to the .xml annotation file

    Returns:
        np.ndarray of dtype float32, shape (NUM_CLASSES,), or None when
        the annotation is missing or unreadable. An existing, valid XML file
        with no objects is a legitimate all-zero label vector.
    """
    label_vector = np.zeros(NUM_CLASSES, dtype=np.float32)
    if not os.path.exists(xml_path):
        return None

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            name_node = obj.find("name")
            if name_node is None or not name_node.text:
                continue
            class_name = name_node.text.strip()
            if class_name in CLASSES:
                label_vector[CLASSES.index(class_name)] = 1.0
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
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"\n[WARNING] Generating {num_samples} SYNTHETIC training samples. "
        "This run must not be reported as real-data model evaluation."
    )

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

def collect_pairs():
    """
    Scan IMAGE_DIR for supported images, parse each matching XML
    to build a multi-hot vector, and return two parallel lists:
      image_paths : list of absolute file paths (str)
      label_matrix: np.ndarray of shape (N, NUM_CLASSES), float32

    This is purely a CPU/disk scan — no images are decoded here.
    """
    global SYNTHETIC_DATA_USED
    extensions = {".jpg", ".jpeg", ".png"}

    def scan_labeled_images():
        image_paths, label_rows = [], []
        candidate_count = 0
        if not IMAGE_DIR.is_dir():
            return image_paths, label_rows, candidate_count
        for fname in sorted(os.listdir(IMAGE_DIR)):
            if os.path.splitext(fname)[1].lower() not in extensions:
                continue
            candidate_count += 1
            img_path = IMAGE_DIR / fname
            xml_path = ANNOTATIONS_DIR / (Path(fname).stem + ".xml")
            label_vector = parse_annotation(xml_path)
            if label_vector is None:
                print(f"[WARN] Skipping image without a readable annotation: {img_path}")
                continue
            image_paths.append(str(img_path))
            label_rows.append(label_vector)
        return image_paths, label_rows, candidate_count

    image_paths, label_rows, candidate_count = scan_labeled_images()
    if candidate_count == 0 and ALLOW_SYNTHETIC_DATA:
        SYNTHETIC_DATA_USED = True
        generate_sample_data()
        image_paths, label_rows, candidate_count = scan_labeled_images()

    if not image_paths:
        raise ValueError(
            "No labeled training images were found. Put image/XML pairs in "
            f"{IMAGE_DIR} and {ANNOTATIONS_DIR}. To generate explicitly "
            "synthetic samples for a pipeline smoke test only, set "
            "TRAFFIC_ALLOW_SYNTHETIC_DATA=1."
        )

    label_matrix = np.asarray(label_rows, dtype=np.float32)
    print(
        f"[INFO] Found {len(image_paths)} images with valid annotations"
        f" ({candidate_count - len(image_paths)} skipped)."
    )
    if SYNTHETIC_DATA_USED:
        print("[WARNING] SYNTHETIC_DATA_USED=True")
    return image_paths, label_matrix


def build_datasets():
    """
    Build train and validation tf.data.Datasets from the collected
    image-path / multi-hot-label pairs.

    Returns:
        train_ds, val_ds — optimized tf.data.Dataset objects
    """
    image_paths, label_matrix = collect_pairs()
    N = len(image_paths)

    if N < 2:
        raise ValueError("At least two labeled images are required for a train/validation split.")

    # A seeded random image-level split replaces the old filesystem-order split.
    # Location-disjoint evaluation still requires location IDs, which this
    # archive does not provide.
    indices = np.random.default_rng(SEED).permutation(N)
    val_size = min(max(1, int(N * VAL_SPLIT)), N - 1)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    train_size = len(train_indices)
    train_ds = tf.data.Dataset.from_tensor_slices(
        ([image_paths[i] for i in train_indices], label_matrix[train_indices])
    )
    val_ds = tf.data.Dataset.from_tensor_slices(
        ([image_paths[i] for i in val_indices], label_matrix[val_indices])
    )

    print(
        f"[INFO] Train: {train_size} images | Val: {val_size} images "
        f"(seeded random image-level split; not location-disjoint)"
    )

    # ── 4d: Define the per-image decode function.
    # This runs inside .map() in parallel across CPU cores.
    # tf.io.read_file + tf.image.decode_* keeps everything in
    # the TF graph (no Python overhead per image).
    def load_and_preprocess(path, label):
        raw   = tf.io.read_file(path)                        # read bytes
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMG_SIZE)             # 224×224
        # Pixel values stay in [0, 255] here.
        # EfficientNetV2 preprocess_input (inside the model) handles
        # normalization to [-1, 1], so we don't rescale here.
        return image, label

    # ── 4e: Apply the decode function in parallel, then cache.
    # num_parallel_calls=AUTOTUNE → TF uses all available CPU cores.
    # .cache() stores decoded tensors after the first epoch so
    # disk I/O + JPEG decoding only happens once per run.
    train_ds = (
        train_ds
        .map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
        .cache()                          # ← saves fully decoded images
        .shuffle(buffer_size=1000, seed=SEED)  # re-shuffle each epoch
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)               # CPU prepares next batch while
    )                                     # GPU trains on current batch

    val_ds = (
        val_ds
        .map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
        .cache()
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)               # no shuffle for val
    )

    return train_ds, val_ds


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
      RandomRotation      — camera not perfectly level
      RandomZoom          — varying camera height / mounting distance
      RandomContrast      — glare, shadow, time-of-day variation
      RandomBrightness    — dawn vs. noon vs. overcast
      RandomTranslation   — marking at edge of frame
    """
    return keras.Sequential([
        # Do not flip horizontally: that changes left-turn arrows to right-
        # turn arrows without changing their labels.
        layers.RandomRotation(0.05),        # ±18°
        layers.RandomZoom(0.10),            # ±10%
        layers.RandomContrast(0.20),        # ±20% contrast
        layers.RandomBrightness(0.20),      # ±20% brightness
        layers.RandomTranslation(0.10, 0.10),
    ], name="augmentation")


@keras.utils.register_keras_serializable(package="TrafficPipeline")
class SubsetAccuracy(keras.metrics.Metric):
    """Exact-match multi-label accuracy (all labels must match per image)."""

    def __init__(self, threshold=THRESHOLD, name="subset_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred >= self.threshold, tf.float32)
        matches = tf.cast(tf.reduce_all(tf.equal(y_true, y_pred), axis=-1), tf.float32)
        if sample_weight is not None:
            weights = tf.cast(sample_weight, tf.float32)
            matches *= weights
            self.count.assign_add(tf.reduce_sum(weights))
        else:
            self.count.assign_add(tf.cast(tf.size(matches), tf.float32))
        self.correct.assign_add(tf.reduce_sum(matches))

    def result(self):
        return tf.math.divide_no_nan(self.correct, self.count)

    def reset_state(self):
        self.correct.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        return {**super().get_config(), "threshold": self.threshold}


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
#     output neuron and averages them.  It treats
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

def build_model(augmentation):
    """
    Multi-label EfficientNetV2S with two-phase training support.

    Returns:
        model      — full compiled Keras model
        base_model — reference to the backbone (for fine-tuning)
    """

    # ── 6a: Load EfficientNetV2S backbone, pre-trained on ImageNet.
    # include_top=False strips the ImageNet 1000-class head so we
    # can attach our own five-class sigmoid head.
    base_model = keras.applications.EfficientNetV2S(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
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

    # EfficientNetV2 built-in preprocessing: [0,255] → [-1, 1]
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
    outputs = layers.Dense(
        NUM_CLASSES,
        activation="sigmoid",           # ← multi-label output
        name="multi_label_output"
    )(x)

    model = keras.Model(inputs, outputs, name="MultiLabelRoadMarking_EfficientNetV2S")

    # ── 6d: Compile — Phase 1  ─────────────────────────────────
    # binary_crossentropy: treats each label as an independent
    # binary classification problem, averages loss across all classes.
    # AUC with multi_label=True reports the average ROC AUC
    # across the five classes — a much better signal than accuracy
    # for imbalanced multi-label problems.
    model.compile(
        optimizer=keras.optimizers.Adam(LR_HEAD),
        loss="binary_crossentropy",                          # ← multi-label loss
        metrics=[
            keras.metrics.AUC(name="auc", multi_label=True, num_labels=NUM_CLASSES),
            SubsetAccuracy(threshold=THRESHOLD),
            keras.metrics.BinaryAccuracy(name="mean_label_accuracy", threshold=THRESHOLD),
        ],
    )

    return model, base_model


# ─────────────────────────────────────────────────────────────
# SECTION 7 — FINE-TUNING  (REBUILT for EfficientNetV2S)
# ─────────────────────────────────────────────────────────────

def fine_tune(model, base_model, train_ds, val_ds):
    """
    Unfreeze the top 50 layers of EfficientNetV2S and continue
    training at LR=1e-5 to adapt high-level texture features
    to road marking patterns without catastrophic forgetting.
    """
    base_model.trainable = True

    # Keep early layers (edges, colours, textures) frozen.
    # Only the high-level feature layers near the output are
    # updated — they're already close to what we need.
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    # Very low LR avoids large gradient updates that would
    # destroy the carefully learned ImageNet representations.
    model.compile(
        optimizer=keras.optimizers.Adam(LR_FINE),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="auc", multi_label=True, num_labels=NUM_CLASSES),
            SubsetAccuracy(threshold=THRESHOLD),
            keras.metrics.BinaryAccuracy(name="mean_label_accuracy", threshold=THRESHOLD),
        ],
    )

    print("\n[Phase 2] Fine-tuning top 50 EfficientNetV2S layers...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max",
                patience=4, restore_best_weights=True
            ),
        ],
    )
    return history


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
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"cv2.imread returned None for {image_path}")

        img_rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMG_SIZE)
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
    print("Available Restrictions:")
    for i in range(0, len(RESTRICTIONS), 4):
        row = RESTRICTIONS[i:i+4]
        print("  ".join(f"{i+j+1}. {item:<30}" for j, item in enumerate(row)))
    print()

    details = {
        "name":            get_input("1. Road name: ", default="Unnamed Road"),
        "lanes":           get_input("2. Number of lanes: ", int),
        "current_cars":    get_input("3. Current cars on road: ", int),
        "capacity":        get_input("4. Maximum car capacity: ", int),
        "avg_speed":       get_input(
            "5. Observed mean travel speed (km/h), not posted speed limit: ", float
        ),
        "posted_speed_limit": get_input(
            "5a. Posted speed limit (km/h; recorded for reference only): ", float
        ),
        "intersection_type": get_input(
            "6. Intersection control type (Traffic light/Traffic signal/Signalized intersection/"
            "4-way stop/Uncontrolled/Other): ",
            str,
            ["Traffic light", "Traffic signal", "Signalized intersection", "4-way stop", "Uncontrolled", "Other"],
        ),
        "congestion_times":get_input("7. Peak congestion times (e.g., '7-9 AM, 4-6 PM'): "),
        "restrictions":    [],
        "special_events":  [],
        "has_crosswalk":   get_input("10. Pedestrian crossing? (y/n/N/A): ", bool),
        "has_bikelane":    get_input("11. Bike lane? (y/n/N/A): ", bool),
        "condition":       get_input("12. Road condition (Good/Fair/Poor/N/A): ", str, ["Good","Fair","Poor"]),
    }

    if get_input("Any special events? (y/n/N/A): ", bool):
        raw = get_input("9. Special events (comma separated): ", str)
        details["special_events"] = [e.strip() for e in raw.split(",") if e.strip()]

    if get_input("8. Any restrictions? (y/n/N/A): ", bool):
        print("\nSelect restrictions by number (comma separated):")
        while True:
            choices = input("Enter restriction numbers: ").strip()
            if choices.upper() == "N/A":
                break
            try:
                selected = [int(c.strip()) for c in choices.split(",") if c.strip()]
                valid    = [i for i in selected if 1 <= i <= len(RESTRICTIONS)]
                details["restrictions"] = [RESTRICTIONS[i-1] for i in valid]
                break
            except ValueError:
                print(f"Enter numbers between 1–{len(RESTRICTIONS)} separated by commas")

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

    tags = set(solution.get("tags", SOLUTION_TAGS.get(solution["name"], ())))
    # Context multipliers intentionally stack when multiple conditions match.
    if road_info.get("has_crosswalk") is True and "pedestrian" in tags:
        score *= 1.1
    if road_info.get("has_bikelane") is True and "bike_facility" in tags:
        score *= 1.1
    lanes = road_info.get("lanes")
    if lanes is not None and 0 < lanes <= 2 and "lane_related" in tags:
        score *= 1.15
    if road_info.get("congestion_times") and "signal" in tags:
        score *= 1.1
    return score


def _detected_marking_names(detected_elements):
    """Normalize inference tuples or plain names into exact canonical labels."""
    aliases = {
        "left arrow": "left turn arrow",
        "right arrow": "right turn arrow",
        "stopline": "stop line",
        "crosswalk": "crosswalk markings",
    }
    names = set()
    for element in detected_elements or ():
        raw_name = element[0] if isinstance(element, (tuple, list)) else element
        normalized = " ".join(
            str(raw_name).strip().lower().replace("_", " ").replace("-", " ").split()
        )
        names.add(aliases.get(normalized, normalized))
    return names


def analyze_traffic(road_info, detected_elements):
    congestion_score = 0.0

    current_cars = road_info.get("current_cars")
    capacity = road_info.get("capacity")
    if current_cars is not None and capacity is not None and capacity > 0:
        congestion_score += min(max(current_cars / capacity, 0), 1) * 0.5
    elif capacity is not None and capacity <= 0:
        print("[WARN] Capacity must be positive; occupancy term omitted.")

    avg_speed = road_info.get("avg_speed")
    if avg_speed is not None:
        congestion_score += max(0, (50 - avg_speed) / 50) * 0.3
    if road_info.get("congestion_times"):
        congestion_score += 0.1
    special_events = road_info.get("special_events") or []
    if isinstance(special_events, str):
        special_events = [event.strip() for event in special_events.split(",") if event.strip()]
    congestion_score += 0.1 * min(len(special_events), 3)

    if congestion_score >= 0.7:
        congestion = "Severely Congested"
    elif congestion_score >= 0.4:
        congestion = "Congested"
    else:
        congestion = "Not Congested"

    reasons = [
        f"Additive congestion index: {congestion_score:.2f} (theoretical maximum 1.20) → {congestion}"
    ]

    detected_names = _detected_marking_names(detected_elements)
    priority_factors = {}
    has_both_turn_arrows = {
        "left turn arrow", "right turn arrow"
    }.issubset(detected_names)
    if has_both_turn_arrows:
        priority_factors["Add dedicated turn lanes"] = 1.25
        reasons.append(
            "Both left- and right-turn arrows were detected; dedicated turn lanes "
            "are elevated for review, not asserted as an engineering conclusion."
        )

    crosswalk_detected = "crosswalk markings" in detected_names
    if crosswalk_detected and road_info.get("has_crosswalk") is not True:
        for name in (
            "Add pedestrian refuge island",
            "Add pedestrian signals",
            "Add crosswalk markings",
        ):
            priority_factors[name] = max(priority_factors.get(name, 1.0), 1.25)
        reasons.append(
            "Survey conflict: crosswalk markings were detected but the survey says "
            "no crossing or has no answer; pedestrian options are elevated for verification."
        )

    intersection_type = " ".join(
        str(road_info.get("intersection_type") or "").strip().lower().replace("_", " ").split()
    )
    stop_line_not_detected = "stop line" not in detected_names
    signalized_or_stop_controlled = any(
        marker in intersection_type
        for marker in ("traffic light", "traffic signal", "signalized", "4 way stop", "four way stop")
    )
    if stop_line_not_detected and signalized_or_stop_controlled:
        priority_factors["Refresh lane markings"] = max(
            priority_factors.get("Refresh lane markings", 1.0), 1.2
        )
        reasons.append(
            "No stop line was detected at a signalized or four-way-stop intersection; "
            "lane-marking refresh is elevated for manual inspection only."
        )

    all_solutions = [
        {**solution, "tags": SOLUTION_TAGS.get(solution["name"], set())}
        for category in SOLUTIONS.values()
        for solution in category
    ]
    restrictions = road_info.get("restrictions") or []
    canonical_restrictions = {
        normalize_restriction(item)
        for item in restrictions
        if normalize_restriction(item) in RESTRICTION_TAGS
    }
    unknown_restrictions = [
        item for item in restrictions
        if normalize_restriction(item) not in RESTRICTION_TAGS
    ]
    for item in unknown_restrictions:
        reasons.append(f"Unrecognized restriction ignored: {item}")
    for restriction in sorted(canonical_restrictions):
        tags = RESTRICTION_TAGS[restriction]
        if not any(
            tags.intersection(solution["tags"])
            for solution in all_solutions
        ):
            reasons.append(
                f"Restriction {restriction} is recognized, but no current option "
                "in the intervention library matches it."
            )

    viable = []
    excluded_count = 0
    for solution in all_solutions:
        blocked_by = blocked_by_restrictions(solution, restrictions)
        if blocked_by:
            excluded_count += 1
            reasons.append(
                f"Excluded {solution['name']} by {', '.join(blocked_by)}."
            )
            continue
        score = calculate_solution_score(solution, congestion, road_info)
        score *= priority_factors.get(solution["name"], 1.0)
        viable.append({**solution, "score": score})

    if excluded_count == 0 and restrictions and not canonical_restrictions:
        reasons.append("Selected restrictions did not exclude any option in the current library.")

    if congestion == "Not Congested":
        reasons.append("No intervention recommendations are issued below the congestion threshold.")
        return congestion, reasons, []

    viable.sort(key=lambda solution: solution["score"], reverse=True)
    n_top = 5 if congestion == "Severely Congested" else 4
    top_solutions = [solution["name"] for solution in viable[:n_top]]

    return congestion, reasons, top_solutions


def visualize_solution(original_img, road_info, solutions, detected_elements):
    img    = original_img.copy()
    h, w   = img.shape[:2]
    pil    = Image.fromarray(img)
    draw   = ImageDraw.Draw(pil)
    font   = ImageFont.load_default()
    large  = ImageFont.load_default()

    draw.text((20, 20), road_info.get("name", "Unnamed Road"), fill=(0,0,0), font=large)
    draw.text((20, 50), f"Condition: {road_info.get('condition', 'N/A')}", fill=(0,0,0), font=font)
    posted_limit = road_info.get("posted_speed_limit")
    if posted_limit is not None:
        draw.text(
            (20, 70),
            f"Posted speed limit: {posted_limit:g} km/h (reference)",
            fill=(0,0,0),
            font=font,
        )

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
    visualized = []
    for sol in solutions:
        if sol in visual_map and applied < 3:
            visual_map[sol]()
            applied += 1
            visualized.append(sol)

    not_visualized = [sol for sol in solutions if sol not in visualized]
    return img, visualized, not_visualized


def write_output_manifest(image_path):
    """Write reproducibility metadata beside a generated visualization."""
    digest = None
    if MODEL_PATH.is_file():
        hasher = hashlib.sha256()
        with MODEL_PATH.open("rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()

    manifest = {
        "image_file": Path(image_path).name,
        "model_sha256": digest,
        "class_schema_version": CLASS_SCHEMA_VERSION,
        "classes": CLASSES,
        "dataset_version": DATASET_VERSION,
        "threshold": THRESHOLD,
        "random_seed": SEED,
        "synthetic_data_used": SYNTHETIC_DATA_USED,
    }
    manifest_path = Path(str(image_path) + ".manifest.json")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest_path


def _binary_classification_metrics(labels, scores, threshold, ece_bins=10):
    """Compute per-label metrics without requiring scikit-learn."""
    count = len(labels)
    positives = sum(labels)
    negatives = count - positives
    predicted = [int(score >= threshold) for score in scores]
    true_positive = sum(y == 1 and p == 1 for y, p in zip(labels, predicted))
    false_positive = sum(y == 0 and p == 1 for y, p in zip(labels, predicted))
    false_negative = sum(y == 1 and p == 0 for y, p in zip(labels, predicted))
    true_negative = sum(y == 0 and p == 0 for y, p in zip(labels, predicted))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / positives if positives else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if recall is not None and precision + recall
        else (0.0 if recall is not None else None)
    )

    # Tie-aware rank AUC; undefined when the sample has only one class.
    roc_auc = None
    if positives and negatives:
        ordered = sorted(zip(scores, labels), key=lambda pair: pair[0])
        positive_rank_sum = 0.0
        i = 0
        while i < count:
            j = i + 1
            while j < count and ordered[j][0] == ordered[i][0]:
                j += 1
            average_rank = ((i + 1) + j) / 2.0
            positive_rank_sum += average_rank * sum(label for _, label in ordered[i:j])
            i = j
        roc_auc = (
            positive_rank_sum - positives * (positives + 1) / 2
        ) / (positives * negatives)

    # Average precision with tied scores grouped at the same threshold.
    average_precision = None
    if positives:
        ordered = sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=True)
        cumulative_tp = cumulative_fp = 0
        previous_recall = 0.0
        average_precision = 0.0
        i = 0
        while i < count:
            j = i + 1
            while j < count and ordered[j][0] == ordered[i][0]:
                j += 1
            group_tp = sum(label for _, label in ordered[i:j])
            group_fp = (j - i) - group_tp
            cumulative_tp += group_tp
            cumulative_fp += group_fp
            recall_at_threshold = cumulative_tp / positives
            precision_at_threshold = cumulative_tp / (cumulative_tp + cumulative_fp)
            average_precision += (
                recall_at_threshold - previous_recall
            ) * precision_at_threshold
            previous_recall = recall_at_threshold
            i = j

    brier = sum((score - label) ** 2 for label, score in zip(labels, scores)) / count if count else None
    calibration_bins = []
    ece = 0.0
    for bin_index in range(ece_bins):
        lower = bin_index / ece_bins
        upper = (bin_index + 1) / ece_bins
        members = [
            (label, score)
            for label, score in zip(labels, scores)
            if lower <= score < upper or (bin_index == ece_bins - 1 and score == 1.0)
        ]
        if members:
            mean_score = sum(score for _, score in members) / len(members)
            observed_rate = sum(label for label, _ in members) / len(members)
            ece += len(members) / count * abs(mean_score - observed_rate)
        else:
            mean_score = observed_rate = None
        calibration_bins.append({
            "lower": lower,
            "upper": upper,
            "count": len(members),
            "mean_score": mean_score,
            "observed_positive_rate": observed_rate,
        })

    return {
        "sample_count": count,
        "positive_count": positives,
        "negative_count": negatives,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "brier_score": brier,
        "ece": ece if count else None,
        "calibration_bins": calibration_bins,
    }


def evaluate_multilabel_scores(y_true, y_scores, class_names, threshold=THRESHOLD, seed=SEED):
    """Report per-class metrics and baseline comparisons for a labeled split."""
    if len(y_true) == 0 or len(y_true) != len(y_scores):
        raise ValueError("y_true and y_scores must contain the same non-zero number of samples.")
    class_count = len(class_names)
    if any(len(row) != class_count for row in y_true) or any(
        len(row) != class_count for row in y_scores
    ):
        raise ValueError("Each truth and score row must match the class schema length.")

    truth = [[int(value >= 0.5) for value in row] for row in y_true]
    scores = [[float(value) for value in row] for row in y_scores]

    def evaluate(scores_for_model):
        per_class = {}
        for class_index, class_name in enumerate(class_names):
            labels = [row[class_index] for row in truth]
            class_scores = [row[class_index] for row in scores_for_model]
            per_class[class_name] = _binary_classification_metrics(
                labels, class_scores, threshold
            )

        flattened_truth = [value for row in truth for value in row]
        flattened_scores = [value for row in scores_for_model for value in row]
        micro = _binary_classification_metrics(
            flattened_truth, flattened_scores, threshold
        )
        sample_exact = sum(
            all(
                int(score >= threshold) == label
                for label, score in zip(label_row, score_row)
            )
            for label_row, score_row in zip(truth, scores_for_model)
        ) / len(truth)
        macro_classes = [
            metrics for metrics in per_class.values()
            if metrics["positive_count"] and metrics["negative_count"]
        ]

        def macro(key):
            values = [metrics[key] for metrics in macro_classes if metrics[key] is not None]
            return sum(values) / len(values) if values else None

        return {
            "threshold": threshold,
            "subset_accuracy": sample_exact,
            "mean_label_accuracy": sum(
                int(label == int(score >= threshold))
                for label_row, score_row in zip(truth, scores_for_model)
                for label, score in zip(label_row, score_row)
            ) / (len(truth) * class_count),
            "micro": micro,
            "macro": {
                "class_count_with_both_labels": len(macro_classes),
                "precision": macro("precision"),
                "recall": macro("recall"),
                "f1": macro("f1"),
                "roc_auc": macro("roc_auc"),
                "average_precision": macro("average_precision"),
            },
            "per_class": per_class,
        }

    prevalence = [
        sum(row[class_index] for row in truth) / len(truth)
        for class_index in range(class_count)
    ]
    rng = random.Random(seed)
    baseline_scores = {
        "random_uniform": [
            [rng.random() for _ in range(class_count)] for _ in truth
        ],
        "prevalence_probability": [
            list(prevalence) for _ in truth
        ],
        "majority_class": [
            [float(rate >= 0.5) for rate in prevalence] for _ in truth
        ],
        "all_labels_positive": [
            [1.0] * class_count for _ in truth
        ],
        "heuristic_stop_line_and_solid_white_line": [
            [
                float(class_name in {"Stop Line", "Solid White Line"})
                for class_name in class_names
            ]
            for _ in truth
        ],
    }
    return {
        "score_interpretation": (
            "Raw model scores are uncalibrated; ECE, Brier score, and reliability-bin "
            "values are diagnostics, not proof of calibration."
        ),
        "model": evaluate(scores),
        "baselines": {
            name: evaluate(baseline)
            for name, baseline in baseline_scores.items()
        },
    }


def evaluate_validation_split(model, dataset):
    """Evaluate the training run's validation split; this is not a test set."""
    y_true, y_scores = [], []
    for batch_images, batch_labels in dataset:
        predictions = model.predict(batch_images, verbose=0)
        y_true.extend(batch_labels.numpy().tolist())
        y_scores.extend(predictions.tolist())
    return evaluate_multilabel_scores(y_true, y_scores, CLASSES)


# ─────────────────────────────────────────────────────────────
# SECTION 10 — MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=== AI Traffic Optimization System (Multi-Label) ===\n")
    print(
        "[WARNING] EfficientNetV2S has no archived training log. "
        "Metrics from this run are new results and do not reproduce the preprint."
    )
    print(
        "[WARNING] Intervention rankings are heuristic research outputs; "
        "visual overlays are schematic and are not engineering designs."
    )

    # ── Build datasets
    print("[Step 1/4] Building tf.data pipelines...")
    train_ds, val_ds = build_datasets()

    # ── Build model
    print("[Step 2/4] Building EfficientNetV2S multi-label model...")
    augmentation     = build_augmentation()
    model, base_model = build_model(augmentation)
    model.summary()

    # ── Phase 1: train head only
    print("\n[Step 3/4] Phase 1 — Training classification head (backbone frozen)...")
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                str(MODEL_PATH), save_best_only=True, monitor="val_auc", mode="max"
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=5, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_auc", mode="max", factor=0.5, patience=3
            ),
        ],
    )

    # ── Phase 2: fine-tune top backbone layers
    print("[Step 4/4] Phase 2 — Fine-tuning top backbone layers...")
    h2 = fine_tune(model, base_model, train_ds, val_ds)

    model.save(str(MODEL_PATH))
    print(f"\n[Done] Model saved → {MODEL_PATH}")

    validation_report = evaluate_validation_split(model, val_ds)
    validation_report.update({
        "split": "validation",
        "is_held_out_test": False,
        "dataset_version": DATASET_VERSION,
        "synthetic_data_used": SYNTHETIC_DATA_USED,
        "warning": (
            "This split is used for early stopping and is not an independent test set. "
            "Do not use these results as unbiased generalization estimates."
        ),
    })
    metrics_path = OUTPUT_DIR / (
        f"validation_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(validation_report, metrics_file, indent=2)
        metrics_file.write("\n")
    print(f"[Done] Per-class metrics and baseline report saved → {metrics_path}")

    # ── Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (m, vm), title in zip(
        axes,
        [("auc", "val_auc"), ("loss", "val_loss")],
        ["AUC (multi-label)", "Binary Cross-Entropy Loss"]
    ):
        for h, label in [(h1, "phase-1"), (h2, "phase-2")]:
            ax.plot(h.history[m],  label=f"train ({label})")
            ax.plot(h.history[vm], label=f"val ({label})",  linestyle="--")
        ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves_multilabel.png", dpi=150)
    print("[Done] Curves → training_curves_multilabel.png")

    # ── Run inference on test images
    test_imgs = sorted(
        f for f in os.listdir(TEST_IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ) if TEST_IMAGES_DIR.is_dir() else []
    if not test_imgs:
        print("[WARN] No inference images found in TEST_IMAGES_DIR.")
        return
    print(
        "[INFO] These images are used for inference demonstration only. "
        "Without paired ground-truth annotations they are not an evaluation test set."
    )

    road_info = get_road_details()
    image_path = os.path.join(TEST_IMAGES_DIR, test_imgs[0])
    print(f"\n[Inference] Processing: {test_imgs[0]}")

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
    if not solutions:
        print("    None")
    for i, sol in enumerate(solutions, 1):
        print(f"    {i}. {sol}")

    modified, visualized, not_visualized = visualize_solution(
        original_img, road_info, solutions, detected
    )
    if visualized:
        print(f"\n  Visualized interventions: {', '.join(visualized)}")
    if not_visualized:
        print(
            "\n[INFO] Ranked but not visualized (no validated drawing routine or "
            f"the 3-overlay limit): {', '.join(not_visualized)}"
        )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_road_name = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(road_info.get("name") or "unnamed_road")
    ).strip("_") or "unnamed_road"
    out_path = OUTPUT_DIR / f"{safe_road_name}_{ts}.jpg"
    if not cv2.imwrite(str(out_path), cv2.cvtColor(modified, cv2.COLOR_RGB2BGR)):
        raise IOError(f"Failed to save visualization image: {out_path}")
    manifest_path = write_output_manifest(out_path)
    print(f"\n[Done] Annotated image saved → {out_path}")
    print(f"[Done] Reproducibility manifest saved → {manifest_path}")


if __name__ == "__main__":
    main()