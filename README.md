# Constraint-Aware Traffic Intervention Visualization Using Multi-Label Road-Marking Recognition

A research prototype combining multi-label road-marking recognition, survey-based congestion categorization, constraint-filtered intervention ranking, and schematic visualization.

Developed between March and July 2025 as an exploration of peak-hour traffic near a school campus; revised and evaluated in October 2026. The development archive and revision are documented in the companion preprint at [`paper/research-1.pdf`](paper/research-1.pdf).

**Project status:** Proof of concept. Not a validated traffic-management or engineering system.

---

## Overview

The project consists of four independent modules that contribute to a combined report.

### 1. Road-marking recognition

An ImageNet-pretrained EfficientNetV2S backbone, initially frozen, with a trained sigmoid classification head over five road-marking classes:

- Left Turn Arrow
- Right Turn Arrow
- Stop Line
- Solid White Line
- Double Solid Yellow Line

The model performs image-level multi-label classification: it predicts whether a marking class is present or absent in an image.

It does not perform object localization, segmentation, or road-geometry extraction.

### 2. Congestion categorization

A structured roadway survey produces an additive congestion score, `C`, using:

- Road occupancy
- Average speed
- Reported peak periods
- Special events

The score categorizes conditions as:

- Not Congested
- Congested
- Severely Congested

### 3. Intervention ranking

A library of 30 physical, operational, and policy interventions is filtered against nine enforced restrictions.

Remaining interventions are ranked using hand-assigned impact, safety, and cost values. Detected road markings enter the ranking process as soft multipliers, as documented in the research paper.

### 4. Visualization

The system generates schematic overlays at fixed, image-relative positions.

These visualizations are illustrative only. They are not engineering drawings and are not derived from actual road geometry.

---

## Research Findings

The October 2026 evaluation reports repeated 5-fold stratified cross-validation on 76 images, with three repeats and bootstrap 95% confidence intervals.

Reported findings include:

- **Macro AUC: 0.684** (95% CI: 0.63–0.73) for the trained classification head.
- A **logistic-regression probe on the same frozen features reached 0.738** (95% CI: 0.67–0.79), outperforming the trained head.
- **Double Solid Yellow Line performed near chance**, with an AUC of 0.575 (95% CI: 0.46–0.70).
- At the fixed 0.35 threshold, recall was near 1.0 while precision was near class prevalence. The thresholded predictions therefore provided little useful information, as did the detection rules that consumed them.

These findings do not establish calibrated recognition, congestion reduction, or safe roadway designs.

See [`paper/research-1.pdf`](paper/research-1.pdf) for the full evaluation and discussion.

---

## Repository Structure

```text
TrafficCongestionWithAI/
│
├── archive/
│   ├── images/
│   ├── programs/
│   └── yolov & keras/
│
├── data/
│   ├── img/
│   │   └── road1.jpg ... road76.jpg
│   ├── annotations/
│   │   └── road1.xml ... road76.xml
│   ├── test/
│   │   └── test_Image-1.png
│   └── validation/
│       ├── img/
│       └── annotations/
│
├── october 2026/
│   └── Additional project files and programs
│
├── paper/
│   └── research-1.pdf
│
├── image_processing_pipeline.py
├── README.md
├── LICENSE
├── DATASET_LICENSE.md
├── CITATION.cff
├── requirements.txt
└── .gitignore
```

### Directory descriptions

| Directory | Contents |
|---|---|
| `archive/` | Archived images, Python programs, trained models, and YOLO weights |
| `data/img/` | 76 road images used for training and evaluation |
| `data/annotations/` | Pascal VOC XML annotations corresponding to the road images |
| `data/test/` | Separate sample test image |
| `data/validation/img/` | Validation road images |
| `data/validation/annotations/` | Corresponding validation annotations |
| `october 2026/` | Additional project files and programs organized in October 2026 |
| `paper/` | Research preprint and documentation |

---

## Installation

### Environment

The evaluation environment used Python 3.11.

Install dependencies:

```bash
pip install -r requirements.txt
```

### Clone the repository

```bash
git clone https://github.com/KingFarooq/TrafficCongestionWithAI.git
cd TrafficCongestionWithAI
```

---

## Reproduction

The October 2026 research document describes the following evaluation commands for the revised pipeline.

**Repeated cross-validation:**

```bash
python image_processing_pipeline.py --mode cv --repeats 3 --auto-group --seed 42
```

**Ablation experiments:**

```bash
python image_processing_pipeline.py --mode ablate \
  --ablations flip_naive,flip_none,finetune_10,finetune_50 \
  --repeats 3 --auto-group --seed 42
```

**Train the deployment model:**

```bash
python image_processing_pipeline.py --mode train
```

**Analyze an image:**

```bash
python image_processing_pipeline.py --mode analyze --image data/img/road1.jpg
```

These commands assume the root-level pipeline supports the documented options. Check the script's actual command-line interface if you encounter an argument or path error.

---

## Dataset

The training dataset contains 76 annotated aerial-map screenshots of roads in Georgia, USA.

Each training image has a corresponding Pascal VOC XML annotation.

The October 2026 evaluation uses repeated 5-fold stratified cross-validation across the 76 images. It does not use a fixed held-out test set for those reported cross-validation results.

The repository also contains a separate sample test image and a validation directory. These should not be confused with the evaluation protocol reported in the paper.

Refer to [`DATASET_LICENSE.md`](DATASET_LICENSE.md) for image and annotation usage terms.

---

## Known Limitations

1. **Heuristic decision logic:** Congestion weights, category cutoffs, impact/safety/cost values, and detection-rule factors are hand-assigned design inputs, not estimates derived from comprehensive traffic observations.

2. **Image-level recognition:** The classifier predicts the presence of road-marking classes but does not determine their precise locations or geometry.

3. **Uninformative thresholded outputs:** At the fixed 0.35 threshold, precision is near class prevalence and recall is near 1.0. The resulting detection rules can therefore fire with little dependence on image content.

4. **Weak class performance:** Double Solid Yellow Line recognition is near chance for the trained head.

5. **Evaluation limitations:** There is no location-disjoint evaluation split. Bootstrap confidence intervals treat images as independent, while repeated folds reuse the same images.

6. **Schematic overlays:** Visualizations use fixed, image-relative shapes rather than verified road geometry.

7. **No field validation:** No before-and-after studies, traffic simulations, or crash analyses establish that the system reduces congestion, delay, or crash risk.

### Defects identified in the 2025 archive

The October 2026 audit identified the following issues in the original implementation:

- The congestion speed term is skipped when speed is zero due to a truthiness bug.
- The `no new lanes` predicate over-matches and also removes bicycle-lane markings.
- The special-events prompt asks twice, and contradictory answers can change the event variable.
- The user-entered congestion label is not reconciled with the computed category.
- Outputs do not record model, class-schema, or dataset versions.

The system is not a substitute for professional traffic engineering or transportation planning.

---

## Future Work

Potential directions include:

- Versioned, location-disjoint evaluation splits with location metadata.
- Further comparison against linear models, including the logistic-regression probe.
- Class-specific precision-recall analysis.
- Probability calibration and threshold selection using nested cross-validation.
- Higher-resolution input for thin road markings.
- Explicit intervention attributes and a complete, unit-tested constraint registry.
- Geometry-aware visualization using segmentation or surveyed road geometry.
- Outcome evaluation through simulation and site-specific before-and-after studies.

---

## License

Source code is released under the MIT License, covering author-created source code only.

Dataset terms are separate. See [`DATASET_LICENSE.md`](DATASET_LICENSE.md).

---

## Citation

See [`CITATION.cff`](CITATION.cff) for citation metadata.

The research paper's arXiv identifier can be added once it has been assigned.

---

## Author

**Farooq M. Anwar**

GitHub: [@KingFarooq](https://github.com/KingFarooq)
