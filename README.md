# Traffic Congestion Analysis & Intervention Visualization

A research prototype that combines multi-label road-marking recognition, survey-driven congestion categorization, constraint-filtered intervention ranking, and schematic visualization. Developed between March and July 2025 as an exploration of peak-hour traffic near a school campus, and later subjected to a reproducibility audit that is available as an arXiv preprint.

**Status:** Proof of concept. Not a validated traffic-management system. See [Known issues](#known-issues) and the paper for the full audit.

---

## What this project is

The prototype has four independent modules:

1. **Recognition** — A compact convolutional neural network (CNN) trained on 76 annotated aerial-map screenshots. Detects 20 road-marking classes (arrows, stop lines, crosswalks, lane markings) as **image-level** multi-label outputs. Not object detection; not segmentation.

2. **Congestion index** — A user-supplied survey (lane count, current vehicles, capacity, average speed, peak periods, special events) produces an additive score `C` categorized as *Not Congested*, *Congested*, or *Severely Congested*.

3. **Intervention ranking** — A library of 30 physical, operational, and policy interventions is filtered against user-selected restrictions and ranked by hand-assigned impact, safety, and cost values.

4. **Visualization** — Schematic overlays (fixed, image-relative rectangles and circles) are drawn on the input image to illustrate the top-ranked interventions.

The four modules produce a combined report. **Important:** the recognition module's output does not influence the recommendation module's output — the classifier's predictions appear in the report but never reach the ranking logic. This is one of the audit findings documented in the paper; see [Known issues](#known-issues).

---

## What this project is not

- **Not a validated traffic model.** The congestion index and intervention ranking are heuristics. The weights (0.5 occupancy, 0.3 speed, 0.1 peak period, 0.1 special events) and the impact/safety/cost values in the intervention library are hand-assigned design inputs, not estimates derived from traffic observations.
- **Not a detection or segmentation system.** The classifier predicts whether a marking class is *present in an image*. It does not localize markings, measure their geometry, or extract lane boundaries.
- **Not deployment-ready.** The overlays are schematic communication aids drawn at fixed image positions. They are not engineering drawings and do not encode road geometry.
- **Not a mitigation tool.** No before–after studies, simulations, or field measurements were conducted. The prototype has not been shown to reduce congestion, delay, or crash risk.

---

## Quick start

```bash
git clone https://github.com/KingFarooq/TrafficCongestionWithAI.git
cd TrafficCongestionWithAI
pip install tensorflow opencv-python numpy matplotlib pillow
python main.py
