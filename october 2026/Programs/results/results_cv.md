# Results — cv

76 images, 5-fold iterative-stratified CV x 3 repeat(s), pooled out-of-fold predictions, 95% percentile-bootstrap CIs over images (1000 resamples).  Threshold for exact-match / precision / recall / F1 = 0.35.

## Table 0 — dataset (images containing each class)

| Class | Images | Prevalence |
|---|---|---|
| Left Turn Arrow | 38 | 0.50 |
| Right Turn Arrow | 36 | 0.47 |
| Stop Line | 47 | 0.62 |
| Solid White Line | 44 | 0.58 |
| Double Solid Yellow Line | 55 | 0.72 |

## Table A — overall

| Model | Macro AUC [95% CI] | Mean fold AUC ± SD | Macro AUC per repeat | Macro AP | Macro-F1 | Exact-match | Micro-F1 | BCE | Arrow L/R swap rate |
|---|---|---|---|---|---|---|---|---|---|
| EfficientNetV2S, head-only, label-swap flip  [MAIN] | 0.684 [0.63, 0.73] | 0.705 ± 0.062 | 0.691, 0.667, 0.693 | 0.739 | 0.743 | 0.184 | 0.747 | 0.620 | 0.030 |
| Baseline: class prevalence | 0.500 [0.50, 0.50] | — | 0.500, 0.500, 0.500 | 0.579 | 0.729 | 0.184 | 0.733 | 0.666 | 0.000 |
| Baseline: HOG + logistic regression | 0.646 [0.57, 0.71] | 0.645 ± 0.062 | 0.662, 0.650, 0.624 | 0.679 | 0.704 | 0.127 | 0.718 | 0.912 | 0.091 |
| Baseline: frozen EfficientNetV2S features + logistic regression | 0.738 [0.67, 0.79] | 0.734 ± 0.058 | 0.753, 0.741, 0.718 | 0.780 | 0.754 | 0.197 | 0.763 | 0.724 | 0.091 |

Prevalence baseline: constant predictor, so AUC = 0.5 by construction and AP = class prevalence; its value is as the BCE / exact-match / F1 reference.  The linear-probe baseline uses the same frozen ImageNet features as the CNN with a fixed-C logistic regression (no tuning).

## Table B — per-class ROC-AUC [95% CI]

| Model | Left Turn Arrow | Right Turn Arrow | Stop Line | Solid White Line | Double Solid Yellow Line |
|---|---|---|---|---|---|
| EfficientNetV2S, head-only, label-swap flip  [MAIN] | 0.666 [0.56, 0.76] | 0.732 [0.65, 0.82] | 0.672 [0.56, 0.77] | 0.774 [0.68, 0.86] | 0.575 [0.46, 0.70] |
| Baseline: class prevalence | 0.500 [0.50, 0.50] | 0.500 [0.50, 0.50] | 0.500 [0.50, 0.50] | 0.500 [0.50, 0.50] | 0.500 [0.50, 0.50] |
| Baseline: HOG + logistic regression | 0.573 [0.45, 0.69] | 0.584 [0.46, 0.71] | 0.658 [0.54, 0.77] | 0.702 [0.59, 0.80] | 0.711 [0.56, 0.84] |
| Baseline: frozen EfficientNetV2S features + logistic regression | 0.787 [0.68, 0.87] | 0.678 [0.56, 0.79] | 0.739 [0.62, 0.84] | 0.820 [0.72, 0.91] | 0.664 [0.52, 0.80] |

## Table C — per-class average precision (chance level = prevalence)

| Model | Left Turn Arrow | Right Turn Arrow | Stop Line | Solid White Line | Double Solid Yellow Line |
|---|---|---|---|---|---|
| *prevalence (chance)* | 0.500 | 0.474 | 0.618 | 0.579 | 0.724 |
| EfficientNetV2S, head-only, label-swap flip  [MAIN] | 0.662 | 0.702 | 0.753 | 0.808 | 0.769 |
| Baseline: class prevalence | 0.500 | 0.474 | 0.618 | 0.579 | 0.724 |
| Baseline: HOG + logistic regression | 0.537 | 0.525 | 0.744 | 0.760 | 0.831 |
| Baseline: frozen EfficientNetV2S features + logistic regression | 0.773 | 0.644 | 0.815 | 0.840 | 0.830 |

## Table E — per-class precision / recall / F1 at threshold 0.35

n/a = undefined (e.g. the model never predicts that class).

| Model | Left Turn Arrow | Right Turn Arrow | Stop Line | Solid White Line | Double Solid Yellow Line |
|---|---|---|---|---|---|
| EfficientNetV2S, head-only, label-swap flip  [MAIN] | 0.52 / 0.98 / 0.68 | 0.52 / 0.99 / 0.68 | 0.62 / 1.00 / 0.76 | 0.60 / 1.00 / 0.75 | 0.72 / 1.00 / 0.84 |
| Baseline: class prevalence | 0.50 / 1.00 / 0.67 | 0.47 / 1.00 / 0.64 | 0.62 / 1.00 / 0.76 | 0.58 / 1.00 / 0.73 | 0.72 / 1.00 / 0.84 |
| Baseline: HOG + logistic regression | 0.57 / 0.67 / 0.61 | 0.54 / 0.69 / 0.60 | 0.65 / 0.85 / 0.74 | 0.66 / 0.83 / 0.74 | 0.74 / 0.94 / 0.83 |
| Baseline: frozen EfficientNetV2S features + logistic regression | 0.65 / 0.83 / 0.73 | 0.58 / 0.68 / 0.63 | 0.71 / 0.85 / 0.77 | 0.74 / 0.92 / 0.82 | 0.76 / 0.90 / 0.82 |

## Table D — paired difference of the main model minus each alternative (bootstrap, same resamples)

| Alternative | Δ macro AUC [95% CI] | Δ arrow-class AUC [95% CI] | P(main ≤ alt) |
|---|---|---|---|
| Baseline: class prevalence | 0.185 [0.13, 0.23] | 0.199 [0.12, 0.26] | 0.000 |
| Baseline: HOG + logistic regression | 0.039 [-0.03, 0.10] | 0.122 [0.02, 0.23] | 0.125 |
| Baseline: frozen EfficientNetV2S features + logistic regression | -0.053 [-0.09, -0.01] | -0.033 [-0.08, 0.01] | 0.995 |

Positive Δ = main model better.  A CI containing 0 means the data cannot distinguish the two.  P(main ≤ alt) is the bootstrap fraction of resamples where the alternative matched or beat the main model (an approximate one-sided p-value; uncorrected for the number of comparisons — say so in the paper).

## Reading guide / limitations to state in the paper

- Classes whose AUC CI lies entirely above chance (0.5): Left Turn Arrow, Right Turn Arrow, Stop Line, Solid White Line.
- NOT distinguishable from chance: Double Solid Yellow Line.
- Main macro AUC 0.684 vs HOG + logistic regression 0.646: NOT distinguishable (CI contains 0): the trained CNN has not been shown to beat it.
- Main macro AUC 0.684 vs frozen EfficientNetV2S features + logistic regression 0.738: the CNN is significantly WORSE (CI excludes 0).
- Near-duplicate audit (r>0.97): 4 pairs, 8 images in clusters; fold grouping = auto near-duplicate clusters (r>0.97); pairs split across folds in repeat 1 = 0.  Splits that separate near-duplicates make all numbers optimistic.
- Dataset is small (76 images) and image-level (presence/absence), not localisation; images are treated as independent for the bootstrap unless grouped.
- Labels are derived from bounding-box XMLs; class-alias merges are listed in the config JSON.
- The THRESHOLD is fixed a priori, not tuned on evaluation data; AUC/AP are threshold-free.
- Scores are uncalibrated sigmoid outputs: do not describe them as probabilities in the paper.
