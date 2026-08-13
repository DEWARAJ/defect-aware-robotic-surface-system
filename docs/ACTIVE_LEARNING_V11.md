# Risk-Aware Human Review (v0.11)

## Why this milestone exists

An industrial perception system should know when a human needs to inspect its output. v0.11 adds
a reproducible review-acquisition layer around the selected v0.10 compact model. It calibrates
probabilities, estimates input shift, learns which validation samples tend to fail, and ranks a
fixed annotation budget without reading pool masks, anomaly flags, or defect labels.

This is a **retrospective development study**, not a claim of production active learning. The
MVTec test split is label-free while a queue is built, then labels are revealed for audit. Pool
results were inspected while developing this milestone, so a new untouched external dataset is
required to confirm the measured lift.

## Leakage boundary

1. Fit one temperature on validation pixels by minimizing binary NLL.
2. Preserve the original deployed logit boundary by transforming its probability threshold.
3. Compute label-free RGB statistics and model-output statistics for validation and pool images.
4. Define validation failure severity from validation masks only.
5. Search kNN neighborhood and acquisition-weight candidates with leave-one-out validation.
6. Freeze the selected policy and produce `review_queue.json` from pool images and predictions.
7. Reveal pool labels once for the retrospective comparison.

The queue artifact contains hashed review IDs, acquisition scores, predicted failure risk,
uncertainty, input-shift score, and reasons. It does not contain source paths, masks, anomaly flags,
defect types, or failure measurements.

## Measured development result

The study used 175 validation images and a 137-image retrospective pool at 128 x 128 resolution.
The selected policy used eight inverse-distance validation neighbors, 0.8 failure-risk weight, 0.2
entropy weight, no shift weight, and 0.15 diversity weight.

| Measure | Result |
|---|---:|
| Fitted temperature | 0.6193 |
| Pool pixel ECE, before -> after | 0.0706 -> 0.0478 |
| Pool pixel NLL, before -> after | 0.1140 -> 0.0985 |
| Thresholded-mask agreement after calibration | 100% |
| Review budget | 20 / 137 (14.6%) |
| Worst-quartile failures found | 8 / 35 |
| Worst-quartile recall | 22.9% |
| Hit lift vs. random expectation | 1.57x |
| Risk-only diagnostic ablation | 12 / 35, 2.35x random expectation |
| Entropy-only diagnostic ablation | 3 / 35, 0.59x random expectation |
| Shift-only diagnostic ablation | 0 / 35 |

The validation-selected mixed policy is the reported primary result even though the retrospective
risk-only ablation did better on this pool. Selecting the risk-only result after seeing pool labels
would be cherry-picking.

Calibration improves aggregate pixel NLL and ECE but worsens class-balanced NLL from 0.3703 to
0.4141. Background pixels dominate this sparse-defect task, so the calibrated probabilities must
not be described as uniformly better for both classes. The original binary masks are unchanged.

## Controlled input-shift probes

The label-free shift score is a ridge-regularized Mahalanobis distance over RGB summary,
grayscale-histogram, and gradient features fitted on validation images.

| Controlled shift | Clean-vs-shift AUROC | Shift detection rate |
|---|---:|---:|
| Gaussian noise (sigma 0.18) | 0.8249 | 100% |
| Brightness scale 0.45 | 0.9991 | 100% |
| Gaussian blur (radius 2.0) | 0.7324 | 68.6% |

The same validation-percentile threshold flags 36.5% of clean development-pool images. This
reflects category/domain mismatch and makes the detector too noisy for an automatic stop signal.
The controlled probes demonstrate sensitivity to known perturbations; they do not establish
open-world out-of-distribution detection.

## Reproduce

Install the optional ML dependencies and make sure the v0.4 manifest and selected v0.10 compact
checkpoint exist, then run:

```powershell
./scripts/run_active_learning.ps1 -PythonExecutable python
```

Primary outputs:

- `runs/active_learning_v11/active_learning_study.json`: complete aggregate and sample audit.
- `runs/active_learning_v11/review_queue.json`: label-free operational queue.
- `runs/active_learning_v11/retrospective_evaluation.json`: labels revealed after queue freeze.
- `runs/active_learning_v11/review_gallery/`: RGB, probability, entropy, and mask-overlay panels.
- `artifacts/reference/active_learning_v11.json`: portable aggregate evidence without sample data.

## Next gates

1. Freeze the v0.11 policy and confirm queue lift on an untouched external surface dataset.
2. Add deep-ensemble or MC-dropout uncertainty and compare it against failure-risk kNN.
3. Annotate reviewed samples, retrain, and plot model improvement versus annotation budget.
4. Calibrate per category or with class-aware objectives while preserving deployment decisions.
5. Replace handcrafted shift features with a validated embedding detector and target false-alarm
   budget.
