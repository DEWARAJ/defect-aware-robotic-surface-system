# Real-Data Segmentation Protocol

## Dataset sources

The training pipeline accepts two sources:

1. An official MVTec AD download in its original directory structure.
2. User-owned RGB photographs paired with same-size binary masks.

MVTec AD is released under CC BY-NC-SA 4.0 and is restricted to non-commercial use. The repository
does not redistribute its images. Dataset provenance and license details must remain attached to
any report or model trained with it.

## Protocol disclosure

MVTec AD's official task is unsupervised anomaly detection: training images are defect-free and
anomalies occur in the test collection. A supervised segmentation network requires positive masks
during development. Therefore this project labels its protocol `supervised-development` and
deterministically reallocates anomaly images into train, validation, and held-out test groups.

Results from this split must never be reported as official MVTec AD benchmark results. They answer
a different engineering question: whether a supervised segmentation and deployment pipeline works
when annotated defect examples are available.

## Leakage controls

- Every source image appears in exactly one split.
- Splits are stable for a given seed and recorded in `manifest.json`.
- Manifest contents are hashed into the model checkpoint and training report.
- Threshold selection uses validation data only.
- Test evaluation is a separate command after checkpoint selection.
- Reports retain per-category and per-defect measurements.

## Augmentation

Training applies deterministic paired flips and 90-degree rotations to images and masks. Brightness,
contrast, and sensor-noise changes affect the RGB image only. The sample identifier, epoch, and
global seed determine every transformation, which makes a run reproducible while still varying
augmentation across epochs.

## Model selection and evaluation

The initial deep model is a compact U-Net trained with weighted binary cross-entropy plus soft Dice
loss. Positive-pixel weighting is calculated from the training split and capped to avoid unstable
gradients. Validation IoU selects the checkpoint and decision threshold. The final report includes
IoU, Dice, precision, recall, pixel accuracy, inference latency, category/defect breakdowns, and a
failure gallery.

## Deployment gates

1. Export the selected checkpoint to ONNX.
2. Require maximum PyTorch/ONNX logit error at or below `1e-4`.
3. Require binary mask agreement of at least `99.9%` on parity samples.
4. Benchmark inference latency after warmup.
5. Re-run segmentation metrics using the deployed runtime before connecting it to planning.

## Real-world evidence still required

- Annotated aircraft-like aluminum or composite test coupons.
- Multiple cameras, exposure levels, viewing angles, and surface finishes.
- Separate workpieces for train, validation, and test—not merely different crops.
- Annotation review by a second person.
- Calibration, uncertainty margins, and failure analysis before robotic use.
