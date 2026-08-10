# Model Card: Surface Perception Models

## Model

The first runnable model is a balanced pixel-level logistic classifier over RGB, luminance,
gradient, and local-contrast features. It exists to validate data generation, deterministic
training, threshold selection, metric computation, artifact handling, latency measurement, and
ROS/deployment interfaces before the CNN is introduced.

## Intended use

- Research and software-pipeline validation.
- Demonstrating reproducibility and segmentation evaluation.
- Establishing a benchmark that a U-Net or DeepLab model must beat.

## Not intended for

- Production inspection or safety decisions.
- Claims about real aircraft surfaces.
- Comparison with commercial defect-detection systems.
- Directly commanding a sanding tool or safety-critical robot motion.

## Data

Version 0.1 uses procedurally generated brushed-metal images containing synthetic scratches,
corrosion patches, and pits. The test split is deterministic and disjoint from training and
validation, but it comes from the same generator. High scores therefore measure pipeline
correctness, not real-world generalization.

## Current measured result

The local reference run uses 120 images at 96 x 96 pixels, 84 for training, 18 for validation,
and 18 for testing. The generated `runs/mvp/reports/test_report.json` file is the source of truth
for current metrics and latency.

## Real-data model status

Version 0.4 adds a measured MVTec AD supervised-development experiment using `grid`, `metal_nut`,
and `screw`. It is not the official unsupervised MVTec AD protocol because official anomaly test
images were deterministically reallocated across train, validation, and held-out test splits. Raw
MVTec data and model weights are not redistributed.

The global 29,921-parameter Tiny U-Net used 128 x 128 inputs and reached 0.286 held-out IoU and
0.445 Dice across 137 images. Its high 0.762 recall came with 0.314 precision. Per-category review
showed that aggregate performance was dominated by `metal_nut`; the global model reached zero IoU
on the grid-only held-out subset. A grid-specialized retraining recovered grid IoU to 0.165, with
0.300 precision and 0.268 recall across 33 images, but did not resolve every defect type.

Both exported ONNX models achieved 100% thresholded-mask agreement with PyTorch over ten parity
samples. On CPU, global ONNX mean/P95 latency was 3.58/14.49 ms and grid-specialized latency was
1.56/3.36 ms. These timings are environment-specific and exclude camera capture, preprocessing,
postprocessing, ROS transport, and robot control.

The machine-readable evidence is `artifacts/reference/real_mvtec_v04.json`; detailed limitations
and failure-driven iteration are in `docs/MVTEC_V04_RESULTS.md`.

## Next validation gates

1. Add aircraft-like surface coupons with documented permissions and untouched external tests.
2. Compare a pretrained lightweight architecture against Tiny U-Net using identical splits.
3. Run 256 px training and benchmark CUDA, FP16 TensorRT, and INT8 TensorRT.
4. Evaluate domain shift from Isaac Sim synthetic data to held-out real images.
5. Validate the coverage planner after camera calibration, 3D surface projection, MoveIt 2
   collision checking, and force-control integration.
