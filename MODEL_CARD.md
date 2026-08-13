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

## INT8 optimization status

Version 0.6 applies validation-calibrated static S8S8 QDQ quantization to the global ONNX model.
Calibration uses 64 validation images, while quality is measured on all 137 held-out test images.
At the unchanged threshold, INT8 produced 0.293 IoU versus 0.286 for FP32 and 99.76% mean
binary-mask agreement. The model file decreased from 127,709 to 60,539 bytes.

INT8 did not improve latency on the measured CPU. Across five interleaved trials, mean inference
increased from 1.163 ms for FP32 to 2.534 ms for INT8 QDQ. These paired v0.6 timings use a newer
ONNX Runtime environment than the historical v0.4 benchmark and should not be compared across
versions as though the runtime and system state were identical. They exclude preprocessing,
postprocessing, I/O, ROS transport, and robot control.

The CPU INT8 model is therefore not selected as the latency deployment artifact. TensorRT FP16 and
INT8 remain target-hardware validation gates. See `docs/INT8_OPTIMIZATION_V06.md` and
`artifacts/reference/onnx_int8_v06.json`.

## Production pipeline status

Version 0.7 adds data and experiment provenance, container definitions, and safe cloud-transfer
planning. These features improve repeatability and traceability; they do not improve model
accuracy by themselves. A dataset fingerprint proves that the referenced bytes match the recorded
inventory, not that labels are correct, representative, unbiased, licensed for a particular use,
or suitable for aircraft inspection.

No real S3 upload, cloud GPU training, container runtime result, Isaac Sim capture, or TensorRT
benchmark is represented by the local v0.7 implementation. GitHub-hosted container jobs and an
authorized private-bucket test are separate evidence gates. See `docs/PRODUCTION_PIPELINE_V07.md`.

## Model compression status

Version 0.10 compares the v0.4 Tiny U-Net teacher against a one-level Compact U-Net trained with
and without soft-target knowledge distillation, plus a 25% structured-filter-pruned teacher. The
student changes the physical graph from ten to six spatial convolutions and, at eight base
channels, reduces parameters from 29,921 to 6,689. Each candidate uses the same manifest, receives
validation-only threshold selection, and is evaluated on the same held-out test split before ONNX
export and parity verification.

The deployment policy selects compact candidates using a 0.03 absolute validation-IoU budget,
then applies the same budget once as a held-out guardrail. It falls back to the teacher when either
gate fails. The pruning experiment preserves
dense tensor shapes. Its inactive-filter count is an accuracy/sparsity ablation and must not be
described as a size or latency improvement without structural compaction and target-runtime
measurement. Full methodology and limitations are in `docs/MODEL_COMPRESSION_V10.md`.

In the measured CPU run, the distilled student reached 0.328 held-out IoU and 0.494 Dice versus
0.286/0.445 for the teacher. It used 6,689 rather than 29,921 parameters and a 31,334-byte rather
than 127,709-byte ONNX file. The normally trained student reached only 0.142 IoU, so architecture
reduction alone was not sufficient. The selected model's mean ONNX Runtime latency improved from
6.496 ms to 5.358 ms, but the shared-host p95 distribution was wide; no control-loop deadline claim
is made. The portable evidence is `artifacts/reference/model_compression_v10.json`.

## Next validation gates

1. Add aircraft-like surface coupons with documented permissions and untouched external tests.
2. Compare a pretrained lightweight architecture against Tiny U-Net using identical splits.
3. Run 256 px training and benchmark CUDA, FP16 TensorRT, and INT8 TensorRT.
4. Evaluate domain shift from Isaac Sim synthetic data to held-out real images.
5. Validate the coverage planner after camera calibration, 3D surface projection, MoveIt 2
   collision checking, and force-control integration.
6. Benchmark the selected compact candidate on the target Jetson/RTX device and, separately,
   compact pruned channels into a physically smaller graph before making sparse-speedup claims.
