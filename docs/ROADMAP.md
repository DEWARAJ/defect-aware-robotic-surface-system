# Engineering Roadmap

## Milestone 1 - Reproducible baseline (implemented)

- Deterministic synthetic RGB images and binary masks.
- Explicit train, validation, and test splits.
- Learned baseline with balanced sampling and threshold tuning.
- IoU, Dice/F1, precision, recall, accuracy, and latency reports.
- Saved prediction previews and a model artifact.
- Unit tests and CI workflow.

## Milestone 1.5 - Perception-to-action planning (implemented)

- Aerospace-panel-like workcell scene with workpiece, sanding, protected, and defect masks.
- Tool-radius-aware erosion and swept-area safety verification.
- Boustrophedon coverage path with alternating lane direction.
- Reduced feed scale on defect-intersecting segments and local cleanup-pass support.
- JSON plan, normalized waypoint CSV, semantic masks, and recruiter-ready visual preview.
- Modern C++17 ROS2 planning core and normalized path publisher foundation.

## Milestone 2 - Deep segmentation (measured v0.4 run complete)

- Implemented MVTec AD/custom-pair validation, deterministic splitting, and leakage controls.
- Implemented paired augmentation and a Tiny U-Net with weighted BCE + Dice loss.
- Implemented early stopping, threshold selection, metadata checkpoints, and manifest hashing.
- Implemented held-out metrics, per-defect analysis, failure galleries, ONNX export, and parity.
- Completed licensed MVTec AD download and measured CPU run at 128px on 1,157 images.
- Completed held-out evaluation, per-defect failure analysis, and global-versus-specialized study.
- Completed ONNX parity and CPU latency benchmarks for both trained checkpoints.
- Pending 256px GPU training and an untouched external dataset; the current split is a non-official
  supervised-development protocol.
- Pending DeepLabV3 or another lightweight segmentation architecture comparison.

## Milestone 3 - Deployment optimization

- Export to ONNX with dynamic batch and image dimensions.
- Verify PyTorch/ONNX numerical parity.
- Benchmark ONNX Runtime CPU and CUDA providers.
- Implemented validation-calibrated static S8S8 QDQ quantization with held-out quality checks.
- Measured a 52.6% file-size reduction but a 2.18x CPU latency regression; retained FP32 as the
  current CPU latency choice and documented the negative result.
- Build TensorRT FP16 and INT8 engines and record accuracy/latency/memory trade-offs.
- Profile preprocessing, inference, and postprocessing independently.

## Milestone 2.5 - Isaac Sim surface digital twin (v0.5 development)

- Implemented a versioned 1,200-frame Replicator configuration for an aircraft-like surface.
- Implemented scratch, corrosion, pit, fastener, and seam semantics with camera, light, material,
  pose, and scale randomization.
- Implemented aligned RGB, semantic, metric-depth, normal, and camera-parameter output requests.
- Implemented deterministic split planning, config snapshots, offline previews, output discovery,
  manifest generation, and incomplete-capture rejection.
- Added a 50-frame RTX smoke-test wrapper and explicit sensor-alignment acceptance gates.
- Pending the first real Isaac Sim capture on an RTX machine; no rendered synthetic-data result is
  claimed yet.
- Pending curved CAD/USD geometry, raw semantic-ID remapping, and synthetic-to-real ablation.

## Milestone 4 - ROS2 and simulation

- Build the included ROS2 package with `colcon`.
- Publish camera masks and diagnostic latency from the ONNX node.
- Add launch files, rosbag regression tests, lifecycle behavior, and QoS configuration.
- Execute and visually approve the implemented 50-frame Isaac Sim Replicator smoke capture.
- Scale the approved capture to the planned 1,200-frame dataset.
- Measure the synthetic-to-real performance gap.
- Calibrate normalized image-plane paths into a surface coordinate frame.
- Add reachability, acceleration, singularity, force, and collision constraints through MoveIt 2.

## Milestone 3.5 - Production ML workflow (v0.7)

- Implemented byte-level dataset inventories with portable canonical fingerprints.
- Implemented experiment contracts linking config, dataset, Git state, runtime, and dependencies.
- Implemented dry-run-first, content-addressed S3-compatible upload planning with no delete path.
- Added pre-upload re-hashing, idempotent matching-object skips, and encryption options.
- Added non-root multi-stage Docker targets and read-only Compose execution profiles.
- Added GitHub Actions provenance and container smoke jobs with immutable workflow artifacts.
- Pending an actual private-bucket upload and idempotency verification with least-privilege access.
- Pending cloud GPU training with measured cost, utilization, duration, and result provenance.

## Milestone 5 - Recruiter-ready evidence

- Record a 90-second demo with ROS2 diagnostics and live segmentation.
- Publish an architecture diagram and benchmark table.
- Add a portfolio case study focused on failures, measurements, and optimizations.
- Produce a tagged release with reproducible setup instructions.
