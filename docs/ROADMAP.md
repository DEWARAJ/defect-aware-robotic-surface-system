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
- Implemented a fail-closed TensorRT FP32/FP16/explicit-QDQ-INT8 benchmark harness with
  accuracy, latency, FPS, GPU utilization, memory, power, and temperature contracts (v0.9).
- Execute the harness on the intended RTX/Jetson target; no GPU result is claimed yet.
- Profile preprocessing, inference, and postprocessing independently.
- Implemented a four-candidate model-compression study with a fixed teacher, physically smaller
  six-convolution student, supervised-versus-distilled ablation, and 25% structured filter
  pruning (v0.10).
- Added held-out quality budgeting, architecture/parameter evidence, ONNX parity, randomized-order
  latency trials, and fail-safe deployment selection for compression candidates.
- Pending compact-model measurement on TensorRT/Jetson and structural graph compaction for the
  pruned candidate; dense filter zeros are not claimed as a hardware speedup.

## Milestone 2.75 - Risk-aware human review (implemented v0.11 development)

- Implemented validation-fitted pixel temperature scaling with exact preservation of the deployed
  logit decision boundary.
- Implemented label-free per-image entropy, observable summaries, and handcrafted input-shift
  scoring.
- Implemented validation-label-trained kNN failure-risk prediction and leave-one-out acquisition
  policy selection.
- Implemented a label-free review queue, diversity-aware ranking, retrospective comparisons, and
  controlled noise/brightness/blur probes.
- Measured 8/35 worst failures found at a 20/137 review budget (1.57x random expected hits) in the
  retrospective development pool.
- Pending confirmation on a new untouched external surface dataset; v0.11 pool results were
  inspected during development and are not represented as a final holdout.
- Pending retraining-versus-annotation-budget learning curves and ensemble uncertainty.

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
- Implemented a dependency-light C++17 image normalization, mask postprocessing, and rolling
  telemetry core with deterministic GoogleTests (v0.8 development).
- Implemented an optional native ONNX Runtime ROS2 node with sensor QoS, camera masks, defect
  fraction, p50/p95 latency, FPS, deadline warnings, and failure diagnostics (v0.8 development).
- Added ROS2 Humble CI for the dependency-light C++ core and pinned native ONNX Runtime node.
- Added a deterministic two-frame rosbag regression contract for exact masks, fractions,
  timestamps, and healthy diagnostics.
- Connected each timestamped perception mask to the C++ coverage planner and added path, feed,
  and protected-contact verification (v0.9).
- Add lifecycle behavior and expanded QoS/failure-mode testing.
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
