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
- Build TensorRT FP16 and INT8 engines and record accuracy/latency/memory trade-offs.
- Profile preprocessing, inference, and postprocessing independently.

## Milestone 4 - ROS2 and simulation

- Build the included ROS2 package with `colcon`.
- Publish camera masks and diagnostic latency from the ONNX node.
- Add launch files, rosbag regression tests, lifecycle behavior, and QoS configuration.
- Generate RGB, depth, masks, and bounding boxes using Isaac Sim Replicator.
- Randomize lighting, material, texture, camera pose, and defect placement.
- Measure the synthetic-to-real performance gap.
- Calibrate normalized image-plane paths into a surface coordinate frame.
- Add reachability, acceleration, singularity, force, and collision constraints through MoveIt 2.

## Milestone 5 - Recruiter-ready evidence

- Record a 90-second demo with ROS2 diagnostics and live segmentation.
- Publish an architecture diagram and benchmark table.
- Add a portfolio case study focused on failures, measurements, and optimizations.
- Produce a tagged release with reproducible setup instructions.
