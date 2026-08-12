# NVIDIA TensorRT Deployment - v0.9 Development

## Objective

Version 0.9 adds a fail-closed, target-hardware benchmark for the surface-segmentation model and
connects the ROS2 perception output to the protected-region-aware C++ coverage planner. The
benchmark compares TensorRT FP32, FP16, and explicit-QDQ INT8 at batch 1. The current development
machine has no NVIDIA driver or TensorRT installation, so this repository does **not** claim a GPU
performance result yet.

## Measured boundary

The TensorRT report deliberately separates two measurement scopes:

1. `trtexec` measures engine throughput, host latency, transfers, enqueue time, and GPU compute.
2. The ROS2 node measures preprocessing, inference, postprocessing, end-to-end p50/p95, and
   effective FPS inside the application.

These numbers are not interchangeable. Camera acquisition and actuation are outside both current
contracts.

## Precision matrix

| Variant | Source | Build rule |
|---|---|---|
| FP32 | FP32 ONNX | TF32 explicitly disabled for a stable FP32 baseline |
| FP16 | FP32 ONNX | TensorRT `--fp16` engine build |
| INT8 | Static S8S8 QDQ ONNX | Explicit Q/DQ semantics; no deprecated implicit calibration |

The INT8 graph reuses the v0.6 validation-only calibration result. Held-out quality evidence is
loaded from `artifacts/reference/onnx_int8_v06.json` and remains a required acceptance input.

## Run on an RTX or Jetson target

Install TensorRT so `trtexec` is on `PATH`, verify `nvidia-smi` is available on a discrete GPU,
and place the FP32 and INT8 ONNX files at the configured paths. On Jetson, use the platform's
supported telemetry tool if `nvidia-smi` is unavailable; the current runner targets a discrete
NVIDIA GPU and fails closed otherwise.

```bash
bash scripts/run_tensorrt_v09.sh \
  configs/tensorrt_deployment_v09.json \
  runs/tensorrt_v09
```

The runner performs five trials per precision and writes:

- engine and model SHA-256 hashes;
- throughput plus host-latency p50/p95/p99;
- GPU-compute time;
- GPU utilization and peak device memory;
- host-process CPU utilization and peak resident memory;
- power and temperature telemetry in raw trial files;
- layer/profile exports and a machine-readable acceptance report.

Serialized TensorRT engines are excluded from Git. They contain target-specific native code and
must only be loaded from a trusted build pipeline.

## Acceptance gates

- at least 30 FPS for every precision;
- at most 33.3 ms mean trial p95 host latency;
- no more than 0.02 absolute INT8 IoU loss against FP32;
- at least 97% minimum INT8-versus-FP32 binary-mask agreement.

These are portfolio-development gates, not EMMA production specifications. Real system limits
must be negotiated with controls, safety, hardware, and customer stakeholders.

## Perception-to-planner ROS2 contract

The deterministic ROS bag now publishes sanding and protected masks before camera frames. The
native ONNX Runtime node emits a timestamped defect mask, and that mask triggers the C++ coverage
planner. CI verifies two complete perception-to-plan cycles:

- exact output masks and defect fractions;
- matching inference/planner timestamps;
- non-empty normalized path messages;
- nominal feed for the defect-free frame;
- reduced feed for the defect frame;
- zero protected-region contact.

This proves the image-plane message and safety contract. It does not prove calibrated robot-space
motion, collision avoidance, reachability, force control, or aircraft-process qualification.

## Remaining evidence

1. Run the precision matrix on the intended RTX or Jetson target.
2. Benchmark the real 128-pixel model and a representative camera recording.
3. Compare TensorRT results with ROS2 end-to-end telemetry under sustained load.
4. Add dropped-frame, thermal-throttling, restart, and malformed-input tests.
5. Calibrate image coordinates to the surface frame and integrate MoveIt 2 safety constraints.
