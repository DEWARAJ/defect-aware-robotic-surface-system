# Model Compression Study (v0.10)

Version 0.10 measures three distinct approaches for reducing the deployment cost of the surface
segmentation model: physical architecture reduction, knowledge distillation, and structured
filter pruning. All candidates use the same manifest and held-out test split. The experiment keeps
the v0.4 29,921-parameter Tiny U-Net as the fixed teacher and baseline.

## Why this matters

Small industrial and robotic deployments are constrained by compute, memory, power, and control
cycle deadlines. A smaller network is valuable only if the size and latency gains are measured and
its segmentation loss is acceptable. This study therefore treats compression as a multi-objective
engineering decision rather than assuming that fewer weights automatically create a better model.

The compact student uses one encoder level instead of two. It contains six spatial 3 x 3
convolutions rather than ten, while retaining a skip connection and full-resolution binary logits.
With eight base channels it has 6,689 parameters, 77.6% fewer than the 29,921-parameter teacher.

## Candidates

| Candidate | Training or transformation | Purpose |
|---|---|---|
| `teacher` | Existing Tiny U-Net checkpoint | Fixed accuracy and deployment baseline |
| `student_supervised` | Compact U-Net with weighted BCE + Dice | Isolate architecture reduction |
| `student_distilled` | Same student plus soft teacher targets | Measure the value of distillation |
| `teacher_pruned` | 25% per-layer low-L1 output filters disabled | Measure structured sparsity sensitivity |

The pruning branch zeros whole convolution output filters and their paired BatchNorm affine
channels. It deliberately retains the original dense tensor shapes. Its quality result is useful,
but it is **not** evidence of smaller tensors or faster dense inference; a later graph-compaction
pass or sparse backend would be required for that claim.

## Loss and selection rule

The supervised term combines positive-class-weighted binary cross entropy with soft Dice loss.
The distilled student adds temperature-scaled binary cross entropy between its logits and the
teacher's soft probabilities:

```text
total = (1 - alpha) * supervised + alpha * distillation
alpha = 0.35, temperature = 2.0
```

Thresholds and compact-candidate selection use only the validation split. A student is eligible
only when its validation IoU is no more than 0.03 below the teacher. Among eligible students, the
smallest model is selected, then validation IoU and measured latency break ties. The provisionally
selected student receives one final held-out qualification gate with the same 0.03 budget. A
failed validation gate or held-out guardrail keeps the teacher selected; test results are never
used to tune between students.

## Measurement protocol

Each checkpoint is evaluated through the same PyTorch data path, exported independently to ONNX,
and checked for PyTorch/ONNX numerical parity. ONNX Runtime latency uses a shared seeded input,
per-model warmup, three timed trials, and seeded randomized candidate order. Reports capture:

- parameter, layer, serialized-state, zero-parameter, and inactive-filter counts;
- held-out IoU, Dice/F1, precision, recall, and pixel accuracy;
- checkpoint and ONNX byte sizes plus SHA-256 digests;
- PyTorch timing and interleaved ONNX Runtime p50/p95/mean latency;
- maximum logit error and minimum binary-mask agreement;
- per-candidate IoU retention, parameter reduction, ONNX reduction, and measured speedup.

## Measured result

The real-data experiment completed on the original 1,157-image v0.4 manifest. The held-out split
contained 137 images. Distillation was essential: the normally trained student lost substantial
quality, while the same compact graph with teacher guidance passed both selection gates and
outperformed the teacher on held-out IoU.

| Candidate | Params | Spatial convs | Validation IoU | Test IoU | Test F1 | ONNX size | Mean CPU latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Teacher | 29,921 | 10 | 0.2706 | 0.2860 | 0.4448 | 127,709 B | 6.496 ms |
| Supervised student | 6,689 | 6 | 0.1972 | 0.1421 | 0.2488 | 31,334 B | 5.763 ms |
| Distilled student | 6,689 | 6 | 0.3436 | 0.3276 | 0.4935 | 31,334 B | 5.358 ms |
| 25% pruned teacher | 29,921 | 10 | 0.2619 | 0.2461 | 0.3949 | 127,709 B | 6.400 ms |

Relative to the teacher, the selected distilled student reduced parameters by 77.6%, reduced ONNX
size by 75.5%, improved held-out IoU by 0.0416, and was 1.21x faster by mean latency. Its ONNX
export reached 100% thresholded-mask agreement with PyTorch over the parity sample and a maximum
absolute logit error of 3.34e-6.

The latency distribution was noisy on this shared CPU host: teacher/student p95 was 35.04/29.35
ms even though p50 was 1.91/1.51 ms. These are environment-specific model-only numbers and must
not be treated as deterministic control-loop deadlines. The definitive portable evidence is
[`artifacts/reference/model_compression_v10.json`](../artifacts/reference/model_compression_v10.json).
Generated checkpoints, raw data, and full run directories remain outside version control.

## Run the measured study

Install the optional stack and execute from the repository root:

```powershell
python -m pip install -e ".[ml]"
.\scripts\run_model_compression.ps1 `
  -PythonExecutable python `
  -Config configs/model_compression_v10.json
```

The default config expects the existing v0.4 assets:

```text
runs/mvtec_v04/manifest.json
runs/mvtec_v04/model_128/best.pt
```

For an execution-only smoke test, use the fixture builder and pass one epoch plus reduced ONNX
iterations through the module CLI. The GitHub `optional-ml-smoke` workflow does this and uploads
`compression_study.json` as workflow evidence.

## Acceptance and honesty boundaries

- MVTec AD is used under its CC BY-NC-SA 4.0 terms; data and weights are not redistributed.
- The split remains the v0.4 supervised-development protocol, not the official unsupervised
  MVTec benchmark.
- CPU timings are specific to the measured machine and exclude camera I/O, preprocessing,
  postprocessing, ROS transport, planning, and robot control.
- Filter sparsity does not imply latency improvement on a dense runtime.
- No aircraft-surface, production-inspection, GPU, or safety-critical performance claim is made.
