# Static INT8 ONNX Optimization (v0.6)

## Question

Can post-training INT8 quantization reduce the Tiny U-Net deployment footprint and improve CPU
inference speed without materially reducing held-out segmentation quality?

The answer on the measured system is mixed: the model became 52.6% smaller and retained similar
quality, but inference became slower. This negative latency result is retained because deployment
formats must be selected from target-hardware measurements, not assumptions.

## Leakage-controlled protocol

- Source: the v0.4 global Tiny U-Net trained on the supervised-development MVTec AD protocol.
- Calibration: 64 images from the validation split only.
- Evaluation: all 137 held-out test images; no test image was used for calibration.
- Input: batch size 1 at 128 x 128.
- Quantization: static post-training QDQ, signed INT8 activations and weights, per-channel weights,
  MinMax calibration.
- Benchmark: five alternating FP32/INT8 trials, each with 20 warmups and 200 timed inferences.
- Runtime: ONNX Runtime 1.28.0 `CPUExecutionProvider` on Windows/AMD64.
- Decision threshold: the unchanged FP32 validation-selected threshold of 0.8.

ONNX Runtime recommends static quantization for CNNs and S8S8 QDQ as the first CPU configuration.
See the official
[ONNX Runtime quantization guidance](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

## Measured results

| Measure | FP32 | INT8 QDQ | Change |
|---|---:|---:|---:|
| Held-out IoU | 0.2860 | 0.2929 | +0.0069 |
| Held-out F1 | 0.4448 | 0.4531 | +0.0083 |
| Precision | 0.3140 | 0.3241 | +0.0101 |
| Recall | 0.7623 | 0.7527 | -0.0096 |
| Mean latency | 1.163 ms | 2.534 ms | +117.9% |
| Mean P95 latency | 1.730 ms | 3.495 ms | +102.0% |
| Throughput from mean latency | 859.7 FPS | 394.6 FPS | 0.459x |
| Model file size | 127,709 bytes | 60,539 bytes | -52.6% |

Across the 137 test images, mean binary-mask agreement was 99.76% and the worst image agreement
was 97.41%. The slight quality increase is an observed result at the fixed threshold, not evidence
that quantization generally improves accuracy.

## Why INT8 was slower

The 29,921-parameter model is already very small. QDQ quantization reduced weight storage but
expanded the ONNX graph from 63 to 111 nodes, including 40 `DequantizeLinear` and 18
`QuantizeLinear` nodes. On this CPU and runtime, conversion overhead outweighed accelerated INT8
convolution. Different CPUs with VNNI, edge accelerators, or TensorRT-capable GPUs may produce a
different result and must be benchmarked separately.

## Reproduce

Install the optional ML stack, retain the locally downloaded MVTec dataset/checkpoint, and run:

```powershell
.\scripts\run_int8_experiment.ps1 `
  -PythonExecutable "C:\path\to\python.exe" `
  -Output "runs/mvtec_v06_int8"
```

Or invoke the module directly:

```powershell
python -m surface_perception.quantize_onnx `
  --model runs/mvtec_v04/model_128/global_model.onnx `
  --manifest runs/mvtec_v04/manifest.json `
  --training-report runs/mvtec_v04/model_128/training_report.json `
  --output runs/mvtec_v06_int8 `
  --image-size 128 --calibration-samples 64 `
  --iterations 200 --warmup 20 --trials 5
```

The tool writes the preprocessed FP32 model, INT8 QDQ model, full per-class report, and a sanitized
summary. Raw MVTec data and model weights remain excluded from Git.

## Deployment decision

Do not select this CPU INT8 format as the current latency-optimized deployment artifact. Retain
FP32 ONNX for this CPU, and run the next benchmark matrix on target hardware:

1. TensorRT FP32, FP16 and INT8 on an RTX GPU or Jetson.
2. Accuracy, mean/P95 latency, FPS, model/engine size, GPU memory and utilization.
3. Identical preprocessing, batch size, image set and threshold.
4. At least five interleaved trials after warmup.
5. Record engine-build time and calibration cache provenance.

The machine-readable evidence is
[`artifacts/reference/onnx_int8_v06.json`](../artifacts/reference/onnx_int8_v06.json).
