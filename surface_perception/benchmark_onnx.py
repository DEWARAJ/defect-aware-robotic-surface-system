from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def benchmark(model_path: Path, image_size: int, iterations: int, warmup: int) -> dict:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Install onnxruntime with the optional ML dependencies.") from exc

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    available = set(ort.get_available_providers())
    selected = [provider for provider in providers if provider in available]
    session = ort.InferenceSession(str(model_path), providers=selected)
    input_name = session.get_inputs()[0].name
    sample = np.random.default_rng(42).random((1, 3, image_size, image_size), dtype=np.float32)
    for _ in range(warmup):
        session.run(None, {input_name: sample})
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        session.run(None, {input_name: sample})
        latencies.append((time.perf_counter() - start) * 1000.0)
    return {
        "model": str(model_path),
        "providers": session.get_providers(),
        "image_size": image_size,
        "iterations": iterations,
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
        },
        "fps_from_mean_latency": float(1000.0 / np.mean(latencies)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark an ONNX segmentation model")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = benchmark(args.model, args.image_size, args.iterations, args.warmup)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()

