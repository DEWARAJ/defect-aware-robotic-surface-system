from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .benchmark_onnx import benchmark
from .metrics import add_counts, confusion_counts, metrics_from_counts
from .real_data import load_manifest
from .torch_data import load_record_arrays


class ManifestCalibrationReader:
    """Feed a deterministic, non-test manifest split to ONNX Runtime calibration."""

    def __init__(
        self,
        manifest_path: Path,
        input_name: str,
        image_size: int,
        *,
        split: str = "validation",
        max_samples: int = 64,
    ) -> None:
        records = [
            record
            for record in load_manifest(manifest_path)["records"]
            if record["split"] == split
        ]
        if not records:
            raise ValueError(f"manifest contains no records for calibration split '{split}'")
        if split == "test":
            raise ValueError("the held-out test split cannot be used for quantization calibration")
        self.records = records[:max_samples]
        self.input_name = input_name
        self.image_size = int(image_size)
        self.index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.index >= len(self.records):
            return None
        image, _ = load_record_arrays(
            self.records[self.index],
            image_size=self.image_size,
        )
        self.index += 1
        return {self.input_name: image[None, ...].astype(np.float32)}

    def rewind(self) -> None:
        self.index = 0


def _providers(ort) -> list[str]:
    available = set(ort.get_available_providers())
    return [
        provider
        for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in available
    ]


def quantize_static_int8(
    source_model: Path,
    output_model: Path,
    manifest_path: Path,
    *,
    image_size: int,
    calibration_split: str = "validation",
    calibration_samples: int = 64,
    preprocessed_model: Path | None = None,
) -> dict:
    try:
        import onnxruntime as ort
        from onnxruntime.quantization import (
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
        from onnxruntime.quantization.shape_inference import quant_pre_process
    except ImportError as exc:
        raise RuntimeError("Install the optional ML dependencies for ONNX quantization.") from exc

    source_model = Path(source_model)
    output_model = Path(output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    preprocessed_model = Path(
        preprocessed_model or output_model.with_name(f"{output_model.stem}_preprocessed.onnx")
    )
    preprocessed_model.parent.mkdir(parents=True, exist_ok=True)

    quant_pre_process(
        input_model=source_model,
        output_model_path=preprocessed_model,
        skip_symbolic_shape=True,
        skip_optimization=False,
        skip_onnx_shape=False,
    )
    session = ort.InferenceSession(str(preprocessed_model), providers=_providers(ort))
    input_name = session.get_inputs()[0].name
    reader = ManifestCalibrationReader(
        manifest_path,
        input_name,
        image_size,
        split=calibration_split,
        max_samples=calibration_samples,
    )
    quantize_static(
        model_input=preprocessed_model,
        model_output=output_model,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        calibrate_method=CalibrationMethod.MinMax,
    )
    return {
        "format": "QDQ",
        "activation_type": "QInt8",
        "weight_type": "QInt8",
        "per_channel": True,
        "calibration_method": "MinMax",
        "calibration_split": calibration_split,
        "calibration_samples": len(reader.records),
        "preprocessed_model": str(preprocessed_model.resolve()),
        "quantized_model": str(output_model.resolve()),
    }


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def evaluate_onnx_pair(
    reference_model: Path,
    candidate_model: Path,
    manifest_path: Path,
    *,
    image_size: int,
    threshold: float,
    split: str = "test",
) -> dict:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Install ONNX Runtime to evaluate optimized models.") from exc

    records = [
        record for record in load_manifest(manifest_path)["records"] if record["split"] == split
    ]
    if not records:
        raise ValueError(f"manifest contains no records for evaluation split '{split}'")

    providers = _providers(ort)
    reference_session = ort.InferenceSession(str(reference_model), providers=providers)
    candidate_session = ort.InferenceSession(str(candidate_model), providers=providers)
    reference_input = reference_session.get_inputs()[0].name
    candidate_input = candidate_session.get_inputs()[0].name

    totals = {"fp32": {}, "int8": {}}
    grouped = {"fp32": defaultdict(dict), "int8": defaultdict(dict)}
    latencies = {"fp32": [], "int8": []}
    agreements: list[float] = []
    maximum_logit_error = 0.0
    mean_logit_errors: list[float] = []

    for record in records:
        image, mask = load_record_arrays(record, image_size=image_size)
        batch = image[None, ...].astype(np.float32)
        target = mask[0] > 0.5

        start = time.perf_counter()
        reference_logits = reference_session.run(None, {reference_input: batch})[0]
        latencies["fp32"].append((time.perf_counter() - start) * 1000.0)
        start = time.perf_counter()
        candidate_logits = candidate_session.run(None, {candidate_input: batch})[0]
        latencies["int8"].append((time.perf_counter() - start) * 1000.0)

        reference_mask = _sigmoid(reference_logits)[0, 0] >= threshold
        candidate_mask = _sigmoid(candidate_logits)[0, 0] >= threshold
        group_key = f"{record['category']}:{record['defect_type']}"
        for name, prediction in (("fp32", reference_mask), ("int8", candidate_mask)):
            counts = confusion_counts(prediction, target)
            totals[name] = add_counts(totals[name], counts)
            grouped[name][group_key] = add_counts(grouped[name][group_key], counts)

        difference = np.abs(reference_logits - candidate_logits)
        maximum_logit_error = max(maximum_logit_error, float(difference.max()))
        mean_logit_errors.append(float(difference.mean()))
        agreements.append(float(np.mean(reference_mask == candidate_mask)))

    quality = {}
    for name in ("fp32", "int8"):
        quality[name] = {
            "aggregate": metrics_from_counts(totals[name]),
            "by_category_and_defect": {
                key: metrics_from_counts(counts)
                for key, counts in sorted(grouped[name].items())
            },
            "latency_ms_single_pass": {
                "mean": float(np.mean(latencies[name])),
                "p50": float(np.percentile(latencies[name], 50)),
                "p95": float(np.percentile(latencies[name], 95)),
            },
        }
    quality["metric_delta_int8_minus_fp32"] = {
        metric: quality["int8"]["aggregate"][metric] - quality["fp32"]["aggregate"][metric]
        for metric in quality["fp32"]["aggregate"]
    }
    quality["parity"] = {
        "maximum_absolute_logit_error": maximum_logit_error,
        "mean_absolute_logit_error": float(np.mean(mean_logit_errors)),
        "minimum_binary_mask_agreement": min(agreements),
        "mean_binary_mask_agreement": float(np.mean(agreements)),
    }
    quality["samples"] = len(records)
    quality["split"] = split
    quality["threshold"] = threshold
    return quality


def _graph_summary(model_path: Path) -> dict:
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("Install ONNX to inspect optimized model graphs.") from exc
    model = onnx.load(model_path, load_external_data=False)
    operators = Counter(node.op_type for node in model.graph.node)
    return {
        "file_bytes": Path(model_path).stat().st_size,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "operators": dict(sorted(operators.items())),
    }


def _paired_benchmark_trials(
    fp32_model: Path,
    int8_model: Path,
    image_size: int,
    iterations: int,
    warmup: int,
    trials: int,
) -> dict:
    if trials < 1:
        raise ValueError("benchmark trials must be at least 1")
    results = {"fp32": [], "int8": []}
    for trial in range(trials):
        order = ("fp32", "int8") if trial % 2 == 0 else ("int8", "fp32")
        paths = {"fp32": fp32_model, "int8": int8_model}
        for name in order:
            result = benchmark(paths[name], image_size, iterations, warmup)
            result["trial"] = trial + 1
            results[name].append(result)

    payload = {}
    for name, trial_results in results.items():
        means = [result["latency_ms"]["mean"] for result in trial_results]
        p95_values = [result["latency_ms"]["p95"] for result in trial_results]
        mean_latency = float(np.mean(means))
        payload[name] = {
            "trials": trial_results,
            "aggregate": {
                "trials": trials,
                "iterations_per_trial": iterations,
                "mean_latency_ms": mean_latency,
                "median_trial_mean_latency_ms": float(np.median(means)),
                "standard_deviation_trial_mean_ms": float(np.std(means)),
                "mean_p95_latency_ms": float(np.mean(p95_values)),
                "fps_from_mean_latency": float(1000.0 / mean_latency),
                "providers": trial_results[0]["providers"],
            },
        }
    return payload


def run_quantization_experiment(
    source_model: Path,
    manifest_path: Path,
    training_report_path: Path,
    output_dir: Path,
    *,
    image_size: int = 128,
    calibration_samples: int = 64,
    iterations: int = 200,
    warmup: int = 20,
    trials: int = 5,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_report = json.loads(Path(training_report_path).read_text(encoding="utf-8"))
    threshold = float(training_report["selected_threshold"])
    int8_model = output_dir / "model_int8_qdq.onnx"
    preprocessed_model = output_dir / "model_fp32_preprocessed.onnx"

    quantization = quantize_static_int8(
        source_model,
        int8_model,
        manifest_path,
        image_size=image_size,
        calibration_split="validation",
        calibration_samples=calibration_samples,
        preprocessed_model=preprocessed_model,
    )
    quality = evaluate_onnx_pair(
        source_model,
        int8_model,
        manifest_path,
        image_size=image_size,
        threshold=threshold,
        split="test",
    )
    benchmark_report = _paired_benchmark_trials(
        source_model,
        int8_model,
        image_size,
        iterations,
        warmup,
        trials,
    )
    fp32_graph = _graph_summary(source_model)
    int8_graph = _graph_summary(int8_model)
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Install ONNX and ONNX Runtime for optimization evidence.") from exc
    report = {
        "experiment": "onnx_static_int8_post_training_quantization",
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor()
            or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
            "logical_cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "source_model": str(Path(source_model).resolve()),
        "manifest": str(Path(manifest_path).resolve()),
        "image_size": image_size,
        "quantization": quantization,
        "quality": quality,
        "benchmark": benchmark_report,
        "graph": {"fp32": fp32_graph, "int8": int8_graph},
        "compression": {
            "size_ratio_fp32_over_int8": fp32_graph["file_bytes"]
            / int8_graph["file_bytes"],
            "size_reduction_fraction": 1.0
            - int8_graph["file_bytes"] / fp32_graph["file_bytes"],
        },
        "performance_delta": {
            "mean_latency_reduction_fraction": 1.0
            - benchmark_report["int8"]["aggregate"]["mean_latency_ms"]
            / benchmark_report["fp32"]["aggregate"]["mean_latency_ms"],
            "fps_multiplier": benchmark_report["int8"]["aggregate"][
                "fps_from_mean_latency"
            ]
            / benchmark_report["fp32"]["aggregate"]["fps_from_mean_latency"],
        },
        "evidence_limit": (
            "Measured with ONNX Runtime CPUExecutionProvider; this is not a TensorRT, CUDA, "
            "GPU, edge-device, or production benchmark."
        ),
    }
    report_path = output_dir / "quantization_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    reference_summary = {
        "experiment": report["experiment"],
        "status": "measured",
        "model": {
            "architecture": training_report.get("model", "Tiny U-Net"),
            "parameters": training_report.get("parameter_count"),
            "image_size": image_size,
        },
        "data_protocol": {
            "calibration_split": quantization["calibration_split"],
            "calibration_samples": quantization["calibration_samples"],
            "evaluation_split": quality["split"],
            "evaluation_samples": quality["samples"],
            "threshold": threshold,
            "test_split_used_for_calibration": False,
        },
        "quality": {
            "fp32": quality["fp32"]["aggregate"],
            "int8": quality["int8"]["aggregate"],
            "delta_int8_minus_fp32": quality["metric_delta_int8_minus_fp32"],
            "parity": quality["parity"],
        },
        "performance": {
            "fp32": benchmark_report["fp32"]["aggregate"],
            "int8": benchmark_report["int8"]["aggregate"],
            **report["performance_delta"],
        },
        "model_size": {
            "fp32_bytes": fp32_graph["file_bytes"],
            "int8_bytes": int8_graph["file_bytes"],
            **report["compression"],
        },
        "quantization": {
            key: quantization[key]
            for key in (
                "format",
                "activation_type",
                "weight_type",
                "per_channel",
                "calibration_method",
            )
        },
        "platform": report["platform"],
        "evidence_limit": report["evidence_limit"],
        "conclusion": (
            "INT8 preserved held-out quality and reduced file size, but QDQ overhead made this "
            "small CNN slower on the measured CPU. Test TensorRT FP16/INT8 on target RTX/Jetson "
            "hardware before selecting a deployment format."
        ),
    }
    (output_dir / "quantization_summary.json").write_text(
        json.dumps(reference_summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate and measure static INT8 ONNX surface segmentation"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--calibration-samples", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    report = run_quantization_experiment(
        args.model,
        args.manifest,
        args.training_report,
        args.output,
        image_size=args.image_size,
        calibration_samples=args.calibration_samples,
        iterations=args.iterations,
        warmup=args.warmup,
        trials=args.trials,
    )
    summary = {
        "fp32_iou": report["quality"]["fp32"]["aggregate"]["iou"],
        "int8_iou": report["quality"]["int8"]["aggregate"]["iou"],
        "minimum_mask_agreement": report["quality"]["parity"][
            "minimum_binary_mask_agreement"
        ],
        "size_reduction_fraction": report["compression"]["size_reduction_fraction"],
        "latency_reduction_fraction": report["performance_delta"][
            "mean_latency_reduction_fraction"
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"report: {(args.output / 'quantization_report.json').resolve()}")


if __name__ == "__main__":
    main()
