from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


PRECISIONS = ("fp32", "fp16", "int8")
METRIC_LABELS = {
    "Latency": "host_latency_ms",
    "Enqueue Time": "enqueue_time_ms",
    "H2D Latency": "h2d_latency_ms",
    "GPU Compute Time": "gpu_compute_time_ms",
    "D2H Latency": "d2h_latency_ms",
}


def _finite_positive(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"models", "input", "benchmark", "acceptance_gates"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"deployment config is missing: {', '.join(missing)}")
    models = config["models"]
    if not models.get("fp32_onnx") or not models.get("int8_qdq_onnx"):
        raise ValueError("both FP32 and explicit-QDQ INT8 ONNX paths are required")
    shape = config["input"].get("shape")
    if not isinstance(shape, list) or len(shape) != 4 or any(int(x) <= 0 for x in shape):
        raise ValueError("input.shape must contain four positive dimensions")
    if int(shape[0]) != 1 or int(shape[1]) != 3:
        raise ValueError("v0.9 requires a batch-1, three-channel input")
    if not str(config["input"].get("name", "")).strip():
        raise ValueError("input.name is required")
    benchmark = config["benchmark"]
    if int(benchmark.get("trials", 0)) < 1 or int(benchmark.get("iterations", 0)) < 1:
        raise ValueError("benchmark trials and iterations must be positive")
    if int(benchmark.get("warmup_ms", -1)) < 0:
        raise ValueError("benchmark warmup_ms cannot be negative")
    gates = config["acceptance_gates"]
    _finite_positive(gates.get("minimum_fps"), "minimum_fps")
    _finite_positive(gates.get("maximum_p95_latency_ms"), "maximum_p95_latency_ms")
    agreement = float(gates.get("minimum_int8_mask_agreement"))
    if not 0.0 <= agreement <= 1.0:
        raise ValueError("minimum_int8_mask_agreement must be in [0, 1]")
    maximum_drop = float(gates.get("maximum_int8_iou_drop"))
    if not 0.0 <= maximum_drop <= 1.0:
        raise ValueError("maximum_int8_iou_drop must be in [0, 1]")
    return config


def _resolved(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (root / candidate).resolve()


def build_commands(config: dict[str, Any], config_root: Path, output: Path) -> dict[str, dict]:
    output = Path(output).resolve()
    input_name = config["input"]["name"]
    shape_text = "x".join(str(int(value)) for value in config["input"]["shape"])
    shape_flag = f"--shapes={input_name}:{shape_text}"
    benchmark = config["benchmark"]
    commands: dict[str, dict] = {}
    for precision in PRECISIONS:
        model_key = "int8_qdq_onnx" if precision == "int8" else "fp32_onnx"
        model = _resolved(config["models"][model_key], config_root)
        engine = output / "engines" / f"surface_segmentation_{precision}.plan"
        build = [
            "trtexec",
            f"--onnx={model}",
            f"--saveEngine={engine}",
            shape_flag,
            "--profilingVerbosity=detailed",
            "--skipInference",
        ]
        if precision == "fp32":
            build.append("--noTF32")
        elif precision == "fp16":
            build.append("--fp16")
        # INT8 is an explicit Q/DQ ONNX graph. Precision flags or calibration are intentionally
        # omitted so TensorRT preserves the graph's quantization semantics.
        run = [
            "trtexec",
            f"--loadEngine={engine}",
            shape_flag,
            f"--warmUp={int(benchmark['warmup_ms'])}",
            "--duration=0",
            f"--iterations={int(benchmark['iterations'])}",
            "--includeDataTransfers",
            "--useCudaGraph",
            "--dumpProfile",
            f"--exportTimes={output / 'raw' / f'{precision}_times.json'}",
            f"--exportProfile={output / 'raw' / f'{precision}_profile.json'}",
            f"--exportLayerInfo={output / 'raw' / f'{precision}_layers.json'}",
        ]
        commands[precision] = {
            "model": str(model),
            "engine": str(engine),
            "build": build,
            "run": run,
        }
    return commands


def _summary_values(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for label in ("min", "max", "mean", "median"):
        match = re.search(rf"\b{label}\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms", text)
        if match:
            values[label] = float(match.group(1))
    for percentile in (90, 95, 99):
        match = re.search(
            rf"percentile\({percentile}%\)\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms", text
        )
        if match:
            values[f"p{percentile}"] = float(match.group(1))
    return values


def parse_trtexec_output(text: str) -> dict[str, Any]:
    throughput_matches = re.findall(
        r"Throughput:\s*([0-9]+(?:\.[0-9]+)?)\s*(?:qps|queries/s)", text
    )
    if not throughput_matches:
        raise ValueError("trtexec output does not contain throughput")
    result: dict[str, Any] = {"throughput_fps": float(throughput_matches[-1])}
    for label, key in METRIC_LABELS.items():
        matches = re.findall(
            rf"^\s*(?:\[[^\]]+\]\s*)*{re.escape(label)}:\s*([^\r\n]+)$",
            text,
            flags=re.MULTILINE,
        )
        if matches:
            values = _summary_values(matches[-1])
            if values:
                result[key] = values
    host = result.get("host_latency_ms", {})
    compute = result.get("gpu_compute_time_ms", {})
    if "p95" not in host or "median" not in host or "mean" not in compute:
        raise ValueError("trtexec output is missing required latency statistics")
    return result


def parse_nvidia_smi_csv(text: str) -> dict[str, Any]:
    rows: list[list[float]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("utilization"):
            continue
        try:
            values = [float(value.strip()) for value in stripped.split(",")]
        except ValueError:
            continue
        if len(values) == 5:
            rows.append(values)
    if not rows:
        raise ValueError("GPU telemetry contains no numeric samples")
    columns = list(zip(*rows))
    return {
        "samples": len(rows),
        "gpu_utilization_percent": {
            "mean": statistics.fmean(columns[0]),
            "maximum": max(columns[0]),
        },
        "memory_used_mib": {"mean": statistics.fmean(columns[1]), "maximum": max(columns[1])},
        "memory_total_mib": columns[2][0],
        "power_draw_watts": {"mean": statistics.fmean(columns[3]), "maximum": max(columns[3])},
        "temperature_celsius": {"mean": statistics.fmean(columns[4]), "maximum": max(columns[4])},
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quality_from_evidence(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    quality = payload["quality"]
    return {
        "fp32_iou": float(quality["fp32"]["iou"]),
        "int8_iou": float(quality["int8"]["iou"]),
        "minimum_int8_mask_agreement": float(quality["parity"]["minimum_binary_mask_agreement"]),
    }


def evaluate_acceptance(report: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for precision in PRECISIONS:
        metrics = report["results"][precision]["aggregate"]
        checks.extend(
            [
                {
                    "name": f"{precision}_minimum_fps",
                    "pass": metrics["throughput_fps_mean"] >= float(gates["minimum_fps"]),
                    "observed": metrics["throughput_fps_mean"],
                    "required": float(gates["minimum_fps"]),
                },
                {
                    "name": f"{precision}_maximum_p95_latency_ms",
                    "pass": metrics["host_latency_p95_ms_mean"]
                    <= float(gates["maximum_p95_latency_ms"]),
                    "observed": metrics["host_latency_p95_ms_mean"],
                    "required": float(gates["maximum_p95_latency_ms"]),
                },
            ]
        )
    quality = report["quality_evidence"]
    checks.extend(
        [
            {
                "name": "int8_iou_drop",
                "pass": quality["fp32_iou"] - quality["int8_iou"]
                <= float(gates["maximum_int8_iou_drop"]),
                "observed": quality["fp32_iou"] - quality["int8_iou"],
                "required": float(gates["maximum_int8_iou_drop"]),
            },
            {
                "name": "int8_minimum_mask_agreement",
                "pass": quality["minimum_int8_mask_agreement"]
                >= float(gates["minimum_int8_mask_agreement"]),
                "observed": quality["minimum_int8_mask_agreement"],
                "required": float(gates["minimum_int8_mask_agreement"]),
            },
        ]
    )
    return {"pass": all(check["pass"] for check in checks), "checks": checks}


def _aggregate(trials: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "trials": len(trials),
        "throughput_fps_mean": statistics.fmean(x["throughput_fps"] for x in trials),
        "host_latency_median_ms_mean": statistics.fmean(
            x["host_latency_ms"]["median"] for x in trials
        ),
        "host_latency_p95_ms_mean": statistics.fmean(
            x["host_latency_ms"]["p95"] for x in trials
        ),
        "host_latency_p99_ms_mean": statistics.fmean(
            x["host_latency_ms"].get("p99", x["host_latency_ms"]["p95"]) for x in trials
        ),
        "gpu_compute_mean_ms": statistics.fmean(
            x["gpu_compute_time_ms"]["mean"] for x in trials
        ),
        "gpu_utilization_percent_mean": statistics.fmean(
            x["gpu_telemetry"]["gpu_utilization_percent"]["mean"] for x in trials
        ),
        "gpu_memory_peak_mib": max(
            x["gpu_telemetry"]["memory_used_mib"]["maximum"] for x in trials
        ),
        "host_process_cpu_percent_mean": statistics.fmean(
            x["host_telemetry"]["process_cpu_percent_mean"] for x in trials
        ),
        "host_process_memory_peak_mib": max(
            x["host_telemetry"]["process_memory_peak_mib"] for x in trials
        ),
    }


def _run(command: list[str], log_path: Path) -> str:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    rendered = completed.stdout + completed.stderr
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(rendered, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}); see {log_path}")
    return rendered


def _run_with_host_telemetry(command: list[str], log_path: Path) -> tuple[str, dict[str, float]]:
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("install the 'nvidia' extra to capture host telemetry") from exc
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_samples: list[float] = []
    peak_rss = 0
    with log_path.open("w", encoding="utf-8") as stream:
        process = psutil.Popen(command, stdout=stream, stderr=subprocess.STDOUT, text=True)
        process.cpu_percent(interval=None)
        while process.poll() is None:
            descendants = process.children(recursive=True)
            active = [process, *descendants]
            sample_cpu = 0.0
            sample_rss = 0
            for observed in active:
                try:
                    sample_cpu += observed.cpu_percent(interval=None)
                    sample_rss += observed.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            cpu_samples.append(sample_cpu)
            peak_rss = max(peak_rss, sample_rss)
            time.sleep(0.1)
        return_code = process.wait()
    rendered = log_path.read_text(encoding="utf-8")
    if return_code != 0:
        raise RuntimeError(f"command failed ({return_code}); see {log_path}")
    return rendered, {
        "samples": len(cpu_samples),
        "process_cpu_percent_mean": statistics.fmean(cpu_samples) if cpu_samples else 0.0,
        "process_cpu_percent_peak": max(cpu_samples, default=0.0),
        "process_memory_peak_mib": peak_rss / (1024.0 * 1024.0),
    }


def _monitor_command() -> list[str]:
    return [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
        "--loop-ms=100",
    ]


def run_benchmark(config_path: Path, output: Path, *, dry_run: bool = False) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    output = Path(output).resolve()
    config = load_config(config_path)
    commands = build_commands(config, config_path.parent, output)
    plan = {
        "schema_version": "tensorrt-deployment-plan-1.0",
        "status": "dry-run" if dry_run else "measured",
        "config": str(config_path),
        "commands": commands,
        "measurement_scope": {
            "trtexec": "batch-1 engine throughput and host/GPU latency",
            "gpu_telemetry": "utilization, memory, power, and temperature sampled at 100 ms",
            "excluded": "camera acquisition, ROS transport, preprocessing, postprocessing, actuation",
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    if dry_run:
        (output / "benchmark_plan.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8"
        )
        return plan
    if shutil.which("trtexec") is None or shutil.which("nvidia-smi") is None:
        raise RuntimeError("a target NVIDIA system with trtexec and nvidia-smi is required")
    for item in commands.values():
        if not Path(item["model"]).is_file():
            raise FileNotFoundError(item["model"])
    quality_path = _resolved(config["models"]["quality_evidence"], config_path.parent)
    quality = _quality_from_evidence(quality_path)
    results: dict[str, Any] = {}
    for precision in PRECISIONS:
        item = commands[precision]
        engine = Path(item["engine"])
        engine.parent.mkdir(parents=True, exist_ok=True)
        (output / "raw").mkdir(parents=True, exist_ok=True)
        _run(item["build"], output / "raw" / f"{precision}_build.log")
        trials = []
        for trial_index in range(int(config["benchmark"]["trials"])):
            telemetry_path = output / "raw" / f"{precision}_trial_{trial_index + 1}_gpu.csv"
            telemetry_stream = telemetry_path.open("w", encoding="utf-8")
            monitor = subprocess.Popen(
                _monitor_command(), stdout=telemetry_stream, stderr=subprocess.STDOUT, text=True
            )
            try:
                time.sleep(0.25)
                log, host_telemetry = _run_with_host_telemetry(
                    item["run"], output / "raw" / f"{precision}_trial_{trial_index + 1}.log"
                )
            finally:
                monitor.terminate()
                try:
                    monitor.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    monitor.kill()
                telemetry_stream.close()
            metrics = parse_trtexec_output(log)
            metrics["host_telemetry"] = host_telemetry
            metrics["gpu_telemetry"] = parse_nvidia_smi_csv(
                telemetry_path.read_text(encoding="utf-8")
            )
            trials.append(metrics)
        results[precision] = {
            "model_sha256": _sha256(Path(item["model"])),
            "engine_sha256": _sha256(engine),
            "engine_bytes": engine.stat().st_size,
            "trials": trials,
            "aggregate": _aggregate(trials),
        }
    hardware = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap,power.limit",
            "--format=csv,noheader,nounits",
        ],
        output / "raw" / "hardware.log",
    ).strip()
    report = {
        **plan,
        "status": "measured",
        "platform": {"system": platform.system(), "release": platform.release(), "gpu": hardware},
        "quality_evidence": quality,
        "results": results,
    }
    report["acceptance"] = evaluate_acceptance(report, config["acceptance_gates"])
    (output / "tensorrt_benchmark.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not report["acceptance"]["pass"]:
        raise RuntimeError("TensorRT benchmark failed one or more acceptance gates")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and benchmark TensorRT deployment formats")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = run_benchmark(args.config, args.output, dry_run=args.dry_run)
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
