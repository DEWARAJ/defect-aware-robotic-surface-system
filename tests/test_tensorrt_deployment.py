from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from surface_perception.tensorrt_deployment import (
    _aggregate,
    build_commands,
    evaluate_acceptance,
    load_config,
    parse_nvidia_smi_csv,
    parse_trtexec_output,
    run_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "tensorrt_deployment_v09.json"


TRTEXEC_OUTPUT = """
[I] Throughput: 412.50 qps
[I] Latency: min = 2.10 ms, max = 3.60 ms, mean = 2.50 ms, median = 2.40 ms, percentile(90%) = 2.80 ms, percentile(95%) = 3.00 ms, percentile(99%) = 3.40 ms
[I] Enqueue Time: min = 0.10 ms, max = 0.20 ms, mean = 0.15 ms, median = 0.14 ms, percentile(90%) = 0.18 ms, percentile(95%) = 0.19 ms, percentile(99%) = 0.20 ms
[I] H2D Latency: min = 0.20 ms, max = 0.30 ms, mean = 0.25 ms, median = 0.24 ms, percentile(90%) = 0.28 ms, percentile(95%) = 0.29 ms, percentile(99%) = 0.30 ms
[I] GPU Compute Time: min = 1.70 ms, max = 2.90 ms, mean = 2.00 ms, median = 1.90 ms, percentile(90%) = 2.40 ms, percentile(95%) = 2.50 ms, percentile(99%) = 2.80 ms
[I] D2H Latency: min = 0.20 ms, max = 0.40 ms, mean = 0.25 ms, median = 0.24 ms, percentile(90%) = 0.30 ms, percentile(95%) = 0.31 ms, percentile(99%) = 0.35 ms
"""


class TensorRtDeploymentTests(unittest.TestCase):
    def test_config_and_commands_preserve_explicit_qdq(self) -> None:
        config = load_config(CONFIG)
        commands = build_commands(config, CONFIG.parent, ROOT / "runs" / "v09")
        self.assertIn("--noTF32", commands["fp32"]["build"])
        self.assertIn("--fp16", commands["fp16"]["build"])
        self.assertNotIn("--int8", commands["int8"]["build"])
        self.assertIn("--includeDataTransfers", commands["int8"]["run"])
        self.assertIn("--useCudaGraph", commands["int8"]["run"])

    def test_parse_trtexec_and_gpu_telemetry(self) -> None:
        metrics = parse_trtexec_output(TRTEXEC_OUTPUT)
        self.assertEqual(metrics["throughput_fps"], 412.5)
        self.assertEqual(metrics["host_latency_ms"]["p95"], 3.0)
        self.assertEqual(metrics["gpu_compute_time_ms"]["mean"], 2.0)
        telemetry = parse_nvidia_smi_csv("80, 1000, 8192, 75, 62\n90, 1200, 8192, 82, 65\n")
        self.assertEqual(telemetry["samples"], 2)
        self.assertEqual(telemetry["gpu_utilization_percent"]["mean"], 85.0)
        self.assertEqual(telemetry["memory_used_mib"]["maximum"], 1200.0)

    def test_missing_metrics_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            parse_trtexec_output("Throughput: 10 qps\n")
        with self.assertRaises(ValueError):
            parse_nvidia_smi_csv("not,a,measurement\n")

    def test_acceptance_checks_every_precision_and_quality(self) -> None:
        aggregate = {
            "throughput_fps_mean": 100.0,
            "host_latency_p95_ms_mean": 10.0,
        }
        report = {
            "results": {name: {"aggregate": aggregate} for name in ("fp32", "fp16", "int8")},
            "quality_evidence": {
                "fp32_iou": 0.286,
                "int8_iou": 0.293,
                "minimum_int8_mask_agreement": 0.974,
            },
        }
        gates = load_config(CONFIG)["acceptance_gates"]
        result = evaluate_acceptance(report, gates)
        self.assertTrue(result["pass"])
        self.assertEqual(len(result["checks"]), 8)

    def test_trial_aggregate_includes_host_and_gpu_resources(self) -> None:
        metrics = parse_trtexec_output(TRTEXEC_OUTPUT)
        metrics["gpu_telemetry"] = parse_nvidia_smi_csv("80, 1000, 8192, 75, 62\n")
        metrics["host_telemetry"] = {
            "process_cpu_percent_mean": 42.0,
            "process_memory_peak_mib": 256.0,
        }
        aggregate = _aggregate([metrics, metrics])
        self.assertEqual(aggregate["host_process_cpu_percent_mean"], 42.0)
        self.assertEqual(aggregate["host_process_memory_peak_mib"], 256.0)
        self.assertEqual(aggregate["gpu_memory_peak_mib"], 1000.0)

    def test_dry_run_writes_machine_readable_plan_without_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = run_benchmark(CONFIG, Path(directory), dry_run=True)
            self.assertEqual(report["status"], "dry-run")
            saved = json.loads((Path(directory) / "benchmark_plan.json").read_text())
            self.assertEqual(set(saved["commands"]), {"fp32", "fp16", "int8"})


if __name__ == "__main__":
    unittest.main()
