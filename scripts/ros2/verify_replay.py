from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray


def uint8_to_int(value: int | bytes | bytearray) -> int:
    """Normalize ROS uint8 fields across rosidl Python representations."""
    if isinstance(value, (bytes, bytearray)):
        if len(value) != 1:
            raise AssertionError(f"expected one diagnostic level byte, got {len(value)}")
        return value[0]
    return int(value)


class ReplayVerifier(Node):
    def __init__(self, size: int) -> None:
        super().__init__("surface_perception_replay_verifier")
        self.size = size
        self.mask_hashes: list[str] = []
        self.mask_stamps: list[int] = []
        self.fractions: list[float] = []
        self.diagnostic_stamps: list[int] = []
        self.diagnostic_levels: list[int] = []
        self.diagnostic_keys: list[list[str]] = []
        self.paths: list[list[float]] = []
        self.planner_stamps: list[int] = []
        self.planner_values: list[dict[str, str]] = []
        self.create_subscription(Image, "/surface_perception/mask", self.on_mask, 10)
        self.create_subscription(
            Float32, "/surface_perception/defect_fraction", self.on_fraction, 10
        )
        self.create_subscription(
            DiagnosticArray, "/surface_perception/diagnostics", self.on_diagnostics, 10
        )
        self.create_subscription(
            Float32MultiArray, "/surface_perception/coverage_path", self.on_path, 10
        )
        self.create_subscription(
            DiagnosticArray,
            "/surface_perception/planner_diagnostics",
            self.on_planner_diagnostics,
            10,
        )

    def on_mask(self, message: Image) -> None:
        if (
            message.encoding != "mono8"
            or message.width != self.size
            or message.height != self.size
        ):
            raise AssertionError("unexpected output mask contract")
        self.mask_hashes.append(hashlib.sha256(bytes(message.data)).hexdigest())
        self.mask_stamps.append(message.header.stamp.sec)

    def on_fraction(self, message: Float32) -> None:
        self.fractions.append(float(message.data))

    def on_diagnostics(self, message: DiagnosticArray) -> None:
        if len(message.status) != 1:
            raise AssertionError("expected exactly one inference diagnostic status")
        status = message.status[0]
        self.diagnostic_stamps.append(message.header.stamp.sec)
        self.diagnostic_levels.append(uint8_to_int(status.level))
        self.diagnostic_keys.append(sorted(value.key for value in status.values))

    def on_path(self, message: Float32MultiArray) -> None:
        if len(message.layout.dim) != 2 or message.layout.dim[1].size != 5:
            raise AssertionError("coverage path must contain five values per segment")
        if len(message.data) == 0 or len(message.data) % 5 != 0:
            raise AssertionError("coverage path must contain at least one complete segment")
        if message.layout.dim[0].size * 5 != len(message.data):
            raise AssertionError("coverage path layout does not match its data")
        self.paths.append([float(value) for value in message.data])

    def on_planner_diagnostics(self, message: DiagnosticArray) -> None:
        if len(message.status) != 1:
            raise AssertionError("expected exactly one planner diagnostic status")
        self.planner_stamps.append(message.header.stamp.sec)
        self.planner_values.append(
            {value.key: value.value for value in message.status[0].values}
        )

    def complete(self) -> bool:
        return (
            len(self.mask_hashes) >= 2
            and len(self.fractions) >= 2
            and len(self.diagnostic_levels) >= 2
            and len(self.paths) >= 2
            and len(self.planner_values) >= 2
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify deterministic ROS2 replay outputs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--size", type=int, default=32)
    args = parser.parse_args()
    rclpy.init()
    node = ReplayVerifier(args.size)
    deadline = time.monotonic() + args.timeout_seconds
    try:
        while time.monotonic() < deadline and not node.complete():
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    required_keys = {
        "provider",
        "inference_mean_ms",
        "total_p95_ms",
        "effective_fps",
        "defect_fraction",
        "failed_frames",
    }
    expected_mask_hashes = [
        hashlib.sha256(bytes([0]) * args.size * args.size).hexdigest(),
        hashlib.sha256(bytes([255]) * args.size * args.size).hexdigest(),
    ]
    if node.mask_hashes != expected_mask_hashes:
        raise AssertionError(f"unexpected mask hashes: {node.mask_hashes}")
    if node.mask_stamps != [1, 2] or node.diagnostic_stamps != [1, 2]:
        raise AssertionError("output timestamps did not preserve the rosbag frame timestamps")
    if node.fractions != [0.0, 1.0]:
        raise AssertionError(f"unexpected defect fractions: {node.fractions}")
    expected_ok = uint8_to_int(DiagnosticStatus.OK)
    if node.diagnostic_levels != [expected_ok, expected_ok]:
        raise AssertionError(f"unhealthy diagnostics: {node.diagnostic_levels}")
    if len(node.diagnostic_keys) != 2 or any(
        not required_keys.issubset(keys) for keys in map(set, node.diagnostic_keys)
    ):
        raise AssertionError(f"missing diagnostic fields: {node.diagnostic_keys}")
    if node.planner_stamps != [1, 2]:
        raise AssertionError("planner diagnostics did not preserve defect-mask timestamps")
    for index, values in enumerate(node.planner_values):
        if int(values.get("segments", "0")) <= 0:
            raise AssertionError("planner generated no path segments")
        if int(values.get("protected_contact_pixels", "-1")) != 0:
            raise AssertionError("coverage plan contacts a protected region")
        if int(values.get("source_stamp_nanoseconds", "0")) != (index + 1) * 1_000_000_000:
            raise AssertionError("planner source timestamp does not match inference output")
    for path in node.paths:
        coordinates = [value for index, value in enumerate(path) if index % 5 != 4]
        if any(value < 0.0 or value > 1.0 for value in coordinates):
            raise AssertionError("coverage path coordinates must be normalized")
    first_feed_scales = node.paths[0][4::5]
    second_feed_scales = node.paths[1][4::5]
    if any(abs(value - 1.0) > 1e-6 for value in first_feed_scales):
        raise AssertionError("defect-free frame should use nominal feed")
    if any(abs(value - 0.55) > 1e-6 for value in second_feed_scales):
        raise AssertionError("defect frame should use reduced feed")

    payload = {
        "pass": True,
        "mask_hashes": node.mask_hashes,
        "mask_stamps": node.mask_stamps,
        "defect_fractions": node.fractions,
        "diagnostic_messages": len(node.diagnostic_levels),
        "diagnostic_levels": node.diagnostic_levels,
        "coverage_paths": len(node.paths),
        "coverage_segments": [len(path) // 5 for path in node.paths],
        "planner_stamps": node.planner_stamps,
        "protected_contact_pixels": [
            int(values["protected_contact_pixels"]) for values in node.planner_values
        ],
        "feed_scales": [sorted(set(path[4::5])) for path in node.paths],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
