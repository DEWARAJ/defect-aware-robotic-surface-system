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
from std_msgs.msg import Float32


EXPECTED_MASK_HASHES = [
    hashlib.sha256(bytes([0]) * 64).hexdigest(),
    hashlib.sha256(bytes([255]) * 64).hexdigest(),
]


def uint8_to_int(value: int | bytes | bytearray) -> int:
    """Normalize ROS uint8 fields across rosidl Python representations."""
    if isinstance(value, (bytes, bytearray)):
        if len(value) != 1:
            raise AssertionError(f"expected one diagnostic level byte, got {len(value)}")
        return value[0]
    return int(value)


class ReplayVerifier(Node):
    def __init__(self) -> None:
        super().__init__("surface_perception_replay_verifier")
        self.mask_hashes: list[str] = []
        self.mask_stamps: list[int] = []
        self.fractions: list[float] = []
        self.diagnostic_stamps: list[int] = []
        self.diagnostic_levels: list[int] = []
        self.diagnostic_keys: list[list[str]] = []
        self.create_subscription(Image, "/surface_perception/mask", self.on_mask, 10)
        self.create_subscription(
            Float32, "/surface_perception/defect_fraction", self.on_fraction, 10
        )
        self.create_subscription(
            DiagnosticArray, "/surface_perception/diagnostics", self.on_diagnostics, 10
        )

    def on_mask(self, message: Image) -> None:
        if message.encoding != "mono8" or message.width != 8 or message.height != 8:
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

    def complete(self) -> bool:
        return (
            len(self.mask_hashes) >= 2
            and len(self.fractions) >= 2
            and len(self.diagnostic_levels) >= 2
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify deterministic ROS2 replay outputs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    args = parser.parse_args()
    rclpy.init()
    node = ReplayVerifier()
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
    if node.mask_hashes != EXPECTED_MASK_HASHES:
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

    payload = {
        "pass": True,
        "mask_hashes": node.mask_hashes,
        "mask_stamps": node.mask_stamps,
        "defect_fractions": node.fractions,
        "diagnostic_messages": len(node.diagnostic_levels),
        "diagnostic_levels": node.diagnostic_levels,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
