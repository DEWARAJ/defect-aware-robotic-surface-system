from __future__ import annotations

import json
import time

import cv2
import numpy as np
import onnxruntime as ort
import rclpy
from cv_bridge import CvBridge
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from sensor_msgs.msg import Image


class SurfacePerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("surface_perception")
        self.declare_parameter("model_path", "model.onnx")
        self.declare_parameter("input_topic", "/camera/image_raw")
        self.declare_parameter("mask_topic", "/surface_perception/mask")
        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("input_size", 256)

        model_path = self.get_parameter("model_path").value
        providers = [
            provider
            for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if provider in ort.get_available_providers()
        ]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.threshold = float(self.get_parameter("threshold").value)
        self.input_size = int(self.get_parameter("input_size").value)
        self.bridge = CvBridge()
        self.mask_publisher = self.create_publisher(
            Image, str(self.get_parameter("mask_topic").value), 10
        )
        self.diagnostic_publisher = self.create_publisher(
            DiagnosticArray, "/surface_perception/diagnostics", 10
        )
        self.subscription = self.create_subscription(
            Image, str(self.get_parameter("input_topic").value), self.on_image, 10
        )
        self.get_logger().info(
            json.dumps({"model": model_path, "providers": self.session.get_providers()})
        )

    def on_image(self, message: Image) -> None:
        bgr = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        tensor = resized.astype(np.float32).transpose(2, 0, 1)[None, ...] / 255.0
        start = time.perf_counter()
        logits = self.session.run(None, {self.input_name: tensor})[0][0, 0]
        latency_ms = (time.perf_counter() - start) * 1000.0
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        mask = (probability >= self.threshold).astype(np.uint8) * 255
        mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_message = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
        mask_message.header = message.header
        self.mask_publisher.publish(mask_message)

        diagnostics = DiagnosticArray()
        diagnostics.header = message.header
        status = DiagnosticStatus()
        status.name = "surface_perception/inference"
        status.level = DiagnosticStatus.OK
        status.message = "inference complete"
        status.values = [
            KeyValue(key="latency_ms", value=f"{latency_ms:.3f}"),
            KeyValue(key="defect_fraction", value=f"{float((mask > 0).mean()):.6f}"),
            KeyValue(key="provider", value=self.session.get_providers()[0]),
        ]
        diagnostics.status = [status]
        self.diagnostic_publisher.publish(diagnostics)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SurfacePerceptionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

