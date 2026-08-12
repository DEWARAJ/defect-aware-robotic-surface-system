from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import rosbag2_py
from onnx import TensorProto, helper, numpy_helper
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Image


def build_model(output_path: Path, size: int) -> None:
    input_info = helper.make_tensor_value_info(
        "image", TensorProto.FLOAT, [1, 3, size, size]
    )
    output_info = helper.make_tensor_value_info(
        "logits", TensorProto.FLOAT, [1, 1, size, size]
    )
    half = numpy_helper.from_array(np.asarray(0.5, dtype=np.float32), "half")
    scale = numpy_helper.from_array(np.asarray(20.0, dtype=np.float32), "scale")
    nodes = [
        helper.make_node("ReduceMean", ["image"], ["mean"], axes=[1], keepdims=1),
        helper.make_node("Sub", ["mean", "half"], ["centered"]),
        helper.make_node("Mul", ["centered", "scale"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes,
        "deterministic_surface_replay",
        [input_info],
        [output_info],
        [half, scale],
    )
    model = helper.make_model(
        graph,
        producer_name="surface-perception-v08-replay",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)


def image_message(size: int, value: int, timestamp_seconds: int) -> Image:
    message = Image()
    message.header.stamp.sec = timestamp_seconds
    message.header.frame_id = "replay_camera"
    message.height = size
    message.width = size
    message.encoding = "rgb8"
    message.is_bigendian = False
    message.step = size * 3
    message.data = [value] * (size * size * 3)
    return message


def build_bag(output_path: Path, size: int) -> None:
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(output_path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    writer.create_topic(
        rosbag2_py.TopicMetadata(
            name="/camera/image_raw",
            type="sensor_msgs/msg/Image",
            serialization_format="cdr",
            offered_qos_profiles="",
        )
    )
    for timestamp_seconds, value in ((1, 0), (2, 255)):
        message = image_message(size, value, timestamp_seconds)
        writer.write(
            "/camera/image_raw",
            serialize_message(message),
            timestamp_seconds * 1_000_000_000,
        )
    del writer


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic ROS2 replay fixtures")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=8)
    args = parser.parse_args()
    if args.size <= 0:
        raise ValueError("size must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "replay_model.onnx"
    bag_path = args.output / "camera_bag"
    build_model(model_path, args.size)
    build_bag(bag_path, args.size)
    print(
        json.dumps(
            {"model": str(model_path), "bag": str(bag_path), "frames": 2},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
