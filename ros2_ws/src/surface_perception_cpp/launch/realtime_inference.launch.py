from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("surface_perception_cpp"))
    default_config = str(package_share / "config" / "realtime_inference.yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            DeclareLaunchArgument("model_path", default_value="model.onnx"),
            Node(
                package="surface_perception_cpp",
                executable="realtime_inference_node",
                name="surface_perception_realtime",
                output="screen",
                parameters=[
                    LaunchConfiguration("config"),
                    {"model_path": LaunchConfiguration("model_path")},
                ],
            ),
        ]
    )
