from setuptools import find_packages, setup


package_name = "surface_perception_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Dewaraj Raparthi",
    maintainer_email="dewarajraparthi@gmail.com",
    description="ROS2 ONNX surface-defect segmentation node.",
    license="MIT",
    entry_points={
        "console_scripts": ["surface_perception_node = surface_perception_ros.node:main"]
    },
)

