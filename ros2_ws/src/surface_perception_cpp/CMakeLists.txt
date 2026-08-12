cmake_minimum_required(VERSION 3.16)
project(surface_perception_cpp)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

add_library(inference_core src/inference_core.cpp)
target_include_directories(inference_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_options(inference_core PRIVATE -Wall -Wextra -Wpedantic)

add_library(coverage_planner src/coverage_planner.cpp)
target_include_directories(coverage_planner PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_options(coverage_planner PRIVATE -Wall -Wextra -Wpedantic)

add_executable(coverage_planner_node src/coverage_planner_node.cpp)
target_link_libraries(coverage_planner_node coverage_planner)
ament_target_dependencies(coverage_planner_node
  diagnostic_msgs rclcpp sensor_msgs std_msgs)

option(SURFACE_PERCEPTION_ENABLE_ONNX "Build the ONNX Runtime inference node" OFF)
if(SURFACE_PERCEPTION_ENABLE_ONNX)
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS ENV ONNXRUNTIME_ROOT
    PATH_SUFFIXES include include/onnxruntime/core/session)
  find_library(ONNXRUNTIME_LIBRARY onnxruntime
    HINTS ENV ONNXRUNTIME_ROOT
    PATH_SUFFIXES lib lib64)
  if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR
      "ONNX Runtime C++ headers/library not found. Set ONNXRUNTIME_ROOT or disable "
      "SURFACE_PERCEPTION_ENABLE_ONNX.")
  endif()

  add_executable(realtime_inference_node src/realtime_inference_node.cpp)
  target_include_directories(realtime_inference_node PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(realtime_inference_node inference_core ${ONNXRUNTIME_LIBRARY})
  target_compile_options(realtime_inference_node PRIVATE -Wall -Wextra -Wpedantic)
  ament_target_dependencies(realtime_inference_node
    diagnostic_msgs rclcpp sensor_msgs std_msgs)
  install(TARGETS realtime_inference_node RUNTIME DESTINATION lib/${PROJECT_NAME})
endif()

install(TARGETS inference_core coverage_planner coverage_planner_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_coverage_planner test/test_coverage_planner.cpp)
  target_link_libraries(test_coverage_planner coverage_planner)
  ament_add_gtest(test_inference_core test/test_inference_core.cpp)
  target_link_libraries(test_inference_core inference_core)
endif()

ament_export_include_directories(include)
ament_export_libraries(coverage_planner inference_core)
ament_package()
