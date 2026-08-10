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

add_library(coverage_planner src/coverage_planner.cpp)
target_include_directories(coverage_planner PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_options(coverage_planner PRIVATE -Wall -Wextra -Wpedantic)

add_executable(coverage_planner_node src/coverage_planner_node.cpp)
target_link_libraries(coverage_planner_node coverage_planner)
ament_target_dependencies(coverage_planner_node
  diagnostic_msgs rclcpp sensor_msgs std_msgs)

install(TARGETS coverage_planner coverage_planner_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_coverage_planner test/test_coverage_planner.cpp)
  target_link_libraries(test_coverage_planner coverage_planner)
endif()

ament_export_include_directories(include)
ament_export_libraries(coverage_planner)
ament_package()
