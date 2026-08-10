#include "surface_perception_cpp/coverage_planner.hpp"

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace surface_perception_cpp {

class CoveragePlannerNode final : public rclcpp::Node {
 public:
  CoveragePlannerNode() : Node("surface_coverage_planner_cpp") {
    tool_radius_ = declare_parameter<int>("tool_radius_pixels", 6);
    lane_spacing_ = declare_parameter<int>("lane_spacing_pixels", 9);
    minimum_segment_length_ = declare_parameter<int>("minimum_segment_length_pixels", 12);
    path_publisher_ = create_publisher<std_msgs::msg::Float32MultiArray>(
        "/surface_perception/coverage_path", 10);
    diagnostic_publisher_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
        "/surface_perception/planner_diagnostics", 10);
    protected_subscription_ = create_subscription<sensor_msgs::msg::Image>(
        "/surface_perception/protected_mask", 10,
        [this](const sensor_msgs::msg::Image::SharedPtr message) {
          if (message->encoding != "mono8" || message->step != message->width) {
            RCLCPP_WARN(get_logger(), "Protected mask must be tightly packed mono8");
            return;
          }
          protected_mask_ = message->data;
          protected_width_ = static_cast<int>(message->width);
          protected_height_ = static_cast<int>(message->height);
        });
    defect_subscription_ = create_subscription<sensor_msgs::msg::Image>(
        "/surface_perception/mask", 10,
        [this](const sensor_msgs::msg::Image::SharedPtr message) {
          if (message->encoding != "mono8" || message->step != message->width) {
            RCLCPP_WARN(get_logger(), "Defect mask must be tightly packed mono8");
            return;
          }
          defect_mask_ = message->data;
          defect_width_ = static_cast<int>(message->width);
          defect_height_ = static_cast<int>(message->height);
        });
    sanding_subscription_ = create_subscription<sensor_msgs::msg::Image>(
        "/surface_perception/sanding_mask", 10,
        std::bind(&CoveragePlannerNode::onSandingMask, this, std::placeholders::_1));
  }

 private:
  static std::vector<std::uint8_t> binaryMask(const sensor_msgs::msg::Image& message) {
    std::vector<std::uint8_t> result(message.data.size());
    std::transform(message.data.begin(), message.data.end(), result.begin(),
                   [](const std::uint8_t value) { return value == 0 ? 0U : 1U; });
    return result;
  }

  void onSandingMask(const sensor_msgs::msg::Image::SharedPtr message) {
    const int width = static_cast<int>(message->width);
    const int height = static_cast<int>(message->height);
    const auto pixels = static_cast<std::size_t>(width * height);
    if (message->encoding != "mono8" || message->step != message->width ||
        protected_width_ != width || protected_height_ != height ||
        defect_width_ != width || defect_height_ != height ||
        protected_mask_.size() != pixels || defect_mask_.size() != pixels) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
                           "Waiting for aligned mono8 sanding, protected, and defect masks");
      return;
    }

    CoveragePlanner planner(width, height, tool_radius_, lane_spacing_, minimum_segment_length_);
    const auto sanding = binaryMask(*message);
    std::vector<std::uint8_t> protected_binary(protected_mask_.size());
    std::vector<std::uint8_t> defect_binary(defect_mask_.size());
    std::transform(protected_mask_.begin(), protected_mask_.end(), protected_binary.begin(),
                   [](const std::uint8_t value) { return value == 0 ? 0U : 1U; });
    std::transform(defect_mask_.begin(), defect_mask_.end(), defect_binary.begin(),
                   [](const std::uint8_t value) { return value == 0 ? 0U : 1U; });
    const Plan plan = planner.plan(sanding, protected_binary, defect_binary);

    std_msgs::msg::Float32MultiArray path_message;
    path_message.layout.dim.resize(2);
    path_message.layout.dim[0].label = "segments";
    path_message.layout.dim[0].size = plan.segments.size();
    path_message.layout.dim[0].stride = plan.segments.size() * 5;
    path_message.layout.dim[1].label = "start_x,start_y,end_x,end_y,feed_scale";
    path_message.layout.dim[1].size = 5;
    path_message.layout.dim[1].stride = 5;
    path_message.data.reserve(plan.segments.size() * 5);
    for (const auto& segment : plan.segments) {
      path_message.data.push_back(static_cast<float>(segment.start.x) / static_cast<float>(width - 1));
      path_message.data.push_back(static_cast<float>(segment.start.y) / static_cast<float>(height - 1));
      path_message.data.push_back(static_cast<float>(segment.end.x) / static_cast<float>(width - 1));
      path_message.data.push_back(static_cast<float>(segment.end.y) / static_cast<float>(height - 1));
      path_message.data.push_back(static_cast<float>(segment.feed_scale));
    }
    path_publisher_->publish(path_message);

    diagnostic_msgs::msg::DiagnosticArray diagnostics;
    diagnostics.header = message->header;
    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name = "surface_perception/coverage_planner_cpp";
    status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
    status.message = "safe normalized image-plane path generated";
    diagnostic_msgs::msg::KeyValue segment_value;
    segment_value.key = "segments";
    segment_value.value = std::to_string(plan.segments.size());
    diagnostic_msgs::msg::KeyValue path_value;
    path_value.key = "process_path_pixels";
    path_value.value = std::to_string(plan.process_path_pixels);
    diagnostic_msgs::msg::KeyValue contact_value;
    contact_value.key = "protected_contact_pixels";
    contact_value.value = std::to_string(plan.protected_contact_pixels);
    status.values = {segment_value, path_value, contact_value};
    diagnostics.status.push_back(std::move(status));
    diagnostic_publisher_->publish(diagnostics);
  }

  int tool_radius_{};
  int lane_spacing_{};
  int minimum_segment_length_{};
  int protected_width_{};
  int protected_height_{};
  int defect_width_{};
  int defect_height_{};
  std::vector<std::uint8_t> protected_mask_;
  std::vector<std::uint8_t> defect_mask_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr path_publisher_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostic_publisher_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sanding_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr protected_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr defect_subscription_;
};

}  // namespace surface_perception_cpp

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<surface_perception_cpp::CoveragePlannerNode>());
  rclcpp::shutdown();
  return 0;
}
