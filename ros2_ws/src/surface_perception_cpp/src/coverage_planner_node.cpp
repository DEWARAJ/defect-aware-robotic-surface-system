#include "surface_perception_cpp/coverage_planner.hpp"

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
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
    sanding_topic_ =
        declare_parameter<std::string>("sanding_topic", "/surface_perception/sanding_mask");
    protected_topic_ = declare_parameter<std::string>(
        "protected_topic", "/surface_perception/protected_mask");
    defect_topic_ =
        declare_parameter<std::string>("defect_topic", "/surface_perception/mask");
    path_topic_ = declare_parameter<std::string>(
        "path_topic", "/surface_perception/coverage_path");
    validateParameters();
    path_publisher_ = create_publisher<std_msgs::msg::Float32MultiArray>(
        path_topic_, 10);
    diagnostic_publisher_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
        "/surface_perception/planner_diagnostics", 10);
    protected_subscription_ = create_subscription<sensor_msgs::msg::Image>(
        protected_topic_, 10,
        [this](const sensor_msgs::msg::Image::SharedPtr message) {
          if (!validMask(*message, "protected")) {
            return;
          }
          protected_mask_ = message->data;
          protected_width_ = static_cast<int>(message->width);
          protected_height_ = static_cast<int>(message->height);
        });
    sanding_subscription_ = create_subscription<sensor_msgs::msg::Image>(
        sanding_topic_, 10,
        [this](const sensor_msgs::msg::Image::SharedPtr message) {
          if (!validMask(*message, "sanding")) {
            return;
          }
          sanding_mask_ = message->data;
          sanding_width_ = static_cast<int>(message->width);
          sanding_height_ = static_cast<int>(message->height);
        });
    defect_subscription_ = create_subscription<sensor_msgs::msg::Image>(
        defect_topic_, 10,
        [this](const sensor_msgs::msg::Image::SharedPtr message) { onDefectMask(message); });

    RCLCPP_INFO(get_logger(), "sanding=%s protected=%s defect=%s path=%s",
                sanding_topic_.c_str(), protected_topic_.c_str(), defect_topic_.c_str(),
                path_topic_.c_str());
  }

 private:
  void validateParameters() const {
    if (tool_radius_ < 1 || lane_spacing_ < 1 ||
        lane_spacing_ > tool_radius_ * 2 || minimum_segment_length_ < 2) {
      throw std::invalid_argument("invalid coverage planner parameters");
    }
    if (sanding_topic_.empty() || protected_topic_.empty() || defect_topic_.empty() ||
        path_topic_.empty()) {
      throw std::invalid_argument("coverage planner topics cannot be empty");
    }
  }

  bool validMask(const sensor_msgs::msg::Image& message, const char* role) const {
    const auto pixels = static_cast<std::size_t>(message.width) * message.height;
    if (message.encoding != "mono8" || message.width < 2 || message.height < 2 ||
        message.step != message.width || message.data.size() != pixels) {
      RCLCPP_WARN(get_logger(), "%s mask must be non-empty, tightly packed mono8", role);
      return false;
    }
    return true;
  }

  static std::vector<std::uint8_t> binaryMask(const sensor_msgs::msg::Image& message) {
    std::vector<std::uint8_t> result(message.data.size());
    std::transform(message.data.begin(), message.data.end(), result.begin(),
                   [](const std::uint8_t value) { return value == 0 ? 0U : 1U; });
    return result;
  }

  void onDefectMask(const sensor_msgs::msg::Image::SharedPtr message) {
    if (!validMask(*message, "defect")) {
      return;
    }
    const int width = static_cast<int>(message->width);
    const int height = static_cast<int>(message->height);
    const auto pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (sanding_width_ != width || sanding_height_ != height ||
        protected_width_ != width || protected_height_ != height ||
        sanding_mask_.size() != pixels || protected_mask_.size() != pixels) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
                           "Waiting for dimension-aligned sanding and protected masks");
      return;
    }

    CoveragePlanner planner(width, height, tool_radius_, lane_spacing_, minimum_segment_length_);
    std::vector<std::uint8_t> sanding_binary(sanding_mask_.size());
    std::vector<std::uint8_t> protected_binary(protected_mask_.size());
    std::transform(sanding_mask_.begin(), sanding_mask_.end(), sanding_binary.begin(),
                   [](const std::uint8_t value) { return value == 0 ? 0U : 1U; });
    std::transform(protected_mask_.begin(), protected_mask_.end(), protected_binary.begin(),
                   [](const std::uint8_t value) { return value == 0 ? 0U : 1U; });
    const auto defect_binary = binaryMask(*message);
    const Plan plan = planner.plan(sanding_binary, protected_binary, defect_binary);

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
    diagnostic_msgs::msg::KeyValue source_stamp_value;
    source_stamp_value.key = "source_stamp_nanoseconds";
    source_stamp_value.value = std::to_string(
        rclcpp::Time(message->header.stamp).nanoseconds());
    status.values = {segment_value, path_value, contact_value, source_stamp_value};
    diagnostics.status.push_back(std::move(status));
    diagnostic_publisher_->publish(diagnostics);
  }

  int tool_radius_{};
  int lane_spacing_{};
  int minimum_segment_length_{};
  int protected_width_{};
  int protected_height_{};
  int sanding_width_{};
  int sanding_height_{};
  std::string sanding_topic_;
  std::string protected_topic_;
  std::string defect_topic_;
  std::string path_topic_;
  std::vector<std::uint8_t> sanding_mask_;
  std::vector<std::uint8_t> protected_mask_;
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
