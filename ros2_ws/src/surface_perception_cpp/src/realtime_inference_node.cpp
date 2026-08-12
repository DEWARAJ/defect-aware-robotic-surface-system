#include "surface_perception_cpp/inference_core.hpp"

#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32.hpp>

namespace surface_perception_cpp {
namespace {

using SteadyClock = std::chrono::steady_clock;

double elapsedMilliseconds(SteadyClock::time_point start,
                           SteadyClock::time_point finish) {
  return std::chrono::duration<double, std::milli>(finish - start).count();
}

diagnostic_msgs::msg::KeyValue keyValue(const std::string& key, double value,
                                        int precision = 3) {
  diagnostic_msgs::msg::KeyValue result;
  result.key = key;
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(precision) << value;
  result.value = stream.str();
  return result;
}

diagnostic_msgs::msg::KeyValue keyValue(const std::string& key,
                                        const std::string& value) {
  diagnostic_msgs::msg::KeyValue result;
  result.key = key;
  result.value = value;
  return result;
}

}  // namespace

class RealtimeInferenceNode final : public rclcpp::Node {
 public:
  RealtimeInferenceNode()
      : Node("surface_perception_realtime"),
        environment_(ORT_LOGGING_LEVEL_WARNING, "surface_perception"),
        telemetry_(120) {
    model_path_ = declare_parameter<std::string>("model_path", "model.onnx");
    input_topic_ = declare_parameter<std::string>("input_topic", "/camera/image_raw");
    mask_topic_ =
        declare_parameter<std::string>("mask_topic", "/surface_perception/mask");
    threshold_ = declare_parameter<double>("threshold", 0.5);
    input_width_ = declare_parameter<int>("input_width", 256);
    input_height_ = declare_parameter<int>("input_height", 256);
    maximum_p95_ms_ = declare_parameter<double>("maximum_p95_ms", 50.0);
    intra_op_threads_ = declare_parameter<int>("intra_op_threads", 1);
    const int telemetry_window = declare_parameter<int>("telemetry_window", 120);

    validateParameters(telemetry_window);
    telemetry_ = RollingTelemetry(static_cast<std::size_t>(telemetry_window));
    configureSession();

    mask_publisher_ = create_publisher<sensor_msgs::msg::Image>(mask_topic_, 10);
    fraction_publisher_ = create_publisher<std_msgs::msg::Float32>(
        "/surface_perception/defect_fraction", 10);
    diagnostics_publisher_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
        "/surface_perception/diagnostics", 10);
    image_subscription_ = create_subscription<sensor_msgs::msg::Image>(
        input_topic_, rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::Image::ConstSharedPtr message) { onImage(*message); });

    RCLCPP_INFO(get_logger(),
                "model=%s input=%s mask=%s shape=[1,3,%d,%d] provider=%s",
                model_path_.c_str(), input_topic_.c_str(), mask_topic_.c_str(),
                input_height_, input_width_, provider_.c_str());
  }

 private:
  void validateParameters(int telemetry_window) const {
    if (!std::filesystem::is_regular_file(model_path_)) {
      throw std::invalid_argument("model_path must point to a readable ONNX file");
    }
    if (input_width_ <= 0 || input_height_ <= 0) {
      throw std::invalid_argument("input dimensions must be positive");
    }
    if (threshold_ < 0.0 || threshold_ > 1.0) {
      throw std::invalid_argument("threshold must be in [0, 1]");
    }
    if (maximum_p95_ms_ <= 0.0 || intra_op_threads_ <= 0 || telemetry_window <= 0) {
      throw std::invalid_argument(
          "latency target, thread count, and telemetry window must be positive");
    }
  }

  void configureSession() {
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options_.SetIntraOpNumThreads(intra_op_threads_);
    session_ = std::make_unique<Ort::Session>(environment_, model_path_.c_str(),
                                              session_options_);
    if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1) {
      throw std::runtime_error("model must expose exactly one input and one output");
    }

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session_->GetInputNameAllocated(0, allocator);
    auto output_name = session_->GetOutputNameAllocated(0, allocator);
    input_name_ = input_name.get();
    output_name_ = output_name.get();

    const auto input_type = session_->GetInputTypeInfo(0);
    const auto input_info = input_type.GetTensorTypeAndShapeInfo();
    if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw std::runtime_error("model input must be float32");
    }
    const auto shape = input_info.GetShape();
    if (shape.size() != 4 || (shape[0] > 0 && shape[0] != 1) ||
        (shape[1] > 0 && shape[1] != 3) ||
        (shape[2] > 0 && shape[2] != input_height_) ||
        (shape[3] > 0 && shape[3] != input_width_)) {
      throw std::runtime_error("model input must match [1,3,input_height,input_width]");
    }
  }

  PixelOrder pixelOrder(const std::string& encoding) const {
    if (encoding == sensor_msgs::image_encodings::RGB8) {
      return PixelOrder::kRgb;
    }
    if (encoding == sensor_msgs::image_encodings::BGR8) {
      return PixelOrder::kBgr;
    }
    throw std::invalid_argument("only rgb8 and bgr8 camera images are supported");
  }

  void onImage(const sensor_msgs::msg::Image& message) {
    try {
      const auto total_start = SteadyClock::now();
      const auto preprocess_start = total_start;
      auto input = preprocessImage(
          message.data, message.width, message.height, message.step,
          static_cast<std::size_t>(input_width_), static_cast<std::size_t>(input_height_),
          pixelOrder(message.encoding));
      const auto preprocess_finish = SteadyClock::now();

      auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      auto input_tensor = Ort::Value::CreateTensor<float>(
          memory, input.tensor.data(), input.tensor.size(), input.shape.data(),
          input.shape.size());
      const char* input_names[] = {input_name_.c_str()};
      const char* output_names[] = {output_name_.c_str()};
      const auto inference_start = SteadyClock::now();
      auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
                                   output_names, 1);
      const auto inference_finish = SteadyClock::now();

      const auto output_info = outputs.front().GetTensorTypeAndShapeInfo();
      if (output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("model output must be float32");
      }
      const auto output_shape = output_info.GetShape();
      if (output_shape.size() != 4 || output_shape[0] != 1 || output_shape[1] != 1 ||
          output_shape[2] <= 0 || output_shape[3] <= 0) {
        throw std::runtime_error("model output must have shape [1,1,H,W]");
      }
      const std::size_t model_height = static_cast<std::size_t>(output_shape[2]);
      const std::size_t model_width = static_cast<std::size_t>(output_shape[3]);
      const float* output_data = outputs.front().GetTensorData<float>();
      std::vector<float> logits(output_data, output_data + model_height * model_width);

      const auto postprocess_start = SteadyClock::now();
      auto result = postprocessLogits(logits, model_width, model_height, message.width,
                                      message.height, threshold_);
      const auto postprocess_finish = SteadyClock::now();
      telemetry_.add({elapsedMilliseconds(preprocess_start, preprocess_finish),
                      elapsedMilliseconds(inference_start, inference_finish),
                      elapsedMilliseconds(postprocess_start, postprocess_finish),
                      elapsedMilliseconds(total_start, postprocess_finish)});
      publishResult(message, std::move(result));
    } catch (const std::exception& error) {
      ++failed_frames_;
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "inference failed: %s",
                            error.what());
      publishFailure(message, error.what());
    }
  }

  void publishResult(const sensor_msgs::msg::Image& source,
                     SegmentationResult result) {
    sensor_msgs::msg::Image mask;
    mask.header = source.header;
    mask.height = source.height;
    mask.width = source.width;
    mask.encoding = sensor_msgs::image_encodings::MONO8;
    mask.is_bigendian = false;
    mask.step = source.width;
    mask.data = std::move(result.mask);
    mask_publisher_->publish(mask);

    std_msgs::msg::Float32 fraction;
    fraction.data = static_cast<float>(result.defect_fraction);
    fraction_publisher_->publish(fraction);

    const TimingSummary timing = telemetry_.summary();
    diagnostic_msgs::msg::DiagnosticArray diagnostics;
    diagnostics.header = source.header;
    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name = "surface_perception/realtime_inference";
    status.hardware_id = provider_;
    const bool deadline_missed = timing.p95_total_ms > maximum_p95_ms_;
    status.level = deadline_missed ? diagnostic_msgs::msg::DiagnosticStatus::WARN
                                   : diagnostic_msgs::msg::DiagnosticStatus::OK;
    status.message = deadline_missed ? "p95 latency target exceeded" : "inference healthy";
    status.values = {
        keyValue("provider", provider_),
        keyValue("model_path", model_path_),
        keyValue("preprocess_mean_ms", timing.mean_preprocess_ms),
        keyValue("inference_mean_ms", timing.mean_inference_ms),
        keyValue("postprocess_mean_ms", timing.mean_postprocess_ms),
        keyValue("total_p50_ms", timing.p50_total_ms),
        keyValue("total_p95_ms", timing.p95_total_ms),
        keyValue("effective_fps", timing.effective_fps),
        keyValue("defect_fraction", result.defect_fraction, 6),
        keyValue("mean_probability", result.mean_probability, 6),
        keyValue("peak_probability", result.peak_probability, 6),
        keyValue("failed_frames", static_cast<double>(failed_frames_), 0),
    };
    diagnostics.status.push_back(std::move(status));
    diagnostics_publisher_->publish(diagnostics);
  }

  void publishFailure(const sensor_msgs::msg::Image& source, const std::string& error) {
    diagnostic_msgs::msg::DiagnosticArray diagnostics;
    diagnostics.header = source.header;
    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name = "surface_perception/realtime_inference";
    status.hardware_id = provider_;
    status.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
    status.message = error;
    status.values = {
        keyValue("failed_frames", static_cast<double>(failed_frames_), 0),
    };
    diagnostics.status.push_back(std::move(status));
    diagnostics_publisher_->publish(diagnostics);
  }

  Ort::Env environment_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_;
  RollingTelemetry telemetry_;
  std::string model_path_;
  std::string input_topic_;
  std::string mask_topic_;
  std::string input_name_;
  std::string output_name_;
  std::string provider_{"CPUExecutionProvider"};
  double threshold_{0.5};
  double maximum_p95_ms_{50.0};
  int input_width_{256};
  int input_height_{256};
  int intra_op_threads_{1};
  std::size_t failed_frames_{0};
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr mask_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr fraction_publisher_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
      diagnostics_publisher_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
};

}  // namespace surface_perception_cpp

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(
        std::make_shared<surface_perception_cpp::RealtimeInferenceNode>());
  } catch (const std::exception& error) {
    RCLCPP_FATAL(rclcpp::get_logger("surface_perception_realtime"), "%s",
                 error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
