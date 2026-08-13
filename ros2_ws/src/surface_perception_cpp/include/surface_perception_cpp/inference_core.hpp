#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

namespace surface_perception_cpp {

enum class PixelOrder { kRgb, kBgr };

struct ModelInput {
  std::vector<float> tensor;
  std::array<std::int64_t, 4> shape{};
};

struct SegmentationResult {
  std::vector<std::uint8_t> mask;
  double defect_fraction{0.0};
  double mean_probability{0.0};
  double peak_probability{0.0};
};

struct TimingSample {
  double preprocess_ms{0.0};
  double inference_ms{0.0};
  double postprocess_ms{0.0};
  double total_ms{0.0};
};

struct TimingSummary {
  std::size_t samples{0};
  double mean_preprocess_ms{0.0};
  double mean_inference_ms{0.0};
  double mean_postprocess_ms{0.0};
  double mean_total_ms{0.0};
  double p50_total_ms{0.0};
  double p95_total_ms{0.0};
  double max_total_ms{0.0};
  double effective_fps{0.0};
};

[[nodiscard]] ModelInput preprocessImage(
    const std::vector<std::uint8_t>& image, std::size_t source_width,
    std::size_t source_height, std::size_t source_step,
    std::size_t target_width, std::size_t target_height, PixelOrder order);

[[nodiscard]] SegmentationResult postprocessLogits(
    const std::vector<float>& logits, std::size_t model_width,
    std::size_t model_height, std::size_t output_width,
    std::size_t output_height, double threshold);

class RollingTelemetry {
 public:
  explicit RollingTelemetry(std::size_t window_size);

  void add(const TimingSample& sample);
  [[nodiscard]] TimingSummary summary() const;

 private:
  std::size_t window_size_;
  std::deque<TimingSample> samples_;
};

}  // namespace surface_perception_cpp
