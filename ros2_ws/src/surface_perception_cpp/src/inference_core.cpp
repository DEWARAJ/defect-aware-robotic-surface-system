#include "surface_perception_cpp/inference_core.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace surface_perception_cpp {
namespace {

std::size_t checkedProduct(std::size_t first, std::size_t second,
                           const char* description) {
  if (first != 0 && second > std::numeric_limits<std::size_t>::max() / first) {
    throw std::overflow_error(description);
  }
  return first * second;
}

double sigmoid(double value) {
  if (value >= 0.0) {
    const double exponential = std::exp(-value);
    return 1.0 / (1.0 + exponential);
  }
  const double exponential = std::exp(value);
  return exponential / (1.0 + exponential);
}

double nearestRankPercentile(std::vector<double> values, double percentile) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const auto rank = static_cast<std::size_t>(
      std::ceil(percentile * static_cast<double>(values.size())));
  const std::size_t index = std::min(values.size() - 1, std::max<std::size_t>(1, rank) - 1);
  return values[index];
}

void validateTiming(double value, const char* name) {
  if (!std::isfinite(value) || value < 0.0) {
    throw std::invalid_argument(name);
  }
}

}  // namespace

ModelInput preprocessImage(const std::vector<std::uint8_t>& image,
                           std::size_t source_width,
                           std::size_t source_height,
                           std::size_t source_step,
                           std::size_t target_width,
                           std::size_t target_height, PixelOrder order) {
  if (source_width == 0 || source_height == 0 || target_width == 0 ||
      target_height == 0) {
    throw std::invalid_argument("image dimensions must be positive");
  }
  const std::size_t packed_step = checkedProduct(source_width, 3, "source row overflow");
  if (source_step < packed_step) {
    throw std::invalid_argument("source step is smaller than a packed RGB row");
  }
  const std::size_t required_bytes =
      checkedProduct(source_step, source_height, "source image size overflow");
  if (image.size() < required_bytes) {
    throw std::invalid_argument("source image buffer is smaller than height times step");
  }
  const std::size_t target_pixels =
      checkedProduct(target_width, target_height, "target image size overflow");
  const std::size_t tensor_values = checkedProduct(target_pixels, 3, "tensor size overflow");

  ModelInput output;
  output.tensor.resize(tensor_values);
  output.shape = {1, 3, static_cast<std::int64_t>(target_height),
                  static_cast<std::int64_t>(target_width)};

  for (std::size_t target_y = 0; target_y < target_height; ++target_y) {
    const double mapped_y = std::clamp(
        (static_cast<double>(target_y) + 0.5) *
                static_cast<double>(source_height) / static_cast<double>(target_height) -
            0.5,
        0.0, static_cast<double>(source_height - 1));
    const auto y0 = static_cast<std::size_t>(std::floor(mapped_y));
    const auto y1 = std::min(y0 + 1, source_height - 1);
    const double y_weight = mapped_y - static_cast<double>(y0);

    for (std::size_t target_x = 0; target_x < target_width; ++target_x) {
      const double mapped_x = std::clamp(
          (static_cast<double>(target_x) + 0.5) *
                  static_cast<double>(source_width) / static_cast<double>(target_width) -
              0.5,
          0.0, static_cast<double>(source_width - 1));
      const auto x0 = static_cast<std::size_t>(std::floor(mapped_x));
      const auto x1 = std::min(x0 + 1, source_width - 1);
      const double x_weight = mapped_x - static_cast<double>(x0);

      for (std::size_t channel = 0; channel < 3; ++channel) {
        const std::size_t source_channel =
            order == PixelOrder::kRgb ? channel : 2 - channel;
        const auto pixel = [&](std::size_t x, std::size_t y) {
          return static_cast<double>(image[y * source_step + x * 3 + source_channel]);
        };
        const double top = pixel(x0, y0) * (1.0 - x_weight) +
                           pixel(x1, y0) * x_weight;
        const double bottom = pixel(x0, y1) * (1.0 - x_weight) +
                              pixel(x1, y1) * x_weight;
        const double value = top * (1.0 - y_weight) + bottom * y_weight;
        output.tensor[channel * target_pixels + target_y * target_width + target_x] =
            static_cast<float>(value / 255.0);
      }
    }
  }
  return output;
}

SegmentationResult postprocessLogits(const std::vector<float>& logits,
                                     std::size_t model_width,
                                     std::size_t model_height,
                                     std::size_t output_width,
                                     std::size_t output_height,
                                     double threshold) {
  if (model_width == 0 || model_height == 0 || output_width == 0 ||
      output_height == 0) {
    throw std::invalid_argument("model and output dimensions must be positive");
  }
  if (!std::isfinite(threshold) || threshold < 0.0 || threshold > 1.0) {
    throw std::invalid_argument("threshold must be finite and in [0, 1]");
  }
  const std::size_t model_pixels =
      checkedProduct(model_width, model_height, "model output size overflow");
  if (logits.size() != model_pixels) {
    throw std::invalid_argument("logit count does not match the model output shape");
  }
  const std::size_t output_pixels =
      checkedProduct(output_width, output_height, "mask output size overflow");

  SegmentationResult result;
  result.mask.resize(output_pixels);
  std::size_t defect_pixels = 0;
  double probability_sum = 0.0;
  for (std::size_t y = 0; y < output_height; ++y) {
    const std::size_t source_y =
        std::min(model_height - 1, y * model_height / output_height);
    for (std::size_t x = 0; x < output_width; ++x) {
      const std::size_t source_x =
          std::min(model_width - 1, x * model_width / output_width);
      const double probability = sigmoid(logits[source_y * model_width + source_x]);
      const bool defect = probability >= threshold;
      result.mask[y * output_width + x] = defect ? 255 : 0;
      defect_pixels += defect ? 1 : 0;
      probability_sum += probability;
      result.peak_probability = std::max(result.peak_probability, probability);
    }
  }
  result.defect_fraction =
      static_cast<double>(defect_pixels) / static_cast<double>(output_pixels);
  result.mean_probability = probability_sum / static_cast<double>(output_pixels);
  return result;
}

RollingTelemetry::RollingTelemetry(std::size_t window_size) : window_size_(window_size) {
  if (window_size_ == 0) {
    throw std::invalid_argument("telemetry window must be positive");
  }
}

void RollingTelemetry::add(const TimingSample& sample) {
  validateTiming(sample.preprocess_ms, "preprocess time must be finite and non-negative");
  validateTiming(sample.inference_ms, "inference time must be finite and non-negative");
  validateTiming(sample.postprocess_ms, "postprocess time must be finite and non-negative");
  validateTiming(sample.total_ms, "total time must be finite and non-negative");
  samples_.push_back(sample);
  while (samples_.size() > window_size_) {
    samples_.pop_front();
  }
}

TimingSummary RollingTelemetry::summary() const {
  TimingSummary result;
  result.samples = samples_.size();
  if (samples_.empty()) {
    return result;
  }
  std::vector<double> totals;
  totals.reserve(samples_.size());
  for (const auto& sample : samples_) {
    result.mean_preprocess_ms += sample.preprocess_ms;
    result.mean_inference_ms += sample.inference_ms;
    result.mean_postprocess_ms += sample.postprocess_ms;
    result.mean_total_ms += sample.total_ms;
    totals.push_back(sample.total_ms);
  }
  const double count = static_cast<double>(samples_.size());
  result.mean_preprocess_ms /= count;
  result.mean_inference_ms /= count;
  result.mean_postprocess_ms /= count;
  result.mean_total_ms /= count;
  result.p50_total_ms = nearestRankPercentile(totals, 0.50);
  result.p95_total_ms = nearestRankPercentile(totals, 0.95);
  result.max_total_ms = *std::max_element(totals.begin(), totals.end());
  if (result.mean_total_ms > 0.0) {
    result.effective_fps = 1000.0 / result.mean_total_ms;
  }
  return result;
}

}  // namespace surface_perception_cpp
