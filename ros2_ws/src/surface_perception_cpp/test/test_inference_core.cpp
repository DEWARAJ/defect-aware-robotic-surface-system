#include "surface_perception_cpp/inference_core.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace surface_perception_cpp {

TEST(InferenceCore, ConvertsBgrToNormalizedRgbNchw) {
  const std::vector<std::uint8_t> bgr = {10, 20, 30};
  const auto input = preprocessImage(bgr, 1, 1, 3, 1, 1, PixelOrder::kBgr);

  EXPECT_EQ(input.shape, (std::array<std::int64_t, 4>{1, 3, 1, 1}));
  ASSERT_EQ(input.tensor.size(), 3U);
  EXPECT_NEAR(input.tensor[0], 30.0 / 255.0, 1e-6);
  EXPECT_NEAR(input.tensor[1], 20.0 / 255.0, 1e-6);
  EXPECT_NEAR(input.tensor[2], 10.0 / 255.0, 1e-6);
}

TEST(InferenceCore, RejectsTruncatedImageBuffers) {
  const std::vector<std::uint8_t> truncated(5, 0);
  EXPECT_THROW(preprocessImage(truncated, 2, 1, 6, 2, 1, PixelOrder::kRgb),
               std::invalid_argument);
}

TEST(InferenceCore, ThresholdsAndResizesLogitsDeterministically) {
  const std::vector<float> logits = {-10.0F, 0.0F, 10.0F, -10.0F};
  const auto result = postprocessLogits(logits, 2, 2, 4, 2, 0.5);

  EXPECT_EQ(result.mask,
            (std::vector<std::uint8_t>{0, 0, 255, 255, 255, 255, 0, 0}));
  EXPECT_DOUBLE_EQ(result.defect_fraction, 0.5);
  EXPECT_GT(result.peak_probability, 0.999);
}

TEST(InferenceCore, RejectsInvalidThresholdAndShape) {
  EXPECT_THROW(postprocessLogits({0.0F}, 1, 1, 1, 1, 1.1),
               std::invalid_argument);
  EXPECT_THROW(postprocessLogits({0.0F}, 2, 1, 1, 1, 0.5),
               std::invalid_argument);
}

TEST(InferenceCore, ReportsRollingLatencyAndFps) {
  RollingTelemetry telemetry(3);
  telemetry.add({1.0, 8.0, 1.0, 10.0});
  telemetry.add({2.0, 16.0, 2.0, 20.0});
  telemetry.add({3.0, 24.0, 3.0, 30.0});
  telemetry.add({4.0, 32.0, 4.0, 40.0});

  const auto summary = telemetry.summary();
  EXPECT_EQ(summary.samples, 3U);
  EXPECT_DOUBLE_EQ(summary.mean_total_ms, 30.0);
  EXPECT_DOUBLE_EQ(summary.p50_total_ms, 30.0);
  EXPECT_DOUBLE_EQ(summary.p95_total_ms, 40.0);
  EXPECT_NEAR(summary.effective_fps, 1000.0 / 30.0, 1e-9);
}

TEST(InferenceCore, RejectsInvalidTelemetry) {
  EXPECT_THROW(RollingTelemetry(0), std::invalid_argument);
  RollingTelemetry telemetry(2);
  EXPECT_THROW(
      telemetry.add({0.0, std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0}),
      std::invalid_argument);
}

}  // namespace surface_perception_cpp
