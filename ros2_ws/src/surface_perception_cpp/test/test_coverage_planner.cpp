#include "surface_perception_cpp/coverage_planner.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

TEST(CoveragePlanner, GeneratesSafeBoustrophedonSegments) {
  constexpr int width = 80;
  constexpr int height = 60;
  std::vector<std::uint8_t> sanding(width * height, 0);
  std::vector<std::uint8_t> protected_mask(width * height, 0);
  std::vector<std::uint8_t> defects(width * height, 0);
  for (int y = 8; y < height - 8; ++y) {
    for (int x = 8; x < width - 8; ++x) {
      sanding[static_cast<std::size_t>(y * width + x)] = 1;
    }
  }
  for (int y = 20; y < 32; ++y) {
    for (int x = 35; x < 45; ++x) {
      protected_mask[static_cast<std::size_t>(y * width + x)] = 1;
      sanding[static_cast<std::size_t>(y * width + x)] = 0;
    }
  }
  defects[static_cast<std::size_t>(15 * width + 20)] = 1;

  surface_perception_cpp::CoveragePlanner planner(width, height, 3, 5, 6);
  const auto plan = planner.plan(sanding, protected_mask, defects);
  ASSERT_FALSE(plan.segments.empty());
  EXPECT_GT(plan.process_path_pixels, 0.0);
  EXPECT_TRUE(plan.segments.front().start.x < plan.segments.front().end.x);
  bool has_priority = false;
  for (const auto& segment : plan.segments) {
    has_priority = has_priority || segment.priority;
  }
  EXPECT_TRUE(has_priority);
}
