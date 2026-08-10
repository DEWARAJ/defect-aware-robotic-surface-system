#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace surface_perception_cpp {

struct Point {
  int x{};
  int y{};
};

struct Segment {
  Point start;
  Point end;
  double feed_scale{1.0};
  bool priority{false};
};

struct Plan {
  std::vector<Segment> segments;
  std::size_t protected_contact_pixels{0};
  double process_path_pixels{0.0};
};

class CoveragePlanner {
 public:
  CoveragePlanner(int width, int height, int tool_radius, int lane_spacing,
                  int minimum_segment_length);

  [[nodiscard]] Plan plan(const std::vector<std::uint8_t>& sanding_mask,
                          const std::vector<std::uint8_t>& protected_mask,
                          const std::vector<std::uint8_t>& defect_mask) const;

 private:
  [[nodiscard]] std::vector<std::uint8_t> safeCenters(
      const std::vector<std::uint8_t>& sanding_mask,
      const std::vector<std::uint8_t>& protected_mask) const;
  [[nodiscard]] std::size_t index(int x, int y) const;

  int width_;
  int height_;
  int tool_radius_;
  int lane_spacing_;
  int minimum_segment_length_;
};

}  // namespace surface_perception_cpp
