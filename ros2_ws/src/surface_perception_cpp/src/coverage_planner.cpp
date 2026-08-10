#include "surface_perception_cpp/coverage_planner.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace surface_perception_cpp {

CoveragePlanner::CoveragePlanner(const int width, const int height,
                                 const int tool_radius, const int lane_spacing,
                                 const int minimum_segment_length)
    : width_(width),
      height_(height),
      tool_radius_(tool_radius),
      lane_spacing_(lane_spacing),
      minimum_segment_length_(minimum_segment_length) {
  if (width_ <= 0 || height_ <= 0 || tool_radius_ < 1 || lane_spacing_ < 1 ||
      lane_spacing_ > tool_radius_ * 2 || minimum_segment_length_ < 2) {
    throw std::invalid_argument("invalid coverage planner geometry");
  }
}

std::size_t CoveragePlanner::index(const int x, const int y) const {
  return static_cast<std::size_t>(y * width_ + x);
}

std::vector<std::uint8_t> CoveragePlanner::safeCenters(
    const std::vector<std::uint8_t>& sanding_mask,
    const std::vector<std::uint8_t>& protected_mask) const {
  const auto expected = static_cast<std::size_t>(width_ * height_);
  if (sanding_mask.size() != expected || protected_mask.size() != expected) {
    throw std::invalid_argument("mask dimensions do not match planner dimensions");
  }
  std::vector<std::uint8_t> safe(expected, 0);
  for (int y = tool_radius_; y < height_ - tool_radius_; ++y) {
    for (int x = tool_radius_; x < width_ - tool_radius_; ++x) {
      bool valid = true;
      for (int dy = -tool_radius_; dy <= tool_radius_ && valid; ++dy) {
        for (int dx = -tool_radius_; dx <= tool_radius_; ++dx) {
          if (dx * dx + dy * dy > tool_radius_ * tool_radius_) {
            continue;
          }
          const auto neighbor = index(x + dx, y + dy);
          if (sanding_mask[neighbor] == 0 || protected_mask[neighbor] != 0) {
            valid = false;
            break;
          }
        }
      }
      safe[index(x, y)] = valid ? 1U : 0U;
    }
  }
  return safe;
}

Plan CoveragePlanner::plan(const std::vector<std::uint8_t>& sanding_mask,
                           const std::vector<std::uint8_t>& protected_mask,
                           const std::vector<std::uint8_t>& defect_mask) const {
  const auto expected = static_cast<std::size_t>(width_ * height_);
  if (defect_mask.size() != expected) {
    throw std::invalid_argument("defect mask dimensions do not match planner dimensions");
  }
  const auto safe = safeCenters(sanding_mask, protected_mask);
  Plan result;
  bool forward = true;
  for (int y = tool_radius_; y < height_ - tool_radius_; y += lane_spacing_) {
    std::vector<std::pair<int, int>> runs;
    int x = 0;
    while (x < width_) {
      while (x < width_ && safe[index(x, y)] == 0) {
        ++x;
      }
      const int start = x;
      while (x < width_ && safe[index(x, y)] != 0) {
        ++x;
      }
      const int end = x - 1;
      if (end - start + 1 >= minimum_segment_length_) {
        runs.emplace_back(start, end);
      }
    }
    if (runs.empty()) {
      continue;
    }
    if (!forward) {
      std::reverse(runs.begin(), runs.end());
    }
    for (const auto& [start_x, end_x] : runs) {
      bool priority = false;
      for (int check_x = start_x; check_x <= end_x; ++check_x) {
        for (int dy = -tool_radius_; dy <= tool_radius_; ++dy) {
          const int check_y = y + dy;
          if (check_y >= 0 && check_y < height_ && defect_mask[index(check_x, check_y)] != 0) {
            priority = true;
            break;
          }
        }
        if (priority) {
          break;
        }
      }
      const Point start = forward ? Point{start_x, y} : Point{end_x, y};
      const Point end = forward ? Point{end_x, y} : Point{start_x, y};
      result.segments.push_back(Segment{start, end, priority ? 0.55 : 1.0, priority});
      result.process_path_pixels += std::hypot(
          static_cast<double>(end.x - start.x), static_cast<double>(end.y - start.y));
    }
    forward = !forward;
  }
  std::vector<std::uint8_t> swept(expected, 0);
  for (const auto& segment : result.segments) {
    const int first_x = std::min(segment.start.x, segment.end.x);
    const int last_x = std::max(segment.start.x, segment.end.x);
    for (int center_x = first_x; center_x <= last_x; ++center_x) {
      for (int dy = -tool_radius_; dy <= tool_radius_; ++dy) {
        for (int dx = -tool_radius_; dx <= tool_radius_; ++dx) {
          if (dx * dx + dy * dy > tool_radius_ * tool_radius_) {
            continue;
          }
          const int x = center_x + dx;
          const int y = segment.start.y + dy;
          if (x >= 0 && x < width_ && y >= 0 && y < height_) {
            swept[index(x, y)] = 1;
          }
        }
      }
    }
  }
  for (std::size_t pixel = 0; pixel < expected; ++pixel) {
    if (swept[pixel] != 0 && protected_mask[pixel] != 0) {
      ++result.protected_contact_pixels;
    }
  }
  return result;
}

}  // namespace surface_perception_cpp
