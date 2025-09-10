/******************************************************************************
 * Copyright 2024 Desay SV Automotive Co., Ltd.
 * Copyright 2018 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#include "modules/planning/open_space/rl_policy/image_extractor.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace swift {
namespace planning {
namespace open_space {
namespace rl_policy {

std::vector<float> ImageExtractor::ExtractOccupancyGrid(
    const swift::common::VehicleState &vehicle_state,
    const std::vector<swift::planning::Obstacle> &obstacles, int width,
    int height, int channels, double view_range) {

  double resolution = view_range / std::max(width, height);
  return ExtractOccupancyGridWithResolution(vehicle_state, obstacles, width,
                                            height, channels, view_range,
                                            resolution);
}

std::vector<float> ImageExtractor::ExtractOccupancyGridWithResolution(
    const swift::common::VehicleState &vehicle_state,
    const std::vector<swift::planning::Obstacle> &obstacles, int width,
    int height, int channels, double view_range, double resolution) {

  // Initialize grid with zeros
  std::vector<float> grid(width * height * channels, 0.0f);

  double vehicle_x = vehicle_state.x();
  double vehicle_y = vehicle_state.y();
  double vehicle_yaw = vehicle_state.heading();

  // Rasterize each obstacle
  for (const auto &obstacle : obstacles) {
    RasterizeObstacle(obstacle, grid, width, height, resolution, vehicle_x,
                      vehicle_y, vehicle_yaw);
  }

  return grid;
}

std::vector<float> ImageExtractor::CreateEmptyGrid(int width, int height,
                                                   int channels) {
  return std::vector<float>(width * height * channels, 0.0f);
}

void ImageExtractor::RasterizeObstacle(
    const swift::planning::Obstacle &obstacle, std::vector<float> &grid,
    int width, int height, double resolution, double vehicle_x,
    double vehicle_y, double vehicle_yaw) {

  const auto &bounding_box = obstacle.PerceptionBoundingBox();
  RasterizeBoundingBox(bounding_box, grid, width, height, resolution, vehicle_x,
                       vehicle_y, vehicle_yaw);
}

void ImageExtractor::RasterizeBoundingBox(
    const swift::common::math::Box2d &bounding_box, std::vector<float> &grid,
    int width, int height, double resolution, double vehicle_x,
    double vehicle_y, double vehicle_yaw) {

  // Get obstacle corners
  auto corners = bounding_box.GetAllCorners();

  // Find bounding rectangle in grid coordinates
  int min_grid_x = width, max_grid_x = -1;
  int min_grid_y = height, max_grid_y = -1;

  for (const auto &corner : corners) {
    auto grid_coord = WorldToGrid(corner.x(), corner.y(), vehicle_x, vehicle_y,
                                  vehicle_yaw, resolution, width, height);

    if (IsValidGridCoordinate(grid_coord.first, grid_coord.second, width,
                              height)) {
      min_grid_x = std::min(min_grid_x, grid_coord.first);
      max_grid_x = std::max(max_grid_x, grid_coord.first);
      min_grid_y = std::min(min_grid_y, grid_coord.second);
      max_grid_y = std::max(max_grid_y, grid_coord.second);
    }
  }

  // Clamp to grid bounds
  min_grid_x = std::max(0, min_grid_x);
  max_grid_x = std::min(width - 1, max_grid_x);
  min_grid_y = std::max(0, min_grid_y);
  max_grid_y = std::min(height - 1, max_grid_y);

  // Fill grid cells within bounding rectangle
  for (int y = min_grid_y; y <= max_grid_y; ++y) {
    for (int x = min_grid_x; x <= max_grid_x; ++x) {
      // Convert grid coordinates back to world coordinates
      double world_x = vehicle_x + (x - width / 2) * resolution;
      double world_y = vehicle_y + (y - height / 2) * resolution;

      // Rotate to world frame
      double cos_yaw = std::cos(vehicle_yaw);
      double sin_yaw = std::sin(vehicle_yaw);
      double rotated_x = world_x * cos_yaw - world_y * sin_yaw;
      double rotated_y = world_x * sin_yaw + world_y * cos_yaw;

      // Check if point is inside obstacle bounding box
      if (bounding_box.IsPointIn(
              swift::common::math::Vec2d(rotated_x, rotated_y))) {
        // Set occupancy value (1.0 for occupied)
        for (int c = 0; c < 3; ++c) { // Assuming 3 channels
          int index = (y * width + x) * 3 + c;
          if (index < static_cast<int>(grid.size())) {
            grid[index] = 1.0f;
          }
        }
      }
    }
  }
}

std::pair<int, int> ImageExtractor::WorldToGrid(
    double world_x, double world_y, double vehicle_x, double vehicle_y,
    double vehicle_yaw, double resolution, int grid_width, int grid_height) {

  // Transform to vehicle coordinate system
  double dx = world_x - vehicle_x;
  double dy = world_y - vehicle_y;

  // Rotate to vehicle frame
  double cos_yaw = std::cos(-vehicle_yaw);
  double sin_yaw = std::sin(-vehicle_yaw);
  double local_x = dx * cos_yaw - dy * sin_yaw;
  double local_y = dx * sin_yaw + dy * cos_yaw;

  // Convert to grid coordinates
  int grid_x =
      static_cast<int>(std::round(local_x / resolution + grid_width / 2));
  int grid_y =
      static_cast<int>(std::round(local_y / resolution + grid_height / 2));

  return std::make_pair(grid_x, grid_y);
}

bool ImageExtractor::IsValidGridCoordinate(int grid_x, int grid_y, int width,
                                           int height) {
  return grid_x >= 0 && grid_x < width && grid_y >= 0 && grid_y < height;
}

} // namespace rl_policy
} // namespace open_space
} // namespace planning
} // namespace swift
