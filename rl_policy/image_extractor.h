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

/**
 * @file swift_image_extractor.h
 * @brief Extract occupancy grid image from Swift obstacles for RL observation
 */

#pragma once

#include <memory>
#include <vector>

#include "modules/common/math/box2d.h"
#include "modules/common/vehicle_state/vehicle_state_provider.h"
#include "modules/planning/common/obstacle.h"

namespace swift {
namespace planning {
namespace open_space {
namespace rl_policy {

/**
 * @class ImageExtractor
 * @brief Extract 3x64x64 occupancy grid image from Swift obstacles
 */
class ImageExtractor {
public:
  ImageExtractor() = default;
  ~ImageExtractor() = default;

  /**
   * @brief Extract occupancy grid image from obstacles
   * @param vehicle_state Current vehicle state
   * @param obstacles List of obstacles to rasterize
   * @param width Image width (default: 64)
   * @param height Image height (default: 64)
   * @param channels Number of channels (default: 3)
   * @param view_range View range in meters (default: 20.0)
   * @return Vector of image data (width * height * channels)
   */
  std::vector<float>
  ExtractOccupancyGrid(const swift::common::VehicleState &vehicle_state,
                       const std::vector<swift::planning::Obstacle> &obstacles,
                       int width = 64, int height = 64, int channels = 3,
                       double view_range = 20.0);

  /**
   * @brief Extract occupancy grid with custom resolution
   * @param vehicle_state Current vehicle state
   * @param obstacles List of obstacles to rasterize
   * @param width Image width
   * @param height Image height
   * @param channels Number of channels
   * @param view_range View range in meters
   * @param resolution Grid resolution in meters per pixel
   * @return Vector of image data
   */
  std::vector<float> ExtractOccupancyGridWithResolution(
      const swift::common::VehicleState &vehicle_state,
      const std::vector<swift::planning::Obstacle> &obstacles, int width,
      int height, int channels, double view_range, double resolution);

  /**
   * @brief Create empty occupancy grid (all zeros)
   * @param width Image width
   * @param height Image height
   * @param channels Number of channels
   * @return Vector of zeros
   */
  std::vector<float> CreateEmptyGrid(int width, int height, int channels);

private:
  /**
   * @brief Rasterize single obstacle to grid
   * @param obstacle Obstacle to rasterize
   * @param grid Output grid data
   * @param width Grid width
   * @param height Grid height
   * @param resolution Grid resolution (m/pixel)
   * @param vehicle_x Vehicle x position
   * @param vehicle_y Vehicle y position
   * @param vehicle_yaw Vehicle yaw angle
   */
  void RasterizeObstacle(const swift::planning::Obstacle &obstacle,
                         std::vector<float> &grid, int width, int height,
                         double resolution, double vehicle_x, double vehicle_y,
                         double vehicle_yaw);

  /**
   * @brief Rasterize obstacle bounding box to grid
   * @param bounding_box Obstacle bounding box
   * @param grid Output grid data
   * @param width Grid width
   * @param height Grid height
   * @param resolution Grid resolution (m/pixel)
   * @param vehicle_x Vehicle x position
   * @param vehicle_y Vehicle y position
   * @param vehicle_yaw Vehicle yaw angle
   */
  void RasterizeBoundingBox(const swift::common::math::Box2d &bounding_box,
                            std::vector<float> &grid, int width, int height,
                            double resolution, double vehicle_x,
                            double vehicle_y, double vehicle_yaw);

  /**
   * @brief Convert world coordinates to grid coordinates
   * @param world_x World x coordinate
   * @param world_y World y coordinate
   * @param vehicle_x Vehicle x position
   * @param vehicle_y Vehicle y position
   * @param vehicle_yaw Vehicle yaw angle
   * @param resolution Grid resolution (m/pixel)
   * @param grid_width Grid width
   * @param grid_height Grid height
   * @return Grid coordinates [grid_x, grid_y]
   */
  std::pair<int, int> WorldToGrid(double world_x, double world_y,
                                  double vehicle_x, double vehicle_y,
                                  double vehicle_yaw, double resolution,
                                  int grid_width, int grid_height);

  /**
   * @brief Check if grid coordinates are valid
   * @param grid_x Grid x coordinate
   * @param grid_y Grid y coordinate
   * @param width Grid width
   * @param height Grid height
   * @return True if coordinates are valid
   */
  bool IsValidGridCoordinate(int grid_x, int grid_y, int width, int height);

  // Default configuration
  static constexpr int kDefaultWidth = 64;
  static constexpr int kDefaultHeight = 64;
  static constexpr int kDefaultChannels = 3;
  static constexpr double kDefaultViewRange = 20.0;
  static constexpr double kDefaultResolution = 0.3125; // 20m / 64 pixels
};

} // namespace rl_policy
} // namespace open_space
} // namespace planning
} // namespace swift
