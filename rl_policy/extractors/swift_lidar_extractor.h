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
 * @file swift_lidar_extractor.h
 * @brief Extract lidar data from Swift perception system for RL observation
 */

#pragma once

#include <memory>
#include <vector>

#include "modules/common/vehicle_state/vehicle_state_provider.h"
#include "modules/perception/base/point_cloud.h"
#include "modules/planning/common/obstacle.h"

namespace swift {
namespace planning {
namespace open_space {
namespace rl_policy {

/**
 * @class SwiftLidarExtractor
 * @brief Extract 120-dimensional lidar data from Swift point cloud and
 * obstacles
 */
class SwiftLidarExtractor {
public:
  SwiftLidarExtractor() = default;
  ~SwiftLidarExtractor() = default;

  /**
   * @brief Extract lidar beams from Swift point cloud and obstacles
   * @param point_cloud Swift point cloud data
   * @param vehicle_state Current vehicle state
   * @param obstacles List of obstacles for ray casting
   * @param max_range Maximum lidar range (default: 10.0m)
   * @param num_beams Number of lidar beams (default: 120)
   * @param fov Field of view in radians (default: 2*PI)
   * @return Vector of lidar distances (120 elements)
   */
  std::vector<float>
  ExtractLidarBeams(const swift::perception::base::PointDCloud &point_cloud,
                    const swift::common::VehicleState &vehicle_state,
                    const std::vector<swift::planning::Obstacle> &obstacles,
                    double max_range = 10.0, int num_beams = 120,
                    double fov = 2.0 * M_PI);

  /**
   * @brief Extract lidar beams using only obstacles (for simulation)
   * @param vehicle_state Current vehicle state
   * @param obstacles List of obstacles for ray casting
   * @param max_range Maximum lidar range (default: 10.0m)
   * @param num_beams Number of lidar beams (default: 120)
   * @param fov Field of view in radians (default: 2*PI)
   * @return Vector of lidar distances (120 elements)
   */
  std::vector<float> ExtractLidarBeamsFromObstacles(
      const swift::common::VehicleState &vehicle_state,
      const std::vector<swift::planning::Obstacle> &obstacles,
      double max_range = 10.0, int num_beams = 120, double fov = 2.0 * M_PI);

private:
  /**
   * @brief Cast ray to obstacles and return distance
   * @param start_x Ray start x coordinate
   * @param start_y Ray start y coordinate
   * @param yaw Vehicle yaw angle
   * @param angle Ray angle relative to vehicle heading
   * @param obstacles List of obstacles
   * @param max_range Maximum ray range
   * @return Distance to closest obstacle, or max_range if no obstacle
   */
  double
  RaycastToObstacles(double start_x, double start_y, double yaw, double angle,
                     const std::vector<swift::planning::Obstacle> &obstacles,
                     double max_range);

  /**
   * @brief Check if ray intersects with obstacle bounding box
   * @param ray_start_x Ray start x coordinate
   * @param ray_start_y Ray start y coordinate
   * @param ray_end_x Ray end x coordinate
   * @param ray_end_y Ray end y coordinate
   * @param obstacle Obstacle to check intersection with
   * @return Distance to intersection, or -1 if no intersection
   */
  double RayObstacleIntersection(double ray_start_x, double ray_start_y,
                                 double ray_end_x, double ray_end_y,
                                 const swift::planning::Obstacle &obstacle);

  /**
   * @brief Extract lidar data from point cloud (if available)
   * @param point_cloud Swift point cloud data
   * @param vehicle_state Current vehicle state
   * @param max_range Maximum lidar range
   * @param num_beams Number of lidar beams
   * @param fov Field of view
   * @return Vector of lidar distances
   */
  std::vector<float>
  ExtractFromPointCloud(const swift::perception::base::PointDCloud &point_cloud,
                        const swift::common::VehicleState &vehicle_state,
                        double max_range, int num_beams, double fov);

  // Configuration parameters
  static constexpr int kDefaultNumBeams = 120;
  static constexpr double kDefaultMaxRange = 10.0;
  static constexpr double kDefaultFOV = 2.0 * M_PI;
  static constexpr double kRayStepSize = 0.1; // Ray casting step size
};

} // namespace rl_policy
} // namespace open_space
} // namespace planning
} // namespace swift
