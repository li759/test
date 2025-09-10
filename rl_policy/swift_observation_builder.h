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
 * @file swift_observation_builder.h
 * @brief Build 12455-dimensional RL observation from Swift data sources
 */

#pragma once

#include <memory>
#include <vector>

#include "modules/common/math/vec2d.h"
#include "modules/common/vehicle_state/vehicle_state_provider.h"
#include "modules/perception/base/point_cloud.h"
#include "modules/planning/common/obstacle.h"
#include "modules/planning/reference_line/reference_line.h"

#include "modules/planning/open_space/rl_policy/extractors/swift_action_mask_extractor.h"
#include "modules/planning/open_space/rl_policy/extractors/swift_image_extractor.h"
#include "modules/planning/open_space/rl_policy/extractors/swift_lidar_extractor.h"
#include "modules/planning/open_space/rl_policy/extractors/swift_target_extractor.h"

namespace swift {
namespace planning {
namespace open_space {
namespace rl_policy {

/**
 * @struct SwiftObservation
 * @brief Complete RL observation structure (12455 dimensions)
 */
struct SwiftObservation {
  std::vector<float> lidar;       // 120 dimensions - lidar beams
  std::vector<float> target;      // 5 dimensions - target information
  std::vector<float> img;         // 12288 dimensions - occupancy grid (3x64x64)
  std::vector<float> action_mask; // 42 dimensions - action mask
  std::vector<float> flattened;   // 12455 dimensions - concatenated observation

  SwiftObservation() {
    lidar.resize(120, 0.0f);
    target.resize(5, 0.0f);
    img.resize(12288, 0.0f);
    action_mask.resize(42, 1.0f);
    flattened.resize(12455, 0.0f);
  }

  /**
   * @brief Flatten all components into single vector
   */
  void Flatten() {
    flattened.clear();
    flattened.reserve(12455);

    // Concatenate in order: lidar + target + img + action_mask
    flattened.insert(flattened.end(), lidar.begin(), lidar.end());
    flattened.insert(flattened.end(), target.begin(), target.end());
    flattened.insert(flattened.end(), img.begin(), img.end());
    flattened.insert(flattened.end(), action_mask.begin(), action_mask.end());
  }

  /**
   * @brief Get observation dimension
   */
  static constexpr size_t GetObservationDim() { return 12455; }

  /**
   * @brief Get component dimensions
   */
  static constexpr size_t GetLidarDim() { return 120; }
  static constexpr size_t GetTargetDim() { return 5; }
  static constexpr size_t GetImgDim() { return 12288; }
  static constexpr size_t GetActionMaskDim() { return 42; }
};

/**
 * @class SwiftObservationBuilder
 * @brief Build complete RL observation from Swift data sources
 */
class SwiftObservationBuilder {
public:
  SwiftObservationBuilder() = default;
  ~SwiftObservationBuilder() = default;

  /**
   * @brief Build complete observation from Swift data
   * @param point_cloud Swift point cloud data (optional)
   * @param vehicle_state Current vehicle state
   * @param obstacles List of obstacles
   * @param target_position Target position in global coordinates
   * @param target_yaw Target yaw angle (rad)
   * @param reference_line Reference line (optional)
   * @return Complete SwiftObservation structure
   */
  SwiftObservation BuildObservation(
      const swift::perception::base::PointDCloud &point_cloud,
      const swift::common::VehicleState &vehicle_state,
      const std::vector<swift::planning::Obstacle> &obstacles,
      const swift::common::math::Vec2d &target_position, double target_yaw,
      const std::shared_ptr<swift::planning::ReferenceLine> &reference_line =
          nullptr);

  /**
   * @brief Build observation without point cloud (obstacle-based only)
   * @param vehicle_state Current vehicle state
   * @param obstacles List of obstacles
   * @param target_position Target position in global coordinates
   * @param target_yaw Target yaw angle (rad)
   * @param reference_line Reference line (optional)
   * @return Complete SwiftObservation structure
   */
  SwiftObservation BuildObservationFromObstacles(
      const swift::common::VehicleState &vehicle_state,
      const std::vector<swift::planning::Obstacle> &obstacles,
      const swift::common::math::Vec2d &target_position, double target_yaw,
      const std::shared_ptr<swift::planning::ReferenceLine> &reference_line =
          nullptr);

  /**
   * @brief Build observation with custom parameters
   * @param point_cloud Swift point cloud data (optional)
   * @param vehicle_state Current vehicle state
   * @param obstacles List of obstacles
   * @param target_position Target position in global coordinates
   * @param target_yaw Target yaw angle (rad)
   * @param reference_line Reference line (optional)
   * @param lidar_max_range Maximum lidar range (default: 10.0m)
   * @param img_view_range Image view range (default: 20.0m)
   * @return Complete SwiftObservation structure
   */
  SwiftObservation BuildObservationWithParams(
      const swift::perception::base::PointDCloud &point_cloud,
      const swift::common::VehicleState &vehicle_state,
      const std::vector<swift::planning::Obstacle> &obstacles,
      const swift::common::math::Vec2d &target_position, double target_yaw,
      const std::shared_ptr<swift::planning::ReferenceLine> &reference_line,
      double lidar_max_range = 10.0, double img_view_range = 20.0);

  /**
   * @brief Validate observation dimensions
   * @param observation Observation to validate
   * @return True if dimensions are correct
   */
  bool ValidateObservation(const SwiftObservation &observation);

  /**
   * @brief Get observation statistics for debugging
   * @param observation Observation to analyze
   * @return String with statistics
   */
  std::string GetObservationStats(const SwiftObservation &observation);

private:
  // Component extractors
  SwiftLidarExtractor lidar_extractor_;
  SwiftTargetExtractor target_extractor_;
  SwiftImageExtractor image_extractor_;
  SwiftActionMaskExtractor action_mask_extractor_;

  // Default parameters
  static constexpr double kDefaultLidarMaxRange = 10.0;
  static constexpr double kDefaultImgViewRange = 20.0;
  static constexpr int kDefaultLidarBeams = 120;
  static constexpr int kDefaultImgWidth = 64;
  static constexpr int kDefaultImgHeight = 64;
  static constexpr int kDefaultImgChannels = 3;
};

} // namespace rl_policy
} // namespace open_space
} // namespace planning
} // namespace swift
