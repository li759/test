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
 * @file swift_to_hope_adapter.h
 * @brief Adapter to convert Swift data to HOPE+ format for RL inference
 */

#pragma once

#include <memory>
#include <vector>

#include "modules/common/vehicle_state/vehicle_state_provider.h"
#include "modules/planning/common/obstacle.h"
#include "modules/planning/open_space/rl_policy/swift_observation_builder.h"

namespace swift {
namespace planning {
namespace open_space {
namespace rl_policy {

/**
 * @struct HopeVehicleState
 * @brief Simplified vehicle state for HOPE+ compatibility
 */
struct HopeVehicleState {
  double x;
  double y;
  double yaw;
  double speed;
  double steering_angle;

  HopeVehicleState()
      : x(0.0), y(0.0), yaw(0.0), speed(0.0), steering_angle(0.0) {}
  HopeVehicleState(double x, double y, double yaw, double speed,
                   double steering_angle)
      : x(x), y(y), yaw(yaw), speed(speed), steering_angle(steering_angle) {}
};

/**
 * @struct HopeObstacle
 * @brief Simplified obstacle for HOPE+ compatibility
 */
struct HopeObstacle {
  double x;
  double y;
  double length;
  double width;
  double yaw;
  bool is_static;

  HopeObstacle()
      : x(0.0), y(0.0), length(0.0), width(0.0), yaw(0.0), is_static(true) {}
  HopeObstacle(double x, double y, double length, double width, double yaw,
               bool is_static)
      : x(x), y(y), length(length), width(width), yaw(yaw),
        is_static(is_static) {}
};

/**
 * @class SwiftToHopeAdapter
 * @brief Convert Swift data to HOPE+ format for RL inference
 */
class SwiftToHopeAdapter {
public:
  SwiftToHopeAdapter() = default;
  ~SwiftToHopeAdapter() = default;

  /**
   * @brief Convert Swift observation to HOPE+ format
   * @param swift_obs Swift observation structure
   * @return Vector of 12455 float values for HOPE+ inference
   */
  std::vector<float>
  ConvertToHopeObservation(const SwiftObservation &swift_obs);

  /**
   * @brief Convert Swift vehicle state to HOPE+ format
   * @param swift_state Swift vehicle state
   * @return HopeVehicleState structure
   */
  HopeVehicleState
  ConvertToHopeVehicleState(const swift::common::VehicleState &swift_state);

  /**
   * @brief Convert Swift obstacles to HOPE+ format
   * @param swift_obstacles Swift obstacles
   * @return Vector of HopeObstacle structures
   */
  std::vector<HopeObstacle> ConvertToHopeObstacles(
      const std::vector<swift::planning::Obstacle> &swift_obstacles);

  /**
   * @brief Convert HOPE+ action to Swift format
   * @param hope_action HOPE+ action vector [steering, step_length]
   * @return Swift action structure
   */
  struct SwiftAction {
    double steering_angle;
    double step_length;

    SwiftAction() : steering_angle(0.0), step_length(0.0) {}
    SwiftAction(double steering, double step)
        : steering_angle(steering), step_length(step) {}
  };

  SwiftAction ConvertToSwiftAction(const std::vector<float> &hope_action);

  /**
   * @brief Validate HOPE+ observation format
   * @param hope_obs HOPE+ observation vector
   * @return True if format is valid
   */
  bool ValidateHopeObservation(const std::vector<float> &hope_obs);

  /**
   * @brief Get HOPE+ observation statistics
   * @param hope_obs HOPE+ observation vector
   * @return String with statistics
   */
  std::string GetHopeObservationStats(const std::vector<float> &hope_obs);

  /**
   * @brief Create empty HOPE+ observation (all zeros)
   * @return Vector of 12455 zeros
   */
  std::vector<float> CreateEmptyHopeObservation();

private:
  /**
   * @brief Normalize angle to [-PI, PI] range
   * @param angle Input angle (rad)
   * @return Normalized angle (rad)
   */
  static double NormalizeAngle(double angle);

  // HOPE+ observation dimensions
  static constexpr size_t kHopeObservationDim = 12455;
  static constexpr size_t kHopeLidarDim = 120;
  static constexpr size_t kHopeTargetDim = 5;
  static constexpr size_t kHopeImgDim = 12288;
  static constexpr size_t kHopeActionMaskDim = 42;
};

} // namespace rl_policy
} // namespace open_space
} // namespace planning
} // namespace swift
