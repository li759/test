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
 * @file swift_target_extractor.h
 * @brief Extract target information from Swift vehicle state for RL observation
 */

#pragma once

#include <vector>

#include "modules/common/math/vec2d.h"
#include "modules/common/vehicle_state/vehicle_state_provider.h"
#include "modules/planning/open_space/rl_policy/parking_endpoint_calculator.h"

namespace swift {
namespace planning {
namespace open_space {
namespace rl_policy {

/**
 * @struct TargetInfo
 * @brief Target information for RL observation (5-dimensional)
 */
struct TargetInfo {
  float dx;            // Target relative position x (m)
  float dy;            // Target relative position y (m)
  float heading_error; // Heading error (rad)
  float current_speed; // Current vehicle speed (m/s)
  float curvature;     // Path curvature (1/m)

  TargetInfo()
      : dx(0.0f), dy(0.0f), heading_error(0.0f), current_speed(0.0f),
        curvature(0.0f) {}

  TargetInfo(float dx, float dy, float heading_error, float current_speed,
             float curvature)
      : dx(dx), dy(dy), heading_error(heading_error),
        current_speed(current_speed), curvature(curvature) {}
};

/**
 * @class TargetExtractor
 * @brief Extract 5-dimensional target information from Swift vehicle state
 */
class TargetExtractor {
public:
  TargetExtractor() = default;
  ~TargetExtractor() = default;

  /**
   * @brief Extract target information from vehicle state and target position
   * @param vehicle_state Current vehicle state
   * @param target_position Target position in global coordinates
   * @param target_yaw Target yaw angle (rad)
   * @return TargetInfo structure with 5-dimensional target data
   */
  TargetInfo
  ExtractTargetInfo(const swift::common::VehicleState &vehicle_state,
                    const swift::common::math::Vec2d &target_position,
                    double target_yaw);

  /**
   * @brief Extract target information with reference line curvature
   * @param vehicle_state Current vehicle state
   * @param target_position Target position in global coordinates
   * @param target_yaw Target yaw angle (rad)
   * @param reference_curvature Curvature from reference line (1/m)
   * @return TargetInfo structure with 5-dimensional target data
   */
  TargetInfo ExtractTargetInfoWithCurvature(
      const swift::common::VehicleState &vehicle_state,
      const swift::common::math::Vec2d &target_position, double target_yaw,
      double reference_curvature = 0.0);

  /**
   * @brief Extract target information from parking slot
   * @param vehicle_state Current vehicle state
   * @param parking_slot Parking slot information
   * @param obstacles Obstacle information for optimization
   * @param is_wheel_stop_valid Whether wheel stop is valid
   * @return TargetInfo structure with 5-dimensional target data
   */
  TargetInfo ExtractTargetInfoFromParkingSlot(
      const swift::common::VehicleState &vehicle_state,
      const ParkingSlot &parking_slot,
      const std::vector<ObstacleInfo> &obstacles = {},
      bool is_wheel_stop_valid = false);

  /**
   * @brief Convert TargetInfo to vector format for RL observation
   * @param target_info Target information structure
   * @return Vector of 5 float values [dx, dy, heading_error, speed, curvature]
   */
  std::vector<float> ToVector(const TargetInfo &target_info);

  /**
   * @brief Calculate heading error between current and target yaw
   * @param current_yaw Current vehicle yaw (rad)
   * @param target_yaw Target yaw (rad)
   * @return Heading error in [-PI, PI] (rad)
   */
  static double CalculateHeadingError(double current_yaw, double target_yaw);

  /**
   * @brief Calculate relative position in vehicle coordinate system
   * @param vehicle_state Current vehicle state
   * @param target_position Target position in global coordinates
   * @return Relative position [dx, dy] in vehicle frame
   */
  static std::pair<double, double>
  CalculateRelativePosition(const swift::common::VehicleState &vehicle_state,
                            const swift::common::math::Vec2d &target_position);

private:
  /**
   * @brief Normalize angle to [-PI, PI] range
   * @param angle Input angle (rad)
   * @return Normalized angle (rad)
   */
  static double NormalizeAngle(double angle);

  // Parking endpoint calculator
  ParkingEndpointCalculator parking_calculator_;
};

} // namespace rl_policy
} // namespace open_space
} // namespace planning
} // namespace swift
