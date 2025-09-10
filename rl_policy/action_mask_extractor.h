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
 * @file swift_action_mask_extractor.h
 * @brief Extract action mask from Swift constraints for RL observation
 */

#pragma once

#include <memory>
#include <vector>

#include "modules/common/vehicle_state/vehicle_state_provider.h"
#include "modules/planning/common/obstacle.h"
#include "modules/planning/reference_line/reference_line.h"

namespace swift {
namespace planning {
namespace open_space {
namespace rl_policy {

/**
 * @class ActionMaskExtractor
 * @brief Extract 42-dimensional action mask from Swift constraints
 */
class ActionMaskExtractor {
public:
  ActionMaskExtractor() = default;
  ~ActionMaskExtractor() = default;

  /**
   * @brief Extract action mask from vehicle state and obstacles
   * @param vehicle_state Current vehicle state
   * @param obstacles List of obstacles for constraint checking
   * @param reference_line Reference line for path constraints (optional)
   * @return Vector of action mask values (42 elements)
   */
  std::vector<float> ExtractActionMask(
      const swift::common::VehicleState &vehicle_state,
      const std::vector<swift::planning::Obstacle> &obstacles,
      const std::shared_ptr<swift::planning::ReferenceLine> &reference_line =
          nullptr);

  /**
   * @brief Extract action mask with custom action space
   * @param vehicle_state Current vehicle state
   * @param obstacles List of obstacles for constraint checking
   * @param steering_actions List of steering angles to check
   * @param step_lengths List of step lengths to check
   * @param reference_line Reference line for path constraints (optional)
   * @return Vector of action mask values
   */
  std::vector<float> ExtractActionMaskWithCustomActions(
      const swift::common::VehicleState &vehicle_state,
      const std::vector<swift::planning::Obstacle> &obstacles,
      const std::vector<double> &steering_actions,
      const std::vector<double> &step_lengths,
      const std::shared_ptr<swift::planning::ReferenceLine> &reference_line =
          nullptr);

  /**
   * @brief Create default action mask (all actions allowed)
   * @param num_actions Number of actions (default: 42)
   * @return Vector of ones (all actions allowed)
   */
  std::vector<float> CreateDefaultActionMask(int num_actions = 42);

  /**
   * @brief Check if specific action is valid
   * @param vehicle_state Current vehicle state
   * @param obstacles List of obstacles
   * @param steering_angle Steering angle to check
   * @param step_length Step length to check
   * @param reference_line Reference line (optional)
   * @return True if action is valid
   */
  bool IsActionValid(const swift::common::VehicleState &vehicle_state,
                     const std::vector<swift::planning::Obstacle> &obstacles,
                     double steering_angle, double step_length,
                     const std::shared_ptr<swift::planning::ReferenceLine>
                         &reference_line = nullptr);

private:
  /**
   * @brief Check collision with obstacles for given action
   * @param vehicle_state Current vehicle state
   * @param obstacles List of obstacles
   * @param steering_angle Steering angle
   * @param step_length Step length
   * @return True if no collision
   */
  bool CheckCollision(const swift::common::VehicleState &vehicle_state,
                      const std::vector<swift::planning::Obstacle> &obstacles,
                      double steering_angle, double step_length);

  /**
   * @brief Predict vehicle position after action
   * @param vehicle_state Current vehicle state
   * @param steering_angle Steering angle
   * @param step_length Step length
   * @return Predicted vehicle state
   */
  swift::common::VehicleState
  PredictVehicleState(const swift::common::VehicleState &vehicle_state,
                      double steering_angle, double step_length);

  /**
   * @brief Check if predicted state is within reference line bounds
   * @param predicted_state Predicted vehicle state
   * @param reference_line Reference line
   * @return True if within bounds
   */
  bool CheckReferenceLineBounds(
      const swift::common::VehicleState &predicted_state,
      const std::shared_ptr<swift::planning::ReferenceLine> &reference_line);

  /**
   * @brief Get default action space (steering angles and step lengths)
   * @param steering_actions Output vector for steering angles
   * @param step_lengths Output vector for step lengths
   */
  void GetDefaultActionSpace(std::vector<double> &steering_actions,
                             std::vector<double> &step_lengths);

  // Default action space configuration
  static constexpr int kDefaultNumSteeringActions = 7; // -1.0 to 1.0 in steps
  static constexpr int kDefaultNumStepLengths = 6;     // 0.1 to 1.0 in steps
  static constexpr int kDefaultTotalActions = 42;      // 7 * 6 = 42
  static constexpr double kDefaultMinSteering = -1.0;
  static constexpr double kDefaultMaxSteering = 1.0;
  static constexpr double kDefaultMinStepLength = 0.1;
  static constexpr double kDefaultMaxStepLength = 1.0;
  static constexpr double kVehicleLength =
      4.8; // Vehicle length for collision checking
  static constexpr double kVehicleWidth =
      2.0; // Vehicle width for collision checking
};

} // namespace rl_policy
} // namespace open_space
} // namespace planning
} // namespace swift
