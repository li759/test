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
#include <array>
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
      const std::vector<double> &speeds,
      const std::shared_ptr<swift::planning::ReferenceLine> &reference_line =
          nullptr);

  /**
   * @brief Create default action mask (all actions allowed)
   * @param num_actions Number of actions (default: 42)
   * @return Vector of ones (all actions allowed)
   */
  std::vector<float> CreateDefaultActionMask(int num_actions = 42);

  // HOPE风格：直接由lidar生成42维连续mask
  std::vector<float> ExtractActionMaskFromLidar(
      const std::vector<float>& raw_lidar_obs);

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
                     double steering_angle, double signed_speed,
                     const std::shared_ptr<swift::planning::ReferenceLine>
                         &reference_line = nullptr);

  /**
   * @brief Compute continuous action score in [0,1] (HOPE-style soft mask)
   * @details 1.0 = safest, 0.0 = invalid/collision. Uses proximity to obstacles
   *          and optional reference line constraints.
   */
  double ComputeActionScore(const swift::common::VehicleState &vehicle_state,
                            const std::vector<swift::planning::Obstacle> &obstacles,
                            double steering_angle, double signed_speed,
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
                      double steering_angle, double signed_speed);

  /**
   * @brief Predict vehicle position after action
   * @param vehicle_state Current vehicle state
   * @param steering_angle Steering angle
   * @param step_length Step length
   * @return Predicted vehicle state
   */
  swift::common::VehicleState
  PredictVehicleState(const swift::common::VehicleState &vehicle_state,
                      double steering_angle, double signed_speed);

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
                             std::vector<double> &speeds);

  // Default action space configuration
  // HOPE-aligned: 21 steering angles in [+0.75, -0.75] with step 0.75/10, and two speeds {+1.0, -1.0}
  static constexpr int kHopePrecision = 10;            // -> 2*10+1 = 21
  static constexpr double kHopeValidSteerMax = 0.75;   // radians (normalized)
  static constexpr double kHopeValidSpeed = 1.0;       // normalized speed unit
  static constexpr int kDefaultNumSteeringActions = 2 * kHopePrecision + 1; // 21
  static constexpr int kDefaultNumSpeeds = 2;          // +1.0, -1.0
  static constexpr int kDefaultTotalActions = 42;      // 21 * 2 = 42
  static constexpr double kVehicleLength =
      4.8; // Vehicle length for collision checking
  static constexpr double kVehicleWidth =
      2.0; // Vehicle width for collision checking

  // ====== HOPE对齐的Lidar-based掩码计算所需参数与缓存 ======
  static constexpr int kLidarNum = 120;
  static constexpr double kLidarRange = 10.0;
  static constexpr double kWheelBase = 2.8;
  static constexpr int kNumIter = 10;
  static constexpr int kUpSample = 10;
  static constexpr int kNumSteers = 21;
  static constexpr int kNumSpeedsHope = 2;
  static constexpr int kNumActionsHope = 42;
  static constexpr double kSteerMax = 0.75;
  static constexpr double kValidSpeed = 1.0;

  bool lidar_mask_initialized_ = false;
  std::vector<std::array<double,2>> am_action_space_;
  std::vector<std::vector<std::array<double,2>>> am_vehicle_boxes_;
  std::vector<double> am_vehicle_lidar_base_;
  std::vector<std::vector<std::vector<double>>> am_dist_star_;

  void AMInitializeActionSpace();
  void AMInitializeVehicleBoxes();
  void AMPrecomputeVehicleLidarBase();
  void AMPrecomputeDistStar();
  void AMEnsureInitialized();
  static std::vector<double> AMLinearInterpolate1D(const std::vector<double>& x,
                                                   int upsample);
  static std::vector<float> AMMinimumFilter1DClipped(const std::vector<float>& in,
                                                     int kernel, int clip_max);
  static void AMBuildRectangleCorners(double cx, double cy, double yaw,
                                      double length, double width,
                                      std::vector<std::array<double,2>>& out4);
  static double AMRaySegmentIntersectionDistance(double sx, double sy,
                                                 double ex, double ey,
                                                 double x1, double y1,
                                                 double x2, double y2);
};

} // namespace rl_policy
} // namespace open_space
} // namespace planning
} // namespace swift
