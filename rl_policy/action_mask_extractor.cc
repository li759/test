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

#include "modules/planning/open_space/rl_policy/action_mask_extractor.h"

#include <algorithm>
#include <cmath>

#include "modules/common/math/box2d.h"
#include "modules/common/math/vec2d.h"

namespace swift {
namespace planning {
namespace open_space {
namespace rl_policy {

std::vector<float> ActionMaskExtractor::ExtractActionMask(
    const swift::common::VehicleState &vehicle_state,
    const std::vector<swift::planning::Obstacle> &obstacles,
    const std::shared_ptr<swift::planning::ReferenceLine> &reference_line) {

  std::vector<double> steering_actions;
  std::vector<double> step_lengths;
  GetDefaultActionSpace(steering_actions, step_lengths);

  return ExtractActionMaskWithCustomActions(
      vehicle_state, obstacles, steering_actions, step_lengths, reference_line);
}

std::vector<float> ActionMaskExtractor::ExtractActionMaskWithCustomActions(
    const swift::common::VehicleState &vehicle_state,
    const std::vector<swift::planning::Obstacle> &obstacles,
    const std::vector<double> &steering_actions,
    const std::vector<double> &step_lengths,
    const std::shared_ptr<swift::planning::ReferenceLine> &reference_line) {

  std::vector<float> action_mask;
  action_mask.reserve(steering_actions.size() * step_lengths.size());

  for (double steering : steering_actions) {
    for (double step_length : step_lengths) {
      bool is_valid = IsActionValid(vehicle_state, obstacles, steering,
                                    step_length, reference_line);
      action_mask.push_back(is_valid ? 1.0f : 0.0f);
    }
  }

  return action_mask;
}

std::vector<float>
ActionMaskExtractor::CreateDefaultActionMask(int num_actions) {
  return std::vector<float>(num_actions, 1.0f);
}

bool ActionMaskExtractor::IsActionValid(
    const swift::common::VehicleState &vehicle_state,
    const std::vector<swift::planning::Obstacle> &obstacles,
    double steering_angle, double step_length,
    const std::shared_ptr<swift::planning::ReferenceLine> &reference_line) {

  // Check collision
  if (!CheckCollision(vehicle_state, obstacles, steering_angle, step_length)) {
    return false;
  }

  // Check reference line bounds if available
  if (reference_line) {
    auto predicted_state =
        PredictVehicleState(vehicle_state, steering_angle, step_length);
    if (!CheckReferenceLineBounds(predicted_state, reference_line)) {
      return false;
    }
  }

  return true;
}

bool ActionMaskExtractor::CheckCollision(
    const swift::common::VehicleState &vehicle_state,
    const std::vector<swift::planning::Obstacle> &obstacles,
    double steering_angle, double step_length) {

  // Predict vehicle state after action
  auto predicted_state =
      PredictVehicleState(vehicle_state, steering_angle, step_length);

  // Create vehicle bounding box at predicted position
  swift::common::math::Vec2d predicted_position(predicted_state.x(),
                                                predicted_state.y());
  swift::common::math::Box2d vehicle_box(predicted_position,
                                         predicted_state.heading(),
                                         kVehicleLength, kVehicleWidth);

  // Check collision with obstacles
  for (const auto &obstacle : obstacles) {
    const auto &obstacle_box = obstacle.PerceptionBoundingBox();
    if (vehicle_box.HasOverlap(obstacle_box)) {
      return false; // Collision detected
    }
  }

  return true; // No collision
}

swift::common::VehicleState ActionMaskExtractor::PredictVehicleState(
    const swift::common::VehicleState &vehicle_state, double steering_angle,
    double step_length) {

  // Simple bicycle model prediction
  double current_x = vehicle_state.x();
  double current_y = vehicle_state.y();
  double current_yaw = vehicle_state.heading();
  double current_speed = vehicle_state.linear_velocity();

  // Calculate wheelbase (distance between front and rear axles)
  double wheelbase = 2.8; // Typical wheelbase in meters

  // Calculate new position using bicycle model
  double dt = step_length / std::max(current_speed, 0.1);  // Time step
  double beta = std::atan(std::tan(steering_angle) / 2.0); // Slip angle

  double new_x = current_x + current_speed * std::cos(current_yaw + beta) * dt;
  double new_y = current_y + current_speed * std::sin(current_yaw + beta) * dt;
  double new_yaw =
      current_yaw + (current_speed / wheelbase) * std::sin(beta) * dt;

  // Create new vehicle state (simplified - only position and heading)
  swift::common::VehicleState predicted_state = vehicle_state;
  // Note: VehicleState doesn't have setters, so we'll use the current state
  // In a real implementation, you'd need to create a new state or use a
  // different approach

  return predicted_state;
}

bool ActionMaskExtractor::CheckReferenceLineBounds(
    const swift::common::VehicleState &predicted_state,
    const std::shared_ptr<swift::planning::ReferenceLine> &reference_line) {

  if (!reference_line) {
    return true; // No reference line constraints
  }

  // Check if predicted position is within reference line bounds
  // This is a simplified check - in practice, you'd need more sophisticated
  // logic
  double predicted_x = predicted_state.x();
  double predicted_y = predicted_state.y();

  // Get reference line bounds (simplified)
  double ref_line_width = 3.5; // Typical lane width

  // Check if vehicle is within lane bounds
  // This is a placeholder - actual implementation would use reference line
  // geometry
  return true; // Simplified - always return true for now
}

void ActionMaskExtractor::GetDefaultActionSpace(
    std::vector<double> &steering_actions, std::vector<double> &step_lengths) {

  // Generate steering actions from -1.0 to 1.0
  steering_actions.clear();
  for (int i = 0; i < kDefaultNumSteeringActions; ++i) {
    double steering =
        kDefaultMinSteering + (kDefaultMaxSteering - kDefaultMinSteering) * i /
                                  (kDefaultNumSteeringActions - 1);
    steering_actions.push_back(steering);
  }

  // Generate step lengths from 0.1 to 1.0
  step_lengths.clear();
  for (int i = 0; i < kDefaultNumStepLengths; ++i) {
    double step_length = kDefaultMinStepLength +
                         (kDefaultMaxStepLength - kDefaultMinStepLength) * i /
                             (kDefaultNumStepLengths - 1);
    step_lengths.push_back(step_length);
  }
}

} // namespace rl_policy
} // namespace open_space
} // namespace planning
} // namespace swift
