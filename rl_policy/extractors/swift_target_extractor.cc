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

#include "modules/planning/open_space/rl_policy/extractors/swift_target_extractor.h"

#include <algorithm>
#include <cmath>

namespace swift {
namespace planning {
namespace open_space {
namespace rl_policy {

TargetInfo SwiftTargetExtractor::ExtractTargetInfo(
    const swift::common::VehicleState &vehicle_state,
    const swift::common::math::Vec2d &target_position, double target_yaw) {

  return ExtractTargetInfoWithCurvature(vehicle_state, target_position,
                                        target_yaw, 0.0);
}

TargetInfo SwiftTargetExtractor::ExtractTargetInfoWithCurvature(
    const swift::common::VehicleState &vehicle_state,
    const swift::common::math::Vec2d &target_position, double target_yaw,
    double reference_curvature) {

  // Calculate relative position in vehicle coordinate system
  auto relative_pos = CalculateRelativePosition(vehicle_state, target_position);
  double dx = relative_pos.first;
  double dy = relative_pos.second;

  // Calculate heading error
  double heading_error =
      CalculateHeadingError(vehicle_state.heading(), target_yaw);

  // Get current speed
  double current_speed = vehicle_state.linear_velocity();

  return TargetInfo(static_cast<float>(dx), static_cast<float>(dy),
                    static_cast<float>(heading_error),
                    static_cast<float>(current_speed),
                    static_cast<float>(reference_curvature));
}

std::vector<float>
SwiftTargetExtractor::ToVector(const TargetInfo &target_info) {
  std::vector<float> target_vector(5);
  target_vector[0] = target_info.dx;
  target_vector[1] = target_info.dy;
  target_vector[2] = target_info.heading_error;
  target_vector[3] = target_info.current_speed;
  target_vector[4] = target_info.curvature;
  return target_vector;
}

double SwiftTargetExtractor::CalculateHeadingError(double current_yaw,
                                                   double target_yaw) {
  double error = target_yaw - current_yaw;
  return NormalizeAngle(error);
}

std::pair<double, double> SwiftTargetExtractor::CalculateRelativePosition(
    const swift::common::VehicleState &vehicle_state,
    const swift::common::math::Vec2d &target_position) {

  double vehicle_x = vehicle_state.x();
  double vehicle_y = vehicle_state.y();
  double vehicle_yaw = vehicle_state.heading();

  // Calculate global position difference
  double global_dx = target_position.x() - vehicle_x;
  double global_dy = target_position.y() - vehicle_y;

  // Transform to vehicle coordinate system
  double cos_yaw = std::cos(-vehicle_yaw);
  double sin_yaw = std::sin(-vehicle_yaw);

  double dx = global_dx * cos_yaw - global_dy * sin_yaw;
  double dy = global_dx * sin_yaw + global_dy * cos_yaw;

  return std::make_pair(dx, dy);
}

double SwiftTargetExtractor::NormalizeAngle(double angle) {
  while (angle > M_PI) {
    angle -= 2.0 * M_PI;
  }
  while (angle < -M_PI) {
    angle += 2.0 * M_PI;
  }
  return angle;
}

} // namespace rl_policy
} // namespace open_space
} // namespace planning
} // namespace swift
