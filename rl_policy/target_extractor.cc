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

#include "modules/planning/open_space/rl_policy/target_extractor.h"

#include <algorithm>
#include <cmath>

#include "core/common/log.h"

namespace swift {
namespace planning {
namespace open_space {
namespace rl_policy {

TargetInfo TargetExtractor::ExtractTargetInfo(
    const swift::common::VehicleState &vehicle_state,
    const swift::common::math::Vec2d &target_position, double target_yaw) {

  return ExtractTargetInfoWithCurvature(vehicle_state, target_position,
                                        target_yaw, 0.0);
}

TargetInfo TargetExtractor::ExtractTargetInfoWithCurvature(
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

std::vector<float> TargetExtractor::ToVector(const TargetInfo &target_info) {
  // HOPE 表示: [rel_distance, cos(rel_angle), sin(rel_angle), cos(rel_dest_heading), cos(rel_dest_heading)]
  // 1) 车辆坐标系下相对位姿: (dx, dy) 已在上游计算
  const double dx = static_cast<double>(target_info.dx);
  const double dy = static_cast<double>(target_info.dy);
  const double rel_distance = std::sqrt(dx * dx + dy * dy);
  const double rel_angle = std::atan2(dy, dx);


  const double rel_dest_heading = static_cast<double>(target_info.heading_error);

  std::vector<float> target_vector(5);
  target_vector[0] = static_cast<float>(rel_distance);
  target_vector[1] = static_cast<float>(std::cos(rel_angle));
  target_vector[2] = static_cast<float>(std::sin(rel_angle));
  target_vector[3] = static_cast<float>(std::cos(rel_dest_heading));
  target_vector[4] = static_cast<float>(std::cos(rel_dest_heading)); // 按你提供的HOPE格式保留两次cos
  return target_vector;
}

double TargetExtractor::CalculateHeadingError(double current_yaw,
                                              double target_yaw) {
  double error = target_yaw - current_yaw;
  return NormalizeAngle(error);
}

std::pair<double, double> TargetExtractor::CalculateRelativePosition(
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

double TargetExtractor::NormalizeAngle(double angle) {
  while (angle > M_PI) {
    angle -= 2.0 * M_PI;
  }
  while (angle < -M_PI) {
    angle += 2.0 * M_PI;
  }
  return angle;
}

TargetInfo TargetExtractor::ExtractTargetInfoFromParkingSlot(
    const swift::common::VehicleState &vehicle_state,
    const ParkingSlot &parking_slot, const std::vector<ObstacleInfo> &obstacles,
    bool is_wheel_stop_valid) {

  // Calculate parking endpoint using APA planner logic
  ParkingEndpoint endpoint = parking_calculator_.CalculateParkingEndpoint(
      parking_slot, obstacles, is_wheel_stop_valid);

  if (!endpoint.is_valid) {
    AERROR << "Failed to calculate parking endpoint";
    return TargetInfo(); // Return default invalid target
  }

  // Convert parking endpoint to target position
  swift::common::math::Vec2d target_position(endpoint.position.x(),
                                             endpoint.position.y());
  double target_yaw = endpoint.yaw;

  // Extract target information using existing method
  return ExtractTargetInfoWithCurvature(vehicle_state, target_position,
                                        target_yaw, 0.0);
}

} // namespace rl_policy
} // namespace open_space
} // namespace planning
} // namespace swift
