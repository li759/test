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

#include "modules/planning/open_space/rl_policy/parking_endpoint_calculator.h"

#include <algorithm>
#include <cmath>

#include "core/common/log.h"

namespace swift {
namespace planning {
namespace open_space {
namespace rl_policy {

ParkingEndpoint ParkingEndpointCalculator::CalculateVerticalParkingEndpoint(
    const ParkingSlot &slot, const std::vector<ObstacleInfo> &obstacles,
    bool is_wheel_stop_valid) {

  ParkingEndpoint endpoint;

  // Get vehicle configuration
  const auto &config = VehicleConfigManager::GetInstance().GetConfig();

  // Calculate slot center and heading
  swift::common::math::Vec2d slot_center;
  double slot_heading;
  CalculateSlotCenterAndHeading(slot, slot_center, slot_heading);

  // Calculate safe distance
  double safe_distance = CalculateSafeDistance(slot, obstacles);

  // Calculate offset based on slot width and angle
  double offset = std::abs(slot.width * std::cos(slot.angle)) / 4.0;

  // Calculate rear-to-front distance (r2f)
  double r2f = config.rear_to_front_axle;

  // Calculate endpoint position
  if (!is_wheel_stop_valid ||
      (is_wheel_stop_valid &&
       (safe_distance - config.wheel_radius) > (r2f + offset))) {
    // No obstacle constraint: use standard distance
    endpoint.position.set_x(slot_center.x() -
                            (r2f + offset) * std::cos(slot_heading));
    endpoint.position.set_y(slot_center.y() -
                            (r2f + offset) * std::sin(slot_heading));
  } else {
    // Obstacle constraint: use safe distance
    endpoint.position.set_x(slot_center.x() -
                            (safe_distance - config.wheel_radius) *
                                std::cos(slot_heading));
    endpoint.position.set_y(slot_center.y() -
                            (safe_distance - config.wheel_radius) *
                                std::sin(slot_heading));
  }

  endpoint.yaw = slot_heading;
  endpoint.confidence = 1.0;
  endpoint.is_valid = true;

  return endpoint;
}

ParkingEndpoint ParkingEndpointCalculator::CalculateParallelParkingEndpoint(
    const ParkingSlot &slot, const std::vector<ObstacleInfo> &obstacles,
    bool is_wheel_stop_valid) {

  ParkingEndpoint endpoint;

  // Get vehicle configuration
  const auto &config = VehicleConfigManager::GetInstance().GetConfig();

  // Calculate slot center and heading
  swift::common::math::Vec2d slot_center;
  double slot_heading;
  CalculateSlotCenterAndHeading(slot, slot_center, slot_heading);

  // Calculate endpoint position
  if (!is_wheel_stop_valid) {
    // No obstacle constraint
    endpoint.position.set_x(slot_center.x() - (config.car_length / 2.0 - 1.0) *
                                                  std::cos(slot_heading));
    endpoint.position.set_y(slot_center.y() - (config.car_length / 2.0 - 1.0) *
                                                  std::sin(slot_heading));
  } else {
    // Obstacle constraint: use safe distance
    double safe_distance = CalculateSafeDistance(slot, obstacles);
    if (slot.p0.x() > slot.p1.x()) {
      endpoint.position.set_x(slot.p0.x() -
                              (safe_distance - 0.30) * std::cos(slot_heading));
      endpoint.position.set_y(slot.p0.y() -
                              (safe_distance - 0.30) * std::sin(slot_heading));
    } else {
      endpoint.position.set_x(slot.p1.x() -
                              (safe_distance - 0.30) * std::cos(slot_heading));
      endpoint.position.set_y(slot.p1.y() -
                              (safe_distance - 0.30) * std::sin(slot_heading));
    }
  }

  // Lateral position adjustment
  double euclidean_distance = CalculateDistance(slot.p3, slot.p0);
  if (slot.p0.y() > slot.p3.y()) {
    endpoint.position.set_x(endpoint.position.x() -
                            (euclidean_distance / 2) *
                                std::cos(slot_heading + M_PI / 2));
    endpoint.position.set_y(endpoint.position.y() -
                            (euclidean_distance / 2) *
                                std::sin(slot_heading + M_PI / 2));
  } else {
    endpoint.position.set_x(endpoint.position.x() +
                            (euclidean_distance / 2) *
                                std::cos(slot_heading + M_PI / 2));
    endpoint.position.set_y(endpoint.position.y() +
                            (euclidean_distance / 2) *
                                std::sin(slot_heading + M_PI / 2));
  }

  endpoint.yaw = slot_heading;
  endpoint.confidence = 1.0;
  endpoint.is_valid = true;

  return endpoint;
}

ParkingEndpoint ParkingEndpointCalculator::CalculateParkingEndpoint(
    const ParkingSlot &slot, const std::vector<ObstacleInfo> &obstacles,
    bool is_wheel_stop_valid) {

  ParkingType type = DetectParkingType(slot);

  switch (type) {
  case ParkingType::VERTICAL:
    return CalculateVerticalParkingEndpoint(slot, obstacles,
                                            is_wheel_stop_valid);
  case ParkingType::PARALLEL:
    return CalculateParallelParkingEndpoint(slot, obstacles,
                                            is_wheel_stop_valid);
  default:
    AERROR << "Unsupported parking type: " << static_cast<int>(type);
    return ParkingEndpoint();
  }
}

ParkingEndpoint ParkingEndpointCalculator::OptimizeEndpointWithObstacles(
    const ParkingEndpoint &initial_endpoint,
    const std::vector<ObstacleInfo> &obstacles) {

  // TODO: Implement obstacle-based optimization similar to UpdateEPts
  // For now, return the initial endpoint
  return initial_endpoint;
}

ParkingType
ParkingEndpointCalculator::DetectParkingType(const ParkingSlot &slot) {
  // Calculate slot dimensions
  double width = CalculateDistance(slot.p0, slot.p1);
  double length = CalculateDistance(slot.p0, slot.p3);

  // Simple heuristic: if width > length, it's parallel parking
  if (width > length) {
    return ParkingType::PARALLEL;
  } else {
    return ParkingType::VERTICAL;
  }
}

bool ParkingEndpointCalculator::ValidateEndpoint(
    const ParkingEndpoint &endpoint, const ParkingSlot &slot) {
  // Check if endpoint is within reasonable bounds
  if (!endpoint.is_valid) {
    return false;
  }

  // Check if endpoint is within slot bounds (with some tolerance)
  double tolerance = 2.0; // 2 meters tolerance
  swift::common::math::Vec2d slot_center =
      (slot.p0 + slot.p1 + slot.p2 + slot.p3) / 4.0;
  double distance = CalculateDistance(endpoint.position, slot_center);

  return distance <= tolerance;
}

void ParkingEndpointCalculator::CalculateSlotCenterAndHeading(
    const ParkingSlot &slot, swift::common::math::Vec2d &center,
    double &heading) {

  // Calculate slot center
  center = (slot.p0 + slot.p1 + slot.p2 + slot.p3) / 4.0;

  // Calculate slot heading based on slot orientation
  swift::common::math::Vec2d front_center = (slot.p0 + slot.p1) / 2.0;
  swift::common::math::Vec2d rear_center = (slot.p2 + slot.p3) / 2.0;

  swift::common::math::Vec2d direction = front_center - rear_center;
  heading = std::atan2(direction.y(), direction.x());
}

double ParkingEndpointCalculator::CalculateSafeDistance(
    const ParkingSlot &slot, const std::vector<ObstacleInfo> &obstacles) {

  if (obstacles.empty()) {
    return 10.0; // Default safe distance
  }

  // Calculate minimum distance to obstacles
  double min_distance = std::numeric_limits<double>::max();
  swift::common::math::Vec2d slot_center =
      (slot.p0 + slot.p1 + slot.p2 + slot.p3) / 4.0;

  for (const auto &obstacle : obstacles) {
    double distance = CalculateDistance(slot_center, obstacle.position);
    min_distance = std::min(min_distance, distance);
  }

  return min_distance;
}

double ParkingEndpointCalculator::NormalizeAngle(double angle) {
  while (angle > M_PI) {
    angle -= 2.0 * M_PI;
  }
  while (angle < -M_PI) {
    angle += 2.0 * M_PI;
  }
  return angle;
}

double ParkingEndpointCalculator::CalculateDistance(
    const swift::common::math::Vec2d &p1,
    const swift::common::math::Vec2d &p2) {

  return std::sqrt(std::pow(p1.x() - p2.x(), 2) + std::pow(p1.y() - p2.y(), 2));
}

} // namespace rl_policy
} // namespace open_space
} // namespace planning
} // namespace swift
