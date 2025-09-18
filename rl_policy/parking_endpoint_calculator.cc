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
#include <iostream>
#include <limits>
#include "core/common/log.h"

namespace swift {
namespace planning {
namespace open_space {
namespace rl_policy {

ParkingEndpoint ParkingEndpointCalculator::CalculateVerticalParkingEndpoint(
    const ParkingSlot& slot, const std::vector<ObstacleInfo>& obstacles, bool is_wheel_stop_valid) {
  ParkingEndpoint endpoint;

  // Get vehicle configuration
  const auto& config = VehicleConfigManager::GetInstance().GetConfig();

  // APA_planner algorithm for vertical parking
  // Calculate middle points of slot edges
  double middle_x1 = (slot.p0.x() + slot.p1.x()) / 2.0;
  double middle_y1 = (slot.p0.y() + slot.p1.y()) / 2.0;
  double middle_x2 = (slot.p2.x() + slot.p3.x()) / 2.0;
  double middle_y2 = (slot.p2.y() + slot.p3.y()) / 2.0;

  double dif_x = middle_x1 - middle_x2;
  double dif_y = middle_y1 - middle_y2;
  double dif_l = std::sqrt(dif_x * dif_x + dif_y * dif_y);

  // Calculate slot heading (dif_t)
  double dif_t = 0.0;
  if (dif_l > 0) {
    if (dif_y / dif_l >= 1.0) {
      dif_t = M_PI / 2.0;
    } else if (dif_y / dif_l <= -1.0) {
      dif_t = -M_PI / 2.0;
    } else {
      dif_t = std::asin(dif_y / dif_l);
    }

    if (dif_x < 0) {
      if (dif_y > 0) {
        dif_t = M_PI - dif_t;
      } else {
        dif_t = -M_PI - dif_t;
      }
    }
  }

  // Calculate offset
  double offset = std::abs(slot.width * std::cos(slot.angle)) / 4.0;

  // Calculate rear-to-front distance (r2f)
  double r2f = config.rear_to_front_axle;

  // Calculate obstacle constraints if wheel stop is valid
  double ll = 0.0;
  if (is_wheel_stop_valid && !obstacles.empty()) {
    // Find closest obstacle to slot center
    double min_distance = std::numeric_limits<double>::max();
    double middle_dx1 = 0.0, middle_dy1 = 0.0;

    for (const auto& obs : obstacles) {
      double distance =
          std::sqrt(std::pow(obs.position.x() - middle_x1, 2) + std::pow(obs.position.y() - middle_y1, 2));
      if (distance < min_distance) {
        min_distance = distance;
        middle_dx1 = obs.position.x();
        middle_dy1 = obs.position.y();
      }
    }

    if (min_distance < std::numeric_limits<double>::max()) {
      double la = std::sqrt(
          (middle_dx1 - middle_x1) * (middle_dx1 - middle_x1) + (middle_dy1 - middle_y1) * (middle_dy1 - middle_y1));
      double lb = std::sqrt(
          (middle_dx1 - middle_x2) * (middle_dx1 - middle_x2) + (middle_dy1 - middle_y2) * (middle_dy1 - middle_y2));
      ll = (la * dif_l) / (la + lb);
    }
  }

  // Calculate endpoint position
  if (!is_wheel_stop_valid || (is_wheel_stop_valid && ((ll - config.wheel_radius) > (r2f + offset)))) {
    // No obstacle constraint: use standard distance
    endpoint.position.set_x(middle_x1 - (r2f + offset) * std::cos(dif_t));
    endpoint.position.set_y(middle_y1 - (r2f + offset) * std::sin(dif_t));
  } else {
    // Obstacle constraint: use safe distance
    endpoint.position.set_x(middle_x1 - (ll - config.wheel_radius) * std::cos(dif_t));
    endpoint.position.set_y(middle_y1 - (ll - config.wheel_radius) * std::sin(dif_t));
  }

  endpoint.yaw = dif_t;
  endpoint.confidence = 1.0;
  endpoint.is_valid = true;

  return endpoint;
}

ParkingEndpoint ParkingEndpointCalculator::CalculateParkingEndpoint(
    const swift::common::VehicleState& vehicle_state,
    const ParkingSlot& slot,
    const std::vector<ObstacleInfo>& obstacles,
    bool is_wheel_stop_valid) {
  // Build transform from vehicle state (world <-> ego)


  double tx, ty, tyaw;
  BuildTransformFromState(vehicle_state, tx, ty, tyaw);

  // Transform inputs to ego frame to align numerics with APA
  ParkingSlot slot_ego = TransformSlot(slot, tx, ty, tyaw, /*world_to_ego=*/true);
  auto obs_ego = TransformObstacles(obstacles, tx, ty, tyaw, /*world_to_ego=*/true);
  std::cout << "[RL] P0_x: " << slot_ego.p0.x() << ",y:" << slot_ego.p0.y() << std::endl;
  std::cout << "[RL] P1_x: " << slot_ego.p1.x() << ",y:" << slot_ego.p1.y() << std::endl;
  std::cout << "[RL] P2_x: " << slot_ego.p2.x() << ",y:" << slot_ego.p2.y() << std::endl;
  std::cout << "[RL] P3_x: " << slot_ego.p3.x() << ",y:" << slot_ego.p3.y() << std::endl;


  // Compute endpoint in ego frame using existing logic (origin at start)
  ParkingEndpoint endpoint_ego = CalculateParkingEndpoint(slot_ego, obs_ego, is_wheel_stop_valid);

  // Return endpoint in ego coordinates to align with APA (origin at start)
  return endpoint_ego;
}

void ParkingEndpointCalculator::BuildTransformFromState(
    const swift::common::VehicleStateProvider& state, double& tx, double& ty, double& tyaw) {
  tx = state.x();
  ty = state.y();
  tyaw = state.heading();
}

void ParkingEndpointCalculator::BuildTransformFromState(
    const swift::common::VehicleState& state, double& tx, double& ty, double& tyaw) {
  tx = state.x();
  ty = state.y();
  tyaw = state.heading();
}

swift::common::math::Vec2d ParkingEndpointCalculator::TransformPoint(
    const swift::common::math::Vec2d& p, double tx, double ty, double tyaw, bool world_to_ego) {
  double c = std::cos(tyaw), s = std::sin(tyaw);
  if (world_to_ego) {
    double dx = p.x() - tx;
    double dy = p.y() - ty;
    return {dx * c + dy * s, -dx * s + dy * c};
  } else {
    double x = p.x() * c - p.y() * s + tx;
    double y = p.x() * s + p.y() * c + ty;
    return {x, y};
  }
}

double ParkingEndpointCalculator::TransformYaw(double yaw, double tyaw, bool world_to_ego) {
  if (world_to_ego) {
    return yaw - tyaw;
  } else {
    return yaw + tyaw;
  }
}

ParkingSlot ParkingEndpointCalculator::TransformSlot(
    const ParkingSlot& slot, double tx, double ty, double tyaw, bool world_to_ego) {
  ParkingSlot out = slot;
  out.p0 = TransformPoint(slot.p0, tx, ty, tyaw, world_to_ego);
  out.p1 = TransformPoint(slot.p1, tx, ty, tyaw, world_to_ego);
  out.p2 = TransformPoint(slot.p2, tx, ty, tyaw, world_to_ego);
  out.p3 = TransformPoint(slot.p3, tx, ty, tyaw, world_to_ego);
  out.angle = TransformYaw(slot.angle, tyaw, world_to_ego);
  return out;
}

std::vector<ObstacleInfo> ParkingEndpointCalculator::TransformObstacles(
    const std::vector<ObstacleInfo>& obs, double tx, double ty, double tyaw, bool world_to_ego) {
  std::vector<ObstacleInfo> out;
  out.reserve(obs.size());
  for (const auto& o : obs) {
    ObstacleInfo t = o;
    t.position = TransformPoint(o.position, tx, ty, tyaw, world_to_ego);
    t.yaw = TransformYaw(o.yaw, tyaw, world_to_ego);
    out.push_back(t);
  }
  return out;
}

ParkingEndpoint ParkingEndpointCalculator::CalculateParallelParkingEndpoint(
    const ParkingSlot& slot, const std::vector<ObstacleInfo>& obstacles, bool is_wheel_stop_valid) {
  ParkingEndpoint endpoint;

  // Get vehicle configuration
  const auto& config = VehicleConfigManager::GetInstance().GetConfig();

  // APA_planner algorithm for parallel parking
  double dif_t = 0.0;
  double dif_x = 0.0;
  double dif_y = 0.0;
  double middle_x1 = 0.0;
  double middle_y1 = 0.0;
  double middle_x2 = 0.0;
  double middle_y2 = 0.0;

  // Determine middle points based on slot orientation
  if (slot.p0.x() > slot.p1.x()) {
    middle_x1 = (slot.p0.x() + slot.p3.x()) / 2.0;
    middle_y1 = (slot.p0.y() + slot.p3.y()) / 2.0;
    middle_x2 = (slot.p1.x() + slot.p2.x()) / 2.0;
    middle_y2 = (slot.p1.y() + slot.p2.y()) / 2.0;
  } else {
    middle_x1 = (slot.p1.x() + slot.p2.x()) / 2.0;
    middle_y1 = (slot.p1.y() + slot.p2.y()) / 2.0;
    middle_x2 = (slot.p0.x() + slot.p3.x()) / 2.0;
    middle_y2 = (slot.p0.y() + slot.p3.y()) / 2.0;
  }

  dif_x = middle_x1 - middle_x2;
  dif_y = middle_y1 - middle_y2;
  double dif_l = std::sqrt(dif_x * dif_x + dif_y * dif_y);

  // Calculate slot heading (dif_t)

  if (dif_y / dif_l >= 1.0) {
    dif_t = M_PI / 2.0;
  } else if (dif_y / dif_l <= -1.0) {
    dif_t = -M_PI / 2.0;
  } else {
    // Calculate heading based on p0-p1 edge
    double p0p1_dist = std::sqrt(std::pow(slot.p0.x() - slot.p1.x(), 2) + std::pow(slot.p0.y() - slot.p1.y(), 2));
    if (p0p1_dist > 0) {
      if (slot.p0.x() > slot.p1.x()) {
        dif_t = std::asin((slot.p0.y() - slot.p1.y()) / p0p1_dist);
      } else {
        dif_t = std::asin((slot.p1.y() - slot.p0.y()) / p0p1_dist);
      }
    }
  }


  // Calculate obstacle constraints if wheel stop is valid
  double ll = 0.0;
  if (is_wheel_stop_valid && !obstacles.empty()) {
    // Find closest obstacle to slot center
    double min_distance = std::numeric_limits<double>::max();
    double middle_dx1 = 0.0, middle_dy1 = 0.0;

    for (const auto& obs : obstacles) {
      double distance =
          std::sqrt(std::pow(obs.position.x() - middle_x1, 2) + std::pow(obs.position.y() - middle_y1, 2));
      if (distance < min_distance) {
        min_distance = distance;
        middle_dx1 = obs.position.x();
        middle_dy1 = obs.position.y();
      }
    }

    if (min_distance < std::numeric_limits<double>::max()) {
      double la = std::sqrt(
          (middle_dx1 - middle_x1) * (middle_dx1 - middle_x1) + (middle_dy1 - middle_y1) * (middle_dy1 - middle_y1));
      double lb = std::sqrt(
          (middle_dx1 - middle_x2) * (middle_dx1 - middle_x2) + (middle_dy1 - middle_y2) * (middle_dy1 - middle_y2));
      ll = (la * dif_l) / (la + lb);
    }
  }

  // Calculate endpoint position
  if (!is_wheel_stop_valid) {
    // No obstacle constraint
    endpoint.position.set_x((slot.p0.x() + slot.p1.x()) / 2.0 - (config.car_length / 2.0 - 1.0) * std::cos(dif_t));
    endpoint.position.set_y((slot.p0.y() + slot.p1.y()) / 2.0 - (config.car_length / 2.0 - 1.0) * std::sin(dif_t));
  } else {
    // Obstacle constraint: use safe distance
    if (slot.p0.x() > slot.p1.x()) {
      endpoint.position.set_x(slot.p0.x() - (ll - 0.30) * std::cos(dif_t));
      endpoint.position.set_y(slot.p0.y() - (ll - 0.30) * std::sin(dif_t));
    } else {
      endpoint.position.set_x(slot.p1.x() - (ll - 0.30) * std::cos(dif_t));
      endpoint.position.set_y(slot.p1.y() - (ll - 0.30) * std::sin(dif_t));
    }
  }
  std::cout << "[RL] car_length:" << config.car_length << std::endl;
  std::cout << "[RL] endpoint.x:" << endpoint.position.x() << std::endl;
  std::cout << "[RL] endpoint.x:" << endpoint.position.y() << std::endl;
  std::cout << "[RL] endpoint.yaw:" << dif_t << std::endl;
  // Lateral position adjustment
  double euclidean_distance =
      std::sqrt(std::pow(slot.p3.x() - slot.p0.x(), 2) + std::pow(slot.p3.y() - slot.p0.y(), 2));
  if (slot.p0.y() > slot.p3.y()) {
    endpoint.position.set_x(endpoint.position.x() - (euclidean_distance / 2) * std::cos(dif_t + M_PI / 2));
    endpoint.position.set_y(endpoint.position.y() - (euclidean_distance / 2) * std::sin(dif_t + M_PI / 2));
  } else {
    endpoint.position.set_x(endpoint.position.x() + (euclidean_distance / 2) * std::cos(dif_t + M_PI / 2));
    endpoint.position.set_y(endpoint.position.y() + (euclidean_distance / 2) * std::sin(dif_t + M_PI / 2));
  }
  std::cout << "[RL] endpoint.x:" << endpoint.position.x() << std::endl;
  std::cout << "[RL] endpoint.x:" << endpoint.position.y() << std::endl;
  std::cout << "[RL] endpoint.yaw:" << dif_t << std::endl;
  endpoint.yaw = dif_t;
  endpoint.confidence = 1.0;
  endpoint.is_valid = true;

  return endpoint;
}

ParkingEndpoint ParkingEndpointCalculator::CalculateParkingEndpoint(
    const ParkingSlot& slot, const std::vector<ObstacleInfo>& obstacles, bool is_wheel_stop_valid) {
  ParkingType type = DetectParkingType(slot);

  switch (type) {
    case ParkingType::VERTICAL:
      std::cout << "[RL] VERTICAL" << std::endl;
      return CalculateVerticalParkingEndpoint(slot, obstacles, is_wheel_stop_valid);
    case ParkingType::PARALLEL:
      std::cout << "[RL] PARALLEL" << std::endl;
      return CalculateParallelParkingEndpoint(slot, obstacles, is_wheel_stop_valid);
    default:
      AERROR << "Unsupported parking type: " << static_cast<int>(type);
      return ParkingEndpoint();
  }
}

ParkingEndpoint ParkingEndpointCalculator::OptimizeEndpointWithObstacles(
    const ParkingEndpoint& initial_endpoint, const std::vector<ObstacleInfo>& obstacles) {
  // TODO: Implement obstacle-based optimization similar to UpdateEPts
  // For now, return the initial endpoint
  return initial_endpoint;
}

ParkingType ParkingEndpointCalculator::DetectParkingType(const ParkingSlot& slot) {
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

bool ParkingEndpointCalculator::ValidateEndpoint(const ParkingEndpoint& endpoint, const ParkingSlot& slot) {
  // Check if endpoint is within reasonable bounds
  if (!endpoint.is_valid) {
    return false;
  }

  // Check if endpoint is within slot bounds (with some tolerance)
  double tolerance = 2.0;  // 2 meters tolerance
  swift::common::math::Vec2d slot_center = (slot.p0 + slot.p1 + slot.p2 + slot.p3) / 4.0;
  double distance = CalculateDistance(endpoint.position, slot_center);

  return distance <= tolerance;
}

void ParkingEndpointCalculator::CalculateSlotCenterAndHeading(
    const ParkingSlot& slot, swift::common::math::Vec2d& center, double& heading) {
  // Calculate slot center
  center = (slot.p0 + slot.p1 + slot.p2 + slot.p3) / 4.0;

  // Calculate slot heading based on slot orientation
  swift::common::math::Vec2d front_center = (slot.p0 + slot.p1) / 2.0;
  swift::common::math::Vec2d rear_center = (slot.p2 + slot.p3) / 2.0;

  swift::common::math::Vec2d direction = front_center - rear_center;
  heading = std::atan2(direction.y(), direction.x());
}

double ParkingEndpointCalculator::CalculateSafeDistance(
    const ParkingSlot& slot, const std::vector<ObstacleInfo>& obstacles) {
  if (obstacles.empty()) {
    return 10.0;  // Default safe distance
  }

  // Calculate minimum distance to obstacles
  double min_distance = std::numeric_limits<double>::max();
  swift::common::math::Vec2d slot_center = (slot.p0 + slot.p1 + slot.p2 + slot.p3) / 4.0;

  for (const auto& obstacle : obstacles) {
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
    const swift::common::math::Vec2d& p1, const swift::common::math::Vec2d& p2) {
  return std::sqrt(std::pow(p1.x() - p2.x(), 2) + std::pow(p1.y() - p2.y(), 2));
}

}  // namespace rl_policy
}  // namespace open_space
}  // namespace planning
}  // namespace swift
