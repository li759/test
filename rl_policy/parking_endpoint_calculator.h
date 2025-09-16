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
 * @file swift_parking_endpoint_calculator.h
 * @brief Parking endpoint calculator for RL policy module
 */

#pragma once

#include <vector>

#include "modules/common/math/vec2d.h"
#include "modules/planning/open_space/rl_policy/vehicle_config_manager.h"
#include "modules/common/vehicle_state/vehicle_state_provider.h"
#include "modules/common/vehicle_state/proto/vehicle_state.pb.h"

namespace swift {
namespace planning {
namespace open_space {
namespace rl_policy {

/**
 * @enum ParkingType
 * @brief Parking type enumeration
 */
enum class ParkingType {
  VERTICAL = 0, // Vertical parking
  PARALLEL = 1, // Parallel parking
  DIAGONAL = 2  // Diagonal parking
};

/**
 * @struct ParkingSlot
 * @brief Parking slot information
 */
struct ParkingSlot {
  swift::common::math::Vec2d p0; // Corner point 0
  swift::common::math::Vec2d p1; // Corner point 1
  swift::common::math::Vec2d p2; // Corner point 2
  swift::common::math::Vec2d p3; // Corner point 3
  double angle = 0.0;            // Slot angle (rad)
  double width = 0.0;            // Slot width (m)
  ParkingType type = ParkingType::VERTICAL;

  ParkingSlot() = default;

  ParkingSlot(const swift::common::math::Vec2d &corner0,
              const swift::common::math::Vec2d &corner1,
              const swift::common::math::Vec2d &corner2,
              const swift::common::math::Vec2d &corner3,
              double slot_angle = 0.0, double slot_width = 0.0)
      : p0(corner0), p1(corner1), p2(corner2), p3(corner3), angle(slot_angle),
        width(slot_width) {}
};

/**
 * @struct ParkingEndpoint
 * @brief Parking endpoint information
 */
struct ParkingEndpoint {
  swift::common::math::Vec2d position; // Endpoint position (rear axle center)
  double yaw = 0.0;                    // Endpoint yaw angle (rad)
  double confidence = 1.0;             // Confidence level [0, 1]
  bool is_valid = false;               // Validity flag

  ParkingEndpoint() = default;

  ParkingEndpoint(const swift::common::math::Vec2d &pos, double heading,
                  double conf = 1.0, bool valid = true)
      : position(pos), yaw(heading), confidence(conf), is_valid(valid) {}
};

/**
 * @struct ObstacleInfo
 * @brief Obstacle information for endpoint optimization
 */
struct ObstacleInfo {
  swift::common::math::Vec2d position; // Obstacle position
  double width = 0.0;                  // Obstacle width (m)
  double length = 0.0;                 // Obstacle length (m)
  double yaw = 0.0;                    // Obstacle yaw angle (rad)

  ObstacleInfo() = default;

  ObstacleInfo(const swift::common::math::Vec2d &pos, double w = 0.0,
               double l = 0.0, double heading = 0.0)
      : position(pos), width(w), length(l), yaw(heading) {}
};

/**
 * @class SwiftParkingEndpointCalculator
 * @brief Calculate parking endpoints based on APA planner logic
 */
class ParkingEndpointCalculator {
public:
  ParkingEndpointCalculator() = default;
  ~ParkingEndpointCalculator() = default;

  /**
   * @brief Calculate vertical parking endpoint
   * @param slot Parking slot information
   * @param obstacles Obstacle information
   * @param is_wheel_stop_valid Whether wheel stop is valid
   * @return Parking endpoint information
   */
  ParkingEndpoint CalculateVerticalParkingEndpoint(
      const ParkingSlot &slot, const std::vector<ObstacleInfo> &obstacles = {},
      bool is_wheel_stop_valid = false);

  /**
   * @brief Calculate parallel parking endpoint
   * @param slot Parking slot information
   * @param obstacles Obstacle information
   * @param is_wheel_stop_valid Whether wheel stop is valid
   * @return Parking endpoint information
   */
  ParkingEndpoint CalculateParallelParkingEndpoint(
      const ParkingSlot &slot, const std::vector<ObstacleInfo> &obstacles = {},
      bool is_wheel_stop_valid = false);

  /**
   * @brief Calculate parking endpoint based on slot type
   * @param slot Parking slot information
   * @param obstacles Obstacle information
   * @param is_wheel_stop_valid Whether wheel stop is valid
   * @return Parking endpoint information
   */
  ParkingEndpoint
  CalculateParkingEndpoint(const ParkingSlot &slot,
                           const std::vector<ObstacleInfo> &obstacles = {},
                           bool is_wheel_stop_valid = false);

  // Overload with vehicle state to align with APA planner's coordinate handling
  ParkingEndpoint
  CalculateParkingEndpoint(const swift::common::VehicleStateProvider &vehicle_state,
                           const ParkingSlot &slot,
                           const std::vector<ObstacleInfo> &obstacles = {},
                           bool is_wheel_stop_valid = false);

  /**
   * @brief Optimize endpoint with obstacle constraints
   * @param initial_endpoint Initial endpoint
   * @param obstacles Obstacle information
   * @return Optimized endpoint
   */
  ParkingEndpoint
  OptimizeEndpointWithObstacles(const ParkingEndpoint &initial_endpoint,
                                const std::vector<ObstacleInfo> &obstacles);

  /**
   * @brief Detect parking slot type
   * @param slot Parking slot information
   * @return Detected parking type
   */
  ParkingType DetectParkingType(const ParkingSlot &slot);

  /**
   * @brief Validate parking endpoint
   * @param endpoint Endpoint to validate
   * @param slot Parking slot information
   * @return True if valid
   */
  bool ValidateEndpoint(const ParkingEndpoint &endpoint,
                        const ParkingSlot &slot);

private:
  /**
   * @brief Calculate slot center and heading
   * @param slot Parking slot information
   * @param center Output slot center
   * @param heading Output slot heading
   */
  void CalculateSlotCenterAndHeading(const ParkingSlot &slot,
                                     swift::common::math::Vec2d &center,
                                     double &heading);

  /**
   * @brief Calculate safe distance to obstacles
   * @param slot Parking slot information
   * @param obstacles Obstacle information
   * @return Safe distance (m)
   */
  double CalculateSafeDistance(const ParkingSlot &slot,
                               const std::vector<ObstacleInfo> &obstacles);

  /**
   * @brief Normalize angle to [-π, π]
   * @param angle Input angle (rad)
   * @return Normalized angle (rad)
   */
  double NormalizeAngle(double angle);

  /**
   * @brief Calculate distance between two points
   * @param p1 First point
   * @param p2 Second point
   * @return Distance (m)
   */
  double CalculateDistance(const swift::common::math::Vec2d &p1,
                           const swift::common::math::Vec2d &p2);

  // === Coordinate transform helpers (APA-aligned) ===
  static void BuildTransformFromState(const swift::common::VehicleStateProvider &state,
                                      double &tx, double &ty, double &tyaw);
  static swift::common::math::Vec2d TransformPoint(const swift::common::math::Vec2d &p,
                                                   double tx, double ty, double tyaw,
                                                   bool world_to_ego);
  static double TransformYaw(double yaw, double tyaw, bool world_to_ego);
  static ParkingSlot TransformSlot(const ParkingSlot &slot,
                                   double tx, double ty, double tyaw,
                                   bool world_to_ego);
  static std::vector<ObstacleInfo> TransformObstacles(const std::vector<ObstacleInfo> &obs,
                                                      double tx, double ty, double tyaw,
                                                      bool world_to_ego);
};

} // namespace rl_policy
} // namespace open_space
} // namespace planning
} // namespace swift
