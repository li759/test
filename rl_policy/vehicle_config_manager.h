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
 * @file vehicle_config_manager.h
 * @brief Vehicle configuration manager for RL policy module
 */

#pragma once

#include <memory>

#include "modules/common/math/vec2d.h"

namespace swift {
namespace planning {
namespace open_space {
namespace rl_policy {

/**
 * @struct VehicleConfig
 * @brief Vehicle configuration parameters
 */
struct VehicleConfig {
  double car_length = 4.893;      // Vehicle total length (m)
  double car_width = 1.872;       // Vehicle width (m)
  double front_axle = 0.953;      // Distance from front axle to rear axle center (m)
  double rear_axle = 1.047;       // Distance from rear axle to rear edge (m)
  double wheel_radius = 0.386;    // Wheel radius (m)
  double wheel_base = 2.912;      // Wheelbase (m)
  double min_turn_radius = 6.0;   // Minimum turn radius (m)
  double max_steer_angle = 0.48;  // Maximum steering angle (rad)

  // Calculated parameters
  double rear_to_front_axle = car_length - rear_axle;  // Distance from rear to front edge

  VehicleConfig() = default;

  VehicleConfig(
      double length, double width, double front_axle_dist, double rear_axle_dist, double wheel_rad, double wheelbase)
      : car_length(length),
        car_width(width),
        front_axle(front_axle_dist),
        rear_axle(rear_axle_dist),
        wheel_radius(wheel_rad),
        wheel_base(wheelbase),
        rear_to_front_axle(length - rear_axle_dist) {}
};

/**
 * @class VehicleConfigManager
 * @brief Singleton manager for vehicle configuration
 */
class VehicleConfigManager {
 public:
  static VehicleConfigManager& GetInstance() {
    static VehicleConfigManager instance;
    return instance;
  }

  // Disable copy constructor and assignment operator
  VehicleConfigManager(const VehicleConfigManager&) = delete;
  VehicleConfigManager& operator=(const VehicleConfigManager&) = delete;

  /**
   * @brief Initialize vehicle configuration
   * @param config Vehicle configuration parameters
   */
  void Initialize(const VehicleConfig& config) {
    config_ = config;
    initialized_ = true;
  }

  /**
   * @brief Get vehicle configuration
   * @return Reference to vehicle configuration
   */
  const VehicleConfig& GetConfig() const { return config_; }

  /**
   * @brief Check if configuration is initialized
   * @return True if initialized
   */
  bool IsInitialized() const { return initialized_; }

  // Convenience getters
  double GetCarLength() const { return config_.car_length; }
  double GetCarWidth() const { return config_.car_width; }
  double GetFrontAxle() const { return config_.front_axle; }
  double GetRearAxle() const { return config_.rear_axle; }
  double GetWheelRadius() const { return config_.wheel_radius; }
  double GetWheelBase() const { return config_.wheel_base; }
  double GetMinTurnRadius() const { return config_.min_turn_radius; }
  double GetMaxSteerAngle() const { return config_.max_steer_angle; }
  double GetRearToFrontAxle() const { return config_.rear_to_front_axle; }

 private:
  VehicleConfigManager() = default;
  ~VehicleConfigManager() = default;

  VehicleConfig config_;
  bool initialized_ = false;
};

}  // namespace rl_policy
}  // namespace open_space
}  // namespace planning
}  // namespace swift
