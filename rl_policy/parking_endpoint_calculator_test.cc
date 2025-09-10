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

#include <gtest/gtest.h>

#include "modules/common/math/vec2d.h"

namespace swift {
namespace planning {
namespace open_space {
namespace rl_policy {

class ParkingEndpointCalculatorTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize vehicle configuration
    VehicleConfig config;
    config.car_length = 4.912;
    config.car_width = 1.872;
    config.front_axle = 0.953;
    config.rear_axle = 1.047;
    config.wheel_radius = 0.386;
    config.wheel_base = 2.912;
    config.min_turn_radius = 6.0;
    config.max_steer_angle = 0.48;

    VehicleConfigManager::GetInstance().Initialize(config);
  }

  void TearDown() override {}

  ParkingEndpointCalculator calculator_;
};

TEST_F(ParkingEndpointCalculatorTest, TestVerticalParkingEndpoint) {
  // Create a vertical parking slot
  ParkingSlot slot;
  slot.p0 = swift::common::math::Vec2d(0.0, 0.0); // Left top
  slot.p1 = swift::common::math::Vec2d(2.0, 0.0); // Right top
  slot.p2 = swift::common::math::Vec2d(2.0, 5.0); // Right bottom
  slot.p3 = swift::common::math::Vec2d(0.0, 5.0); // Left bottom
  slot.angle = 0.0;
  slot.width = 2.0;
  slot.type = ParkingType::VERTICAL;

  // Calculate endpoint
  ParkingEndpoint endpoint = calculator_.CalculateVerticalParkingEndpoint(slot);

  // Verify endpoint is valid
  EXPECT_TRUE(endpoint.is_valid);
  EXPECT_GT(endpoint.confidence, 0.0);

  // Verify endpoint position is reasonable
  EXPECT_NEAR(endpoint.position.x(), 1.0, 0.1); // Should be near slot center x
  EXPECT_LT(endpoint.position.y(), 0.0);        // Should be behind the slot
  EXPECT_NEAR(endpoint.yaw, M_PI / 2.0, 0.1);   // Should be vertical heading
}

TEST_F(ParkingEndpointCalculatorTest, TestParallelParkingEndpoint) {
  // Create a parallel parking slot
  ParkingSlot slot;
  slot.p0 = swift::common::math::Vec2d(0.0, 0.0); // Left front
  slot.p1 = swift::common::math::Vec2d(5.0, 0.0); // Right front
  slot.p2 = swift::common::math::Vec2d(5.0, 2.0); // Right rear
  slot.p3 = swift::common::math::Vec2d(0.0, 2.0); // Left rear
  slot.angle = 0.0;
  slot.width = 5.0;
  slot.type = ParkingType::PARALLEL;

  // Calculate endpoint
  ParkingEndpoint endpoint = calculator_.CalculateParallelParkingEndpoint(slot);

  // Verify endpoint is valid
  EXPECT_TRUE(endpoint.is_valid);
  EXPECT_GT(endpoint.confidence, 0.0);

  // Verify endpoint position is reasonable
  EXPECT_NEAR(endpoint.position.y(), 1.0, 0.1); // Should be near slot center y
  EXPECT_LT(endpoint.position.x(), 0.0);        // Should be behind the slot
  EXPECT_NEAR(endpoint.yaw, 0.0, 0.1);          // Should be horizontal heading
}

TEST_F(ParkingEndpointCalculatorTest, TestParkingTypeDetection) {
  // Test vertical parking detection
  ParkingSlot vertical_slot;
  vertical_slot.p0 = swift::common::math::Vec2d(0.0, 0.0);
  vertical_slot.p1 = swift::common::math::Vec2d(2.0, 0.0);
  vertical_slot.p2 = swift::common::math::Vec2d(2.0, 5.0);
  vertical_slot.p3 = swift::common::math::Vec2d(0.0, 5.0);

  ParkingType type = calculator_.DetectParkingType(vertical_slot);
  EXPECT_EQ(type, ParkingType::VERTICAL);

  // Test parallel parking detection
  ParkingSlot parallel_slot;
  parallel_slot.p0 = swift::common::math::Vec2d(0.0, 0.0);
  parallel_slot.p1 = swift::common::math::Vec2d(5.0, 0.0);
  parallel_slot.p2 = swift::common::math::Vec2d(5.0, 2.0);
  parallel_slot.p3 = swift::common::math::Vec2d(0.0, 2.0);

  type = calculator_.DetectParkingType(parallel_slot);
  EXPECT_EQ(type, ParkingType::PARALLEL);
}

TEST_F(ParkingEndpointCalculatorTest, TestEndpointValidation) {
  // Create a valid parking slot
  ParkingSlot slot;
  slot.p0 = swift::common::math::Vec2d(0.0, 0.0);
  slot.p1 = swift::common::math::Vec2d(2.0, 0.0);
  slot.p2 = swift::common::math::Vec2d(2.0, 5.0);
  slot.p3 = swift::common::math::Vec2d(0.0, 5.0);

  // Create a valid endpoint
  ParkingEndpoint valid_endpoint;
  valid_endpoint.position = swift::common::math::Vec2d(1.0, -2.0);
  valid_endpoint.yaw = M_PI / 2.0;
  valid_endpoint.is_valid = true;
  valid_endpoint.confidence = 1.0;

  // Validate endpoint
  bool is_valid = calculator_.ValidateEndpoint(valid_endpoint, slot);
  EXPECT_TRUE(is_valid);

  // Create an invalid endpoint (too far from slot)
  ParkingEndpoint invalid_endpoint;
  invalid_endpoint.position = swift::common::math::Vec2d(100.0, 100.0);
  invalid_endpoint.yaw = M_PI / 2.0;
  invalid_endpoint.is_valid = false;
  invalid_endpoint.confidence = 0.0;

  // Validate endpoint
  is_valid = calculator_.ValidateEndpoint(invalid_endpoint, slot);
  EXPECT_FALSE(is_valid);
}

TEST_F(ParkingEndpointCalculatorTest, TestObstacleConstraints) {
  // Create a parking slot
  ParkingSlot slot;
  slot.p0 = swift::common::math::Vec2d(0.0, 0.0);
  slot.p1 = swift::common::math::Vec2d(2.0, 0.0);
  slot.p2 = swift::common::math::Vec2d(2.0, 5.0);
  slot.p3 = swift::common::math::Vec2d(0.0, 5.0);

  // Create obstacles
  std::vector<ObstacleInfo> obstacles;
  ObstacleInfo obstacle;
  obstacle.position = swift::common::math::Vec2d(1.0, -1.0);
  obstacle.width = 1.0;
  obstacle.length = 2.0;
  obstacles.push_back(obstacle);

  // Calculate endpoint with obstacles
  ParkingEndpoint endpoint =
      calculator_.CalculateVerticalParkingEndpoint(slot, obstacles, true);

  // Verify endpoint is valid
  EXPECT_TRUE(endpoint.is_valid);

  // Verify endpoint considers obstacle constraints
  // The endpoint should be adjusted to avoid the obstacle
  EXPECT_GT(endpoint.position.y(), -1.0); // Should be behind the obstacle
}

} // namespace rl_policy
} // namespace open_space
} // namespace planning
} // namespace swift
