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
#include "modules/planning/open_space/rl_policy/matplotlibcpp.h"
#include <iostream>
#include <cmath>

#include "modules/common/math/vec2d.h"
// #include "modules/common/vehicle_state/proto/vehicle_state.pb.h
namespace plt = matplotlibcpp;

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

  // Helper function to visualize parking slot and endpoint
  void VisualizeParkingSlot(
      const ParkingSlot& slot,
      const ParkingEndpoint& endpoint,
      const std::string& title = "Parking Slot and Endpoint") {
    // Extract parking slot corners
    std::vector<double> slot_x = {slot.p0.x(), slot.p1.x(), slot.p2.x(), slot.p3.x(), slot.p0.x()};
    std::vector<double> slot_y = {slot.p0.y(), slot.p1.y(), slot.p2.y(), slot.p3.y(), slot.p0.y()};

    // Plot parking slot
    plt::plot(slot_x, slot_y, "b-");
    // Annotate corner points p0-p3 near their positions
    {
      std::vector<double> px = {slot.p0.x(), slot.p1.x(), slot.p2.x(), slot.p3.x()};
      std::vector<double> py = {slot.p0.y(), slot.p1.y(), slot.p2.y(), slot.p3.y()};
      std::vector<std::string> pnames = {"p0", "p1", "p2", "p3"};
      for (size_t i = 0; i < pnames.size(); ++i) {
        // small offset to avoid overlap with the marker
        double ox = px[i] + 0.1;
        double oy = py[i] + 0.1;
        // point marker
        plt::plot(std::vector<double>{px[i]}, std::vector<double>{py[i]}, "ko");
        // label text
        plt::text(ox, oy, pnames[i]);
      }
    }

    // Plot endpoint
    plt::plot({endpoint.position.x()}, {endpoint.position.y()}, "ro");

    // Draw vehicle orientation at endpoint
    double vehicle_length = 4.912;
    double vehicle_width = 1.872;
    double cos_yaw = std::cos(endpoint.yaw);
    double sin_yaw = std::sin(endpoint.yaw);

    // Vehicle corners
    std::vector<double> vehicle_x = {
        endpoint.position.x() - vehicle_length / 2 * cos_yaw + vehicle_width / 2 * sin_yaw,
        endpoint.position.x() + vehicle_length / 2 * cos_yaw + vehicle_width / 2 * sin_yaw,
        endpoint.position.x() + vehicle_length / 2 * cos_yaw - vehicle_width / 2 * sin_yaw,
        endpoint.position.x() - vehicle_length / 2 * cos_yaw - vehicle_width / 2 * sin_yaw,
        endpoint.position.x() - vehicle_length / 2 * cos_yaw + vehicle_width / 2 * sin_yaw};
    std::vector<double> vehicle_y = {
        endpoint.position.y() - vehicle_length / 2 * sin_yaw - vehicle_width / 2 * cos_yaw,
        endpoint.position.y() + vehicle_length / 2 * sin_yaw - vehicle_width / 2 * cos_yaw,
        endpoint.position.y() + vehicle_length / 2 * sin_yaw + vehicle_width / 2 * cos_yaw,
        endpoint.position.y() - vehicle_length / 2 * sin_yaw + vehicle_width / 2 * cos_yaw,
        endpoint.position.y() - vehicle_length / 2 * sin_yaw - vehicle_width / 2 * cos_yaw};

    plt::plot(vehicle_x, vehicle_y, "g-");

    // Draw heading arrow
    double arrow_length = 2.0;
    double arrow_x = endpoint.position.x() + arrow_length * cos_yaw;
    double arrow_y = endpoint.position.y() + arrow_length * sin_yaw;


    plt::xlabel("X (m)");
    plt::ylabel("Y (m)");
    plt::title(title);
    plt::legend();
    plt::grid(true);
    plt::axis("equal");
    plt::show();
  }
};

TEST_F(ParkingEndpointCalculatorTest, TestVerticalParkingEndpoint) {
  // Create a vertical parking slot
  ParkingSlot slot;
  slot.p0 = swift::common::math::Vec2d(0.0, 0.0);  // Left top
  slot.p1 = swift::common::math::Vec2d(2.0, 0.0);  // Right top
  slot.p2 = swift::common::math::Vec2d(2.0, 5.0);  // Right bottom
  slot.p3 = swift::common::math::Vec2d(0.0, 5.0);  // Left bottom
  slot.angle = 0.0;
  slot.width = 2.0;
  slot.type = ParkingType::VERTICAL;
  // Calculate endpoint
  ParkingEndpoint endpoint = calculator_.CalculateVerticalParkingEndpoint(slot);

  // Verify endpoint is valid
  EXPECT_TRUE(endpoint.is_valid);
  EXPECT_GT(endpoint.confidence, 0.0);

  // Verify endpoint position is reasonable
  EXPECT_NEAR(endpoint.position.x(), 1.0, 0.1);  // Should be near slot center x
  EXPECT_LT(endpoint.position.y(), 0.0);         // Should be behind the slot
  EXPECT_NEAR(endpoint.yaw, M_PI / 2.0, 0.1);    // Should be vertical heading

  // Visualize the result
  VisualizeParkingSlot(slot, endpoint, "Vertical Parking Endpoint Test");
}

/*TEST_F(ParkingEndpointCalculatorTest, TestCoordinateTransformation) {
  // Create a vehicle state (start point as origin)
  swift::common::VehicleState vehicle_state;
  vehicle_state.set_x(0);  // Start at (10, 5)
  vehicle_state.set_y(0);
  vehicle_state.set_heading(M_PI / 4);  // 45 degrees

  // Create a parking slot in world coordinates
  ParkingSlot world_slot;
  world_slot.p0 = swift::common::math::Vec2d(3.703, -2.381);  // World coordinates
  world_slot.p1 = swift::common::math::Vec2d(-1.517, -2.118);
  world_slot.p2 = swift::common::math::Vec2d(-1.562, -4.25);
  world_slot.p3 = swift::common::math::Vec2d(3.658, -4.557);
  world_slot.angle = M_PI / 2;  // 90 degrees
  world_slot.width = 2.0;
  world_slot.type = ParkingType::VERTICAL;

  // Calculate endpoint using coordinate transformation
  ParkingEndpoint endpoint = calculator_.CalculateParkingEndpoint(vehicle_state, world_slot, {}, false);

  // Verify endpoint is valid
  EXPECT_TRUE(endpoint.is_valid);
  EXPECT_GT(endpoint.confidence, 0.0);

  // Print coordinates for debugging
  std::cout << "Vehicle start: (" << vehicle_state.x() << ", " << vehicle_state.y()
            << "), heading: " << vehicle_state.heading() << std::endl;
  std::cout << "World slot corners:" << std::endl;
  std::cout << "  P0: (" << world_slot.p0.x() << ", " << world_slot.p0.y() << ")" << std::endl;
  std::cout << "  P1: (" << world_slot.p1.x() << ", " << world_slot.p1.y() << ")" << std::endl;
  std::cout << "  P2: (" << world_slot.p2.x() << ", " << world_slot.p2.y() << ")" << std::endl;
  std::cout << "  P3: (" << world_slot.p3.x() << ", " << world_slot.p3.y() << ")" << std::endl;
  std::cout << "Endpoint (ego frame): (" << endpoint.position.x() << ", " << endpoint.position.y()
            << "), yaw: " << endpoint.yaw << std::endl;

  // Visualize the result
  // VisualizeParkingSlot(world_slot, endpoint, "Coordinate Transformation Test");
}*/

/*TEST_F(ParkingEndpointCalculatorTest, TestParallelParkingEndpoint) {
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
}*/

}  // namespace rl_policy
}  // namespace open_space
}  // namespace planning
}  // namespace swift
