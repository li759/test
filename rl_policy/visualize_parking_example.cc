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
 * @file visualize_parking_example.cc
 * @brief Example program to demonstrate parking scenario visualization
 */

#include "modules/planning/open_space/rl_policy/observation_builder.h"
#include "modules/planning/open_space/rl_policy/matplotlibcpp.h"
#include "modules/common/math/vec2d.h"
#include "modules/common/vehicle_state/proto/vehicle_state.pb.h"
#include "modules/perception/base/point_cloud.h"
#include <iostream>
#include <vector>

namespace plt = matplotlibcpp;

int main() {
  std::cout << "=== Parking Scenario Visualization Example ===" << std::endl;
  
  // Create observation builder
  swift::planning::open_space::rl_policy::ObservationBuilder builder;
  
  // Create test vehicle state
  swift::common::VehicleState vehicle_state;
  vehicle_state.set_x(0.0);
  vehicle_state.set_y(0.0);
  vehicle_state.set_heading(0.0);
  
  // Create test parking slot (vertical parking)
  swift::planning::open_space::rl_policy::ParkingSlot parking_slot;
  parking_slot.p0 = swift::common::math::Vec2d(5.0, 2.0);
  parking_slot.p1 = swift::common::math::Vec2d(5.0, -2.0);
  parking_slot.p2 = swift::common::math::Vec2d(8.0, -2.0);
  parking_slot.p3 = swift::common::math::Vec2d(8.0, 2.0);
  parking_slot.width = 4.0;
  parking_slot.angle = 0.0;
  
  // Create test obstacles
  std::vector<swift::planning::Obstacle> obstacles;
  
  // Obstacle 1: Left side
  swift::planning::Obstacle obstacle1;
  obstacle1.set_id("obs1");
  auto bbox1 = swift::common::math::Box2d(swift::common::math::Vec2d(3.0, 0.0), 0.0, 2.0, 1.0);
  obstacle1.set_perception_bounding_box(bbox1);
  obstacles.push_back(obstacle1);
  
  // Obstacle 2: Right side
  swift::planning::Obstacle obstacle2;
  obstacle2.set_id("obs2");
  auto bbox2 = swift::common::math::Box2d(swift::common::math::Vec2d(10.0, 0.0), 0.0, 2.0, 1.0);
  obstacle2.set_perception_bounding_box(bbox2);
  obstacles.push_back(obstacle2);
  
  // Obstacle 3: Behind parking slot
  swift::planning::Obstacle obstacle3;
  obstacle3.set_id("obs3");
  auto bbox3 = swift::common::math::Box2d(swift::common::math::Vec2d(6.5, 3.0), 0.0, 3.0, 1.0);
  obstacle3.set_perception_bounding_box(bbox3);
  obstacles.push_back(obstacle3);
  
  // Create empty point cloud
  swift::perception::base::PointDCloud point_cloud;
  
  // Build observation (this will trigger visualization)
  std::cout << "Building observation and generating visualization..." << std::endl;
  auto observation = builder.BuildObservationFromParkingSlot(
      point_cloud, vehicle_state, obstacles, parking_slot);
  
  std::cout << "Visualization complete! Check 'parking_scenario.png' for the plot." << std::endl;
  
  return 0;
}
