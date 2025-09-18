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

#include "modules/planning/open_space/rl_policy/observation_builder.h"

#include <gtest/gtest.h>
#include <memory>

#include "modules/common/math/vec2d.h"
#include "modules/common/vehicle_state/vehicle_state_provider.h"
#include "modules/perception/base/point_cloud.h"
#include "modules/planning/common/obstacle.h"

namespace swift {
namespace planning {
namespace open_space {
namespace rl_policy {

class ObservationBuilderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create test vehicle state
    vehicle_state_ = std::make_unique<swift::common::VehicleState>();
    // Note: VehicleState doesn't have public setters, so we'll use default
    // values

    // Create test obstacles
    CreateTestObstacles();

    // Create test point cloud
    CreateTestPointCloud();

    // Set target position
    target_position_ = swift::common::math::Vec2d(10.0, 5.0);
    target_yaw_ = M_PI / 2.0;
  }

  void CreateTestObstacles() {
    // Create a simple rectangular obstacle
    swift::perception::PerceptionObstacle perception_obs;
    perception_obs.set_id(1);
    perception_obs.set_type(swift::perception::PerceptionObstacle::VEHICLE);
    perception_obs.mutable_position()->set_x(5.0);
    perception_obs.mutable_position()->set_y(2.0);
    perception_obs.mutable_position()->set_z(0.0);
    perception_obs.set_length(4.0);
    perception_obs.set_width(2.0);
    perception_obs.set_height(1.5);
    perception_obs.set_theta(0.0);
    perception_obs.mutable_velocity()->set_x(0.0);
    perception_obs.mutable_velocity()->set_y(0.0);
    perception_obs.mutable_velocity()->set_z(0.0);

    auto obstacle = std::make_unique<swift::planning::Obstacle>(
        "test_obstacle", perception_obs, swift::prediction::ObstaclePriority::NORMAL, true);

    obstacles_.push_back(std::move(obstacle));
  }

  void CreateTestPointCloud() {
    // Create a simple point cloud with a few points
    point_cloud_ = std::make_unique<swift::perception::base::PointDCloud>();

    // Add some test points
    swift::perception::base::PointD point1;
    point1.x = 3.0;
    point1.y = 1.0;
    point1.z = 0.0;
    point_cloud_->push_back(point1);

    swift::perception::base::PointD point2;
    point2.x = 7.0;
    point2.y = 3.0;
    point2.z = 0.0;
    point_cloud_->push_back(point2);
  }

  std::unique_ptr<swift::common::VehicleState> vehicle_state_;
  std::vector<std::unique_ptr<swift::planning::Obstacle>> obstacles_;
  std::unique_ptr<swift::perception::base::PointDCloud> point_cloud_;
  swift::common::math::Vec2d target_position_;
  double target_yaw_;
};

TEST_F(ObservationBuilderTest, BuildObservationFromObstacles) {
  ObservationBuilder builder;

  // Convert obstacles to const references
  std::vector<swift::planning::Obstacle> const_obstacles;
  for (const auto& obs : obstacles_) {
    const_obstacles.push_back(*obs);
  }

  SwiftObservation observation =
      builder.BuildObservationFromObstacles(*vehicle_state_, const_obstacles, target_position_, target_yaw_);

  // Validate observation
  EXPECT_TRUE(builder.ValidateObservation(observation));

  // Check dimensions
  EXPECT_EQ(observation.lidar.size(), SwiftObservation::GetLidarDim());
  EXPECT_EQ(observation.target.size(), SwiftObservation::GetTargetDim());
  EXPECT_EQ(observation.img.size(), SwiftObservation::GetImgDim());
  EXPECT_EQ(observation.action_mask.size(), SwiftObservation::GetActionMaskDim());
  EXPECT_EQ(observation.flattened.size(), SwiftObservation::GetObservationDim());

  // Check that flattened observation is correct
  observation.Flatten();
  EXPECT_EQ(observation.flattened.size(), SwiftObservation::GetObservationDim());
}

TEST_F(ObservationBuilderTest, BuildObservationWithPointCloud) {
  ObservationBuilder builder;

  // Convert obstacles to const references
  std::vector<swift::planning::Obstacle> const_obstacles;
  for (const auto& obs : obstacles_) {
    const_obstacles.push_back(*obs);
  }

  SwiftObservation observation =
      builder.BuildObservation(*point_cloud_, *vehicle_state_, const_obstacles, target_position_, target_yaw_);

  // Validate observation
  EXPECT_TRUE(builder.ValidateObservation(observation));

  // Check dimensions
  EXPECT_EQ(observation.lidar.size(), SwiftObservation::GetLidarDim());
  EXPECT_EQ(observation.target.size(), SwiftObservation::GetTargetDim());
  EXPECT_EQ(observation.img.size(), SwiftObservation::GetImgDim());
  EXPECT_EQ(observation.action_mask.size(), SwiftObservation::GetActionMaskDim());
  EXPECT_EQ(observation.flattened.size(), SwiftObservation::GetObservationDim());
}

TEST_F(ObservationBuilderTest, BuildObservationWithParams) {
  ObservationBuilder builder;

  // Convert obstacles to const references
  std::vector<swift::planning::Obstacle> const_obstacles;
  for (const auto& obs : obstacles_) {
    const_obstacles.push_back(*obs);
  }

  SwiftObservation observation = builder.BuildObservationWithParams(
      *point_cloud_,
      *vehicle_state_,
      const_obstacles,
      target_position_,
      target_yaw_,
      nullptr,
      15.0,
      25.0);  // Custom lidar range and image view range

  // Validate observation
  EXPECT_TRUE(builder.ValidateObservation(observation));

  // Check dimensions
  EXPECT_EQ(observation.lidar.size(), SwiftObservation::GetLidarDim());
  EXPECT_EQ(observation.target.size(), SwiftObservation::GetTargetDim());
  EXPECT_EQ(observation.img.size(), SwiftObservation::GetImgDim());
  EXPECT_EQ(observation.action_mask.size(), SwiftObservation::GetActionMaskDim());
  EXPECT_EQ(observation.flattened.size(), SwiftObservation::GetObservationDim());
}

TEST_F(ObservationBuilderTest, GetObservationStats) {
  ObservationBuilder builder;

  // Convert obstacles to const references
  std::vector<swift::planning::Obstacle> const_obstacles;
  for (const auto& obs : obstacles_) {
    const_obstacles.push_back(*obs);
  }

  SwiftObservation observation =
      builder.BuildObservationFromObstacles(*vehicle_state_, const_obstacles, target_position_, target_yaw_);

  std::string stats = builder.GetObservationStats(observation);

  // Check that stats string is not empty
  EXPECT_FALSE(stats.empty());

  // Check that stats contain expected keywords
  EXPECT_TRUE(stats.find("Lidar") != std::string::npos);
  EXPECT_TRUE(stats.find("Target") != std::string::npos);
  EXPECT_TRUE(stats.find("Image") != std::string::npos);
  EXPECT_TRUE(stats.find("ActionMask") != std::string::npos);
}

TEST_F(ObservationBuilderTest, ValidateObservation) {
  ObservationBuilder builder;

  // Test with valid observation
  SwiftObservation valid_observation;
  valid_observation.lidar.resize(SwiftObservation::GetLidarDim(), 1.0f);
  valid_observation.target.resize(SwiftObservation::GetTargetDim(), 0.5f);
  valid_observation.img.resize(SwiftObservation::GetImgDim(), 0.0f);
  valid_observation.action_mask.resize(SwiftObservation::GetActionMaskDim(), 1.0f);
  valid_observation.Flatten();

  EXPECT_TRUE(builder.ValidateObservation(valid_observation));

  // Test with invalid dimensions
  SwiftObservation invalid_observation;
  invalid_observation.lidar.resize(50, 1.0f);  // Wrong size
  invalid_observation.target.resize(SwiftObservation::GetTargetDim(), 0.5f);
  invalid_observation.img.resize(SwiftObservation::GetImgDim(), 0.0f);
  invalid_observation.action_mask.resize(SwiftObservation::GetActionMaskDim(), 1.0f);
  invalid_observation.Flatten();

  EXPECT_FALSE(builder.ValidateObservation(invalid_observation));
}

TEST_F(ObservationBuilderTest, TestParkingScenarioVisualization) {
  // Create test vehicle state
  swift::common::VehicleState vehicle_state;
  vehicle_state.set_x(0.0);
  vehicle_state.set_y(0.0);
  vehicle_state.set_heading(0.0);

  // Create test parking slot
  ParkingSlot parking_slot;
  parking_slot.p0 = swift::common::math::Vec2d(5.0, 2.0);
  parking_slot.p1 = swift::common::math::Vec2d(5.0, -2.0);
  parking_slot.p2 = swift::common::math::Vec2d(8.0, -2.0);
  parking_slot.p3 = swift::common::math::Vec2d(8.0, 2.0);
  parking_slot.width = 4.0;
  parking_slot.angle = 0.0;

  // Create test obstacles
  std::vector<swift::planning::Obstacle> obstacles;

  // Create a simple obstacle
  swift::planning::Obstacle obstacle1;
  obstacle1.SetId("obs1");
  auto bbox1 = swift::common::math::Box2d(swift::common::math::Vec2d(3.0, 0.0), 0.0, 2.0, 1.0);
  obstacle1.MutablePerceptionBoundingBox() = bbox1;
  obstacles.push_back(obstacle1);

  // Create empty point cloud
  swift::perception::base::PointDCloud point_cloud;
  ObservationBuilder builder;

  // Build observation (this will trigger visualization)
  auto observation = builder.BuildObservationFromParkingSlot(point_cloud, vehicle_state, obstacles, parking_slot);

  // Verify observation is valid
  EXPECT_TRUE(builder.ValidateObservation(observation));

  // Check that the observation has correct dimensions
  EXPECT_EQ(observation.lidar.size(), SwiftObservation::GetLidarDim());
  EXPECT_EQ(observation.target.size(), SwiftObservation::GetTargetDim());
  EXPECT_EQ(observation.img.size(), SwiftObservation::GetImgDim());
  EXPECT_EQ(observation.action_mask.size(), SwiftObservation::GetActionMaskDim());
}

}  // namespace rl_policy
}  // namespace open_space
}  // namespace planning
}  // namespace swift
