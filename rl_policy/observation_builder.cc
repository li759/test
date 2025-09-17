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

#include <algorithm>
#include <numeric>
#include <sstream>
#include <iostream>

namespace swift {
namespace planning {
namespace open_space {
namespace rl_policy {

SwiftObservation ObservationBuilder::BuildObservation(
    const swift::perception::base::PointDCloud& point_cloud,
    const swift::common::VehicleState& vehicle_state,
    const std::vector<swift::planning::Obstacle>& obstacles,
    const swift::common::math::Vec2d& target_position,
    double target_yaw,
    const std::shared_ptr<swift::planning::ReferenceLine>& reference_line) {
  return BuildObservationWithParams(
      point_cloud,
      vehicle_state,
      obstacles,
      target_position,
      target_yaw,
      reference_line,
      kDefaultLidarMaxRange,
      kDefaultImgViewRange);
}

SwiftObservation ObservationBuilder::BuildObservationFromObstacles(
    const swift::common::VehicleState& vehicle_state,
    const std::vector<swift::planning::Obstacle>& obstacles,
    const swift::common::math::Vec2d& target_position,
    double target_yaw,
    const std::shared_ptr<swift::planning::ReferenceLine>& reference_line) {
  // Create empty point cloud
  swift::perception::base::PointDCloud empty_point_cloud;

  return BuildObservationWithParams(
      empty_point_cloud,
      vehicle_state,
      obstacles,
      target_position,
      target_yaw,
      reference_line,
      kDefaultLidarMaxRange,
      kDefaultImgViewRange);
}

SwiftObservation ObservationBuilder::BuildObservationWithParams(
    const swift::perception::base::PointDCloud& point_cloud,
    const swift::common::VehicleState& vehicle_state,
    const std::vector<swift::planning::Obstacle>& obstacles,
    const swift::common::math::Vec2d& target_position,
    double target_yaw,
    const std::shared_ptr<swift::planning::ReferenceLine>& reference_line,
    double lidar_max_range,
    double img_view_range) {
  SwiftObservation observation;

  // Extract lidar data
  observation.lidar =
      lidar_extractor_.ExtractLidarBeams(point_cloud, vehicle_state, obstacles, lidar_max_range, kDefaultLidarBeams);

  // Extract target information
  auto target_info = target_extractor_.ExtractTargetInfo(vehicle_state, target_position, target_yaw);
  observation.target = target_extractor_.ToVector(target_info);

  // Extract occupancy grid image
  observation.img = image_extractor_.ExtractOccupancyGrid(
      vehicle_state, obstacles, kDefaultImgWidth, kDefaultImgHeight, kDefaultImgChannels, img_view_range);

  // Extract action mask
  observation.action_mask = action_mask_extractor_.ExtractActionMask(vehicle_state, obstacles, reference_line);

  // Flatten all components
  observation.Flatten();

  return observation;
}

bool ObservationBuilder::ValidateObservation(const SwiftObservation& observation) {
  // Check dimensions
  if (observation.lidar.size() != SwiftObservation::GetLidarDim()) {
    return false;
  }
  if (observation.target.size() != SwiftObservation::GetTargetDim()) {
    return false;
  }
  if (observation.img.size() != SwiftObservation::GetImgDim()) {
    return false;
  }
  if (observation.action_mask.size() != SwiftObservation::GetActionMaskDim()) {
    return false;
  }
  if (observation.flattened.size() != SwiftObservation::GetObservationDim()) {
    return false;
  }

  // Check for NaN or infinite values
  auto has_invalid_value = [](const std::vector<float>& vec) {
    return std::any_of(vec.begin(), vec.end(), [](float val) { return std::isnan(val) || std::isinf(val); });
  };

  if (has_invalid_value(observation.lidar) || has_invalid_value(observation.target) ||
      has_invalid_value(observation.img) || has_invalid_value(observation.action_mask) ||
      has_invalid_value(observation.flattened)) {
    return false;
  }

  return true;
}

std::string ObservationBuilder::GetObservationStats(const SwiftObservation& observation) {
  std::stringstream ss;

  // Lidar statistics
  auto lidar_sum = std::accumulate(observation.lidar.begin(), observation.lidar.end(), 0.0f);
  auto lidar_min = *std::min_element(observation.lidar.begin(), observation.lidar.end());
  auto lidar_max = *std::max_element(observation.lidar.begin(), observation.lidar.end());

  // Target statistics
  auto target_sum = std::accumulate(observation.target.begin(), observation.target.end(), 0.0f);

  // Image statistics
  auto img_sum = std::accumulate(observation.img.begin(), observation.img.end(), 0.0f);
  auto img_nonzero =
      std::count_if(observation.img.begin(), observation.img.end(), [](float val) { return val > 0.0f; });

  // Action mask statistics
  auto action_mask_sum = std::accumulate(observation.action_mask.begin(), observation.action_mask.end(), 0.0f);
  auto action_mask_available = std::count_if(
      observation.action_mask.begin(), observation.action_mask.end(), [](float val) { return val > 0.5f; });

  ss << "SwiftObservation Stats:\n";
  ss << "  Lidar: sum=" << lidar_sum << ", min=" << lidar_min << ", max=" << lidar_max << "\n";
  ss << "  Target: sum=" << target_sum << "\n";
  ss << "  Image: sum=" << img_sum << ", nonzero_pixels=" << img_nonzero << "\n";
  ss << "  ActionMask: sum=" << action_mask_sum << ", available_actions=" << action_mask_available << "\n";
  ss << "  Total dim: " << observation.flattened.size() << "\n";

  return ss.str();
}

SwiftObservation ObservationBuilder::BuildObservationFromParkingSlot(
    const swift::perception::base::PointDCloud& point_cloud,
    const swift::common::VehicleState& vehicle_state,
    const std::vector<swift::planning::Obstacle>& obstacles,
    const ParkingSlot& parking_slot,
    const std::shared_ptr<swift::planning::ReferenceLine>& reference_line,
    double lidar_max_range,
    double img_view_range,
    bool is_wheel_stop_valid) {
  SwiftObservation observation;
  std::cout << "BuildObservationFromParkingSlot start" << std::endl;

  // Convert Swift obstacles to ObstacleInfo format
  std::vector<ObstacleInfo> obstacle_infos;
  for (const auto& obstacle : obstacles) {
    ObstacleInfo info;
    info.position.set_x(obstacle.PerceptionBoundingBox().center().x());
    info.position.set_y(obstacle.PerceptionBoundingBox().center().y());
    info.width = obstacle.PerceptionBoundingBox().width();
    info.length = obstacle.PerceptionBoundingBox().length();
    info.yaw = obstacle.PerceptionBoundingBox().heading();
    obstacle_infos.push_back(info);
  }

  // Extract target information from parking slot
  TargetInfo target_info = target_extractor_.ExtractTargetInfoFromParkingSlot(
      vehicle_state, parking_slot, obstacle_infos, is_wheel_stop_valid);

  std::cout << "[RL] x:" << vehicle_state.x() << std::endl;
  std::cout << "[RL] x:" << vehicle_state.y() << std::endl;

  // Extract lidar data
  observation.lidar =
      lidar_extractor_.ExtractLidarBeams(point_cloud, vehicle_state, obstacles, lidar_max_range, kDefaultLidarBeams);

  // Extract occupancy grid image
  observation.img = image_extractor_.ExtractOccupancyGrid(
      vehicle_state, obstacles, kDefaultImgWidth, kDefaultImgHeight, kDefaultImgChannels, img_view_range);

  // Extract action mask
  observation.action_mask = action_mask_extractor_.ExtractActionMask(vehicle_state, obstacles, reference_line);

  // Flatten all components
  observation.Flatten();

  return observation;
}

}  // namespace rl_policy
}  // namespace open_space
}  // namespace planning
}  // namespace swift
