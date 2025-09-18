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
#include "modules/planning/open_space/rl_policy/matplotlibcpp.h"

namespace plt = matplotlibcpp;

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

  // Extract target information (HOPE-compatible 5-dimensional vector)
  observation.target = target_extractor_.ExtractTargetVector(vehicle_state, target_position, target_yaw);

  // Extract occupancy grid image
  observation.img = image_extractor_.ExtractOccupancyGrid(
      vehicle_state, obstacles, kDefaultImgWidth, kDefaultImgHeight, kDefaultImgChannels, img_view_range);

  // Extract action mask
  observation.action_mask = action_mask_extractor_.ExtractActionMaskFromLidar(observation.lidar);

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

  // Calculate parking endpoint first
  ParkingEndpoint endpoint = parking_endpoint_calculator_.CalculateParkingEndpoint(
      vehicle_state, parking_slot, obstacle_infos, is_wheel_stop_valid);

  // Extract target information from parking slot
  if (!endpoint.is_valid) {
    AERROR << "Failed to calculate parking endpoint";
    observation.target = std::vector<float>(5, 0.0f);
  } else {
    observation.target = target_extractor_.ExtractTargetVector(vehicle_state, endpoint.position, endpoint.yaw);
  }

  // Extract lidar data
  observation.lidar =
      lidar_extractor_.ExtractLidarBeams(point_cloud, vehicle_state, obstacles, lidar_max_range, kDefaultLidarBeams);

  // Extract occupancy grid image
  observation.img = image_extractor_.ExtractOccupancyGrid(
      vehicle_state, obstacles, kDefaultImgWidth, kDefaultImgHeight, kDefaultImgChannels, img_view_range);

  // Extract action mask
  observation.action_mask = action_mask_extractor_.ExtractActionMaskFromLidar(observation.lidar);

  // Visualize parking scenario
  // VisualizeParkingScenario(vehicle_state, parking_slot, obstacles, endpoint);

  // Flatten all components
  observation.Flatten();

  return observation;
}

void ObservationBuilder::VisualizeParkingScenario(
    const swift::common::VehicleState& vehicle_state,
    const ParkingSlot& parking_slot,
    const std::vector<swift::planning::Obstacle>& obstacles,
    const ParkingEndpoint& endpoint) {
  // Set up the plot
  plt::figure_size(800, 600);
  plt::clf();

  // Plot vehicle position (start point)
  std::vector<double> vehicle_x = {vehicle_state.x()};
  std::vector<double> vehicle_y = {vehicle_state.y()};
  plt::scatter(vehicle_x, vehicle_y, 100, {{"color", "blue"}, {"label", "Vehicle Start"}});

  // Plot vehicle orientation
  double vehicle_heading = vehicle_state.heading();
  double arrow_length = 2.0;
  double vehicle_end_x = vehicle_state.x() + arrow_length * std::cos(vehicle_heading);
  double vehicle_end_y = vehicle_state.y() + arrow_length * std::sin(vehicle_heading);


  // Plot parking slot corners
  std::vector<double> slot_x = {
      parking_slot.p0.x(), parking_slot.p1.x(), parking_slot.p2.x(), parking_slot.p3.x(), parking_slot.p0.x()};
  std::vector<double> slot_y = {
      parking_slot.p0.y(), parking_slot.p1.y(), parking_slot.p2.y(), parking_slot.p3.y(), parking_slot.p0.y()};
  plt::plot(slot_x, slot_y, "g_");

  // Annotate parking slot corners
  plt::annotate("P0", parking_slot.p0.x(), parking_slot.p0.y());
  plt::annotate("P1", parking_slot.p1.x(), parking_slot.p1.y());
  plt::annotate("P2", parking_slot.p2.x(), parking_slot.p2.y());
  plt::annotate("P3", parking_slot.p3.x(), parking_slot.p3.y());

  // Plot parking endpoint
  std::vector<double> endpoint_x = {endpoint.position.x()};
  std::vector<double> endpoint_y = {endpoint.position.y()};
  plt::scatter(endpoint_x, endpoint_y, 100, {{"color", "red"}, {"label", "Parking Endpoint"}});

  // Plot endpoint orientation
  double endpoint_heading = endpoint.yaw;
  double endpoint_end_x = endpoint.position.x() + arrow_length * std::cos(endpoint_heading);
  double endpoint_end_y = endpoint.position.y() + arrow_length * std::sin(endpoint_heading);

  // Plot obstacles
  for (size_t i = 0; i < obstacles.size(); ++i) {
    const auto& obstacle = obstacles[i];
    const auto& bbox = obstacle.PerceptionBoundingBox();
    const auto& center = bbox.center();
    const auto& corners = bbox.GetAllCorners();

    // Plot obstacle bounding box
    std::vector<double> obs_x, obs_y;
    for (const auto& corner : corners) {
      obs_x.push_back(corner.x());
      obs_y.push_back(corner.y());
    }
    obs_x.push_back(corners[0].x());  // Close the rectangle
    obs_y.push_back(corners[0].y());

    plt::plot(obs_x, obs_y, "b_");

    // Fill obstacle are
  }

  // Set plot properties
  plt::xlabel("X (m)");
  plt::ylabel("Y (m)");
  plt::title("Parking Scenario Visualization");
  plt::legend();
  plt::grid(true);
  plt::axis("equal");

  // Set reasonable plot limits
  double min_x = std::min({vehicle_state.x(), endpoint.position.x()});
  double max_x = std::max({vehicle_state.x(), endpoint.position.x()});
  double min_y = std::min({vehicle_state.y(), endpoint.position.y()});
  double max_y = std::max({vehicle_state.y(), endpoint.position.y()});

  // Include parking slot bounds
  for (const auto& corner : {parking_slot.p0, parking_slot.p1, parking_slot.p2, parking_slot.p3}) {
    min_x = std::min(min_x, corner.x());
    max_x = std::max(max_x, corner.x());
    min_y = std::min(min_y, corner.y());
    max_y = std::max(max_y, corner.y());
  }

  // Add some margin
  double margin = 5.0;
  plt::xlim(min_x - margin, max_x + margin);
  plt::ylim(min_y - margin, max_y + margin);

  // Save the plot
  plt::save("parking_scenario.png");
  plt::show();

  // Print debug information
  std::cout << "=== Parking Scenario Visualization ===" << std::endl;
  std::cout << "Vehicle Start: (" << vehicle_state.x() << ", " << vehicle_state.y() << "), heading: " << vehicle_heading
            << " rad" << std::endl;
  std::cout << "Parking Endpoint: (" << endpoint.position.x() << ", " << endpoint.position.y()
            << "), heading: " << endpoint_heading << " rad" << std::endl;
  std::cout << "Parking Slot: P0(" << parking_slot.p0.x() << ", " << parking_slot.p0.y() << "), "
            << "P1(" << parking_slot.p1.x() << ", " << parking_slot.p1.y() << "), "
            << "P2(" << parking_slot.p2.x() << ", " << parking_slot.p2.y() << "), "
            << "P3(" << parking_slot.p3.x() << ", " << parking_slot.p3.y() << ")" << std::endl;
  std::cout << "Number of obstacles: " << obstacles.size() << std::endl;
  std::cout << "Plot saved as: parking_scenario.png" << std::endl;
  std::cout << "=====================================" << std::endl;
}

}  // namespace rl_policy
}  // namespace open_space
}  // namespace planning
}  // namespace swift
