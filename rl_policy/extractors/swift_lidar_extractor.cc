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

#include "modules/planning/open_space/rl_policy/extractors/swift_lidar_extractor.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "modules/common/math/box2d.h"
#include "modules/common/math/vec2d.h"

namespace swift {
namespace planning {
namespace open_space {
namespace rl_policy {

std::vector<float> SwiftLidarExtractor::ExtractLidarBeams(
    const swift::perception::base::PointDCloud &point_cloud,
    const swift::common::VehicleState &vehicle_state,
    const std::vector<swift::planning::Obstacle> &obstacles, double max_range,
    int num_beams, double fov) {

  std::vector<float> lidar_beams(num_beams, max_range);

  // Try to extract from point cloud first
  if (!point_cloud.empty()) {
    auto point_cloud_beams = ExtractFromPointCloud(point_cloud, vehicle_state,
                                                   max_range, num_beams, fov);
    if (!point_cloud_beams.empty()) {
      return point_cloud_beams;
    }
  }

  // Fallback to obstacle-based ray casting
  return ExtractLidarBeamsFromObstacles(vehicle_state, obstacles, max_range,
                                        num_beams, fov);
}

std::vector<float> SwiftLidarExtractor::ExtractLidarBeamsFromObstacles(
    const swift::common::VehicleState &vehicle_state,
    const std::vector<swift::planning::Obstacle> &obstacles, double max_range,
    int num_beams, double fov) {

  std::vector<float> lidar_beams(num_beams, max_range);

  double vehicle_x = vehicle_state.x();
  double vehicle_y = vehicle_state.y();
  double vehicle_yaw = vehicle_state.heading();

  double angle_step = fov / num_beams;
  double start_angle = -fov / 2.0;

  for (int i = 0; i < num_beams; ++i) {
    double ray_angle = start_angle + i * angle_step;
    double distance = RaycastToObstacles(vehicle_x, vehicle_y, vehicle_yaw,
                                         ray_angle, obstacles, max_range);
    lidar_beams[i] = static_cast<float>(distance);
  }

  return lidar_beams;
}

double SwiftLidarExtractor::RaycastToObstacles(
    double start_x, double start_y, double yaw, double angle,
    const std::vector<swift::planning::Obstacle> &obstacles, double max_range) {

  double ray_angle = yaw + angle;
  double end_x = start_x + max_range * std::cos(ray_angle);
  double end_y = start_y + max_range * std::sin(ray_angle);

  double min_distance = max_range;

  for (const auto &obstacle : obstacles) {
    double distance =
        RayObstacleIntersection(start_x, start_y, end_x, end_y, obstacle);
    if (distance > 0 && distance < min_distance) {
      min_distance = distance;
    }
  }

  return min_distance;
}

double SwiftLidarExtractor::RayObstacleIntersection(
    double ray_start_x, double ray_start_y, double ray_end_x, double ray_end_y,
    const swift::planning::Obstacle &obstacle) {

  const auto &bounding_box = obstacle.PerceptionBoundingBox();

  // Get obstacle corners
  std::vector<swift::common::math::Vec2d> corners;
  corners.push_back(bounding_box.GetAllCorners()[0]);
  corners.push_back(bounding_box.GetAllCorners()[1]);
  corners.push_back(bounding_box.GetAllCorners()[2]);
  corners.push_back(bounding_box.GetAllCorners()[3]);

  double min_distance = -1.0;

  // Check intersection with each edge of the bounding box
  for (size_t i = 0; i < corners.size(); ++i) {
    size_t next_i = (i + 1) % corners.size();

    double edge_start_x = corners[i].x();
    double edge_start_y = corners[i].y();
    double edge_end_x = corners[next_i].x();
    double edge_end_y = corners[next_i].y();

    // Line-line intersection
    double denom = (ray_start_x - ray_end_x) * (edge_start_y - edge_end_y) -
                   (ray_start_y - ray_end_y) * (edge_start_x - edge_end_x);

    if (std::abs(denom) < 1e-10) {
      continue; // Lines are parallel
    }

    double t = ((ray_start_x - edge_start_x) * (edge_start_y - edge_end_y) -
                (ray_start_y - edge_start_y) * (edge_start_x - edge_end_x)) /
               denom;

    double u = -((ray_start_x - ray_end_x) * (ray_start_y - edge_start_y) -
                 (ray_start_y - ray_end_y) * (ray_start_x - edge_start_x)) /
               denom;

    // Check if intersection is within both line segments
    if (t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0) {
      double intersection_x = ray_start_x + t * (ray_end_x - ray_start_x);
      double intersection_y = ray_start_y + t * (ray_end_y - ray_start_y);

      double distance = std::sqrt(std::pow(intersection_x - ray_start_x, 2) +
                                  std::pow(intersection_y - ray_start_y, 2));

      if (min_distance < 0 || distance < min_distance) {
        min_distance = distance;
      }
    }
  }

  return min_distance;
}

std::vector<float> SwiftLidarExtractor::ExtractFromPointCloud(
    const swift::perception::base::PointDCloud &point_cloud,
    const swift::common::VehicleState &vehicle_state, double max_range,
    int num_beams, double fov) {

  std::vector<float> lidar_beams(num_beams, max_range);

  double vehicle_x = vehicle_state.x();
  double vehicle_y = vehicle_state.y();
  double vehicle_yaw = vehicle_state.heading();

  double angle_step = fov / num_beams;
  double start_angle = -fov / 2.0;

  // Create angle bins for each beam
  std::vector<std::vector<double>> beam_distances(num_beams);

  for (size_t i = 0; i < point_cloud.size(); ++i) {
    const auto &point = point_cloud[i];

    // Transform point to vehicle coordinate system
    double dx = point.x - vehicle_x;
    double dy = point.y - vehicle_y;

    // Rotate to vehicle frame
    double cos_yaw = std::cos(-vehicle_yaw);
    double sin_yaw = std::sin(-vehicle_yaw);
    double local_x = dx * cos_yaw - dy * sin_yaw;
    double local_y = dx * sin_yaw + dy * cos_yaw;

    // Calculate distance and angle
    double distance = std::sqrt(local_x * local_x + local_y * local_y);
    double angle = std::atan2(local_y, local_x);

    // Check if point is within range and FOV
    if (distance <= max_range && std::abs(angle) <= fov / 2.0) {
      // Find which beam this point belongs to
      int beam_index = static_cast<int>((angle - start_angle) / angle_step);
      if (beam_index >= 0 && beam_index < num_beams) {
        beam_distances[beam_index].push_back(distance);
      }
    }
  }

  // For each beam, take the minimum distance
  for (int i = 0; i < num_beams; ++i) {
    if (!beam_distances[i].empty()) {
      auto min_it =
          std::min_element(beam_distances[i].begin(), beam_distances[i].end());
      lidar_beams[i] = static_cast<float>(*min_it);
    }
  }

  return lidar_beams;
}

} // namespace rl_policy
} // namespace open_space
} // namespace planning
} // namespace swift
