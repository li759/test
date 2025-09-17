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

#include "modules/planning/open_space/rl_policy/to_hope_adapter.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

namespace swift {
namespace planning {
namespace open_space {
namespace rl_policy {

std::vector<float> ToHopeAdapter::ConvertToHopeObservation(const SwiftObservation& swift_obs) {
  // Swift observation is already in the correct format (12455 dimensions)
  // Just return the flattened vector
  return swift_obs.flattened;
}

HopeVehicleState ToHopeAdapter::ConvertToHopeVehicleState(const swift::common::VehicleState& swift_state) {
  return HopeVehicleState(
      swift_state.x(),
      swift_state.y(),
      swift_state.heading(),
      swift_state.linear_velocity(),
      0.0  // Steering angle not available in VehicleState
  );
}

std::vector<HopeObstacle> ToHopeAdapter::ConvertToHopeObstacles(
    const std::vector<swift::planning::Obstacle>& swift_obstacles) {
  std::vector<HopeObstacle> hope_obstacles;
  hope_obstacles.reserve(swift_obstacles.size());

  for (const auto& swift_obs : swift_obstacles) {
    const auto& bounding_box = swift_obs.PerceptionBoundingBox();
    const auto& center = bounding_box.center();

    HopeObstacle hope_obs(
        center.x(),
        center.y(),
        bounding_box.length(),
        bounding_box.width(),
        bounding_box.heading(),
        swift_obs.IsStatic());

    hope_obstacles.push_back(hope_obs);
  }

  return hope_obstacles;
}

ToHopeAdapter::SwiftAction ToHopeAdapter::ConvertToSwiftAction(const std::vector<float>& hope_action) {
  if (hope_action.size() < 2) {
    return SwiftAction(0.0, 0.0);
  }

  // HOPE+ action format: [steering_angle, step_length]
  double steering_angle = static_cast<double>(hope_action[0]);
  double step_length = static_cast<double>(hope_action[1]);

  // Clamp steering angle to valid range
  steering_angle = std::max(-1.0, std::min(1.0, steering_angle));

  // Clamp step length to valid range
  step_length = std::max(0.1, std::min(1.0, step_length));

  return SwiftAction(steering_angle, step_length);
}

bool ToHopeAdapter::ValidateHopeObservation(const std::vector<float>& hope_obs) {
  // Check dimension
  if (hope_obs.size() != kHopeObservationDim) {
    return false;
  }

  // Check for NaN or infinite values
  auto has_invalid_value = [](const std::vector<float>& vec) {
    return std::any_of(vec.begin(), vec.end(), [](float val) { return std::isnan(val) || std::isinf(val); });
  };

  if (has_invalid_value(hope_obs)) {
    return false;
  }

  return true;
}

std::string ToHopeAdapter::GetHopeObservationStats(const std::vector<float>& hope_obs) {
  std::stringstream ss;

  if (hope_obs.size() != kHopeObservationDim) {
    ss << "Invalid HOPE observation dimension: " << hope_obs.size() << " (expected " << kHopeObservationDim << ")";
    return ss.str();
  }

  // Calculate statistics for each component
  size_t offset = 0;

  // Lidar statistics (120 dimensions)
  auto lidar_begin = hope_obs.begin() + offset;
  auto lidar_end = lidar_begin + kHopeLidarDim;
  auto lidar_sum = std::accumulate(lidar_begin, lidar_end, 0.0f);
  auto lidar_min = *std::min_element(lidar_begin, lidar_end);
  auto lidar_max = *std::max_element(lidar_begin, lidar_end);
  offset += kHopeLidarDim;

  // Target statistics (5 dimensions)
  auto target_begin = hope_obs.begin() + offset;
  auto target_end = target_begin + kHopeTargetDim;
  auto target_sum = std::accumulate(target_begin, target_end, 0.0f);
  offset += kHopeTargetDim;

  // Image statistics (12288 dimensions)
  auto img_begin = hope_obs.begin() + offset;
  auto img_end = img_begin + kHopeImgDim;
  auto img_sum = std::accumulate(img_begin, img_end, 0.0f);
  auto img_nonzero = std::count_if(img_begin, img_end, [](float val) { return val > 0.0f; });
  offset += kHopeImgDim;

  // Action mask statistics (42 dimensions)
  auto action_mask_begin = hope_obs.begin() + offset;
  auto action_mask_end = action_mask_begin + kHopeActionMaskDim;
  auto action_mask_sum = std::accumulate(action_mask_begin, action_mask_end, 0.0f);
  auto action_mask_available = std::count_if(action_mask_begin, action_mask_end, [](float val) { return val > 0.5f; });

  ss << "HOPE Observation Stats:\n";
  ss << "  Lidar: sum=" << lidar_sum << ", min=" << lidar_min << ", max=" << lidar_max << "\n";
  ss << "  Target: sum=" << target_sum << "\n";
  ss << "  Image: sum=" << img_sum << ", nonzero_pixels=" << img_nonzero << "\n";
  ss << "  ActionMask: sum=" << action_mask_sum << ", available_actions=" << action_mask_available << "\n";
  ss << "  Total dim: " << hope_obs.size() << "\n";

  return ss.str();
}

std::vector<float> ToHopeAdapter::CreateEmptyHopeObservation() {
  return std::vector<float>(kHopeObservationDim, 0.0f);
}

double ToHopeAdapter::NormalizeAngle(double angle) {
  while (angle > M_PI) {
    angle -= 2.0 * M_PI;
  }
  while (angle < -M_PI) {
    angle += 2.0 * M_PI;
  }
  return angle;
}

}  // namespace rl_policy
}  // namespace open_space
}  // namespace planning
}  // namespace swift
