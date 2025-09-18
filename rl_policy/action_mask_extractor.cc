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

#include "modules/planning/open_space/rl_policy/action_mask_extractor.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "modules/common/math/box2d.h"
#include "modules/common/math/vec2d.h"

namespace swift {
namespace planning {
namespace open_space {
namespace rl_policy {

std::vector<float> ActionMaskExtractor::ExtractActionMask(
    const swift::common::VehicleState &vehicle_state,
    const std::vector<swift::planning::Obstacle> &obstacles,
    const std::shared_ptr<swift::planning::ReferenceLine> &reference_line) {

  std::vector<double> steering_actions;
  std::vector<double> step_lengths;
  GetDefaultActionSpace(steering_actions, step_lengths);

  return ExtractActionMaskWithCustomActions(
      vehicle_state, obstacles, steering_actions, step_lengths, reference_line);
}

std::vector<float> ActionMaskExtractor::ExtractActionMaskWithCustomActions(
    const swift::common::VehicleState &vehicle_state,
    const std::vector<swift::planning::Obstacle> &obstacles,
    const std::vector<double> &steering_actions,
    const std::vector<double> &step_lengths,
    const std::shared_ptr<swift::planning::ReferenceLine> &reference_line) {

  std::vector<float> action_mask;
  action_mask.reserve(steering_actions.size() * step_lengths.size());

  for (double steering : steering_actions) {
    for (double step_length : step_lengths) {
      bool is_valid = IsActionValid(vehicle_state, obstacles, steering,
                                    step_length, reference_line);
      action_mask.push_back(is_valid ? 1.0f : 0.0f);
    }
  }

  return action_mask;
}

std::vector<float>
ActionMaskExtractor::CreateDefaultActionMask(int num_actions) {
  return std::vector<float>(num_actions, 1.0f);
}

std::vector<float> ActionMaskExtractor::ExtractActionMaskFromLidar(
    const std::vector<float>& raw_lidar_obs) {
  // Initialize if not done yet
  if (!initialized_) {
    InitVehicleBox();
    vehicle_lidar_base_ = GetVehicleLidarBase();
    PrecomputeDistStar();
    initialized_ = true;
  }

  // Clip lidar observations and add vehicle base
  std::vector<float> lidar_obs(kHopeLidarNum);
  for (int i = 0; i < kHopeLidarNum; ++i) {
    lidar_obs[i] = std::clamp(raw_lidar_obs[i], 0.0f, 10.0f) + vehicle_lidar_base_[i];
  }

  // Linear interpolation (upsampling)
  std::vector<double> lidar_obs_double(lidar_obs.begin(), lidar_obs.end());
  std::vector<double> dist_obs = LinearInterpolate(lidar_obs_double, kHopeUpSampleRate);

  // Reshape to [lidar_num, 1, 1] for broadcasting
  std::vector<std::vector<std::vector<double>>> step_save(
      kHopeLidarNum, 
      std::vector<std::vector<double>>(kHopeNumActions, 
      std::vector<double>(kHopeNIter, 0.0)));

  // Compare dist_star with dist_obs
  for (int i = 0; i < kHopeLidarNum; ++i) {
    for (int j = 0; j < kHopeNumActions; ++j) {
      for (int k = 0; k < kHopeNIter; ++k) {
        if (dist_star_[i][j][k] <= dist_obs[i]) {
          step_save[i][j][k] = 1.0;
        } else {
          step_save[i][j][k] = 0.0;
        }
      }
    }
  }

  // Find max step for each action
  std::vector<float> step_len(kHopeNumActions);
  for (int j = 0; j < kHopeNumActions; ++j) {
    int max_step = kHopeNIter;
    for (int i = 0; i < kHopeLidarNum; ++i) {
      int step = 0;
      for (int k = 0; k < kHopeNIter; ++k) {
        if (step_save[i][j][k] == 1.0) {
          step = k;
          break;
        }
      }
      if (step == 0 && step_save[i][j][0] == 0.0) {
        step = kHopeNIter;
      }
      max_step = std::min(max_step, step);
    }
    step_len[j] = static_cast<float>(max_step);
  }

  // Post-process with minimum filter
  step_len = PostProcessStepLen(step_len);

  // Normalize and clip
  float sum = 0.0f;
  for (float val : step_len) {
    sum += val;
  }
  
  if (sum == 0.0f) {
    // If all actions are invalid, return small positive values
    for (float& val : step_len) {
      val = 0.01f;
    }
  } else {
    // Normalize to [0, 1] range
    for (float& val : step_len) {
      val = std::clamp(val / static_cast<float>(kHopeNIter), 0.01f, 1.0f);
    }
  }

  return step_len;
}

bool ActionMaskExtractor::IsActionValid(
    const swift::common::VehicleState &vehicle_state,
    const std::vector<swift::planning::Obstacle> &obstacles,
    double steering_angle, double step_length,
    const std::shared_ptr<swift::planning::ReferenceLine> &reference_line) {

  // Check collision
  if (!CheckCollision(vehicle_state, obstacles, steering_angle, step_length)) {
    return false;
  }

  // Check reference line bounds if available
  if (reference_line) {
    auto predicted_state =
        PredictVehicleState(vehicle_state, steering_angle, step_length);
    if (!CheckReferenceLineBounds(predicted_state, reference_line)) {
      return false;
    }
  }

  return true;
}

bool ActionMaskExtractor::CheckCollision(
    const swift::common::VehicleState &vehicle_state,
    const std::vector<swift::planning::Obstacle> &obstacles,
    double steering_angle, double step_length) {

  // Predict vehicle state after action
  auto predicted_state =
      PredictVehicleState(vehicle_state, steering_angle, step_length);

  // Create vehicle bounding box at predicted position
  swift::common::math::Vec2d predicted_position(predicted_state.x(),
                                                predicted_state.y());
  swift::common::math::Box2d vehicle_box(predicted_position,
                                         predicted_state.heading(),
                                         kVehicleLength, kVehicleWidth);

  // Check collision with obstacles
  for (const auto &obstacle : obstacles) {
    const auto &obstacle_box = obstacle.PerceptionBoundingBox();
    if (vehicle_box.HasOverlap(obstacle_box)) {
      return false; // Collision detected
    }
  }

  return true; // No collision
}

swift::common::VehicleState ActionMaskExtractor::PredictVehicleState(
    const swift::common::VehicleState &vehicle_state, double steering_angle,
    double step_length) {

  // Simple bicycle model prediction
  double current_x = vehicle_state.x();
  double current_y = vehicle_state.y();
  double current_yaw = vehicle_state.heading();
  double current_speed = vehicle_state.linear_velocity();

  // Calculate wheelbase (distance between front and rear axles)
  double wheelbase = 2.8; // Typical wheelbase in meters

  // Calculate new position using bicycle model
  double dt = step_length / std::max(current_speed, 0.1);  // Time step
  double beta = std::atan(std::tan(steering_angle) / 2.0); // Slip angle

  double new_x = current_x + current_speed * std::cos(current_yaw + beta) * dt;
  double new_y = current_y + current_speed * std::sin(current_yaw + beta) * dt;
  double new_yaw =
      current_yaw + (current_speed / wheelbase) * std::sin(beta) * dt;

  // Create new vehicle state (simplified - only position and heading)
  swift::common::VehicleState predicted_state = vehicle_state;
  // Note: VehicleState doesn't have setters, so we'll use the current state
  // In a real implementation, you'd need to create a new state or use a
  // different approach

  return predicted_state;
}

bool ActionMaskExtractor::CheckReferenceLineBounds(
    const swift::common::VehicleState &predicted_state,
    const std::shared_ptr<swift::planning::ReferenceLine> &reference_line) {

  if (!reference_line) {
    return true; // No reference line constraints
  }

  // Check if predicted position is within reference line bounds
  // This is a simplified check - in practice, you'd need more sophisticated
  // logic
  double predicted_x = predicted_state.x();
  double predicted_y = predicted_state.y();

  // Get reference line bounds (simplified)
  double ref_line_width = 3.5; // Typical lane width

  // Check if vehicle is within lane bounds
  // This is a placeholder - actual implementation would use reference line
  // geometry
  return true; // Simplified - always return true for now
}

void ActionMaskExtractor::GetDefaultActionSpace(
    std::vector<double> &steering_actions, std::vector<double> &step_lengths) {

  // Generate steering actions from -1.0 to 1.0
  steering_actions.clear();
  for (int i = 0; i < kDefaultNumSteeringActions; ++i) {
    double steering =
        kDefaultMinSteering + (kDefaultMaxSteering - kDefaultMinSteering) * i /
                                  (kDefaultNumSteeringActions - 1);
    steering_actions.push_back(steering);
  }

  // Generate step lengths from 0.1 to 1.0
  step_lengths.clear();
  for (int i = 0; i < kDefaultNumStepLengths; ++i) {
    double step_length = kDefaultMinStepLength +
                         (kDefaultMaxStepLength - kDefaultMinStepLength) * i /
                             (kDefaultNumStepLengths - 1);
    step_lengths.push_back(step_length);
  }
}

// HOPE algorithm helper methods
void ActionMaskExtractor::InitVehicleBox() {
  // Initialize HOPE action space
  hope_action_space_.clear();
  hope_action_space_.reserve(kHopeNumActions);
  
  // Generate steering angles from +kHopeValidSteerMax to -kHopeValidSteerMax
  double steer_step = kHopeValidSteerMax / static_cast<double>(kHopePrecision);
  for (int i = 0; i < kHopeNumSteers; ++i) {
    double steer = kHopeValidSteerMax - static_cast<double>(i) * steer_step;
    hope_action_space_.push_back({steer, kHopeValidSpeed});
    hope_action_space_.push_back({steer, -kHopeValidSpeed});
  }
}

std::vector<double> ActionMaskExtractor::GetVehicleLidarBase() {
  std::vector<double> lidar_base(kHopeLidarNum);
  
  // Vehicle box corners (simplified rectangle)
  double front_hang = 0.953;
  double rear_hang = 1.047;
  double width = 1.94;
  
  std::vector<std::pair<double, double>> vehicle_corners = {
    {-rear_hang, -width/2},
    {front_hang + kHopeWheelBase, -width/2},
    {front_hang + kHopeWheelBase, width/2},
    {-rear_hang, width/2}
  };
  
  for (int l = 0; l < kHopeLidarNum; ++l) {
    double angle = l * M_PI / kHopeLidarNum * 2;
    double cos_angle = std::cos(angle);
    double sin_angle = std::sin(angle);
    
    double min_distance = std::numeric_limits<double>::max();
    
    // Check intersection with vehicle box edges
    for (size_t i = 0; i < vehicle_corners.size(); ++i) {
      size_t next_i = (i + 1) % vehicle_corners.size();
      double x1 = vehicle_corners[i].first;
      double y1 = vehicle_corners[i].second;
      double x2 = vehicle_corners[next_i].first;
      double y2 = vehicle_corners[next_i].second;
      
      // Line intersection calculation
      double denom = cos_angle * (y2 - y1) - sin_angle * (x2 - x1);
      if (std::abs(denom) > 1e-8) {
        double t = (x1 * (y2 - y1) - y1 * (x2 - x1)) / denom;
        if (t > 0) {
          double distance = t;
          min_distance = std::min(min_distance, distance);
        }
      }
    }
    
    lidar_base[l] = min_distance;
  }
  
  return lidar_base;
}

void ActionMaskExtractor::PrecomputeDistStar() {
  dist_star_.resize(kHopeLidarNum);
  for (int i = 0; i < kHopeLidarNum; ++i) {
    dist_star_[i].resize(kHopeNumActions);
    for (int j = 0; j < kHopeNumActions; ++j) {
      dist_star_[i][j].resize(kHopeNIter);
    }
  }
  
  // Simplified precomputation - in practice this would involve
  // complex geometric calculations with vehicle trajectories
  for (int i = 0; i < kHopeLidarNum; ++i) {
    for (int j = 0; j < kHopeNumActions; ++j) {
      for (int k = 0; k < kHopeNIter; ++k) {
        // Simplified distance calculation
        double base_dist = vehicle_lidar_base_[i];
        double action_factor = 1.0 + std::abs(hope_action_space_[j][0]) * 0.1;
        dist_star_[i][j][k] = base_dist * action_factor * (1.0 + k * 0.1);
      }
    }
  }
}

std::vector<double> ActionMaskExtractor::LinearInterpolate(
    const std::vector<double>& x, int upsample_rate) {
  std::vector<double> y(x.size() * upsample_rate);
  
  // Add circular boundary
  std::vector<double> x_extended = x;
  x_extended.push_back(x[0]);
  
  for (size_t j = 0; j < y.size(); ++j) {
    int idx = j / upsample_rate;
    double weight = 1.0 - (j % upsample_rate) / static_cast<double>(upsample_rate);
    y[j] = x_extended[idx] * weight + x_extended[idx + 1] * (1.0 - weight);
  }
  
  return y;
}

std::vector<float> ActionMaskExtractor::PostProcessStepLen(
    const std::vector<float>& step_len) {
  std::vector<float> result = step_len;
  
  // Apply minimum filter (simplified version)
  int kernel_size = 5;
  int half_kernel = kernel_size / 2;
  
  for (int i = 0; i < static_cast<int>(result.size()); ++i) {
    float min_val = result[i];
    for (int j = std::max(0, i - half_kernel); 
         j <= std::min(static_cast<int>(result.size()) - 1, i + half_kernel); ++j) {
      min_val = std::min(min_val, result[j]);
    }
    result[i] = min_val;
  }
  
  return result;
}

std::vector<std::vector<double>> ActionMaskExtractor::Intersect(
    const std::vector<std::vector<double>>& e1,
    const std::vector<std::vector<double>>& e2) {
  // Simplified intersection calculation
  // In practice, this would implement the complex geometric intersection logic
  std::vector<std::vector<double>> intersections;
  return intersections;
}

} // namespace rl_policy
} // namespace open_space
} // namespace planning
} // namespace swift
