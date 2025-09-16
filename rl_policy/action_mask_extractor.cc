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
  std::vector<double> speeds;
  GetDefaultActionSpace(steering_actions, speeds);

  // 确保生成固定42维的action_mask，与HOPE保持一致
  auto action_mask = ExtractActionMaskWithCustomActions(
      vehicle_state, obstacles, steering_actions, speeds, reference_line);
  
  // 如果生成的action_mask不是42维，调整到42维
  if (action_mask.size() != kDefaultTotalActions) {
    action_mask.resize(kDefaultTotalActions, 0.0f);
  }
  
  return action_mask;
}

// ====== HOPE风格：由lidar直接生成连续mask ======
std::vector<float> ActionMaskExtractor::ExtractActionMaskFromLidar(
    const std::vector<float>& raw_lidar_obs) {
  AMEnsureInitialized();

  // 1) lidar_obs = clip(raw, 0, 10) + base
  std::vector<double> lidar_obs(kLidarNum, 0.0);
  for (int i = 0; i < kLidarNum && i < static_cast<int>(raw_lidar_obs.size()); ++i) {
    double v = std::min<double>(std::max<double>(raw_lidar_obs[i], 0.0), 10.0);
    lidar_obs[i] = v + (i < static_cast<int>(am_vehicle_lidar_base_.size()) ? am_vehicle_lidar_base_[i] : 0.0);
  }

  // 2) upsample lidar
  auto lidar_up = AMLinearInterpolate1D(lidar_obs, kUpSample);

  // 3) compare with dist_star to compute max_step per action
  std::vector<int> max_step(kNumActionsHope, kNumIter);
  for (int a = 0; a < kNumActionsHope; ++a) {
    int min_over_lidar = kNumIter;
    for (int l = 0; l < kLidarNum; ++l) {
      const auto& ds = am_dist_star_[l][a];
      auto ds_up = AMLinearInterpolate1D(ds, kUpSample);
      int first_break = kNumIter;
      for (int k = 0; k < kNumIter; ++k) {
        int idx = k;
        if (idx >= static_cast<int>(ds_up.size()) || idx >= static_cast<int>(lidar_up.size())) break;
        if (ds_up[idx] > lidar_up[idx]) { first_break = k; break; }
      }
      if (first_break < min_over_lidar) min_over_lidar = first_break;
    }
    max_step[a] = min_over_lidar;
  }

  // 4) post process
  std::vector<float> step_len(kNumActionsHope, 0.0f);
  for (int a = 0; a < kNumActionsHope; ++a) step_len[a] = static_cast<float>(max_step[a]);
  int half = kNumActionsHope / 2;
  if (half * 2 == kNumActionsHope && half > 0) {
    step_len[0] = std::max(0.0f, step_len[0] - 1.0f);
    step_len[half - 1] = std::max(0.0f, step_len[half - 1] - 1.0f);
    step_len[half + 0] = std::max(0.0f, step_len[half + 0] - 1.0f);
    step_len[kNumActionsHope - 1] = std::max(0.0f, step_len[kNumActionsHope - 1] - 1.0f);
    auto forward = std::vector<float>(step_len.begin(), step_len.begin() + half);
    auto backward = std::vector<float>(step_len.begin() + half, step_len.end());
    forward = AMMinimumFilter1DClipped(forward, 5, kNumIter);
    backward = AMMinimumFilter1DClipped(backward, 5, kNumIter);
    for (int i = 0; i < half; ++i) step_len[i] = forward[i];
    for (int i = 0; i < half; ++i) step_len[half + i] = backward[i];
  }
  for (auto& v : step_len) v = std::min<float>(1.0f, std::max<float>(0.0f, v / static_cast<float>(kNumIter)));
  float sum = 0.0f; for (auto v : step_len) sum += v;
  if (sum == 0.0f) {
    for (auto& v : step_len) v = std::min<float>(1.0f, std::max<float>(0.01f, v));
  }
  return step_len;
}

void ActionMaskExtractor::AMEnsureInitialized() {
  if (lidar_mask_initialized_) return;
  AMInitializeActionSpace();
  AMInitializeVehicleBoxes();
  AMPrecomputeVehicleLidarBase();
  AMPrecomputeDistStar();
  lidar_mask_initialized_ = true;
}

void ActionMaskExtractor::AMInitializeActionSpace() {
  am_action_space_.clear();
  am_action_space_.reserve(kNumActionsHope);
  const double step = kSteerMax / 10.0;
  for (int i = 0; i < kNumSteers; ++i) {
    double steer = kSteerMax - i * step;
    am_action_space_.push_back({steer, +kValidSpeed});
  }
  for (int i = 0; i < kNumSteers; ++i) {
    double steer = kSteerMax - i * step;
    am_action_space_.push_back({steer, -kValidSpeed});
  }
}

void ActionMaskExtractor::AMBuildRectangleCorners(double cx, double cy, double yaw,
                                      double length, double width,
                                      std::vector<std::array<double,2>>& out4) {
  const double hl = length * 0.5;
  const double hw = width * 0.5;
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  const double local[4][2] = {{+hl, +hw}, {+hl, -hw}, {-hl, -hw}, {-hl, +hw}};
  out4.resize(4);
  for (int i = 0; i < 4; ++i) {
    double x = local[i][0] * c - local[i][1] * s + cx;
    double y = local[i][0] * s + local[i][1] * c + cy;
    out4[i] = {x, y};
  }
}

void ActionMaskExtractor::AMInitializeVehicleBoxes() {
  am_vehicle_boxes_.clear();
  am_vehicle_boxes_.resize(kNumActionsHope);
  for (int a = 0; a < kNumActionsHope; ++a) {
    double steer = am_action_space_[a][0];
    double speed = am_action_space_[a][1];
    double x = 0.0, y = 0.0, yaw = 0.0;
    double radius = kWheelBase / std::tan(std::max(1e-4, std::abs(steer)));
    double turn_sign = (steer >= 0.0) ? 1.0 : -1.0;
    double delta_phi = turn_sign * 0.5 * speed / 10.0 / radius;
    std::vector<std::array<double,2>> corners;
    am_vehicle_boxes_[a].reserve(kNumIter * 4);
    for (int it = 0; it < kNumIter; ++it) {
      yaw += delta_phi;
      double Ox = -radius * std::sin(0.0);
      double Oy = +radius * std::cos(0.0);
      x = Ox + radius * std::sin(yaw);
      y = Oy - radius * std::cos(yaw);
      AMBuildRectangleCorners(x, y, yaw, kVehicleLength, kVehicleWidth, corners);
      for (const auto& p : corners) am_vehicle_boxes_[a].push_back(p);
    }
  }
}

void ActionMaskExtractor::AMPrecomputeVehicleLidarBase() {
  am_vehicle_lidar_base_.assign(kLidarNum, 0.0);
  for (int l = 0; l < kLidarNum; ++l) {
    double ang = l * 2.0 * M_PI / static_cast<double>(kLidarNum);
    double sx = 0.0, sy = 0.0;
    double ex = std::cos(ang) * kLidarRange;
    double ey = std::sin(ang) * kLidarRange;
    std::vector<std::array<double,2>> rect;
    AMBuildRectangleCorners(0.0, 0.0, 0.0, kVehicleLength, kVehicleWidth, rect);
    double min_d = kLidarRange;
    for (int i = 0; i < 4; ++i) {
      auto p1 = rect[i];
      auto p2 = rect[(i + 1) % 4];
      double d = AMRaySegmentIntersectionDistance(sx, sy, ex, ey, p1[0], p1[1], p2[0], p2[1]);
      if (d > 0.0 && d < min_d) min_d = d;
    }
    am_vehicle_lidar_base_[l] = min_d;
  }
}

double ActionMaskExtractor::AMRaySegmentIntersectionDistance(double sx, double sy,
                                                 double ex, double ey,
                                                 double x1, double y1,
                                                 double x2, double y2) {
  double r_px = sx, r_py = sy;
  double r_dx = ex - sx, r_dy = ey - sy;
  double s_px = x1, s_py = y1;
  double s_dx = x2 - x1, s_dy = y2 - y1;
  double r_mag = std::sqrt(r_dx * r_dx + r_dy * r_dy);
  double s_mag = std::sqrt(s_dx * s_dx + s_dy * s_dy);
  if (r_mag < 1e-8 || s_mag < 1e-8) return -1.0;
  double denom = r_dx * s_dy - r_dy * s_dx;
  if (std::abs(denom) < 1e-10) return -1.0;
  double t = ((s_px - r_px) * s_dy - (s_py - r_py) * s_dx) / denom;
  double u = ((s_px - r_px) * r_dy - (s_py - r_py) * r_dx) / denom;
  if (t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0) {
    double ix = r_px + t * r_dx;
    double iy = r_py + t * r_dy;
    return std::sqrt((ix - sx) * (ix - sx) + (iy - sy) * (iy - sy));
  }
  return -1.0;
}

void ActionMaskExtractor::AMPrecomputeDistStar() {
  am_dist_star_.assign(kLidarNum, std::vector<std::vector<double>>(kNumActionsHope, std::vector<double>(kNumIter, 0.0)));
  for (int l = 0; l < kLidarNum; ++l) {
    double ang = l * 2.0 * M_PI / static_cast<double>(kLidarNum);
    double sx = 0.0, sy = 0.0;
    double ex = std::cos(ang) * kLidarRange * 10.0;
    double ey = std::sin(ang) * kLidarRange * 10.0;
    for (int a = 0; a < kNumActionsHope; ++a) {
      for (int it = 0; it < kNumIter; ++it) {
        int base = it * 4;
        double min_d = 0.0;
        for (int e = 0; e < 4; ++e) {
          double x1 = am_vehicle_boxes_[a][base + e][0];
          double y1 = am_vehicle_boxes_[a][base + e][1];
          double x2 = am_vehicle_boxes_[a][base + ((e + 1) % 4)][0];
          double y2 = am_vehicle_boxes_[a][base + ((e + 1) % 4)][1];
          double d = AMRaySegmentIntersectionDistance(sx, sy, ex, ey, x1, y1, x2, y2);
          if (d > min_d) min_d = d;
        }
        am_dist_star_[l][a][it] = min_d;
      }
    }
  }
}

std::vector<double> ActionMaskExtractor::AMLinearInterpolate1D(const std::vector<double>& x,
                                                   int upsample) {
  if (upsample <= 1) return x;
  std::vector<double> y(x.size() * upsample);
  for (size_t j = 0; j < y.size(); ++j) {
    size_t i0 = j / upsample;
    size_t i1 = (i0 + 1) % x.size();
    double t = static_cast<double>(j % upsample) / static_cast<double>(upsample);
    y[j] = x[i0] * (1.0 - t) + x[i1] * t;
  }
  return y;
}

std::vector<float> ActionMaskExtractor::AMMinimumFilter1DClipped(const std::vector<float>& in,
                                                     int kernel, int clip_max) {
  const int n = static_cast<int>(in.size());
  std::vector<float> out(n);
  int half = std::max(1, kernel);
  for (int i = 0; i < n; ++i) {
    float mn = in[i];
    for (int k = std::max(0, i - half + 1); k <= std::min(n - 1, i + half - 1); ++k) mn = std::min(mn, in[k]);
    out[i] = std::min<float>(mn, static_cast<float>(clip_max));
  }
  return out;
}

std::vector<float> ActionMaskExtractor::ExtractActionMaskWithCustomActions(
    const swift::common::VehicleState &vehicle_state,
    const std::vector<swift::planning::Obstacle> &obstacles,
    const std::vector<double> &steering_actions,
    const std::vector<double> &speeds,
    const std::shared_ptr<swift::planning::ReferenceLine> &reference_line) {

  std::vector<float> action_mask;
  action_mask.reserve(steering_actions.size() * speeds.size());

  for (double steering : steering_actions) {
    for (double signed_speed : speeds) {
      double score = ComputeActionScore(vehicle_state, obstacles, steering,
                                        signed_speed, reference_line);
      action_mask.push_back(static_cast<float>(score));
    }
  }

  // 确保返回的action_mask是42维，与HOPE保持一致
  if (action_mask.size() != kDefaultTotalActions) {
    action_mask.resize(kDefaultTotalActions, 0.0f);
  }

  return action_mask;
}

double ActionMaskExtractor::ComputeActionScore(
    const swift::common::VehicleState &vehicle_state,
    const std::vector<swift::planning::Obstacle> &obstacles,
    double steering_angle, double signed_speed,
    const std::shared_ptr<swift::planning::ReferenceLine> &reference_line) {

  // 基于最近障碍物距离的软评分：d>=margin_max -> 1.0；d<=0 -> 0.0；线性插值
  const double margin_max = 3.0;  // 安全裕度上限（米）

  // 预测一步后的车辆包围盒
  auto predicted_state = PredictVehicleState(vehicle_state, steering_angle, signed_speed);
  swift::common::math::Vec2d predicted_position(predicted_state.x(), predicted_state.y());
  swift::common::math::Box2d vehicle_box(predicted_position, predicted_state.heading(), kVehicleLength, kVehicleWidth);

  // 计算与所有障碍的最近距离（负值视为重叠，直接返回0）
  double min_dist = margin_max;
  for (const auto &ob : obstacles) {
    const auto &ob_box = ob.PerceptionBoundingBox();
    if (vehicle_box.HasOverlap(ob_box)) {
      return 0.0;  // 碰撞 -> 0
    }
    // 近似：用盒中心距离减去半尺寸之和（简化替代精确最近距离）
    const auto vc = vehicle_box.center();
    const auto oc = ob_box.center();
    double dx = vc.x() - oc.x();
    double dy = vc.y() - oc.y();
    double center_dist = std::sqrt(dx * dx + dy * dy);
    double approx_clearance = center_dist - std::max(kVehicleLength, kVehicleWidth);
    min_dist = std::min(min_dist, approx_clearance);
  }

  // 参考线约束（占位：如需严格可扩展）
  if (reference_line) {
    // 可在此处加入对越界程度的扣分逻辑
  }

  // 归一化到[0,1]
  double score = std::clamp(min_dist / margin_max, 0.0, 1.0);
  return score;
}

std::vector<float>
ActionMaskExtractor::CreateDefaultActionMask(int num_actions) {
  // 默认使用42维action_mask，与HOPE保持一致
  int actual_num_actions = (num_actions == 42) ? num_actions : kDefaultTotalActions;
  return std::vector<float>(actual_num_actions, 1.0f);
}

bool ActionMaskExtractor::IsActionValid(
    const swift::common::VehicleState &vehicle_state,
    const std::vector<swift::planning::Obstacle> &obstacles,
    double steering_angle, double signed_speed,
    const std::shared_ptr<swift::planning::ReferenceLine> &reference_line) {

  // Check collision
  if (!CheckCollision(vehicle_state, obstacles, steering_angle, signed_speed)) {
    return false;
  }

  // Check reference line bounds if available
  if (reference_line) {
    auto predicted_state =
        PredictVehicleState(vehicle_state, steering_angle, signed_speed);
    if (!CheckReferenceLineBounds(predicted_state, reference_line)) {
      return false;
    }
  }

  return true;
}

bool ActionMaskExtractor::CheckCollision(
    const swift::common::VehicleState &vehicle_state,
    const std::vector<swift::planning::Obstacle> &obstacles,
    double steering_angle, double signed_speed) {

  // Predict vehicle state after action
  auto predicted_state =
      PredictVehicleState(vehicle_state, steering_angle, signed_speed);

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
    double signed_speed) {

  // Simple bicycle model prediction
  double current_x = vehicle_state.x();
  double current_y = vehicle_state.y();
  double current_yaw = vehicle_state.heading();
  // 使用有符号速度进行预测（与HOPE一致）
  double current_speed = signed_speed;

  // Calculate wheelbase (distance between front and rear axles)
  double wheelbase = 2.8; // Typical wheelbase in meters

  // Calculate new position using bicycle model
  // 统一在单位时间步内推进，signed_speed控制前后
  double dt = 1.0;  // normalized step
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
    std::vector<double> &steering_actions, std::vector<double> &speeds) {

  // 生成21个转角：从 +kHopeValidSteerMax 到 -kHopeValidSteerMax，步长为 /kHopePrecision
  steering_actions.clear();
  steering_actions.reserve(kDefaultNumSteeringActions);
  const double steer_step = kHopeValidSteerMax / static_cast<double>(kHopePrecision);
  for (int i = 0; i < kDefaultNumSteeringActions; ++i) {
    double steering = kHopeValidSteerMax - static_cast<double>(i) * steer_step;
    steering_actions.push_back(steering);
  }

  // 生成两种速度：+1.0 与 -1.0
  speeds.clear();
  speeds.reserve(kDefaultNumSpeeds);
  speeds.push_back(+kHopeValidSpeed);
  speeds.push_back(-kHopeValidSpeed);
}

} // namespace rl_policy
} // namespace open_space
} // namespace planning
} // namespace swift
