/******************************************************************************
 * Copyright 2024 Desay SV Automotive Co., Ltd.
 * Copyright 2019 The Apollo Authors. All Rights Reserved.
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
 * @file
 **/

#pragma once

#include "absl/strings/str_cat.h"
#include <filesystem>
#include <memory>
#include <vector>

#include "modules/common/proto/pnc_point.pb.h"
#include "modules/common/status/status.h"
#include "modules/planning/common/trajectory/discretized_trajectory.h"
#include "modules/planning/open_space/APA_planner/NexPubSub.h"
#include "modules/planning/tasks/optimizers/open_space_trajectory_generation/open_space_trajectory_optimizer.h"
#include "modules/planning/tasks/optimizers/trajectory_optimizer.h"
#include "modules/planning/tasks/task.h"

// 在头文件包含部分添加
#include "modules/perception/base/point_cloud.h"
#include "modules/planning/open_space/rl_policy/observation_builder.h"
#include "modules/planning/open_space/rl_policy/parking_endpoint_calculator.h"
#include "modules/planning/open_space/rl_policy/to_hope_adapter.h"

namespace swift {
namespace planning {

struct OpenSpaceTrajectoryThreadData {
  std::vector<common::TrajectoryPoint> stitching_trajectory;
  std::vector<double> end_pose;
  std::vector<double> XYbounds;
  double rotate_angle;
  swift::common::math::Vec2d translate_origin;
  Eigen::MatrixXi obstacles_edges_num;
  Eigen::MatrixXd obstacles_A;
  Eigen::MatrixXd obstacles_b;
  std::vector<std::vector<common::math::Vec2d>> obstacles_vertices_vec;
};

struct ApaPlannerData {
  msg_SlotsList slots_list;
  msg_LocalMap local_map;
  msg_ApaCtrlToApp apa_ctrl_to_app;
  msg_VehicleChassis_Gear vechicle_chassis;
};

class OpenSpaceTrajectoryProvider : public TrajectoryOptimizer {
public:
  OpenSpaceTrajectoryProvider(
      const TaskConfig &config,
      const std::shared_ptr<DependencyInjector> &injector);

  ~OpenSpaceTrajectoryProvider();

  void Stop();

  void Restart();

private:
  swift::common::Status Process() override;

  void GenerateTrajectoryThread();

  bool IsVehicleNearDestination(const common::VehicleState &vehicle_state,
                                const std::vector<double> &end_pose,
                                double rotate_angle,
                                const common::math::Vec2d &translate_origin);

  bool IsVehicleStopDueToFallBack(const bool is_on_fallback,
                                  const common::VehicleState &vehicle_state);

  void GeneratePreParkStopTrajectory(DiscretizedTrajectory *trajectory_data,
                                     common::VehicleState vehicle_state);

  common::Status
  GenerateApaTrajectory(DiscretizedTrajectory *const trajectory_data,
                        common::math::Polygon2d slot_polygon,
                        std::vector<std::vector<common::math::Vec2d>> Obs_list,
                        bool is_need_replan);

  void ModifiySpeedAndCurvatureForApaTrajectory(
      const msg_ApaPlanInfo apa_plan_info_before,
      msg_ApaPlanInfo &apa_plan_info_after, std::vector<double> &acc_data);

  void InterpolateObstaclesVertices(
      std::vector<std::vector<common::math::Vec2d>> Obs_list,
      msg_LocalMap &local_map);

  void GenerateStopTrajectory(DiscretizedTrajectory *const trajectory_data);

  void
  GenerateLinearStopTrajectory(DiscretizedTrajectory *const trajectory_data);

  void
  GenerateLinearStopPoint_xysvt(DiscretizedTrajectory *const trajectory_data,
                                swift::common::TrajectoryPoint const init_point,
                                const double goal_v, const double dt);

  void LoadResult(DiscretizedTrajectory *const trajectory_data);

  void ReuseLastFrameResult(const Frame *last_frame,
                            DiscretizedTrajectory *const trajectory_data);

  void ReuseLastFrameDebug(const Frame *last_frame);

  std::vector<std::vector<common::math::Vec2d>> LoadObstacleVertices() const;
  void LoadFreeSpaceVertices(msg_LocalMap &local_map) const;
  void CSVWrite();

private:
  bool thread_init_flag_ = false;

  std::unique_ptr<OpenSpaceTrajectoryOptimizer>
      open_space_trajectory_optimizer_;

  size_t optimizer_thread_counter = 0;

  OpenSpaceTrajectoryThreadData thread_data_;
  std::future<void> task_future_;
  std::atomic<bool> is_generation_thread_stop_{false};
  std::atomic<bool> trajectory_updated_{false};
  std::atomic<bool> data_ready_{false};
  std::atomic<bool> trajectory_error_{false};
  std::atomic<bool> trajectory_skipped_{false};
  std::mutex open_space_mutex_;

  bool is_planned_ = false;
  bool is_finished_stop_path_ = false;
  std::unique_ptr<ApaPlanNode> apa_planner_;
  ApaPlannerData apa_planner_data_;

private:
  FRIEND_TEST(OpenSpaceTrajectoryProviderTest, TestInit);
  FRIEND_TEST(OpenSpaceTrajectoryProviderTest,
              TestGeneratePreParkStopTrajectory);
  FRIEND_TEST(OpenSpaceTrajectoryProviderTest, TestAPAplan);
  std::unique_ptr<IterativeAnchoringSmoother> iterative_anchoring_smoother_;
  // FRIEND_TEST(OpenSpaceTrajectoryProviderTest, TestIsParkingSpotFeasible);

  // Q
private:
  std::unique_ptr<swift::planning::open_space::rl_policy::ObservationBuilder>
      observation_builder_;
  std::unique_ptr<swift::planning::open_space::rl_policy::ToHopeAdapter>
      hope_adapter_;
  bool use_rl_policy_ = false;
};

} // namespace planning
} // namespace swift
