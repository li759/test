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

#include "modules/planning/tasks/optimizers/open_space_trajectory_generation/open_space_trajectory_provider.h"
#include "modules/planning/tasks/optimizers/open_space_trajectory_generation/apa_path_smoother.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>

// #include "core/task/task.h"
#include "modules/common/vehicle_state/proto/vehicle_state.pb.h"
#include "modules/planning/common/planning_context.h"
#include "modules/planning/common/planning_gflags.h"
#include "modules/planning/common/trajectory/publishable_trajectory.h"
#include "modules/planning/common/trajectory_stitcher.h"

namespace swift {
namespace planning {

using swift::common::ErrorCode;
using swift::common::Status;
using swift::common::TrajectoryPoint;
using swift::common::math::NormalizeAngle;
using swift::common::math::Polygon2d;
using swift::common::math::Vec2d;
using swift::core::Clock;

OpenSpaceTrajectoryProvider::OpenSpaceTrajectoryProvider(
    const TaskConfig &config,
    const std::shared_ptr<DependencyInjector> &injector)
    : TrajectoryOptimizer(config, injector) {
  open_space_trajectory_optimizer_.reset(new OpenSpaceTrajectoryOptimizer(
      config.open_space_trajectory_provider_config()
          .open_space_trajectory_optimizer_config()));
  apa_planner_ = std::make_unique<ApaPlanNode>();

  // Q
  observation_builder_ = std::make_unique<
      swift::planning::open_space::rl_policy::ObservationBuilder>();
  // hope_adapter_ =
  //      std::make_unique<swift::planning::open_space::rl_policy::ToHopeAdapter>();

  use_rl_policy_ =
      config.open_space_trajectory_provider_config().enable_rl_policy();
}

OpenSpaceTrajectoryProvider::~OpenSpaceTrajectoryProvider() {
  if (FLAGS_enable_open_space_planner_thread) {
    Stop();
  }
}

void OpenSpaceTrajectoryProvider::Stop() {
  if (FLAGS_enable_open_space_planner_thread) {
    is_generation_thread_stop_.store(true);
    if (thread_init_flag_) {
      task_future_.get();
    }
    trajectory_updated_.store(false);
    trajectory_error_.store(false);
    trajectory_skipped_.store(false);
    optimizer_thread_counter = 0;
  }
}

void OpenSpaceTrajectoryProvider::Restart() {
  if (FLAGS_enable_open_space_planner_thread) {
    is_generation_thread_stop_.store(true);
    if (thread_init_flag_) {
      task_future_.get();
    }
    is_generation_thread_stop_.store(false);
    thread_init_flag_ = false;
    trajectory_updated_.store(false);
    trajectory_error_.store(false);
    trajectory_skipped_.store(false);
    optimizer_thread_counter = 0;
  }
}

void OpenSpaceTrajectoryProvider::CSVWrite() {
  std::ofstream Trajectory_data;
  std::ofstream ParkPlace_data;
  std::ofstream Freespace_data;
  std::ofstream Localization_data;

  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);
  // 创建一个东八区的时间偏移量（8小时）
  const int offset_seconds = 8 * 60 * 60;

  // 计算东八区的时间戳
  std::time_t beijing_time = in_time_t + offset_seconds;
  // 将东八区时间戳转换为tm结构体
  std::tm *beijing_tm = std::gmtime(&beijing_time);

  std::ostringstream oss;
  oss << std::put_time(beijing_tm, "%Y%m%d_%H%M");
  // 构建文件保存路径
  namespace fs = std::filesystem;

  // 修复路径拼接：移除开头的斜杠
  fs::path dirPath =
      fs::path(core::common::WorkRoot()) / "modules/planning/debugfile";

  // 添加时间戳子目录
  fs::path timestampDir = dirPath / oss.str();

  try {
    // 创建基础目录（如果不存在）
    if (fs::create_directories(dirPath)) {
      AINFO << "成功创建基础目录: " << dirPath;
    }

    // 创建时间戳目录
    if (fs::create_directory(timestampDir)) {
      AINFO << "成功创建时间戳目录: " << timestampDir;
    } else {
      if (fs::exists(timestampDir)) {
        AINFO << "时间戳目录已存在: " << timestampDir;
      } else {
        AINFO << "创建时间戳目录失败: " << timestampDir;
      }
    }
  } catch (const fs::filesystem_error &e) {
    AINFO << "目录操作失败: " << e.what();
  }

  // 规划成功写入规划路径和感知信息
  if (injector_->frame_history()
          ->Latest()
          ->open_space_info()
          .open_space_provider_success()) {
    fs::path TrajectoryfilePath = fs::path(core::common::WorkRoot()) /
                                  "modules/planning/debugfile/" /
                                  fs::path(oss.str()) / "Trajectory.csv";
    fs::path ParkPlacefilePath = fs::path(core::common::WorkRoot()) /
                                 "modules/planning/debugfile/" /
                                 fs::path(oss.str()) / "ParkPlace.csv";
    fs::path FreeSpacefilePath = fs::path(core::common::WorkRoot()) /
                                 "modules/planning/debugfile/" /
                                 fs::path(oss.str()) / "FreeSpace.csv";
    fs::path LocalizationfilePath = fs::path(core::common::WorkRoot()) /
                                    "modules/planning/debugfile/" /
                                    fs::path(oss.str()) / "Localization.csv";

    // 保存轨迹数据
    if (fs::exists(TrajectoryfilePath)) {
      std::cout << "Trajectory.csv文件已存在, 跳过写入: " << TrajectoryfilePath
                << std::endl;
    } else {
      // 创建目录（如果不存在）
      fs::create_directories(TrajectoryfilePath.parent_path());
      // 打开文件流
      Trajectory_data.open(TrajectoryfilePath, std::ios::out | std::ios::trunc);
      if (Trajectory_data.is_open()) {
        Trajectory_data << std::fixed;
        Trajectory_data << "x(m)" << ',' << "y(m)" << ',' << "theta(rad)" << ','
                        << "v(m_s)" << ',' << "a(m_s^2)" << ',' << "kappa"
                        << ',' << "s(m)" << std::endl;
        const auto &stitched_trajectory_result =
            injector_->frame_history()
                ->Latest()
                ->open_space_info()
                .stitched_trajectory_result();
        for (size_t idm = 0; idm < stitched_trajectory_result.size(); idm++) {
          Trajectory_data
              << stitched_trajectory_result.at(idm).path_point().x() << ','
              << stitched_trajectory_result.at(idm).path_point().y() << ','
              << stitched_trajectory_result.at(idm).path_point().theta() << ','
              << stitched_trajectory_result.at(idm).v() << ','
              << stitched_trajectory_result.at(idm).a() << ','
              << stitched_trajectory_result.at(idm).path_point().kappa() << ','
              << stitched_trajectory_result.at(idm).path_point().s()
              << std::endl;
        }
        Trajectory_data.close();
      } else {
        AINFO << "无法打开文件: " << TrajectoryfilePath << std::endl;
      }
    }

    // 保存车位数据
    if (fs::exists(ParkPlacefilePath)) {
      std::cout << "ParkPlace.csv文件已存在, 跳过写入: " << ParkPlacefilePath
                << std::endl;
    } else {
      Polygon2d slot_polygon = frame_->open_space_info()
                                   .target_parking_space_evaluation()
                                   .parking_space_info;
      // 创建目录（如果不存在）
      fs::create_directories(ParkPlacefilePath.parent_path());
      // 打开文件流
      ParkPlace_data.open(ParkPlacefilePath, std::ios::out | std::ios::trunc);
      if (ParkPlace_data.is_open()) {
        ParkPlace_data << std::fixed;
        ParkPlace_data << "x(m)" << ',' << "y(m)" << std::endl;
        for (size_t idm = 0; idm < 4; idm++) {
          ParkPlace_data << slot_polygon.points().at(idm).x() << ','
                         << slot_polygon.points().at(idm).y() << std::endl;
        }
        ParkPlace_data.close();
      } else {
        AINFO << "无法打开文件: " << ParkPlacefilePath << std::endl;
      }
    }

    // 保存感知数据
    if (fs::exists(FreeSpacefilePath)) {
      std::cout << "FreeSpace.csv文件已存在, 跳过写入: " << FreeSpacefilePath
                << std::endl;
    } else {
      auto obs_vertices = LoadObstacleVertices();
      msg_LocalMap msg_LocalMap = {0};
      InterpolateObstaclesVertices(obs_vertices, msg_LocalMap);
      // 创建目录（如果不存在）
      fs::create_directories(FreeSpacefilePath.parent_path());
      // 打开文件流
      Freespace_data.open(FreeSpacefilePath, std::ios::out | std::ios::trunc);
      if (Freespace_data.is_open()) {
        Freespace_data << std::fixed;
        Freespace_data << "x(m)" << ',' << "y(m)" << std::endl;
        for (size_t idm = 0; idm < msg_LocalMap.num_of_vertices; idm++) {
          Freespace_data << msg_LocalMap.vertices_list[idm].x_m << ','
                         << msg_LocalMap.vertices_list[idm].y_m << std::endl;
        }
        Freespace_data.close();
      } else {
        AINFO << "无法打开文件: " << FreeSpacefilePath << std::endl;
      }
    }

    // 保存定位数据 LocalizationfilePath
    if (fs::exists(LocalizationfilePath)) {
      std::cout << "Localization.csv文件已存在, 继续写入: "
                << LocalizationfilePath << std::endl;
      Localization_data.open(LocalizationfilePath,
                             std::ios::out | std::ios::app);
      if (Localization_data.is_open()) {
        Localization_data << std::fixed;
        Localization_data << frame_->vehicle_state().x() << ','
                          << frame_->vehicle_state().y() << ','
                          << frame_->vehicle_state().heading() << std::endl;
        Localization_data.close();
      }
    } else {
      auto obs_vertices = LoadObstacleVertices();
      msg_LocalMap msg_LocalMap = {0};
      InterpolateObstaclesVertices(obs_vertices, msg_LocalMap);
      // 创建目录（如果不存在）
      fs::create_directories(LocalizationfilePath.parent_path());
      // 打开文件流
      Localization_data.open(LocalizationfilePath,
                             std::ios::out | std::ios::trunc);
      if (Localization_data.is_open()) {
        Localization_data << std::fixed;
        Localization_data << "x(m)" << ',' << "y(m)" << ',' << "theta(rad)"
                          << std::endl;
        Localization_data << frame_->vehicle_state().x() << ','
                          << frame_->vehicle_state().y() << ','
                          << frame_->vehicle_state().heading() << std::endl;
        Localization_data.close();
      } else {
        AINFO << "无法打开文件: " << LocalizationfilePath << std::endl;
      }
    }
  }
}

void OpenSpaceTrajectoryProvider::GenerateLinearStopTrajectory(
    DiscretizedTrajectory *const trajectory_data) {
  // 直线减速轨迹，减速至目标速度时间间隔
  // dt = 0.4
  double relative_time = 0.0;
  // TODO(Jinyun) Move to conf
  static constexpr double relative_stop_time = 0.1;
  // 注意倒车 [default = 0.3 m/s^2]
  static constexpr double vEpsilon = 0.00001;
  // double standstill_acceleration = frame_->vehicle_state().linear_velocity()
  // >= -vEpsilon
  //                                      ?
  //                                      -FLAGS_open_space_standstill_acceleration
  //                                      :
  //                                      FLAGS_open_space_standstill_acceleration;
  double standstill_acceleration = -1.3;

  trajectory_data->clear();
  // 需要 input parma: 目标速度 goal_velocity [default = 1.0 m/s]
  double goal_velocity = 0.4;
  auto scenario_type = injector_->planning_context()
                           ->planning_status()
                           .scenario()
                           .scenario_type();
  AINFO << "GenerateLinearStopTrajectory: scenario_type= " << scenario_type;
  if (scenario_type == ScenarioConfig::VALET_PARKING) {
    goal_velocity = 1.0;
    auto slot_p0x = frame_->open_space_info()
                        .target_parking_space_evaluation()
                        .parking_space_info.points()
                        .at(0)
                        .x();
    auto slot_p0y = frame_->open_space_info()
                        .target_parking_space_evaluation()
                        .parking_space_info.points()
                        .at(0)
                        .y();
    Vec2d vec_slot_p0(slot_p0x, slot_p0y);
    Vec2d vec_loc(frame_->vehicle_state().x(), frame_->vehicle_state().y());
    double dist_diff = vec_loc.DistanceTo(vec_slot_p0);
    AINFO << "GenerateLinearStopTrajectory: dist to slot= " << dist_diff;
    if (dist_diff < 3) {
      goal_velocity = 0.0;
      standstill_acceleration = -1.7;
    }
  } else {
    // 接近终点10m停车
    auto routing_request =
        frame_->local_view().routing_response_msg()->getRoutingRequest();
    const auto &routing_end = *(routing_request.getWaypoint().end() - 1);

    auto diff =
        Vec2d(routing_end.getPose().getX(), routing_end.getPose().getY()) -
        Vec2d(frame_->vehicle_state().x(), frame_->vehicle_state().y());

    // const auto& end_pose = frame_->open_space_info().open_space_end_pose();
    // auto diff = Vec2d(end_pose[0], end_pose[1]) -
    // Vec2d(frame_->vehicle_state().x(), frame_->vehicle_state().y());

    if (diff.Length() <= 5.0) {
      goal_velocity = 0.0;
      standstill_acceleration = -2.5;
    }
  }

  double init_velocity = frame_->vehicle_state().linear_velocity() +
                         standstill_acceleration * relative_stop_time; //
  // 向下取整
  // int stop_trajectory_length = 0;
  // stop_trajectory_length = std::floor(std::fabs((init_velocity -
  // goal_velocity) / standstill_acceleration));

  TrajectoryPoint init_point;
  init_point.mutable_path_point()->set_x(frame_->vehicle_state().x());
  init_point.mutable_path_point()->set_y(frame_->vehicle_state().y());
  init_point.mutable_path_point()->set_theta(frame_->vehicle_state().heading());
  init_point.mutable_path_point()->set_s(0.0);
  init_point.mutable_path_point()->set_kappa(0.0);
  init_point.set_v(init_velocity);
  init_point.set_a(standstill_acceleration);
  init_point.set_relative_time(relative_time);

  trajectory_data->emplace_back(init_point);

  AINFO << "GenerateLinearStopTrajectory: init_velocity= " << init_velocity;

  GenerateLinearStopPoint_xysvt(trajectory_data, init_point, goal_velocity,
                                relative_stop_time);
}

void OpenSpaceTrajectoryProvider::GenerateLinearStopPoint_xysvt(
    DiscretizedTrajectory *const trajectory_data,
    swift::common::TrajectoryPoint const init_point, const double goal_v,
    const double dt) {
  static constexpr int stop_trajectory_length = 20;
  auto temp_point = init_point;

  for (size_t i = 0; i < stop_trajectory_length; i++) {
    double cur_x = temp_point.path_point().x();
    double cur_y = temp_point.path_point().y();
    double cur_theta = temp_point.path_point().theta();
    double cur_s = temp_point.path_point().s();
    double cur_v = temp_point.v();
    double cur_a = temp_point.a();
    double cur_relative_time = temp_point.relative_time();

    double next_v = cur_v + dt * cur_a;

    if (next_v < goal_v) {
      next_v = cur_v;
      cur_a = 0.0;
      temp_point.set_a(0.0);
    }

    double delta_x = dt * (cur_v + 0.5 * dt * cur_a) * std::cos(cur_theta);
    double delta_y = dt * (cur_v + 0.5 * dt * cur_a) * std::sin(cur_theta);

    double next_x = cur_x + delta_x;
    double next_y = cur_y + delta_y;
    double next_s = cur_s + std::sqrt(delta_x * delta_x + delta_y * delta_y);
    double next_relative_time = cur_relative_time + dt;

    temp_point.mutable_path_point()->set_x(next_x);
    temp_point.mutable_path_point()->set_y(next_y);
    temp_point.mutable_path_point()->set_s(next_s);
    temp_point.set_v(next_v);
    temp_point.set_a(cur_a);
    temp_point.set_relative_time(next_relative_time);

    trajectory_data->emplace_back(temp_point);
  }
}

Status OpenSpaceTrajectoryProvider::Process() {
  ADEBUG << "trajectory provider";
  // CSVWrite;
  auto trajectory_data =
      frame_->mutable_open_space_info()->mutable_stitched_trajectory_result();
  const auto scenario_type = injector_->planning_context()
                                 ->planning_status()
                                 .scenario()
                                 .scenario_type();
  const auto parkout_flag = injector_->planning_context()
                                ->planning_status()
                                .valet_parking_status()
                                .parkout_flag();
  if (scenario_type == ScenarioConfig::VALET_PARKING ||
      FLAGS_enable_manual_apa) {
    // if (1) {
    frame_->mutable_open_space_info()->set_fallback_flag(false);
    auto *previous_frame = injector_->frame_history()->Latest();
    uint8_t valid_slot_num = 0;
    for (const auto &slot :
         frame_->open_space_info().parking_space_evaluation_list()) {
      if (slot.is_feasible) {
        valid_slot_num++;
      }
    }
    if (valid_slot_num < 1 && !parkout_flag) {
      AERROR << "No valid parking slot.";
      return Status(ErrorCode::PLANNING_ERROR, "No valid parking slot.");
    }

    // bool apa_stop_path =
    // previous_frame->open_space_info().finished_apa_stop_path();
    // // apa_stop_path = false;
    // AINFO << "apa_stop_path: " << apa_stop_path;
    // // bool apa_stop_path = true;
    // if (!apa_stop_path) {
    //   if (frame_->vehicle_state().linear_velocity() <
    //   common::math::kMathEpsilon) {
    frame_->mutable_open_space_info()->set_finished_apa_stop_path(true);
    //   } else {
    //     AINFO << "Slowing to parking with current speed " <<
    //     frame_->vehicle_state().linear_velocity();
    //     GenerateLinearStopTrajectory(trajectory_data);
    //     frame_->mutable_open_space_info()->set_finished_apa_stop_path(false);
    //     //
    //     frame_->mutable_open_space_info()->set_finished_apa_stop_path(true);

    //     return Status::OK();
    //   }
    // } else {
    // frame_->mutable_open_space_info()->set_finished_apa_stop_path(apa_stop_path);
    // }
    bool is_need_replan = false;
    if (previous_frame != nullptr) {
      is_need_replan = IsVehicleStopDueToFallBack(
          previous_frame->open_space_info().fallback_flag(),
          frame_->vehicle_state());
    }
    frame_->mutable_open_space_info()->set_open_space_provider_success(false);
    Polygon2d slot_polygon = frame_->open_space_info()
                                 .target_parking_space_evaluation()
                                 .parking_space_info;

    double x_offset = 0; // 5.0;
    double y_offset = 0; // 10.0;
    common::math::Vec2d offset = Vec2d(x_offset, y_offset);
    std::vector<common::math::Vec2d> obstacle0_points = {
        {3.6 + x_offset, -3.1 + y_offset},
        {3.6 + x_offset, -5.0 + y_offset},
        {5.0 + x_offset, -5.0 + y_offset},
        {5.0 + x_offset, -3.1 + y_offset}};
    std::vector<common::math::Vec2d> obstacle1_points = {
        {-1.0 + x_offset, -3.0 + y_offset},
        {-3.0 + x_offset, -3.0 + y_offset},
        {-3.0 + x_offset, -5.0 + y_offset},
        {-1.0 + x_offset, -5.0 + y_offset}};
    std::vector<common::math::Vec2d> obstacle2_points = {
        {0.0 + x_offset, 4.0 + y_offset},
        {6.0 + x_offset, 4.0 + y_offset},
        {6.0 + x_offset, 6.0 + y_offset},
        {0.0 + x_offset, 6.0 + y_offset}};
    std::vector<common::math::Vec2d> obstacle3_points = {
        {4.0 + x_offset, 0.0 + y_offset},
        {5.0 + x_offset, 0.0 + y_offset},
        {5.0 + x_offset, 3.0 + y_offset},
        {4.0 + x_offset, 3.0 + y_offset}};
    // obstacle0_points = obstacle0_points+offset;
    // obstacle1_points = obstacle1_points+offset;
    // obstacle2_points = obstacle2_points+offset;
    // obstacle3_points = obstacle3_points+offset;
    auto obs_vertices = LoadObstacleVertices();
    // Obs_list.push_back(obstacle0_points);
    // Obs_list.push_back(obstacle1_points);
    // Obs_list.push_back(obstacle2_points);
    // Obs_list.push_back(obstacle3_points);
    // AINFO << "Start Apa Planning";
    // std::cout << "Start Apa Planning" << std::endl;
    Status status = GenerateApaTrajectory(trajectory_data, slot_polygon,
                                          obs_vertices, is_need_replan);

    if (status.ok()) {
      return status;
    }
  } else {
    // generate stop trajectory at park_and_go check_stage
    if (injector_->planning_context()
            ->mutable_planning_status()
            ->mutable_park_and_go()
            ->in_check_stage()) {
      ADEBUG << "ParkAndGo Stage Check.";
      GenerateStopTrajectory(trajectory_data);
      return Status::OK();
    }

    // Start thread when getting in Process() for the first time
    if (FLAGS_enable_open_space_planner_thread && !thread_init_flag_) {
      task_future_ = std::async(
          &OpenSpaceTrajectoryProvider::GenerateTrajectoryThread, this);
      thread_init_flag_ = true;
    }
    // Get stitching trajectory from last frame
    const common::VehicleState vehicle_state = frame_->vehicle_state();
    auto *previous_frame = injector_->frame_history()->Latest();
    // Use complete raw trajectory from last frame for stitching purpose
    std::vector<TrajectoryPoint> stitching_trajectory;
    if (previous_frame != nullptr &&
        !IsVehicleStopDueToFallBack(
            previous_frame->open_space_info().fallback_flag(), vehicle_state)) {
      const auto &previous_planning =
          previous_frame->open_space_info().stitched_trajectory_result();
      const auto &previous_planning_header =
          previous_frame->current_frame_planned_trajectory()
              .header()
              .timestamp_sec();
      const double planning_cycle_time = FLAGS_open_space_planning_period;
      PublishableTrajectory last_frame_complete_trajectory(
          previous_planning_header, previous_planning);
      // std::cout << "last_frame_complete_trajectory.NumOfPoints() = " <<
      // last_frame_complete_trajectory.NumOfPoints()
      //           << std::endl;
      // for (size_t i = 0; i < last_frame_complete_trajectory.NumOfPoints();
      // i++) {
      //   std::cout << i << " trajectory_data v = " <<
      //   last_frame_complete_trajectory.TrajectoryPointAt(i).v() << std::endl;
      // }
      std::string replan_reason;
      const double start_timestamp = Clock::NowInSeconds();
      stitching_trajectory = TrajectoryStitcher::ComputeStitchingTrajectory(
          false, vehicle_state, start_timestamp, planning_cycle_time,
          FLAGS_open_space_trajectory_stitching_preserved_length, true, 0,
          &last_frame_complete_trajectory, &replan_reason);
      // std::cout << "vehicle x = " << vehicle_state.x() << " y = " <<
      // vehicle_state.y()
      //           << " theta = " << vehicle_state.heading() << " v = " <<
      //           vehicle_state.linear_velocity() << std::endl;
      // std::cout << "predict x = " <<
      // stitching_trajectory.back().path_point().x()
      //           << " y = " << stitching_trajectory.back().path_point().y()
      //           << " theta = " <<
      //           stitching_trajectory.back().path_point().theta()
      //           << " v = " << stitching_trajectory.back().v() << std::endl;
    } else {
      ADEBUG << "Replan due to fallback stop";
      AINFO << "Replan due to fallback stop" << std::endl;
      const double planning_cycle_time =
          1.0 / static_cast<double>(FLAGS_planning_loop_rate);
      AINFO << "FLAGS_planning_loop_rate = " << FLAGS_planning_loop_rate
            << " planning_cycle_time = " << planning_cycle_time << std::endl;
      stitching_trajectory =
          TrajectoryStitcher::ComputeReinitStitchingTrajectory(
              planning_cycle_time, vehicle_state);
      auto *open_space_status = injector_->planning_context()
                                    ->mutable_planning_status()
                                    ->mutable_open_space();
      // open_space_status->set_position_init(false); //
      // 会导致索引点清零，目前不考虑重规划
    }
    // Get open_space_info from current frame
    const auto &open_space_info = frame_->open_space_info();

    if (FLAGS_enable_open_space_planner_thread) {
      ADEBUG << "Open space plan in multi-threads mode";

      if (is_generation_thread_stop_) {
        GenerateStopTrajectory(trajectory_data);
        return Status(ErrorCode::OK, "Parking finished");
      }

      {
        std::lock_guard<std::mutex> lock(open_space_mutex_);
        thread_data_.stitching_trajectory = stitching_trajectory;
        thread_data_.end_pose = open_space_info.open_space_end_pose();
        thread_data_.rotate_angle = open_space_info.origin_heading();
        thread_data_.translate_origin = open_space_info.origin_point();
        thread_data_.obstacles_edges_num =
            open_space_info.obstacles_edges_num();
        thread_data_.obstacles_A = open_space_info.obstacles_A();
        thread_data_.obstacles_b = open_space_info.obstacles_b();
        thread_data_.obstacles_vertices_vec =
            open_space_info.obstacles_vertices_vec();
        thread_data_.XYbounds = open_space_info.ROI_xy_boundary();
        data_ready_.store(true);
      }

      // Check vehicle state
      if (IsVehicleNearDestination(vehicle_state,
                                   open_space_info.open_space_end_pose(),
                                   open_space_info.origin_heading(),
                                   open_space_info.origin_point())) {
        GenerateStopTrajectory(trajectory_data);
        is_generation_thread_stop_.store(true);
        std::cout << "return Vehicle is near to destination" << std::endl;
        return Status(ErrorCode::OK, "Vehicle is near to destination");
      }

      if (previous_frame != nullptr) {
        if (previous_frame->open_space_info().open_space_provider_success()) {
          ReuseLastFrameResult(previous_frame, trajectory_data);
          AINFO << "Reuse last frame result in open_space_trajectory_provider";
          if (FLAGS_enable_record_debug) {
            // copy previous debug to current frame
            ReuseLastFrameDebug(previous_frame);
          }
          // reuse last frame debug when use last frame traj
          AINFO << "return Waiting for open_space_trajectory_optimizer in "
                   "open_space_trajectory_provider"
                << std::endl;
          return Status::OK();
        }
      }

      // Check if trajectory updated
      if (trajectory_updated_) {
        std::lock_guard<std::mutex> lock(open_space_mutex_);
        LoadResult(trajectory_data);
        if (FLAGS_enable_record_debug) {
          // call merge debug ptr, open_space_trajectory_optimizer_
          auto *ptr_debug = frame_->mutable_open_space_info()->mutable_debug();
          open_space_trajectory_optimizer_->UpdateDebugInfo(
              ptr_debug->mutable_planning_data()->mutable_open_space());

          // sync debug instance
          frame_->mutable_open_space_info()->sync_debug_instance();
        }
        data_ready_.store(false);
        trajectory_updated_.store(false);
        return Status::OK();
      }

      if (trajectory_error_) {
        ++optimizer_thread_counter;
        std::lock_guard<std::mutex> lock(open_space_mutex_);
        trajectory_error_.store(false);
        // TODO(Jinyun) Use other fallback mechanism when last iteration
        // smoothing result has out of bound pathpoint which is not allowed for
        // next iteration hybrid astar algorithm which requires start position
        // to be strictly in bound
        if (optimizer_thread_counter > 1000) {
          return Status(ErrorCode::PLANNING_ERROR,
                        "open_space_optimizer failed too many times");
        }
      }

      if (previous_frame != nullptr) {
        if (previous_frame->open_space_info().open_space_provider_success()) {
          ReuseLastFrameResult(previous_frame, trajectory_data);
          if (FLAGS_enable_record_debug) {
            // copy previous debug to current frame
            ReuseLastFrameDebug(previous_frame);
          }
          // reuse last frame debug when use last frame traj
          return Status(ErrorCode::OK,
                        "Waiting for open_space_trajectory_optimizer in "
                        "open_space_trajectory_provider");
        } else {
          GenerateLinearStopTrajectory(trajectory_data);
          AINFO << "GenerateLinearStopTrajectory trajectory_data goal v = "
                << trajectory_data->back().v();
          AINFO << "return Stop due to computation not finished";
          // std:cout << "GenerateLinearStopTrajectory trajectory_data goal v =
          // " << trajectory_data->back().v() << std::endl;
          return Status(ErrorCode::OK, "Stop due to computation not finished");
        }
      }
    } else {
      const auto &end_pose = open_space_info.open_space_end_pose();
      const auto &rotate_angle = open_space_info.origin_heading();
      const auto &translate_origin = open_space_info.origin_point();
      const auto &obstacles_edges_num = open_space_info.obstacles_edges_num();
      const auto &obstacles_A = open_space_info.obstacles_A();
      const auto &obstacles_b = open_space_info.obstacles_b();
      const auto &obstacles_vertices_vec =
          open_space_info.obstacles_vertices_vec();
      const auto &XYbounds = open_space_info.ROI_xy_boundary();

      // Check vehicle state
      if (IsVehicleNearDestination(vehicle_state, end_pose, rotate_angle,
                                   translate_origin)) {
        GenerateStopTrajectory(trajectory_data);
        return Status(ErrorCode::OK, "Vehicle is near to destination");
      }

      // Generate Trajectory;
      double time_latency;
      Status status = open_space_trajectory_optimizer_->Plan(
          stitching_trajectory, end_pose, XYbounds, rotate_angle,
          translate_origin, obstacles_edges_num, obstacles_A, obstacles_b,
          obstacles_vertices_vec, false, &time_latency);
      frame_->mutable_open_space_info()->set_time_latency(time_latency);

      // If status is OK, update vehicle trajectory;
      if (status == Status::OK()) {
        LoadResult(trajectory_data);
        return status;
      } else {
        return status;
      }
    }
  }
  return Status(ErrorCode::PLANNING_ERROR);
}

void OpenSpaceTrajectoryProvider::GenerateTrajectoryThread() {
  if (injector_->frame_history()->Latest() != nullptr) {
    auto *previous_frame = injector_->frame_history()->Latest();
    AINFO << "previous_open_space_provider_success = "
          << previous_frame->open_space_info().open_space_provider_success();
    if (previous_frame->open_space_info().open_space_provider_success())
      return;
  }

  while (!is_generation_thread_stop_) {
    if (!trajectory_updated_ && data_ready_) {
      OpenSpaceTrajectoryThreadData thread_data;
      {
        std::lock_guard<std::mutex> lock(open_space_mutex_);
        thread_data = thread_data_;
      }
      double time_latency;
      Status status = open_space_trajectory_optimizer_->Plan(
          thread_data.stitching_trajectory, thread_data.end_pose,
          thread_data.XYbounds, thread_data.rotate_angle,
          thread_data.translate_origin, thread_data.obstacles_edges_num,
          thread_data.obstacles_A, thread_data.obstacles_b,
          thread_data.obstacles_vertices_vec, false, &time_latency);
      frame_->mutable_open_space_info()->set_time_latency(time_latency);
      if (status == Status::OK()) {
        std::lock_guard<std::mutex> lock(open_space_mutex_);
        trajectory_updated_.store(true);
      } else {
        if (status.ok()) {
          std::lock_guard<std::mutex> lock(open_space_mutex_);
          trajectory_skipped_.store(true);
        } else {
          std::lock_guard<std::mutex> lock(open_space_mutex_);
          trajectory_error_.store(true);
        }
      }
    }
  }
}

bool OpenSpaceTrajectoryProvider::IsVehicleNearDestination(
    const common::VehicleState &vehicle_state,
    const std::vector<double> &goal_pose, double rotate_angle,
    const Vec2d &translate_origin) {
  std::vector<double> end_pose = goal_pose;
  ACHECK_EQ(end_pose.size(), 4U);
  if (injector_->frame_history()->Latest() != nullptr) {
    if (injector_->frame_history()
            ->Latest()
            ->open_space_info()
            .open_space_provider_success()) {
      end_pose[0] = injector_->frame_history()
                        ->Latest()
                        ->open_space_info()
                        .stitched_trajectory_result()
                        .back()
                        .path_point()
                        .x();
      end_pose[1] = injector_->frame_history()
                        ->Latest()
                        ->open_space_info()
                        .stitched_trajectory_result()
                        .back()
                        .path_point()
                        .y();
      end_pose[2] = injector_->frame_history()
                        ->Latest()
                        ->open_space_info()
                        .stitched_trajectory_result()
                        .back()
                        .path_point()
                        .theta();
    }
  }

  Vec2d end_pose_to_world_frame = Vec2d(end_pose[0], end_pose[1]);

  end_pose_to_world_frame.SelfRotate(rotate_angle);
  end_pose_to_world_frame += translate_origin;
  double distance_to_vehicle2 =
      std::sqrt((vehicle_state.x() - end_pose_to_world_frame.x()) *
                    (vehicle_state.x() - end_pose_to_world_frame.x()) +
                (vehicle_state.y() - end_pose_to_world_frame.y()) *
                    (vehicle_state.y() - end_pose_to_world_frame.y()));
  double end_theta_to_world_frame = end_pose[2];
  end_theta_to_world_frame += rotate_angle;
  double distance_to_vehicle1 = std::sqrt(
      (vehicle_state.x() - end_pose[0]) * (vehicle_state.x() - end_pose[0]) +
      (vehicle_state.y() - end_pose[1]) * (vehicle_state.y() - end_pose[1]));
  double distance_to_vehicle =
      std::sqrt((vehicle_state.x() - end_pose_to_world_frame.x()) *
                    (vehicle_state.x() - end_pose_to_world_frame.x()) +
                (vehicle_state.y() - end_pose_to_world_frame.y()) *
                    (vehicle_state.y() - end_pose_to_world_frame.y()));
  double theta_to_vehicle = std::fabs(common::math::AngleDiff(
      vehicle_state.heading(), end_theta_to_world_frame));
  ADEBUG << "distance_to_vehicle1 is: " << distance_to_vehicle1;
  ADEBUG << "distance_to_vehicle2 is: " << distance_to_vehicle2;
  ADEBUG << "theta_to_vehicle: " << theta_to_vehicle
         << "; end_theta_to_world_frame: " << end_theta_to_world_frame
         << "; rotate_angle: " << rotate_angle;
  ADEBUG << "is_near_destination_threshold: "
         << config_.open_space_trajectory_provider_config()
                .open_space_trajectory_optimizer_config()
                .planner_open_space_config()
                .is_near_destination_threshold(); // which config file
  ADEBUG << "is_near_destination_theta_threshold: "
         << config_.open_space_trajectory_provider_config()
                .open_space_trajectory_optimizer_config()
                .planner_open_space_config()
                .is_near_destination_theta_threshold();
  // std::cout << " is_near_destination_threshold: "
  //           << config_.open_space_trajectory_provider_config()
  //                  .open_space_trajectory_optimizer_config()
  //                  .planner_open_space_config()
  //                  .is_near_destination_threshold();  // which config file
  // std::cout << " is_near_destination_theta_threshold: "
  //           << config_.open_space_trajectory_provider_config()
  //                  .open_space_trajectory_optimizer_config()
  //                  .planner_open_space_config()
  //                  .is_near_destination_theta_threshold();

  distance_to_vehicle =
      std::min(std::min(distance_to_vehicle, distance_to_vehicle1),
               distance_to_vehicle2);
  theta_to_vehicle =
      std::min(theta_to_vehicle, std::fabs(vehicle_state.heading()));
  if (distance_to_vehicle < config_.open_space_trajectory_provider_config()
                                .open_space_trajectory_optimizer_config()
                                .planner_open_space_config()
                                .is_near_destination_threshold() &&
      theta_to_vehicle < config_.open_space_trajectory_provider_config()
                             .open_space_trajectory_optimizer_config()
                             .planner_open_space_config()
                             .is_near_destination_theta_threshold()) {
    AERROR << "vehicle reach end_pose";
    frame_->mutable_open_space_info()->set_destination_reached(true);
    return true;
  }
  return false;
}

bool OpenSpaceTrajectoryProvider::IsVehicleStopDueToFallBack(
    const bool is_on_fallback, const common::VehicleState &vehicle_state) {
  static int time_count = 0;
  if (!is_on_fallback) {
    time_count = 0;
    return false;
  }
  static constexpr double kEpsilon = 1.0e-1;
  const double adc_speed = vehicle_state.linear_velocity();
  const double adc_acceleration = vehicle_state.linear_acceleration();
  if (std::fabs(adc_speed) < kEpsilon &&
      std::fabs(adc_acceleration) < kEpsilon && time_count > 30) {
    ADEBUG << "ADC stops due to fallback trajectory";
    return true;
  }
  time_count++;
  return false;
}

void OpenSpaceTrajectoryProvider::GenerateStopTrajectory(
    DiscretizedTrajectory *const trajectory_data) {
  double relative_time = 0.0;
  // TODO(Jinyun) Move to conf
  static constexpr int stop_trajectory_length = 10;
  static constexpr double relative_stop_time = 0.1;
  static constexpr double vEpsilon = 0.00001;
  double standstill_acceleration =
      frame_->vehicle_state().linear_velocity() >= -vEpsilon
          ? -FLAGS_open_space_standstill_acceleration
          : FLAGS_open_space_standstill_acceleration;
  trajectory_data->clear();
  for (size_t i = 0; i < stop_trajectory_length; i++) {
    TrajectoryPoint point;
    point.mutable_path_point()->set_x(frame_->vehicle_state().x());
    point.mutable_path_point()->set_y(frame_->vehicle_state().y());
    point.mutable_path_point()->set_theta(frame_->vehicle_state().heading());
    point.mutable_path_point()->set_s(0.0);
    point.mutable_path_point()->set_kappa(0.0);
    point.set_relative_time(relative_time);
    point.set_v(0.0);
    point.set_a(standstill_acceleration);
    trajectory_data->emplace_back(point);
    relative_time += relative_stop_time;
  }
}

void OpenSpaceTrajectoryProvider::GeneratePreParkStopTrajectory(
    DiscretizedTrajectory *trajectory_data,
    common::VehicleState vehicle_state) {
  // const auto& vehicle_state = frame_->vehicle_state();
  double current_velocity = vehicle_state.linear_velocity();
  double current_x = vehicle_state.x();
  double current_y = vehicle_state.y();
  double current_theta = vehicle_state.heading();

  // 设置减速度和点间距
  const double deceleration = 0.2; // 减速度为0.2
  const double point_distance = 1; // 点间距为0.2米

  // 清空轨迹数据
  trajectory_data->clear();

  // 计算轨迹点
  double relative_time = 0.0;
  double distance_traveled = 0.0;
  while (current_velocity > 0) {
    TrajectoryPoint point;
    point.mutable_path_point()->set_x(current_x);
    point.mutable_path_point()->set_y(current_y);
    point.mutable_path_point()->set_theta(current_theta);
    point.mutable_path_point()->set_s(distance_traveled);
    point.mutable_path_point()->set_kappa(0.0);
    point.set_relative_time(relative_time);
    point.set_v(current_velocity);
    point.set_a(-deceleration);
    trajectory_data->emplace_back(point);

    // 更新位置和速度
    distance_traveled += point_distance;
    // current_velocity -= deceleration * (point_distance / current_velocity);
    double current_velocity_square =
        current_velocity * current_velocity - 2 * deceleration * point_distance;
    current_velocity = std::sqrt(std::max(current_velocity_square, 0.0));
    if (current_velocity < 0) {
      current_velocity = 0;
      relative_time += 10;
    } else {
      relative_time += point_distance / current_velocity;
    }
    current_x += point_distance * cos(current_theta);
    current_y += point_distance * sin(current_theta);
    relative_time += point_distance / current_velocity;
  }

  // 添加最后一个速度为0的点
  TrajectoryPoint stop_point;
  stop_point.mutable_path_point()->set_x(current_x);
  stop_point.mutable_path_point()->set_y(current_y);
  stop_point.mutable_path_point()->set_theta(current_theta);
  stop_point.mutable_path_point()->set_s(distance_traveled);
  stop_point.mutable_path_point()->set_kappa(0.0);
  stop_point.set_relative_time(relative_time);
  stop_point.set_v(0.0);
  stop_point.set_a(0.0);
  trajectory_data->emplace_back(stop_point);
}

Status OpenSpaceTrajectoryProvider::GenerateApaTrajectory(
    DiscretizedTrajectory *const trajectory_data, Polygon2d slot_polygon,
    std::vector<std::vector<common::math::Vec2d>> Obs_list,
    bool is_need_replan) {
  std::ofstream ofs_odom;
  std::ofstream ofs_obs;
  std::ofstream ofs_localmap;
  std::ofstream ofs_traj;
  std::ofstream ofs_slot;

  if (FLAGS_enable_apa_csv_dump) {
    ofs_odom.open("odom.csv");
    ofs_odom << "x,y,theta\n";

    ofs_obs.open("obs.csv");
    ofs_obs << "x,y\n";

    ofs_localmap.open("localmap.csv");
    ofs_localmap << "x,y\n";

    ofs_traj.open("traj.csv");
    ofs_traj << "x,y,theta,v\n";

    ofs_slot.open("slot.csv");
    ofs_slot << "x,y\n";
  }

  auto *previous_frame = injector_->frame_history()->Latest();
  const auto parkout_flag = injector_->planning_context()
                                ->planning_status()
                                .valet_parking_status()
                                .parkout_flag();
  if (slot_polygon.num_points() == 0 && !parkout_flag) {
    std::cout << "Planning slot empty." << std::endl;
    AERROR << "Planning slot empty.";
    if (previous_frame != nullptr) {
      if (previous_frame->open_space_info().open_space_provider_success()) {
        ReuseLastFrameResult(previous_frame, trajectory_data);
        AINFO << "Reuse last frame result in open_space_trajectory_provider";
        if (FLAGS_enable_record_debug) {
          // copy previous debug to current frame
          ReuseLastFrameDebug(previous_frame);
        }
        // reuse last frame debug when use last frame traj
        AINFO << "return Waiting for open_space_trajectory_optimizer in "
                 "open_space_trajectory_provider"
              << std::endl;
        return Status::OK();
      }
    }
    // return Status(ErrorCode::PLANNING_ERROR, "Planning slot empty.");
  }

  // if (slot_polygon.num_points() == 0) {
  //   AERROR << "Planning slot empty.";
  //   return Status(ErrorCode::PLANNING_ERROR, "Planning slot empty.");
  // }
  msg_VehicleChassis_Gear msg_VehicleChassis = {0};
  // msg_VehicleChassis.actv_gear_enum = frame_->vehicle_state().gear();
  // msg_VehicleChassis.standstill = frame_->vehicle_state().linear_velocity() >
  // 0 ? 1 : 0;
  msg_VehicleChassis.actv_gear_enum = 1;
  msg_VehicleChassis.standstill = 1;
  msg_ApaCtrlToApp msg_ApaCtrl = {0};
  msg_ApaCtrl.replan_sts_enum = 0;
  if (previous_frame != nullptr) {
    if (is_need_replan &&
        !previous_frame->open_space_info().apa_replan_flag()) {
      frame_->mutable_open_space_info()->set_apa_replan_flag(true);
      msg_ApaCtrl.replan_sts_enum = 2;
    }
  }
  msg_Odometry msg_Odome = {0};

  // AINFO << "msg_Odome vehicle state x_m: " << frame_->vehicle_state().x() -
  // 241480
  //       << "msg_Odome vehicle state y_m: " << frame_->vehicle_state().y() -
  //       2543770;
  double loc_x = frame_->vehicle_state().x();
  double loc_y = frame_->vehicle_state().y();
  double loc_heading =
      frame_->vehicle_state()
          .heading(); // msg_Odome.pose.position.z_m is defined as yaw
  // msg_Odome.pose.position.x_m = 5.0;
  // msg_Odome.pose.position.y_m = 10.0;
  // msg_Odome.pose.position.z_m = 0 * 3.14 / 180;
  AINFO << "loc_x: " << frame_->vehicle_state().x()
        << "loc_y: " << frame_->vehicle_state().y();

  auto convert_to_vehicle_coordinates = [&](double x, double y) {
    double dx = x - loc_x;
    double dy = y - loc_y;
    double theta = loc_heading;
    double x_vehicle = dx * cos(theta) + dy * sin(theta);
    double y_vehicle = -dx * sin(theta) + dy * cos(theta);
    return std::make_pair(x_vehicle, y_vehicle);
  };

  msg_Odome.pose.position.x_m =
      convert_to_vehicle_coordinates(loc_x, loc_y).first;
  msg_Odome.pose.position.y_m =
      convert_to_vehicle_coordinates(loc_x, loc_y).second;
  msg_Odome.pose.position.z_m = frame_->vehicle_state().heading() - loc_heading;

  if (FLAGS_enable_apa_csv_dump) {
    ofs_odom << std::fixed << std::setprecision(6)
             << msg_Odome.pose.position.x_m << ','   // x_vehicle = 0
             << msg_Odome.pose.position.y_m << ','   // y_vehicle = 0
             << msg_Odome.pose.position.z_m << '\n'; // heading_vehicle = 0

    // --- EXPORT ORIGINAL OBS_LIST ---
    for (const auto &poly : Obs_list) {
      for (const auto &pt : poly) {
        // convert each obstacle vertex to vehicle frame
        auto [x_v, y_v] = convert_to_vehicle_coordinates(pt.x(), pt.y());
        ofs_obs << std::fixed << std::setprecision(6) << x_v << ',' << y_v
                << '\n';
      }
    }
  }

  msg_LocalMap msg_LocalMap = {0};
  InterpolateObstaclesVertices(Obs_list, msg_LocalMap);

  // --- CONVERT LOCALMAP VERTICES TO VEHICLE COORDINATES ---
  for (int i = 0; i < msg_LocalMap.num_of_vertices; ++i) {
    auto [x_v, y_v] = convert_to_vehicle_coordinates(
        msg_LocalMap.vertices_list[i].x_m, msg_LocalMap.vertices_list[i].y_m);
    msg_LocalMap.vertices_list[i].x_m = x_v;
    msg_LocalMap.vertices_list[i].y_m = y_v;
  }

  // --- APPEND FREESPACE VERTICES (adds more vertices to msg_LocalMap) ---
  LoadFreeSpaceVertices(msg_LocalMap);

  // --- EXPORT TO CSV AFTER ALL VERTICES ARE AVAILABLE ---
  if (FLAGS_enable_apa_csv_dump) {
    for (int i = 0; i < msg_LocalMap.num_of_vertices; ++i) {
      ofs_localmap << std::fixed << std::setprecision(6)
                   << msg_LocalMap.vertices_list[i].x_m << ','
                   << msg_LocalMap.vertices_list[i].y_m << '\n';
    }
  }

  msg_ApaFsmInfo msg_ApaFsm = {0};
  msg_ApaFsm.ctrl_run_sts_enmu = 1;
  msg_ApaFsm.plan_run_sts_enum = 1;
  // msg_ApaFsm.park_slot_id =
  // std::stoi(frame_->open_space_info().target_parking_spot_id());
  msg_ApaFsm.park_slot_id = 1;
  msg_SlotsList msg_SlotsList = {0};

  if (!parkout_flag) {
    AINFO << "tartget slot x0: " << slot_polygon.points().at(0).x()
          << " y0: " << slot_polygon.points().at(0).y()
          << "tartget slot x1: " << slot_polygon.points().at(1).x()
          << " y1: " << slot_polygon.points().at(1).y()
          << "tartget slot x2: " << slot_polygon.points().at(2).x()
          << " y2: " << slot_polygon.points().at(2).y()
          << "tartget slot x3: " << slot_polygon.points().at(3).x()
          << " y3: " << slot_polygon.points().at(3).y();
    auto p0_vehicle = convert_to_vehicle_coordinates(
        slot_polygon.points().at(0).x(), slot_polygon.points().at(0).y());
    auto p1_vehicle = convert_to_vehicle_coordinates(
        slot_polygon.points().at(1).x(), slot_polygon.points().at(1).y());
    auto p2_vehicle = convert_to_vehicle_coordinates(
        slot_polygon.points().at(2).x(), slot_polygon.points().at(2).y());
    auto p3_vehicle = convert_to_vehicle_coordinates(
        slot_polygon.points().at(3).x(), slot_polygon.points().at(3).y());
    if (p0_vehicle.second <= 0) {
      msg_SlotsList.slots_list[0].location_enum = 2;
      msg_SlotsList.slots_list[0].p0.x_m = p0_vehicle.first;
      msg_SlotsList.slots_list[0].p0.y_m = p0_vehicle.second;
      msg_SlotsList.slots_list[0].p1.x_m = p1_vehicle.first;
      msg_SlotsList.slots_list[0].p1.y_m = p1_vehicle.second;
      msg_SlotsList.slots_list[0].p2.x_m = p2_vehicle.first;
      msg_SlotsList.slots_list[0].p2.y_m = p2_vehicle.second;
      msg_SlotsList.slots_list[0].p3.x_m = p3_vehicle.first;
      msg_SlotsList.slots_list[0].p3.y_m = p3_vehicle.second;
    } else {
      msg_SlotsList.slots_list[0].location_enum = 1;
      msg_SlotsList.slots_list[0].p0.x_m = p1_vehicle.first;
      msg_SlotsList.slots_list[0].p0.y_m = p1_vehicle.second;
      msg_SlotsList.slots_list[0].p1.x_m = p0_vehicle.first;
      msg_SlotsList.slots_list[0].p1.y_m = p0_vehicle.second;
      msg_SlotsList.slots_list[0].p2.x_m = p3_vehicle.first;
      msg_SlotsList.slots_list[0].p2.y_m = p3_vehicle.second;
      msg_SlotsList.slots_list[0].p3.x_m = p2_vehicle.first;
      msg_SlotsList.slots_list[0].p3.y_m = p2_vehicle.second;
    }
    AINFO << "msg_SlotsList x0: " << msg_SlotsList.slots_list[0].p0.x_m
          << " y0: " << msg_SlotsList.slots_list[0].p0.y_m
          << "msg_SlotsList x1: " << msg_SlotsList.slots_list[0].p1.x_m
          << " y1: " << msg_SlotsList.slots_list[0].p1.y_m
          << "msg_SlotsList x2: " << msg_SlotsList.slots_list[0].p2.x_m
          << " y2: " << msg_SlotsList.slots_list[0].p2.y_m
          << "msg_SlotsList x3: " << msg_SlotsList.slots_list[0].p3.x_m
          << " y3: " << msg_SlotsList.slots_list[0].p3.y_m;
    // Polygon2d slot_polygon =
    // frame_->open_space_info().target_parking_space_evaluation().parking_space_info;
  }

  if (FLAGS_enable_apa_csv_dump) {
    const auto &slot = msg_SlotsList.slots_list[0];
    ofs_slot << slot.p0.x_m << "," << slot.p0.y_m << "\n";
    ofs_slot << slot.p1.x_m << "," << slot.p1.y_m << "\n";
    ofs_slot << slot.p2.x_m << "," << slot.p2.y_m << "\n";
    ofs_slot << slot.p3.x_m << "," << slot.p3.y_m << "\n";
  }

  Vec2d vec_a(msg_SlotsList.slots_list[0].p0.x_m,
              msg_SlotsList.slots_list[0].p0.y_m);
  Vec2d vec_b(msg_SlotsList.slots_list[0].p1.x_m,
              msg_SlotsList.slots_list[0].p1.y_m);
  Vec2d vec_c(msg_SlotsList.slots_list[0].p2.x_m,
              msg_SlotsList.slots_list[0].p2.y_m);
  double dist_ab = vec_b.DistanceTo(vec_a);
  double dist_bc = vec_c.DistanceTo(vec_b);
  double angle_ab = (vec_b - vec_a).Angle();
  double angle_bc = (vec_c - vec_b).Angle();
  AINFO << "dist_ab: " << dist_ab << " dist_bc: " << dist_bc
        << " angle_ab: " << angle_ab << " angle_bc: " << angle_bc;
  std::cout << "dist_ab: " << dist_ab << " dist_bc: " << dist_bc
            << " angle_ab: " << angle_ab << " angle_bc: " << angle_bc
            << std::endl;
  swift::hdmap::ParkingSlot::ShapeType parking_space_type =
      // frame_->open_space_info().target_parking_space_evaluation().parking_space_type;
      swift::hdmap::ParkingSlot::ShapeType::
          ParkingSlot_ShapeType_PARKINGSLOT_SHAPE_RECT;
  switch (parking_space_type) {
  case swift::hdmap::ParkingSlot::ShapeType::
      ParkingSlot_ShapeType_PARKINGSLOT_SHAPE_RECT:
    if (dist_ab < dist_bc) {
      msg_ApaFsm.park_type_enum = 1;
      msg_SlotsList.slots_list[0].type_enum = 2;
      AINFO << "VERTICAL SLOT!!!!!!!!";
      std::cout << "VERTICAL SLOT!!!!!!!!" << std::endl;
    } else {
      msg_ApaFsm.park_type_enum = 1;
      msg_SlotsList.slots_list[0].type_enum = 1;
      AINFO << "HORIZONTAL SLOT!!!!!!!!";
      std::cout << "HORIZONTAL SLOT!!!!!!!!" << std::endl;
    }
    break;
  case swift::hdmap::ParkingSlot::ShapeType::
      ParkingSlot_ShapeType_PARKINGSLOT_SHAPE_DIAGONAL:
    AINFO << "fabs(angle_ab - angle_bc) " << fabs(angle_ab - angle_bc);
    std::cout << "fabs(angle_ab - angle_bc) " << fabs(angle_ab - angle_bc)
              << std::endl;
    if (fabs(angle_ab - angle_bc) < M_PI_2) {
      msg_ApaFsm.park_type_enum = 1;
      msg_SlotsList.slots_list[0].type_enum = 3;
      AINFO << "TAIL IN FISH_BONE SLOT!!!!!!!!";
      std::cout << "TAIL IN FISH_BONE SLOT!!!!!!!!" << std::endl;
    } else {
      msg_ApaFsm.park_type_enum = 2;
      msg_SlotsList.slots_list[0].type_enum = 3;
      AINFO << "HEAD IN ANTI_FISH_BONE SLOT!!!!!!!!";
      std::cout << "HEAD IN ANTI_FISH_BONE SLOT!!!!!!!!" << std::endl;
    }
    break;
  case swift::hdmap::ParkingSlot::ShapeType::
      ParkingSlot_ShapeType_PARKINGSLOT_SHAPE_CIRCLE:
    msg_ApaFsm.park_type_enum = 1;
    msg_SlotsList.slots_list[0].type_enum = 1;
    AINFO << "CIRCLE SLOT!!!!!!!!";
    std::cout << "CIRCLE SLOT!!!!!!!!" << std::endl;
    break;
  default:
    msg_ApaFsm.park_type_enum = 0;
    msg_SlotsList.slots_list[0].type_enum = 0;
    break;
  }
  if (previous_frame != nullptr) {
    if (FLAGS_enable_guardian) {
      auto parkout_dir_req = frame_->local_view()
                                 .guardian_msg()
                                 ->getFsmRequests()
                                 .getParkOutDirSelected();
      if (parkout_flag) {
        frame_->mutable_open_space_info()->set_is_parkout_planned(
            previous_frame->open_space_info().is_parkout_planned());
        AINFO << "previous_frame->open_space_info().is_parkout_planned(): "
              << previous_frame->open_space_info().is_parkout_planned();

        if (!previous_frame->open_space_info().is_parkout_planned()) {
          AINFO << "Planing parkout path for all direction";
          std::cout << "Planing parkout path for all "
                       "direction===================================="
                    << std::endl;
          msg_ApaFsm.park_type_enum = 3;
        } else {
          if (parkout_dir_req == swift_messages::ParkOutDirSelected::NONE) {
            AINFO << "Waiting for direction request===========================";
            std::cout
                << "Waiting for direction request==========================="
                << std::endl;
            msg_ApaFsm.park_type_enum = 3;
          } else {
            AINFO << "Choose parkout direction with " << (int)parkout_dir_req;
            std::cout << "Choose parkout direction with "
                      << (int)parkout_dir_req
                      << " ====================================" << std::endl;
            switch (parkout_dir_req) {
            case swift_messages::ParkOutDirSelected::LEFT:
              msg_ApaFsm.park_type_enum = 4;
              break;
            case swift_messages::ParkOutDirSelected::MIDDLE:
              msg_ApaFsm.park_type_enum = 5;
              break;
            case swift_messages::ParkOutDirSelected::RIGHT:
              msg_ApaFsm.park_type_enum = 6;
              break;
            }
          }
        }
      }
    } else {
      if (parkout_flag) {
        frame_->mutable_open_space_info()->set_is_parkout_planned(
            previous_frame->open_space_info().is_parkout_planned());
        AINFO << "previous_frame->open_space_info().is_parkout_planned(): "
              << previous_frame->open_space_info().is_parkout_planned();
        if (!previous_frame->open_space_info().is_parkout_planned()) {
          AINFO << "Planing parkout path for all direction";
          std::cout << "Planing parkout path for all "
                       "direction===================================="
                    << std::endl;
          msg_ApaFsm.park_type_enum = 3;
          frame_->mutable_open_space_info()->set_is_parkout_planned(true);
        } else {
          AINFO << "Choose parkout direction";
          std::cout
              << "Choose parkout direction===================================="
              << std::endl;
          msg_ApaFsm.park_type_enum = 4;
        }
      }
    }
    if (previous_frame->open_space_info()
            .open_space_provider_success()) // after planning success, reset
                                            // flags
    {
      msg_ApaFsm.ctrl_run_sts_enmu = 0;
      msg_ApaFsm.plan_run_sts_enum = 0;
      msg_ApaFsm.park_type_enum = 0;
    }
  } else {
    AWARN << "no previous frame exist, clear park out info";
    msg_ApaFsm.ctrl_run_sts_enmu = 0;
    msg_ApaFsm.plan_run_sts_enum = 0;
    msg_ApaFsm.park_type_enum = 0;
    frame_->mutable_open_space_info()->set_is_parkout_planned(false);
  }

  AINFO << "ctrl_run_sts_enmu: " << msg_ApaFsm.ctrl_run_sts_enmu
        << " plan_run_sts_enum: " << msg_ApaFsm.plan_run_sts_enum
        << " park_type_enum: " << msg_ApaFsm.park_type_enum;
  std::cout << "ctrl_run_sts_enmu: " << msg_ApaFsm.ctrl_run_sts_enmu
            << " plan_run_sts_enum: " << msg_ApaFsm.plan_run_sts_enum
            << " park_type_enum: " << msg_ApaFsm.park_type_enum << std::endl;
  apa_planner_->FreeSpaceCallback(msg_LocalMap);
  apa_planner_->SlotCallback(msg_SlotsList);
  apa_planner_->ReplanCallback(msg_ApaCtrl);
  apa_planner_->OdomCallback(msg_Odome);
  apa_planner_->FsmCallBack(msg_ApaFsm);

  // Q
  if (use_rl_policy_) {
    AINFO << "Using RL Policy for trajectory planning";
    try {
      // 从车位多边形获取目标位置（计算所有顶点的平均值）
      swift::common::math::Vec2d target_position(0.0, 0.0);
      if (slot_polygon.num_points() > 0) {
        for (int i = 0; i < slot_polygon.num_points(); ++i) {
          target_position += slot_polygon.points()[i];
        }
        target_position /= slot_polygon.num_points();
      }

      // 计算车位中心的角度（使用车位多边形的前两个点计算方向）
      double target_yaw = 0.0;
      if (slot_polygon.num_points() >= 2) {
        auto p0 = slot_polygon.points().at(0);
        auto p1 = slot_polygon.points().at(1);
        target_yaw = std::atan2(p1.y() - p0.y(), p1.x() - p0.x());
      }

      // 创建空的点云数据（如果没有实际点云数据）
      swift::perception::base::PointDCloud empty_point_cloud;

      // 将 Polygon2d 转换为 ParkingSlot
      swift::planning::open_space::rl_policy::ParkingSlot parking_slot;
      if (slot_polygon.num_points() >= 4) {
        parking_slot.p0 = slot_polygon.points().at(0);
        parking_slot.p1 = slot_polygon.points().at(1);
        parking_slot.p2 = slot_polygon.points().at(2);
        parking_slot.p3 = slot_polygon.points().at(3);
        parking_slot.angle = target_yaw;
        // 计算车位宽度（使用前两个点的距离作为宽度估计）
        parking_slot.width =
            slot_polygon.points().at(0).DistanceTo(slot_polygon.points().at(1));
        parking_slot.type =
            swift::planning::open_space::rl_policy::ParkingType::VERTICAL;
      }

      // 将指针向量转换为对象向量
      std::vector<swift::planning::Obstacle> obstacle_objects;
      for (const auto *obstacle_ptr : frame_->obstacles()) {
        if (obstacle_ptr) {
          obstacle_objects.push_back(*obstacle_ptr);
        }
      }

      // 构建观察数据
      auto observation = observation_builder_->BuildObservationFromParkingSlot(
          empty_point_cloud, frame_->vehicle_state(), obstacle_objects,
          parking_slot,
          nullptr, // reference_line
          10.0,    // lidar_max_range
          20.0,    // img_view_range
          false    // is_wheel_stop_valid
      );

      AINFO << "Built RL observation with dimensions: "
            << observation.flattened.size();
      AINFO << "Target position: (" << target_position.x() << ", "
            << target_position.y() << ")";
      AINFO << "Target yaw: " << target_yaw;

      // 使用观察数据获取RL动作建议
      // auto action = hope_adapter_->GetAction(observation);

    } catch (const std::exception &e) {
      AERROR << "Error in RL Policy: " << e.what();
    }
  }

  msg_ApaPlanInfo path_info = apa_planner_->GetStartPlan();

  // 对生成的路径进行二次规划平滑
  if (path_info.num_of_spot > 0) {
    AINFO << "Applying path smoothing to APA generated path with "
          << path_info.num_of_spot << " points";

    // 创建路径平滑器配置
    ApaPathSmootherConfig smoother_config;
    smoother_config.spline_order = 5;
    smoother_config.num_of_total_points = 100;
    smoother_config.weight_smooth = 1.0;
    smoother_config.weight_length = 0.1;
    smoother_config.weight_curvature = 0.1;
    smoother_config.weight_heading = 0.1;
    smoother_config.delta_t = 0.1;
    smoother_config.max_curvature = 0.3;
    smoother_config.max_heading_change = 0.5;

    // 创建路径平滑器实例
    std::unique_ptr<ApaPathSmoother> path_smoother =
        std::make_unique<ApaPathSmoother>(smoother_config);

    // 执行路径平滑
    msg_ApaPlanInfo smoothed_path_info = path_info;
    if (path_smoother->Smooth(path_info, &smoothed_path_info)) {
      AINFO << "Path smoothing successful. Original points: "
            << path_info.num_of_spot
            << ", Smoothed points: " << smoothed_path_info.num_of_spot;

      // 使用平滑后的路径替换原始路径
      path_info = smoothed_path_info;

      // 保存平滑后的路径到CSV文件用于可视化
      auto now = std::chrono::system_clock::now();
      auto time_t = std::chrono::system_clock::to_time_t(now);
      std::stringstream ss;
      ss << "smoothed_apa_path_"
         << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << ".csv";
      std::string filename = ss.str();

      std::ofstream csv_file(filename);
      if (csv_file.is_open()) {
        csv_file << "Point_Index,X,Y,Yaw,Velocity,Direction,Description\n";
        for (int i = 0; i < path_info.num_of_spot; ++i) {
          csv_file << i << "," << path_info.spot_list[i].x_m << ","
                   << path_info.spot_list[i].y_m << ","
                   << path_info.spot_list[i].yaw_rad << ","
                   << path_info.spot_list[i].v_mps << ","
                   << path_info.spot_list[i].dir_enum << ",平滑后路径点" << i
                   << "\n";
        }
        csv_file.close();
        std::cout << "平滑后的路径数据已保存到CSV文件: " << filename
                  << std::endl;
      }
    } else {
      AERROR << "Path smoothing failed, using original path";
    }
  }

  if (FLAGS_enable_guardian && previous_frame != nullptr) {
    auto parkout_dir_req = frame_->local_view()
                               .guardian_msg()
                               ->getFsmRequests()
                               .getParkOutDirSelected();
    if (parkout_dir_req == swift_messages::ParkOutDirSelected::NONE &&
        !previous_frame->open_space_info().is_parkout_planned()) {
      auto *apa_info = injector_->planning_context()
                           ->mutable_planning_status()
                           ->mutable_valet_parking_status()
                           ->mutable_apa_info();
      AINFO << "Waiting for direction request===========================";
      std::cout << "Waiting for direction request==========================="
                << std::endl;
      AINFO << "park_out_direction_num = " << path_info.park_out_direction_enum;
      if (path_info.park_out_direction_enum > 0) {
        std::cout << "ParkOut Plan Success!!!!!!!" << std::endl;
        frame_->mutable_open_space_info()->set_is_parkout_planned(true);
      }
      switch (path_info.park_out_direction_enum) {
      case 0: {
        apa_info->mutable_parkout_avail()->set_parkout_left_avail(false);
        apa_info->mutable_parkout_avail()->set_parkout_right_avail(false);
        apa_info->mutable_parkout_avail()->set_parkout_middle_avail(false);
        break;
      }
      case 1: {
        apa_info->mutable_parkout_avail()->set_parkout_left_avail(true);
        apa_info->mutable_parkout_avail()->set_parkout_right_avail(false);
        apa_info->mutable_parkout_avail()->set_parkout_middle_avail(false);
        break;
      }
      case 2: {
        apa_info->mutable_parkout_avail()->set_parkout_left_avail(false);
        apa_info->mutable_parkout_avail()->set_parkout_right_avail(false);
        apa_info->mutable_parkout_avail()->set_parkout_middle_avail(true);
        break;
      }
      case 3: // 1+2
      {
        apa_info->mutable_parkout_avail()->set_parkout_left_avail(true);
        apa_info->mutable_parkout_avail()->set_parkout_right_avail(false);
        apa_info->mutable_parkout_avail()->set_parkout_middle_avail(true);
        break;
      }
      case 4: // 4
      {
        apa_info->mutable_parkout_avail()->set_parkout_left_avail(false);
        apa_info->mutable_parkout_avail()->set_parkout_right_avail(true);
        apa_info->mutable_parkout_avail()->set_parkout_middle_avail(false);
        break;
      }
      case 5: // 1+4
      {
        apa_info->mutable_parkout_avail()->set_parkout_left_avail(true);
        apa_info->mutable_parkout_avail()->set_parkout_right_avail(true);
        apa_info->mutable_parkout_avail()->set_parkout_middle_avail(false);
        break;
      }
      case 6: // 2+4
      {
        apa_info->mutable_parkout_avail()->set_parkout_left_avail(false);
        apa_info->mutable_parkout_avail()->set_parkout_right_avail(true);
        apa_info->mutable_parkout_avail()->set_parkout_middle_avail(true);
        break;
      }
      case 7: // 1+2+4
      {
        apa_info->mutable_parkout_avail()->set_parkout_left_avail(true);
        apa_info->mutable_parkout_avail()->set_parkout_right_avail(true);
        apa_info->mutable_parkout_avail()->set_parkout_middle_avail(true);
        break;
      }
      default:
        break;
      }
    }
  }
  if (FLAGS_enable_apa_csv_dump) {
    for (int i = 0; i < path_info.num_of_spot; ++i) {
      auto &spot = path_info.spot_list[i];
      double theta_w = NormalizeAngle(spot.yaw_rad + loc_heading);

      ofs_traj << std::fixed << std::setprecision(6) << spot.x_m << ','
               << spot.y_m << ',' << theta_w << ',' << spot.v_mps << '\n';
    }
  }

  if (previous_frame != nullptr) {
    if (previous_frame->open_space_info().open_space_provider_success()) {
      if (!(is_need_replan &&
            !previous_frame->open_space_info().apa_replan_flag())) {
        trajectory_data->clear();
        *trajectory_data =
            previous_frame->open_space_info().stitched_trajectory_result();
        // if(){
        frame_->mutable_open_space_info()->set_open_space_provider_success(
            true);
        frame_->mutable_open_space_info()->set_apa_replan_flag(false);
        return Status::OK();
      }
    } else {
      injector_->planning_context()
          ->mutable_planning_status()
          ->mutable_open_space()
          ->Clear();
    }
  }
  if (!path_info.num_of_spot) {
    if (previous_frame != nullptr) {
      if (parkout_flag &&
          previous_frame->open_space_info().is_parkout_planned()) {
        AINFO << "ready to plan parkout";
        std::cout << "ready to plan parkout" << std::endl;
        trajectory_data->clear();
        return Status::OK();
      }
    }
    AERROR << "Planning fail.";
    return Status(ErrorCode::PLANNING_ERROR, "Planning fail.");
  }

  const auto &open_space_info = frame_->open_space_info();
  const auto &end_pose = open_space_info.open_space_end_pose();
  const auto &rotate_angle = open_space_info.origin_heading();
  const auto &translate_origin = open_space_info.origin_point();
  const auto &obstacles_edges_num = open_space_info.obstacles_edges_num();
  const auto &obstacles_A = open_space_info.obstacles_A();
  const auto &obstacles_b = open_space_info.obstacles_b();
  const auto &obstacles_vertices_vec = open_space_info.obstacles_vertices_vec();
  const auto &XYbounds = open_space_info.ROI_xy_boundary();

  // Generate Trajectory;
  double time_latency;
  std::vector<common::TrajectoryPoint> stitching_trajectory;

  for (int i = 0; i < path_info.num_of_spot; i++) {
    common::TrajectoryPoint point;
    point.mutable_path_point()->set_x(path_info.spot_list[i].x_m);
    point.mutable_path_point()->set_y(path_info.spot_list[i].y_m);
    point.mutable_path_point()->set_theta(path_info.spot_list[i].yaw_rad);
    stitching_trajectory.push_back(point);
  }

  Status status = open_space_trajectory_optimizer_->Plan(
      stitching_trajectory, end_pose, XYbounds, rotate_angle, translate_origin,
      obstacles_edges_num, obstacles_A, obstacles_b, obstacles_vertices_vec,
      true, &time_latency);
  trajectory_data->clear();

  auto optimizer_trajectory_ptr =
      frame_->mutable_open_space_info()->mutable_optimizer_trajectory_data();
  open_space_trajectory_optimizer_->GetOptimizedTrajectory(
      optimizer_trajectory_ptr);
  for (size_t idm = 0; idm < optimizer_trajectory_ptr->size(); idm++) {
    common::TrajectoryPoint Point;
    double dx =
        optimizer_trajectory_ptr->at(idm).path_point().x() * cos(loc_heading) -
        optimizer_trajectory_ptr->at(idm).path_point().y() * sin(loc_heading);
    double dy =
        optimizer_trajectory_ptr->at(idm).path_point().x() * sin(loc_heading) +
        optimizer_trajectory_ptr->at(idm).path_point().y() * cos(loc_heading);
    Point.mutable_path_point()->set_x(dx + loc_x);
    Point.mutable_path_point()->set_y(dy + loc_y);
    double theta_i = NormalizeAngle(
        optimizer_trajectory_ptr->at(idm).path_point().theta() + loc_heading);
    Point.mutable_path_point()->set_theta(theta_i);
    Point.mutable_path_point()->set_kappa(
        optimizer_trajectory_ptr->at(idm).path_point().kappa());
    Point.mutable_path_point()->set_s(
        optimizer_trajectory_ptr->at(idm).path_point().s());
    Point.mutable_path_point()->set_z(
        optimizer_trajectory_ptr->at(idm).path_point().z());
    Point.set_a(optimizer_trajectory_ptr->at(idm).a());
    Point.set_v(optimizer_trajectory_ptr->at(idm).v());
    Point.set_relative_time(optimizer_trajectory_ptr->at(idm).relative_time());
    // std::cout << "idm = " << idm << " time_s = " <<
    // optimizer_trajectory_ptr->at(idm).relative_time() << std::endl;
    trajectory_data->emplace_back(Point);
    // AINFO << "provider idm = " << idm << " kappa = " <<
    // optimizer_trajectory_ptr->at(idm).path_point().kappa();
  }

  if (FLAGS_enable_apa_csv_dump) {
    // --- CLOSE CSV FILES ---
    ofs_odom.close();
    ofs_obs.close();
    ofs_localmap.close();
    ofs_traj.close();
    ofs_slot.close();
  }
  frame_->mutable_open_space_info()->set_open_space_provider_success(true);
  return Status::OK();
}

void OpenSpaceTrajectoryProvider::ModifiySpeedAndCurvatureForApaTrajectory(
    const msg_ApaPlanInfo apa_plan_info_before,
    msg_ApaPlanInfo &apa_plan_info_after, vector<double> &acc_data) {
  // 分割轨迹
  std::vector<msg_ApaSpot> path_points;
  std::vector<std::vector<msg_ApaSpot>> path_list;
  for (int i = 0; i < apa_plan_info_before.num_of_spot; i++) {
    if (apa_plan_info_before.spot_list[i + 1].dir_enum ==
        apa_plan_info_before.spot_list[i].dir_enum) {
      path_points.push_back(apa_plan_info_before.spot_list[i]);
      double temp_angle = apa_plan_info_before.spot_list[i].yaw_rad;
      path_points.back().yaw_rad = NormalizeAngle(temp_angle);
    } else {
      path_points.push_back(apa_plan_info_before.spot_list[i]);
      double temp_angle = apa_plan_info_before.spot_list[i].yaw_rad;
      path_points.back().yaw_rad = NormalizeAngle(temp_angle);
      path_list.push_back(path_points);
      path_points.clear();
    }
  }

  const double pt_internal = config_.open_space_trajectory_provider_config()
                                 .open_space_trajectory_optimizer_config()
                                 .planner_open_space_config()
                                 .apa_point_internal_dist();
  vector<float> curr_data, dist_data, interpdist_data;
  vector<vector<float>> dist_data_list;
  float dist_err, yaw_err, cur_index, cur_diff = 0;
  float disttmp, totaldist = 0;
  uint16_t modified_size = 0;
  for (int idm = 0; idm < path_list.size(); idm++) {
    auto path = path_list[idm];
    dist_data.clear();
    curr_data.clear();
    interpdist_data.clear();
    totaldist = 0;
    dist_data.push_back(totaldist);

    for (int jdm = 0; jdm < path.size() - 1; jdm++) {
      if (path[jdm + 1].x_m != path[jdm].x_m ||
          path[jdm + 1].y_m != path[jdm].y_m) {
        dist_err = sqrtf((path[jdm + 1].x_m - path[jdm].x_m) *
                             (path[jdm + 1].x_m - path[jdm].x_m) +
                         (path[jdm + 1].y_m - path[jdm].y_m) *
                             (path[jdm + 1].y_m - path[jdm].y_m));
        yaw_err = path[jdm + 1].yaw_rad - path[jdm].yaw_rad;
        totaldist = totaldist + dist_err;
        dist_data.push_back(totaldist);
        cur_index = yaw_err / dist_err;
        if (fabs(cur_index) > 0.2) {
          if (curr_data.size() > 0)
            cur_index = curr_data.back();
          else
            cur_index = 0;
        }
        curr_data.push_back(cur_index);
      } else {
        if (curr_data.size() > 0)
          curr_data.push_back(curr_data.back());
        else
          curr_data.push_back(0.0);
      }
    }
    curr_data.push_back(curr_data.back());

    for (int jdm = 0; jdm < path.size(); jdm++) {
      // msg_ApaSpot pt = path[jdm];
      if (jdm < 1) {
        // pt = path[0];
        // apa_plan_info_after.spot_list[modified_size] = pt;
        cur_diff = (curr_data[0] + curr_data[0]) / 2;
      } else if (jdm == path.size() - 1)
        cur_diff = curr_data[jdm];
      else
        cur_diff = (curr_data[jdm] + curr_data[jdm - 1]) / 2;
      if (fabs(cur_diff) > 0.2) {
        if (jdm > 0)
          cur_diff = curr_data[jdm - 1];
        else
          cur_diff = 0;
      }
      // apa_plan_info_after.spot_list[modified_size + jdm].cur_mps2 = cur_diff;
      path[jdm].cur_mps2 = cur_diff;
    }
    double interpdist = pt_internal;
    float L_num = round((totaldist) / interpdist); //  - 0.2 为了纯跟踪
    // if(inp_Pkg4_SlotOut.Slot_Shape == 1 || inp_Pkg4_SlotOut.Slot_Shape == 3)
    //     L_num = round((totaldist) / interpdist);

    for (int i = 0; i <= L_num; i++) {
      interpdist_data.push_back(i * interpdist);
    }
    float k_x = 0;
    float b_x = 0;
    float k_y = 0;
    float b_y = 0;
    float k_yaw = 0;
    float b_yaw = 0;
    float k_cur = 0;
    float b_cur = 0;
    int num = 0;
    vector<float> x_interp, y_interp, yaw_interp, cur_interp;
    for (int i = 0; i < int(interpdist_data.size()); i++) {
      if (interpdist_data[i] > dist_data[num + 1])
        num = num + 1;
      if (num < dist_data.size() - 1) {
        k_x = (path[num + 1].x_m - path[num].x_m) /
              (dist_data[num + 1] - dist_data[num]);
        b_x = path[num + 1].x_m - k_x * dist_data[num + 1];
        k_y = (path[num + 1].y_m - path[num].y_m) /
              (dist_data[num + 1] - dist_data[num]);
        b_y = path[num + 1].y_m - k_y * dist_data[num + 1];
        k_yaw = (path[num + 1].yaw_rad - path[num].yaw_rad) /
                (dist_data[num + 1] - dist_data[num]);
        b_yaw = path[num + 1].yaw_rad - k_yaw * dist_data[num + 1];
        k_cur = (path[num + 1].cur_mps2 - path[num].cur_mps2) /
                (dist_data[num + 1] - dist_data[num]);
        b_cur = path[num + 1].cur_mps2 - k_cur * dist_data[num + 1];
        x_interp.push_back(k_x * interpdist_data[i] + b_x);
        y_interp.push_back(k_y * interpdist_data[i] + b_y);
        yaw_interp.push_back(k_yaw * interpdist_data[i] + b_yaw);
        cur_interp.push_back(k_cur * interpdist_data[i] + b_cur);
      }
    }
    // 保证插值前最后一个点在
    if (!(fabs(x_interp.back() - path[path.size() - 1].x_m) < 0.00001 &&
          fabs(y_interp.back() - path[path.size() - 1].y_m) < 0.00001)) {
      // 保证插值前最后一个点在
      x_interp.push_back(path[path.size() - 1].x_m);
      y_interp.push_back(path[path.size() - 1].y_m);
      yaw_interp.push_back(path[path.size() - 1].yaw_rad);
      cur_interp.push_back(path[path.size() - 1].cur_mps2);
    }

    for (int kdm = 0; kdm < x_interp.size(); kdm++) {
      apa_plan_info_after.spot_list[modified_size + kdm].x_m = x_interp[kdm];
      apa_plan_info_after.spot_list[modified_size + kdm].y_m = y_interp[kdm];
      apa_plan_info_after.spot_list[modified_size + kdm].yaw_rad =
          yaw_interp[kdm];
      apa_plan_info_after.spot_list[modified_size + kdm].cur_mps2 =
          cur_interp[kdm];
      apa_plan_info_after.spot_list[modified_size + kdm].dir_enum =
          path[0].dir_enum;
    }

    // dist_data_list.push_back(dist_data);
    // }

    // 为每段路径规划速度
    const double max_speed = config_.open_space_trajectory_provider_config()
                                 .open_space_trajectory_optimizer_config()
                                 .planner_open_space_config()
                                 .apa_max_speed(); // 最大速度 1.0 m/s
    const double acceleration = config_.open_space_trajectory_provider_config()
                                    .open_space_trajectory_optimizer_config()
                                    .planner_open_space_config()
                                    .apa_acceleration(); // 加速度 0.5 m/s^2
    double deceleration = -acceleration;                 // 减速度 -0.5 m/s^2
    // for (int idm = 0; idm < path_list.size(); idm++) {
    // auto& path = path_list[idm];
    double total_distance = totaldist; // 当前路径段的总长度
    double adjust_max_speed =
        path[0].dir_enum < 0 ? max(0.0, max_speed) : max_speed;
    if (total_distance < 2.0) {
      // acceleration = max(0.15, acceleration - 0.1);
      deceleration = -(acceleration - 0.1);
      adjust_max_speed = adjust_max_speed - 0.2;
    }
    double time_to_max_speed = adjust_max_speed / acceleration;
    double acc_distance =
        0.5 * acceleration * time_to_max_speed * time_to_max_speed;
    double dec_distance = adjust_max_speed * adjust_max_speed /
                          (2 * std::abs(deceleration)); // + pt_internal;
    // AINFO << "before adjust acc_distance: " << acc_distance << "
    // dec_distance: " << dec_distance;
    // 如果总距离小于加速减速距离之和，需要调整最大速度
    if (total_distance < (acc_distance + dec_distance)) {
      // 重新计算峰值速度
      double adjust_max_speed =

          std::sqrt(total_distance * acceleration * std::abs(deceleration) /
                    (acceleration + std::abs(deceleration)));
      acc_distance = adjust_max_speed * adjust_max_speed / (2 * acceleration);
      dec_distance = adjust_max_speed * adjust_max_speed /
                     (2 * std::abs(deceleration)); // + pt_internal;
    }

    // 为每个点分配速度
    for (int jdm = 0; jdm < x_interp.size(); jdm++) {
      if (jdm < 1) {
        apa_plan_info_after.spot_list[modified_size + jdm].v_mps = 0;
        continue;
      }
      double current_dist = interpdist_data[jdm]; // - 1];
      double speed;
      double a;
      // if(jdm < 1){
      //   apa_plan_info_after.spot_list[modified_size].v_mps = 0;
      // }
      if (current_dist <= acc_distance) {
        // 加速阶段
        speed = std::sqrt(2 * acceleration * current_dist);
        a = acceleration;
      } else if (current_dist >= (total_distance - dec_distance)) {
        // 减速阶段
        double dist_to_end = total_distance - current_dist; // - pt_internal;
        if (2 * std::abs(deceleration) * dist_to_end < 0) {
          speed = 0;
          a = 0;
        } else {
          speed = std::sqrt(2 * std::abs(deceleration) * dist_to_end);
          a = deceleration;
        }
      } else {
        // 匀速阶段
        speed = adjust_max_speed;
        a = 0;
      }
      if (speed < 0)
        speed = 0;
      if (speed > adjust_max_speed)
        speed = adjust_max_speed;

      // 设置速度
      apa_plan_info_after.spot_list[modified_size + jdm].v_mps =
          speed * (path[0].dir_enum < 0 ? -1.0 : 1.0);
      acc_data.push_back(a * (path[0].dir_enum < 0 ? -1.0 : 1.0));
    }
    modified_size += x_interp.size();
    apa_plan_info_after.spot_list[modified_size - 1].v_mps = 0;
    acc_data.push_back(0);
    apa_plan_info_after.num_of_spot = modified_size;
  }
  // apa_plan_info_after.num_of_spot = modified_size;
  // apa_plan_info_after.park_type_enum = apa_plan_info_before.park_type_enum;
}
void OpenSpaceTrajectoryProvider::InterpolateObstaclesVertices(
    std::vector<std::vector<Vec2d>> Obs_list, msg_LocalMap &local_map) {
  const double point_distance = 0.3; // 点间距为10cm
  size_t total_num_of_vertices = 0;
  // for (const auto& obstacle :
  // frame_->open_space_info().obstacles_vertices_vec()) {
  for (const auto &obstacle : Obs_list) {
    // std::vector<common::math::Vec2d> interpolated_vertices;
    size_t num_vertices = obstacle.size();

    for (size_t i = 0; i < num_vertices; ++i) {
      const auto &start_vertex = obstacle[i];
      const auto &end_vertex = obstacle[(i + 1) % num_vertices];

      // interpolated_vertices.push_back(start_vertex);

      double distance = (end_vertex - start_vertex).Length();
      size_t num_interpolated_points =
          static_cast<size_t>(distance / point_distance);

      for (size_t j = 1; j <= num_interpolated_points; ++j) {
        double ratio = static_cast<double>(j) / num_interpolated_points;
        common::math::Vec2d interpolated_point =
            start_vertex + (end_vertex - start_vertex) * ratio;
        // interpolated_vertices.push_back(interpolated_point);
        local_map.vertices_list[total_num_of_vertices].x_m =
            interpolated_point.x();
        local_map.vertices_list[total_num_of_vertices].y_m =
            interpolated_point.y();
        total_num_of_vertices += 1;
      }
    }
    local_map.num_of_vertices = total_num_of_vertices;
  }
}

void OpenSpaceTrajectoryProvider::LoadResult(
    DiscretizedTrajectory *const trajectory_data) {
  // Load unstitched two trajectories into frame for debug
  trajectory_data->clear();
  auto optimizer_trajectory_ptr =
      frame_->mutable_open_space_info()->mutable_optimizer_trajectory_data();
  auto stitching_trajectory_ptr =
      frame_->mutable_open_space_info()->mutable_stitching_trajectory_data();
  open_space_trajectory_optimizer_->GetOptimizedTrajectory(
      optimizer_trajectory_ptr);
  open_space_trajectory_optimizer_->GetStitchingTrajectory(
      stitching_trajectory_ptr);
  // Stitch two trajectories and load back to trajectory_data from frame
  size_t optimizer_trajectory_size = optimizer_trajectory_ptr->size();
  double stitching_point_relative_time =
      stitching_trajectory_ptr->back().relative_time();
  double stitching_point_relative_s =
      stitching_trajectory_ptr->back().path_point().s();
  for (size_t i = 0; i < optimizer_trajectory_size; ++i) {
    optimizer_trajectory_ptr->at(i).set_relative_time(
        optimizer_trajectory_ptr->at(i).relative_time() +
        stitching_point_relative_time);
    optimizer_trajectory_ptr->at(i).mutable_path_point()->set_s(
        optimizer_trajectory_ptr->at(i).path_point().s() +
        stitching_point_relative_s);
  }
  *(trajectory_data) = *(optimizer_trajectory_ptr);

  // Last point in stitching trajectory is already in optimized trajectory, so
  // it is deleted
  frame_->mutable_open_space_info()
      ->mutable_stitching_trajectory_data()
      ->pop_back();
  trajectory_data->PrependTrajectoryPoints(
      frame_->open_space_info().stitching_trajectory_data());
  frame_->mutable_open_space_info()->set_open_space_provider_success(true);
  frame_->mutable_open_space_info()->set_finished_apa_stop_path(true);
}

void OpenSpaceTrajectoryProvider::ReuseLastFrameResult(
    const Frame *last_frame, DiscretizedTrajectory *const trajectory_data) {
  *(trajectory_data) =
      last_frame->open_space_info().stitched_trajectory_result();
  frame_->mutable_open_space_info()->set_open_space_provider_success(true);
  frame_->mutable_open_space_info()->set_finished_apa_stop_path(true);
}

void OpenSpaceTrajectoryProvider::ReuseLastFrameDebug(const Frame *last_frame) {
  // reuse last frame's instance
  auto *ptr_debug = frame_->mutable_open_space_info()->mutable_debug_instance();
  ptr_debug->mutable_planning_data()->mutable_open_space()->MergeFrom(
      last_frame->open_space_info()
          .debug_instance()
          .planning_data()
          .open_space());
}

std::vector<std::vector<common::math::Vec2d>>
OpenSpaceTrajectoryProvider::LoadObstacleVertices() const {
  std::vector<std::vector<common::math::Vec2d>> obs_vertices;
  obs_vertices.reserve(frame_->obstacles().size());
  const auto &vehicle_state = frame_->vehicle_state();
  common::math::Vec2d ego_pos =
      common::math::Vec2d(vehicle_state.x(), vehicle_state.y());
  for (const auto &obstacle : frame_->obstacles()) {
    if (obstacle->IsVirtual())
      continue;
    if (obstacle->PerceptionBoundingBox().DistanceTo(ego_pos) < 10.0) {
      obs_vertices.push_back(obstacle->PerceptionBoundingBox().GetAllCorners());
    }
  }
  return obs_vertices;
}

void OpenSpaceTrajectoryProvider::LoadFreeSpaceVertices(
    msg_LocalMap &local_map) const {
  if (!frame_->local_view().freespace_msg()) {
    AWARN << "Free space message is unavailable";
    return;
  }

  // Where to start appending in local_map
  size_t idx = local_map.num_of_vertices;
  constexpr double min_dist = 0.3; // minimum spacing
  constexpr double min_sq = min_dist * min_dist;
  constexpr double max_dist_from_ego = 10.0;
  constexpr double max_dist_from_ego_sq = max_dist_from_ego * max_dist_from_ego;
  constexpr size_t max_vertices = 600;

  // First pass: collect all valid points with downsampling
  std::vector<msg_Vertex> candidates;
  candidates.reserve(1000); // reasonable initial capacity

  msg_Vertex last_pt;
  bool have_last = false;
  size_t freespace_pt_num = 0;
  size_t filtered_freespace_pt_num = 0;
  for (const auto &point :
       frame_->local_view().freespace_msg()->getOccSurfacePoint()) {
    if (point.getClass() == swift_messages::OccTypeOnboard::FREESPACE) {
      continue; // skip pure freespace cells
    }

    // --- Rotate by –π/2: (x,y) → ( y, –x ) ---
    const double raw_x = point.getObjX();
    const double raw_y = point.getObjY();
    msg_Vertex cur;
    cur.x_m = raw_y;
    cur.y_m = -raw_x;
    freespace_pt_num++;
    // --- skip if farther than 10 m from origin ---
    double dist_sq = cur.x_m * cur.x_m + cur.y_m * cur.y_m;
    if (dist_sq > max_dist_from_ego_sq) {
      continue;
    }

    // Apply downsampling: only keep if it's at least 0.3m away from last point
    if (!have_last) {
      have_last = true;
      last_pt = cur;
      candidates.push_back(cur);
      continue;
    }

    const double dx = cur.x_m - last_pt.x_m;
    const double dy = cur.y_m - last_pt.y_m;
    if (dx * dx + dy * dy >= min_sq) {
      last_pt = cur;
      candidates.push_back(cur);
      filtered_freespace_pt_num++;
    }
  }
  AINFO << "FreespacePointNum: " << freespace_pt_num;
  AINFO << "FilteredFreespacePointNum: " << filtered_freespace_pt_num;
  // If we have <= 600 points, use them all
  if (candidates.size() <= max_vertices) {
    for (const auto &vertex : candidates) {
      if (idx >= 600)
        break;
      local_map.vertices_list[idx++] = vertex;
    }
  } else {
    // Sort by distance from ego and take the closest 600
    std::partial_sort(candidates.begin(), candidates.begin() + max_vertices,
                      candidates.end(),
                      [](const msg_Vertex &a, const msg_Vertex &b) {
                        double dist_a_sq = a.x_m * a.x_m + a.y_m * a.y_m;
                        double dist_b_sq = b.x_m * b.x_m + b.y_m * b.y_m;
                        return dist_a_sq < dist_b_sq;
                      });

    // Take the 600 closest points
    for (size_t i = 0; i < max_vertices && idx < 600; ++i) {
      local_map.vertices_list[idx++] = candidates[i];
    }
  }

  // update count
  local_map.num_of_vertices = idx;
}

} // namespace planning
} // namespace swift
