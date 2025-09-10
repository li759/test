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
 * @file base_configure.h
 * @brief Base configuration for RL Policy module
 */

#pragma once

#ifndef base_configure_h
#define base_configure_h

#include <algorithm>
#include <chrono>
#include <eigen3/Eigen/Dense>
#include <math.h>
#include <memory>
#include <thread>
#include <vector>

// RL Policy specific constants
const double Rad2Ang = 57.29578f;
const double Ang2Rad = 57.29578f;
const double Pi = 3.1415926f;

// Vehicle parameters
const double VEHICLE_LENGTH = 4.8;
const double VEHICLE_WIDTH = 2.0;
const double VEHICLE_WHEELBASE = 2.8;
const double VEHICLE_MAX_STEERING = 0.48f;
const double VEHICLE_MIN_RADIUS = 6.0;

// RL Observation dimensions
const int LIDAR_DIM = 120;
const int TARGET_DIM = 5;
const int IMG_DIM = 12288; // 3 * 64 * 64
const int ACTION_MASK_DIM = 42;
const int OBS_DIM = 12455; // LIDAR_DIM + TARGET_DIM + IMG_DIM + ACTION_MASK_DIM

// Lidar parameters
const double LIDAR_MAX_RANGE = 10.0;
const double LIDAR_FOV = 2.0 * Pi;
const int LIDAR_NUM_BEAMS = 120;

// Image parameters
const int IMG_WIDTH = 64;
const int IMG_HEIGHT = 64;
const int IMG_CHANNELS = 3;
const double IMG_VIEW_RANGE = 20.0;
const double IMG_RESOLUTION = IMG_VIEW_RANGE / std::max(IMG_WIDTH, IMG_HEIGHT);

// Action space parameters
const int NUM_STEERING_ACTIONS = 7;
const int NUM_STEP_LENGTHS = 6;
const double MIN_STEERING = -1.0;
const double MAX_STEERING = 1.0;
const double MIN_STEP_LENGTH = 0.1;
const double MAX_STEP_LENGTH = 1.0;

// Environment parameters
const int MAX_EPISODE_LENGTH = 1000;
const double COLLISION_PENALTY = -100.0;
const double SUCCESS_REWARD = 100.0;
const double STEP_PENALTY = -0.1;
const double TARGET_REACHED_THRESHOLD = 0.5;
const double HEADING_ERROR_THRESHOLD = 0.1;

// Debug parameters
const bool ENABLE_VISUALIZATION = true;
const bool SAVE_TRAJECTORY = false;
const bool PRINT_OBSERVATION_STATS = true;

// RL Policy specific structures
struct RLVehicleState {
  double x;
  double y;
  double yaw;
  double speed;
  double steering_angle;

  RLVehicleState()
      : x(0.0), y(0.0), yaw(0.0), speed(0.0), steering_angle(0.0) {}
  RLVehicleState(double x, double y, double yaw, double speed,
                 double steering_angle)
      : x(x), y(y), yaw(yaw), speed(speed), steering_angle(steering_angle) {}
};

struct RLAction {
  double steering_angle;
  double step_length;

  RLAction() : steering_angle(0.0), step_length(0.0) {}
  RLAction(double steering, double step)
      : steering_angle(steering), step_length(step) {}
};

struct RLObservation {
  std::vector<float> lidar;
  std::vector<float> target;
  std::vector<float> img;
  std::vector<float> action_mask;
  std::vector<float> flattened;

  RLObservation() {
    lidar.resize(LIDAR_DIM, 0.0f);
    target.resize(TARGET_DIM, 0.0f);
    img.resize(IMG_DIM, 0.0f);
    action_mask.resize(ACTION_MASK_DIM, 1.0f);
    flattened.resize(OBS_DIM, 0.0f);
  }

  void Flatten() {
    flattened.clear();
    flattened.reserve(OBS_DIM);
    flattened.insert(flattened.end(), lidar.begin(), lidar.end());
    flattened.insert(flattened.end(), target.begin(), target.end());
    flattened.insert(flattened.end(), img.begin(), img.end());
    flattened.insert(flattened.end(), action_mask.begin(), action_mask.end());
  }
};

// Configuration manager for RL Policy
class RLPolicyConfigManager {
public:
  static RLPolicyConfigManager &Get() {
    static RLPolicyConfigManager instance;
    return instance;
  }

  // Vehicle parameters
  double getVehicleLength() const { return VEHICLE_LENGTH; }
  double getVehicleWidth() const { return VEHICLE_WIDTH; }
  double getVehicleWheelbase() const { return VEHICLE_WHEELBASE; }
  double getVehicleMaxSteering() const { return VEHICLE_MAX_STEERING; }
  double getVehicleMinRadius() const { return VEHICLE_MIN_RADIUS; }

  // Observation parameters
  int getLidarDim() const { return LIDAR_DIM; }
  int getTargetDim() const { return TARGET_DIM; }
  int getImgDim() const { return IMG_DIM; }
  int getActionMaskDim() const { return ACTION_MASK_DIM; }
  int getObsDim() const { return OBS_DIM; }

  // Lidar parameters
  double getLidarMaxRange() const { return LIDAR_MAX_RANGE; }
  double getLidarFOV() const { return LIDAR_FOV; }
  int getLidarNumBeams() const { return LIDAR_NUM_BEAMS; }

  // Image parameters
  int getImgWidth() const { return IMG_WIDTH; }
  int getImgHeight() const { return IMG_HEIGHT; }
  int getImgChannels() const { return IMG_CHANNELS; }
  double getImgViewRange() const { return IMG_VIEW_RANGE; }
  double getImgResolution() const { return IMG_RESOLUTION; }

  // Action space parameters
  int getNumSteeringActions() const { return NUM_STEERING_ACTIONS; }
  int getNumStepLengths() const { return NUM_STEP_LENGTHS; }
  double getMinSteering() const { return MIN_STEERING; }
  double getMaxSteering() const { return MAX_STEERING; }
  double getMinStepLength() const { return MIN_STEP_LENGTH; }
  double getMaxStepLength() const { return MAX_STEP_LENGTH; }

  // Environment parameters
  int getMaxEpisodeLength() const { return MAX_EPISODE_LENGTH; }
  double getCollisionPenalty() const { return COLLISION_PENALTY; }
  double getSuccessReward() const { return SUCCESS_REWARD; }
  double getStepPenalty() const { return STEP_PENALTY; }
  double getTargetReachedThreshold() const { return TARGET_REACHED_THRESHOLD; }
  double getHeadingErrorThreshold() const { return HEADING_ERROR_THRESHOLD; }

  // Debug parameters
  bool getEnableVisualization() const { return ENABLE_VISUALIZATION; }
  bool getSaveTrajectory() const { return SAVE_TRAJECTORY; }
  bool getPrintObservationStats() const { return PRINT_OBSERVATION_STATS; }

private:
  RLPolicyConfigManager() = default;
  ~RLPolicyConfigManager() = default;

  // Disable copy constructor and assignment operator
  RLPolicyConfigManager(const RLPolicyConfigManager &) = delete;
  RLPolicyConfigManager &operator=(const RLPolicyConfigManager &) = delete;
};

#endif
