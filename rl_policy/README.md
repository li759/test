# RL Policy Module

## 概述

RL Policy模块是Swift规划系统中的强化学习策略模块，用于在开放空间场景中进行智能决策。该模块从Swift现有的感知、规划和车辆状态数据中提取信息，构建12455维的观测空间，并通过RL模型进行动作推理。

## 目录结构

```
rl_policy/
├── base_configure.h              # 基础配置头文件
├── BUILD                         # Bazel构建文件
├── config/                       # 配置文件目录
│   └── rl_policy_config.yaml    # RL策略配置文件
├── extractors/                   # 数据提取器组件
│   ├── swift_lidar_extractor.h/.cc      # 激光雷达数据提取器
│   ├── swift_target_extractor.h/.cc     # 目标信息提取器
│   ├── swift_image_extractor.h/.cc      # 占用栅格图提取器
│   └── swift_action_mask_extractor.h/.cc # 动作掩码提取器
├── swift_observation_builder.h/.cc      # 统一观测构建器
├── swift_to_hope_adapter.h/.cc          # HOPE+格式适配器
├── swift_observation_builder_test.cc    # 单元测试
└── README.md                     # 本文档
```

## 核心功能

### 1. 数据提取器 (Extractors)

#### SwiftLidarExtractor
- 从Swift点云数据提取120束激光雷达数据
- 支持障碍物射线投射
- 可配置最大距离、束数、视场角

#### SwiftTargetExtractor
- 提取5维目标信息：`[dx, dy, heading_error, speed, curvature]`
- 基于Swift车辆状态计算相对位置和航向误差
- 支持参考线曲率信息

#### SwiftImageExtractor
- 生成3×64×64占用栅格图 (12288维)
- 将Swift障碍物转换为栅格表示
- 支持自定义分辨率和视野范围

#### SwiftActionMaskExtractor
- 生成42维动作掩码 (7个转向角 × 6个步长)
- 基于碰撞检测和约束条件
- 支持参考线边界检查

### 2. 观测构建器 (SwiftObservationBuilder)

整合所有提取器构建完整12455维观测：
- **Lidar**: 120维 (激光雷达距离数据)
- **Target**: 5维 (目标相对位置、航向误差、速度、曲率)
- **Image**: 12288维 (3×64×64占用栅格图)
- **ActionMask**: 42维 (动作可用性掩码)

### 3. HOPE+适配器 (SwiftToHopeAdapter)

将Swift数据转换为HOPE+格式：
- 支持车辆状态、障碍物、动作转换
- 提供格式验证和统计功能
- 兼容HOPE+推理接口

## 使用方法

### 1. 基本使用

```cpp
#include "modules/planning/open_space/rl_policy/swift_observation_builder.h"

// 创建观测构建器
SwiftObservationBuilder builder;

// 构建观测
SwiftObservation observation = builder.BuildObservationFromObstacles(
    vehicle_state, obstacles, target_position, target_yaw);

// 验证观测
if (builder.ValidateObservation(observation)) {
    // 使用观测进行RL推理
    std::vector<float> obs_vector = observation.flattened;
    // ... RL推理逻辑
}
```

### 2. 自定义参数

```cpp
// 使用自定义参数构建观测
SwiftObservation observation = builder.BuildObservationWithParams(
    point_cloud, vehicle_state, obstacles, target_position, target_yaw,
    reference_line, 15.0, 25.0);  // 自定义lidar范围和图像视野
```

### 3. 获取统计信息

```cpp
// 获取观测统计信息
std::string stats = builder.GetObservationStats(observation);
std::cout << stats << std::endl;
```

## 配置参数

### 车辆参数
- `vehicle.length`: 车辆长度 (默认: 4.8m)
- `vehicle.width`: 车辆宽度 (默认: 2.0m)
- `vehicle.wheelbase`: 轴距 (默认: 2.8m)
- `vehicle.max_steering_angle`: 最大转向角 (默认: 0.48rad)

### 观测参数
- `observation.lidar.num_beams`: 激光雷达束数 (默认: 120)
- `observation.lidar.max_range`: 最大探测距离 (默认: 10.0m)
- `observation.image.width/height`: 图像尺寸 (默认: 64×64)
- `observation.image.view_range`: 视野范围 (默认: 20.0m)

### 动作空间参数
- `observation.action_mask.num_steering_actions`: 转向动作数 (默认: 7)
- `observation.action_mask.num_step_lengths`: 步长动作数 (默认: 6)
- `observation.action_mask.min/max_steering`: 转向角范围 (默认: -1.0~1.0)
- `observation.action_mask.min/max_step_length`: 步长范围 (默认: 0.1~1.0)

## 构建和测试

### 构建

```bash
# 构建所有组件
bazel build //modules/planning/open_space/rl_policy:rl_policy_utils

# 构建特定组件
bazel build //modules/planning/open_space/rl_policy:swift_observation_builder
```

### 测试

```bash
# 运行单元测试
bazel test //modules/planning/open_space/rl_policy:swift_observation_builder_test
```

## 依赖关系

### 内部依赖
- `swift_lidar_extractor`
- `swift_target_extractor`
- `swift_image_extractor`
- `swift_action_mask_extractor`
- `swift_observation_builder`
- `swift_to_hope_adapter`

### 外部依赖
- `//core/common:log` - 日志系统
- `//modules/common/math` - 数学工具
- `//modules/common/vehicle_state:vehicle_state_provider` - 车辆状态
- `//modules/perception/base:point_cloud` - 点云数据
- `//modules/planning/common:obstacle` - 障碍物信息
- `//modules/planning/reference_line:reference_line` - 参考线

## 扩展和定制

### 添加新的数据提取器

1. 在`extractors/`目录下创建新的提取器类
2. 在`BUILD`文件中添加相应的构建规则
3. 在`SwiftObservationBuilder`中集成新的提取器

### 修改观测空间维度

1. 更新`base_configure.h`中的常量定义
2. 修改相应的提取器实现
3. 更新`SwiftObservation`结构体

### 添加新的配置参数

1. 在`config/rl_policy_config.yaml`中添加新参数
2. 在`base_configure.h`中添加对应的常量
3. 在`RLPolicyConfigManager`中添加访问方法

## 注意事项

1. **内存管理**: 观测数据较大(12455维)，注意内存使用
2. **性能优化**: 数据提取过程可能较耗时，考虑缓存机制
3. **线程安全**: 当前实现不是线程安全的，多线程使用时需要加锁
4. **数据验证**: 始终使用`ValidateObservation`验证观测数据
5. **错误处理**: 注意处理空指针和异常情况

## 版本历史

- **v1.0.0**: 初始版本，实现基本的12455维观测空间构建
- **v1.1.0**: 添加HOPE+适配器支持
- **v1.2.0**: 优化代码格式，符合Swift/planning代码规范

## 联系方式

如有问题或建议，请联系RL Policy模块开发团队。
