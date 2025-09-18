load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])
PLANNING_COPTS = ["-DMODULE_NAME=\\\"planning\\\"","-O2"]

# Vehicle configuration manager
cc_library(
    name = "vehicle_config_manager",
    srcs = ["vehicle_config_manager.cc"],
    hdrs = ["base_configure.h", "vehicle_config_manager.h"],
    copts = PLANNING_COPTS,
    deps = [
        "//core/common:log",
        "//modules/common/math",
    ],
)

# Parking endpoint calculator
cc_library(
    name = "parking_endpoint_calculator",
    srcs = ["parking_endpoint_calculator.cc"],
    hdrs = ["base_configure.h", "parking_endpoint_calculator.h"],
    copts = PLANNING_COPTS,
    deps = [
        "//core/common:log",
        "//modules/common/math",
        ":vehicle_config_manager",
        "//modules/common/vehicle_state:vehicle_state_provider",
    ],
)

# Data extractors
cc_library(
    name = "lidar_extractor",
    srcs = ["lidar_extractor.cc"],
    hdrs = ["base_configure.h", "lidar_extractor.h"],
    copts = PLANNING_COPTS,
    deps = [
        "//core/common:log",
        "//modules/common/math",
        "//modules/common/vehicle_state:vehicle_state_provider",
        "//modules/perception/base:point_cloud",
        "//modules/planning/common:obstacle",
    ],
)

cc_library(
    name = "target_extractor",
    srcs = ["target_extractor.cc"],
    hdrs = ["base_configure.h", "target_extractor.h"],
    copts = PLANNING_COPTS,
    deps = [
        "//core/common:log",
        "//modules/common/math",
        "//modules/common/vehicle_state:vehicle_state_provider",
        ":parking_endpoint_calculator",
    ],
)

cc_library(
    name = "image_extractor",
    srcs = ["image_extractor.cc"],
    hdrs = ["base_configure.h", "image_extractor.h"],
    copts = PLANNING_COPTS,
    deps = [
        "//core/common:log",
        "//modules/common/math",
        "//modules/common/vehicle_state:vehicle_state_provider",
        "//modules/planning/common:obstacle",
    ],
)

cc_library(
    name = "action_mask_extractor",
    srcs = ["action_mask_extractor.cc"],
    hdrs = ["base_configure.h", "action_mask_extractor.h"],
    copts = PLANNING_COPTS,
    deps = [
        "//core/common:log",
        "//modules/common/math",
        "//modules/common/vehicle_state:vehicle_state_provider",
        "//modules/planning/common:obstacle",
        "//modules/planning/reference_line:reference_line",
    ],
)

# Core observation builder
cc_library(
    name = "observation_builder",
    srcs = ["observation_builder.cc"],
    hdrs = ["base_configure.h", "observation_builder.h"],
    copts = PLANNING_COPTS,
    deps = [
        ":lidar_extractor",
        ":target_extractor", 
        ":image_extractor",
        ":action_mask_extractor",
        ":parking_endpoint_calculator",
        "//core/common:log",
        "//modules/common/math",
        "//modules/common/vehicle_state:vehicle_state_provider",
        "//modules/perception/base:point_cloud",
        "//modules/planning/common:obstacle",
        "//modules/planning/reference_line:reference_line",
        ":matplotlibcpp",  
    ],
)

# HOPE adapter
cc_library(
    name = "to_hope_adapter",
    srcs = ["to_hope_adapter.cc"],
    hdrs = ["base_configure.h", "to_hope_adapter.h"],
    copts = PLANNING_COPTS,
    deps = [
        ":observation_builder",
        "//core/common:log",
        "//modules/common/math",
        "//modules/common/vehicle_state:vehicle_state_provider",
        "//modules/planning/common:obstacle",
    ],
)

# Utility library combining all extractors
cc_library(
    name = "rl_policy_utils",
    copts = PLANNING_COPTS,
    deps = [
        ":lidar_extractor",
        ":target_extractor",
        ":image_extractor", 
        ":action_mask_extractor",
        ":parking_endpoint_calculator",
        ":observation_builder",
        ":to_hope_adapter",
        ":vehicle_config_manager",
    ],
)

# Tests
cc_test(
    name = "observation_builder_test",
    size = "small",
    srcs = ["observation_builder_test.cc"],
    copts = PLANNING_COPTS,
    deps = [
        ":observation_builder",
        "//core/common:log",
        "//modules/common/math",
        "//modules/common/vehicle_state:vehicle_state_provider",
        "//modules/perception/base:point_cloud",
        "//modules/planning/common:obstacle",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "parking_endpoint_calculator_test",
    size = "small",
    srcs = ["parking_endpoint_calculator_test.cc"],
    copts = PLANNING_COPTS,
    deps = [
        ":parking_endpoint_calculator",
        "//modules/planning/planner",
        ":vehicle_config_manager",
        "//modules/common/math",
        "//modules/common/vehicle_state:vehicle_state_provider",
        ":matplotlibcpp",    
    ],
)

cc_library(
    name = "matplotlibcpp",
    hdrs = ["matplotlibcpp.h"],
    copts = [
        "-I/usr/include/python3.8",
        "-I/home/uidq4331/.local/lib/python3.8/site-packages/numpy/core/include",
     ],
    includes = [
        "/usr/include/python3.8",  # 根据你的 Python 版本调整路径
        "/usr/local/lib/python3.8/dist-packages/numpy/core/include",  # 根据你的 NumPy 安装路径调整路径
     ],
    deps = [],
    visibility = ["//visibility:public"],
)