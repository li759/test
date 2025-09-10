load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])
PLANNING_COPTS = ["-DMODULE_NAME=\\\"planning\\\"","-O2"]

cc_library(
    name = "swift_lidar_extractor",
    srcs = ["extractors/swift_lidar_extractor.cc"],
    hdrs = ["base_configure.h",
            "extractors/swift_lidar_extractor.h"],
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
    name = "swift_target_extractor",
    srcs = ["extractors/swift_target_extractor.cc"],
    hdrs = ["base_configure.h",
            "extractors/swift_target_extractor.h"],
    copts = PLANNING_COPTS,
    deps = [
        "//core/common:log",
        "//modules/common/math",
        "//modules/common/vehicle_state:vehicle_state_provider",
    ],
)

cc_library(
    name = "swift_image_extractor",
    srcs = ["extractors/swift_image_extractor.cc"],
    hdrs = ["base_configure.h",
            "extractors/swift_image_extractor.h"],
    copts = PLANNING_COPTS,
    deps = [
        "//core/common:log",
        "//modules/common/math",
        "//modules/common/vehicle_state:vehicle_state_provider",
        "//modules/planning/common:obstacle",
    ],
)

cc_library(
    name = "swift_action_mask_extractor",
    srcs = ["extractors/swift_action_mask_extractor.cc"],
    hdrs = ["base_configure.h",
            "extractors/swift_action_mask_extractor.h"],
    copts = PLANNING_COPTS,
    deps = [
        "//core/common:log",
        "//modules/common/math",
        "//modules/common/vehicle_state:vehicle_state_provider",
        "//modules/planning/common:obstacle",
        "//modules/planning/reference_line:reference_line",
    ],
)

cc_library(
    name = "swift_observation_builder",
    srcs = ["swift_observation_builder.cc"],
    hdrs = ["base_configure.h",
            "swift_observation_builder.h"],
    copts = PLANNING_COPTS,
    deps = [
        ":swift_lidar_extractor",
        ":swift_target_extractor", 
        ":swift_image_extractor",
        ":swift_action_mask_extractor",
        "//core/common:log",
        "//modules/common/math",
        "//modules/common/vehicle_state:vehicle_state_provider",
        "//modules/perception/base:point_cloud",
        "//modules/planning/common:obstacle",
        "//modules/planning/reference_line:reference_line",
    ],
)

cc_library(
    name = "swift_to_hope_adapter",
    srcs = ["swift_to_hope_adapter.cc"],
    hdrs = ["base_configure.h",
            "swift_to_hope_adapter.h"],
    copts = PLANNING_COPTS,
    deps = [
        ":swift_observation_builder",
        "//core/common:log",
        "//modules/common/math",
        "//modules/common/vehicle_state:vehicle_state_provider",
        "//modules/planning/common:obstacle",
    ],
)

cc_library(
    name = "rl_policy_utils",
    copts = PLANNING_COPTS,
    deps = [
        ":swift_lidar_extractor",
        ":swift_target_extractor",
        ":swift_image_extractor", 
        ":swift_action_mask_extractor",
        ":swift_observation_builder",
        ":swift_to_hope_adapter",
    ],
)

cc_test(
    name = "swift_observation_builder_test",
    size = "small",
    srcs = ["swift_observation_builder_test.cc"],
    copts = PLANNING_COPTS,
    deps = [
        ":swift_observation_builder",
        "//core/common:log",
        "//modules/common/math",
        "//modules/common/vehicle_state:vehicle_state_provider",
        "//modules/perception/base:point_cloud",
        "//modules/planning/common:obstacle",
        "@com_google_googletest//:gtest_main",
    ],
)