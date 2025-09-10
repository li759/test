# RL Policy Module Directory Structure

## Overview
The `rl_policy` module follows the standard Apollo/open_space structure with all source files in the main directory, similar to other open_space submodules like `coarse_trajectory_generator` and `trajectory_smoother`.

## Directory Structure

```
rl_policy/
├── base_configure.h                    # Common configuration and data structures
├── BUILD                               # Bazel build file
├── README.md                           # Module documentation
├── DIRECTORY_STRUCTURE.md              # This file
├── config/                             # Configuration files
│   └── rl_policy_config.yaml          # RL policy configuration
├── vehicle_config_manager.h            # Vehicle configuration manager
├── vehicle_config_manager.cc           # Implementation
├── parking_endpoint_calculator.h       # Parking endpoint calculation
├── parking_endpoint_calculator.cc      # Implementation
├── lidar_extractor.h                   # Lidar data extraction
├── lidar_extractor.cc                  # Implementation
├── target_extractor.h                  # Target information extraction
├── target_extractor.cc                 # Implementation
├── image_extractor.h                   # Image data extraction
├── image_extractor.cc                  # Implementation
├── action_mask_extractor.h             # Action mask extraction
├── action_mask_extractor.cc            # Implementation
├── observation_builder.h               # Main observation builder class
├── observation_builder.cc              # Implementation
├── to_hope_adapter.h                   # HOPE format adapter
├── to_hope_adapter.cc                  # Implementation
├── observation_builder_test.cc         # Observation builder tests
└── parking_endpoint_calculator_test.cc # Parking endpoint tests
```

## Module Organization

### Core Components
- **VehicleConfigManager**: Manages vehicle configuration parameters
- **ParkingEndpointCalculator**: Calculates parking endpoints using APA planner logic
- **ObservationBuilder**: Builds 12455-dimensional RL observations from Swift data
- **ToHopeAdapter**: Converts Swift data to HOPE+ format for RL inference

### Data Extractors
- **LidarExtractor**: Extracts 120-dimensional lidar beam data
- **TargetExtractor**: Extracts 5-dimensional target information
- **ImageExtractor**: Extracts 12288-dimensional occupancy grid images
- **ActionMaskExtractor**: Extracts 42-dimensional action masks

### Tests
- **observation_builder_test.cc**: Unit tests for observation builder
- **parking_endpoint_calculator_test.cc**: Unit tests for parking endpoint calculator

### Config
- **rl_policy_config.yaml**: Configuration file for the RL policy module

## Build System

The module uses a single Bazel BUILD file with:
- Individual library targets for each component
- Proper dependency management between components
- Test targets for unit testing
- Utility library combining all components

## Naming Conventions

- Removed `swift_` prefix from all file names
- Removed `Swift` prefix from all class names
- Used descriptive, concise names following Apollo standards
- Maintained consistent naming across header and implementation files

## Integration

The module integrates with:
- Swift planning framework
- APA planner logic for parking endpoint calculation
- HOPE+ RL inference system
- Standard Swift data structures and interfaces

## Structure Alignment

This structure follows the same patterns used by other Apollo/open_space modules:
- **coarse_trajectory_generator**: Flat structure with all .h/.cc files in main directory
- **trajectory_smoother**: Flat structure with specialized components
- **APA_planner**: Flat structure with parking-specific algorithms

The flat structure makes the module easier to navigate and maintain, following Apollo's established conventions for open_space submodules.
