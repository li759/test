/******************************************************************************
 * Coordinate transform utilities shared by RL policy components.
 *****************************************************************************/
#pragma once

#include "modules/common/math/vec2d.h"

namespace swift {
namespace planning {
namespace open_space {
namespace rl_policy {
namespace coord {

// World (map) -> Ego (vehicle) coordinate transform for point
inline swift::common::math::Vec2d WorldToEgo(const swift::common::math::Vec2d& p,
                                             double tx, double ty, double tyaw) {
  const double c = std::cos(tyaw);
  const double s = std::sin(tyaw);
  const double dx = p.x() - tx;
  const double dy = p.y() - ty;
  return {dx * c + dy * s, -dx * s + dy * c};
}

// Ego (vehicle) -> World (map) coordinate transform for point
inline swift::common::math::Vec2d EgoToWorld(const swift::common::math::Vec2d& p,
                                             double tx, double ty, double tyaw) {
  const double c = std::cos(tyaw);
  const double s = std::sin(tyaw);
  const double x = p.x() * c - p.y() * s + tx;
  const double y = p.x() * s + p.y() * c + ty;
  return {x, y};
}

// World (map) -> Ego (vehicle) yaw transform
inline double YawWorldToEgo(double yaw, double tyaw) { return yaw - tyaw; }

// Ego (vehicle) -> World (map) yaw transform
inline double YawEgoToWorld(double yaw, double tyaw) { return yaw + tyaw; }

}  // namespace coord
}  // namespace rl_policy
}  // namespace open_space
}  // namespace planning
}  // namespace swift


