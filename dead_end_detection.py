from typing import List
import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union

# External types expected from caller:
# - start, dest: env.vehicle.State (must provide .loc.x, .loc.y, .heading)
# - obstacles: List[env.map_base.Area] (must provide .shape which is a shapely geometry)

from configs import WIDTH, LENGTH, FRONT_HANG, WHEEL_BASE


def _normalize_angle(angle_rad: float) -> float:
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi  # (-π, π]


def _angle_distance(a: float, b: float) -> float:
    return abs(_normalize_angle(a - b))  # [0, π]


def _check_start_facing_dest_by_angle(start, dest) -> bool:
    sx, sy = start.loc.x, start.loc.y
    dx, dy = dest.loc.x, dest.loc.y
    line_dir = np.arctan2(dy - sy, dx - sx)
    tail_dir_start = start.heading + np.pi
    tail_line_angle_diff = _angle_distance(tail_dir_start, line_dir)
    return tail_line_angle_diff > np.pi / 2


def _create_side_check_area(center_x: float, center_y: float, heading: float, length: float, width: float) -> Polygon:
    half_length = length / 2.0
    half_width = width / 2.0
    vertices = [
        (-half_length, -half_width),
        (half_length, -half_width),
        (half_length, half_width),
        (-half_length, half_width),
    ]
    rotated_vertices = []
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    for vx, vy in vertices:
        rx = vx * cos_h - vy * sin_h
        ry = vx * sin_h + vy * cos_h
        rotated_vertices.append((center_x + rx, center_y + ry))
    return Polygon(rotated_vertices)


def _check_parking_side_coverage(dest, side: str = 'left') -> dict:
    dest_x = dest.loc.x
    dest_y = dest.loc.y
    dest_heading = dest.heading

    rear_to_center_distance = WHEEL_BASE / 2.0
    check_length = LENGTH * 1.2
    check_width = 1.0

    vehicle_center_x = dest_x + rear_to_center_distance * np.cos(dest_heading)
    vehicle_center_y = dest_y + rear_to_center_distance * np.sin(dest_heading)

    front_offset = WHEEL_BASE + FRONT_HANG
    rear_offset = LENGTH / 2.0 + check_length / 2.0
    side_offset = WIDTH / 2.0 + 1.0

    if side == 'left':
        corner_angle = dest_heading + np.pi / 2.0
        if -np.pi / 2.0 <= dest_heading < np.pi / 2.0:
            check_center_x = vehicle_center_x + front_offset * np.cos(dest_heading) + side_offset * np.cos(corner_angle)
            check_center_y = vehicle_center_y + front_offset * np.sin(dest_heading) + side_offset * np.sin(corner_angle)
        else:
            check_center_x = vehicle_center_x + rear_offset * np.cos(dest_heading) + side_offset * np.cos(corner_angle)
            check_center_y = vehicle_center_y + rear_offset * np.sin(dest_heading) + side_offset * np.sin(corner_angle)
    else:
        corner_angle = dest_heading - np.pi / 2.0
        if -np.pi / 2.0 <= dest_heading < np.pi / 2.0:
            check_center_x = vehicle_center_x + front_offset * np.cos(dest_heading) + side_offset * np.cos(corner_angle)
            check_center_y = vehicle_center_y + front_offset * np.sin(dest_heading) + side_offset * np.sin(corner_angle)
        else:
            check_center_x = vehicle_center_x + rear_offset * np.cos(dest_heading) + side_offset * np.cos(corner_angle)
            check_center_y = vehicle_center_y + rear_offset * np.sin(dest_heading) + side_offset * np.sin(corner_angle)

    check_area = _create_side_check_area(check_center_x, check_center_y, dest_heading, check_length, check_width)
    return {'check_area': check_area}


def _get_y_axis_occupancy_at_x(check_area: Polygon, x: float, miny: float, maxy: float, obstacles: List) -> float:
    y_line = LineString([(x, miny), (x, maxy)])
    if not check_area.intersects(y_line):
        return 0.0
    intersection = check_area.intersection(y_line)
    if intersection.is_empty:
        return 0.0

    if hasattr(intersection, 'length'):
        total_y_length = intersection.length
    else:
        total_y_length = 0.0
        if hasattr(intersection, 'geoms'):
            for geom in intersection.geoms:
                if hasattr(geom, 'length'):
                    total_y_length += geom.length
    if total_y_length == 0:
        return 0.0

    occupied_segments = []
    for obstacle in obstacles:
        if not hasattr(obstacle, 'shape'):
            continue
        obstacle_shape = obstacle.shape
        try:
            if hasattr(obstacle_shape, 'coords') and not hasattr(obstacle_shape, 'exterior'):
                obstacle_shape = Polygon(list(obstacle_shape.coords))
        except Exception:
            continue
        if y_line.intersects(obstacle_shape):
            inter = y_line.intersection(obstacle_shape)
            if not inter.is_empty:
                if hasattr(inter, 'geoms'):
                    for geom in inter.geoms:
                        if hasattr(geom, 'length') and geom.length > 0:
                            occupied_segments.append(geom)
                elif hasattr(inter, 'length') and inter.length > 0:
                    occupied_segments.append(inter)

    if occupied_segments:
        try:
            merged_segments = unary_union(occupied_segments)
            if hasattr(merged_segments, 'length'):
                occupied_length = merged_segments.length
            else:
                occupied_length = 0.0
                if hasattr(merged_segments, 'geoms'):
                    for geom in merged_segments.geoms:
                        if hasattr(geom, 'length'):
                            occupied_length += geom.length
        except Exception:
            occupied_length = 0.0
            for geom in occupied_segments:
                if hasattr(geom, 'length'):
                    occupied_length += geom.length
    else:
        occupied_length = 0.0

    if total_y_length > 0:
        return min(occupied_length / total_y_length, 1.0)
    return 0.0


def _check_y_axis_occupancy(check_area: Polygon, obstacles: List, step_size: float = 0.05, occupancy_threshold: float = 0.8) -> bool:
    minx, miny, maxx, maxy = check_area.bounds
    x_positions = np.arange(minx, maxx, step_size)
    y_occupancy_rates = []
    for x in x_positions:
        occ = _get_y_axis_occupancy_at_x(check_area, x, miny, maxy, obstacles)
        y_occupancy_rates.append(occ)
    max_occupancy = np.max(y_occupancy_rates) if y_occupancy_rates else 0.0
    return bool(max_occupancy > occupancy_threshold)


def _is_dead_end_by_bilateral_coverage(dest, obstacles: List) -> bool:
    left_area = _check_parking_side_coverage(dest, side='left')
    right_area = _check_parking_side_coverage(dest, side='right')
    left_blocked = _check_y_axis_occupancy(left_area['check_area'], obstacles, step_size=0.05, occupancy_threshold=0.8)
    right_blocked = _check_y_axis_occupancy(right_area['check_area'], obstacles, step_size=0.05, occupancy_threshold=0.8)
    return bool(left_blocked or right_blocked)


def is_dead_end_scenario(start, dest, obstacles: List) -> bool:
    angle_check = _check_start_facing_dest_by_angle(start, dest)
    coverage_check = _is_dead_end_by_bilateral_coverage(dest, obstacles)
    return bool(angle_check and coverage_check)


