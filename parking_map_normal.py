import sys
sys.path.append('../')
from math import pi, cos, sin, atan2
import os
import numpy as np
from numpy.random import randn, random
from typing import List
from shapely.geometry import LinearRing
from env.vehicle import State
from env.map_base import *
from configs import *
from env.lidar_simulator import LidarSimlator
DEBUG = False
if DEBUG:
    import matplotlib.pyplot as plt
prob_huge_obst = 0.5
n_non_critical_car = 3
prob_non_critical_car = 0.7

global_plot_fig = None
global_plot_ax = None

def visualize_parking_map(start_x, start_y, start_yaw, dest_x, dest_y, dest_yaw, obstacles):
    """
    可视化停车地图，包括起点、终点和障碍物

    参数:
    start_x, start_y, start_yaw: 起始位置和朝向
    dest_x, dest_y, dest_yaw: 目标位置和朝向
    obstacles: 障碍物列表，每个障碍物为LinearRing对象
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(10, 8))

        for i, obstacle in enumerate(obstacles):
            coords = np.array(obstacle.coords)
            ax.plot(coords[:, 0], coords[:, 1], 'gray', linewidth=2)
            ax.fill(coords[:, 0], coords[:, 1], alpha=0.3, color='gray')

        start_arrow_length = 2.0
        start_arrow_dx = start_arrow_length * np.cos(start_yaw)
        start_arrow_dy = start_arrow_length * np.sin(start_yaw)
        ax.plot(start_x, start_y, 'bo', markersize=10)
        ax.arrow(start_x, start_y, start_arrow_dx, start_arrow_dy,
                head_width=0.5, head_length=0.3, fc='blue', ec='blue', linewidth=2)

        dest_arrow_length = 2.0
        dest_arrow_dx = dest_arrow_length * np.cos(dest_yaw)
        dest_arrow_dy = dest_arrow_length * np.sin(dest_yaw)
        ax.plot(dest_x, dest_y, 'go', markersize=10)
        ax.arrow(dest_x, dest_y, dest_arrow_dx, dest_arrow_dy,
                head_width=0.5, head_length=0.3, fc='red', ec='red', linewidth=2)

        def draw_vehicle_outline(ax, x, y, yaw, color, label):
            """
            绘制车辆轮廓，其中(x,y)是后轴中心点
            """
            length, width = 4.8, 2.0
            wheel_base = 2.8  # 轴距
            center_offset = wheel_base / 2
            center_x = x + center_offset * np.cos(yaw)
            center_y = y + center_offset * np.sin(yaw)
            cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
            corners = np.array([
                [-length/2, -width/2],
                [length/2, -width/2],
                [length/2, width/2],
                [-length/2, width/2]
            ])
            rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
            rotated_corners = corners @ rotation_matrix.T
            translated_corners = rotated_corners + np.array([center_x, center_y])
            ax.plot(translated_corners[[0,1,2,3,0], 0],
                   translated_corners[[0,1,2,3,0], 1],
                   color=color, linewidth=2, alpha=0.7)
            ax.fill(translated_corners[:, 0], translated_corners[:, 1],
                   color=color, alpha=0.2)
            ax.plot(x, y, 'o', color=color, markersize=8, alpha=0.8)

        draw_vehicle_outline(ax, start_x, start_y, start_yaw, 'blue', 'Start Vehicle')
        draw_vehicle_outline(ax, dest_x, dest_y, dest_yaw, 'green', 'Dest Vehicle')

        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Parking Map Visualization')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')

        all_x = [start_x, dest_x] + [coord[0] for obstacle in obstacles for coord in obstacle.coords]
        all_y = [start_y, dest_y] + [coord[1] for obstacle in obstacles for coord in obstacle.coords]
        x_margin = (max(all_x) - min(all_x)) * 0.1
        y_margin = (max(all_y) - min(all_y)) * 0.1
        ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
        ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)

        plt.tight_layout()  # 自动调整布局
        plt.ion()  # 开启交互模式
        plt.show(block=False)  # 非阻塞显示，程序会继续运行

        global global_plot_fig, global_plot_ax
        import builtins
        builtins.global_plot_fig = fig
        builtins.global_plot_ax = ax

    except ImportError:
        print("matplotlib未安装，跳过地图可视化")
    except Exception as e:
        print(f"地图可视化失败: {e}")

def random_gaussian_num(mean, std, clip_low, clip_high):
    rand_num = randn() * std + mean
    return np.clip(rand_num, clip_low, clip_high)

def random_uniform_num(clip_low, clip_high):
    rand_num = random() * (clip_high - clip_low) + clip_low
    return rand_num

def get_rand_pos(origin_x, origin_y, angle_min, angle_max, radius_min, radius_max):
    angle_mean = (angle_max + angle_min) / 2
    angle_std = (angle_max - angle_min) / 4
    angle_rand = random_gaussian_num(angle_mean, angle_std, angle_min, angle_max)
    radius_rand = random_gaussian_num((radius_min + radius_max) / 2, (radius_max - radius_min) / 4, radius_min, radius_max)
    return (origin_x + cos(angle_rand) * radius_rand, origin_y + sin(angle_rand) * radius_rand)

def generate_bay_parking_case(map_level):
    '\n    Generate the parameters that a bay parking case need.\n    \n    Returns\n    ----------\n        `start` (list): [x, y, yaw]\n        `dest` (list): [x, y, yaw]\n        `obstacles` (list): [ obstacle (`LinearRing`) , ...]\n    '
    origin = (0.0, 0.0)
    bay_half_len = 15.0
    max_BAY_PARK_LOT_WIDTH = MAX_PARK_LOT_WIDTH_DICT[map_level]
    min_BAY_PARK_LOT_WIDTH = MIN_PARK_LOT_WIDTH_DICT[map_level]
    bay_PARK_WALL_DIST = BAY_PARK_WALL_DIST_DICT[map_level]
    n_obst = N_OBSTACLE_DICT[map_level]
    max_lateral_space = max_BAY_PARK_LOT_WIDTH - WIDTH
    min_lateral_space = min_BAY_PARK_LOT_WIDTH - WIDTH
    generate_success = True
    obstacle_back = LinearRing(((origin[0] + bay_half_len, origin[1]), (origin[0] + bay_half_len, origin[1] - 1), (origin[0] - bay_half_len, origin[1] - 1), (origin[0] - bay_half_len, origin[1])))
    if DEBUG:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        plt.axis('off')
    dest_yaw = random_gaussian_num(pi / 2, pi / 36, pi * 5 / 12, pi * 7 / 12)
    rb, _, _, lb = list(State([origin[0], origin[1], dest_yaw, 0, 0]).create_box().coords)[:-1]
    min_dest_y = -min(rb[1], lb[1]) + MIN_DIST_TO_OBST
    dest_x = origin[0]
    dest_y = random_gaussian_num(min_dest_y + 0.4, 0.2, min_dest_y, min_dest_y + 0.8)
    car_rb, car_rf, car_lf, car_lb = list(State([dest_x, dest_y, dest_yaw, 0, 0]).create_box().coords)[:-1]
    dest_box = LinearRing((car_rb, car_rf, car_lf, car_lb))
    if DEBUG:
        ax.add_patch(plt.Polygon(xy=list([car_rb, car_rf, car_lf, car_lb]), color='b'))
    non_critical_vehicle = []
    if random() < prob_huge_obst:
        max_dist_to_obst = max_lateral_space / 5 * 4
        min_dist_to_obst = max_lateral_space / 5 * 1
        left_obst_rf = get_rand_pos(*car_lf, pi * 11 / 12, pi * 13 / 12, min_dist_to_obst, max_dist_to_obst)
        left_obst_rb = get_rand_pos(*car_lb, pi * 11 / 12, pi * 13 / 12, min_dist_to_obst, max_dist_to_obst)
        obstacle_left = LinearRing((left_obst_rf, left_obst_rb, (origin[0] - bay_half_len, origin[1]), (origin[0] - bay_half_len, left_obst_rf[1])))
    else:
        max_dist_to_obst = max_lateral_space / 5 * 4
        min_dist_to_obst = max_lateral_space / 5 * 1
        left_car_x = origin[0] - (WIDTH + random_uniform_num(min_dist_to_obst, max_dist_to_obst))
        left_car_yaw = random_gaussian_num(pi / 2, pi / 36, pi * 5 / 12, pi * 7 / 12)
        rb, _, _, lb = list(State([left_car_x, origin[1], left_car_yaw, 0, 0]).create_box().coords)[:-1]
        min_left_car_y = -min(rb[1], lb[1]) + MIN_DIST_TO_OBST
        left_car_y = random_gaussian_num(min_left_car_y + 0.4, 0.2, min_left_car_y, min_left_car_y + 0.8)
        obstacle_left = State([left_car_x, left_car_y, left_car_yaw, 0, 0]).create_box()
        for _ in range(n_non_critical_car):
            left_car_x -= WIDTH + MIN_DIST_TO_OBST + random_uniform_num(min_dist_to_obst, max_dist_to_obst)
            left_car_y += random_gaussian_num(0, 0.05, -0.1, 0.1)
            left_car_yaw = random_gaussian_num(pi / 2, pi / 36, pi * 5 / 12, pi * 7 / 12)
            obstacle_left_ = State([left_car_x, left_car_y, left_car_yaw, 0, 0]).create_box()
            if random() < prob_non_critical_car:
                non_critical_vehicle.append(obstacle_left_)
    dist_dest_to_left_obst = dest_box.distance(obstacle_left)
    min_dist_to_obst = max(min_lateral_space - dist_dest_to_left_obst, 0) + MIN_DIST_TO_OBST
    max_dist_to_obst = max(max_lateral_space - dist_dest_to_left_obst, 0) + MIN_DIST_TO_OBST
    if random() < prob_huge_obst:
        right_obst_lf = get_rand_pos(*car_rf, -pi / 12, pi / 12, min_dist_to_obst, max_dist_to_obst)
        right_obst_lb = get_rand_pos(*car_rb, -pi / 12, pi / 12, min_dist_to_obst, max_dist_to_obst)
        obstacle_right = LinearRing(((origin[0] + bay_half_len, right_obst_lf[1]), (origin[0] + bay_half_len, origin[1]), right_obst_lb, right_obst_lf))
    else:
        right_car_x = origin[0] + (WIDTH + random_uniform_num(min_dist_to_obst, max_dist_to_obst))
        right_car_yaw = random_gaussian_num(pi / 2, pi / 36, pi * 5 / 12, pi * 7 / 12)
        rb, _, _, lb = list(State([right_car_x, origin[1], right_car_yaw, 0, 0]).create_box().coords)[:-1]
        min_right_car_y = -min(rb[1], lb[1]) + MIN_DIST_TO_OBST
        right_car_y = random_gaussian_num(min_right_car_y + 0.4, 0.2, min_right_car_y, min_right_car_y + 0.8)
        obstacle_right = State([right_car_x, right_car_y, right_car_yaw, 0, 0]).create_box()
        for _ in range(n_non_critical_car):
            right_car_x += WIDTH + MIN_DIST_TO_OBST + random_uniform_num(min_dist_to_obst, max_dist_to_obst)
            right_car_y += random_gaussian_num(0, 0.05, -0.1, 0.1)
            right_car_yaw = random_gaussian_num(pi / 2, pi / 36, pi * 5 / 12, pi * 7 / 12)
            obstacle_right_ = State([right_car_x, right_car_y, right_car_yaw, 0, 0]).create_box()
            if random() < prob_non_critical_car:
                non_critical_vehicle.append(obstacle_right_)
    dist_dest_to_right_obst = dest_box.distance(obstacle_right)
    if dist_dest_to_right_obst + dist_dest_to_left_obst < min_lateral_space or dist_dest_to_right_obst + dist_dest_to_left_obst > max_lateral_space or dist_dest_to_left_obst < MIN_DIST_TO_OBST or (dist_dest_to_right_obst < MIN_DIST_TO_OBST):
        generate_success = False
    obstacles = [obstacle_back, obstacle_left, obstacle_right]
    obstacles.extend(non_critical_vehicle)
    for obst in obstacles:
        if obst.intersects(dest_box):
            generate_success = False
    max_obstacle_y = max([np.max(np.array(obs.coords)[:, 1]) for obs in obstacles]) + MIN_DIST_TO_OBST
    other_obstcales = []
    if random() < 0.2:
        other_obstcales = [LinearRing(((origin[0] - bay_half_len, bay_PARK_WALL_DIST + max_obstacle_y + MIN_DIST_TO_OBST), (origin[0] + bay_half_len, bay_PARK_WALL_DIST + max_obstacle_y + MIN_DIST_TO_OBST), (origin[0] + bay_half_len, bay_PARK_WALL_DIST + max_obstacle_y + MIN_DIST_TO_OBST + 0.1), (origin[0] - bay_half_len, bay_PARK_WALL_DIST + max_obstacle_y + MIN_DIST_TO_OBST + 0.1)))]
    else:
        other_obstacle_range = LinearRing(((origin[0] - bay_half_len, bay_PARK_WALL_DIST + max_obstacle_y), (origin[0] + bay_half_len, bay_PARK_WALL_DIST + max_obstacle_y), (origin[0] + bay_half_len, bay_PARK_WALL_DIST + max_obstacle_y + 8), (origin[0] - bay_half_len, bay_PARK_WALL_DIST + max_obstacle_y + 8)))
        valid_obst_x_range = (origin[0] - bay_half_len + 2, origin[0] + bay_half_len - 2)
        valid_obst_y_range = (bay_PARK_WALL_DIST + max_obstacle_y + 2, bay_PARK_WALL_DIST + max_obstacle_y + 6)
        for _ in range(n_obst):
            obs_x = random_uniform_num(*valid_obst_x_range)
            obs_y = random_uniform_num(*valid_obst_y_range)
            obs_yaw = random() * pi * 2
            obs_coords = np.array(State([obs_x, obs_y, obs_yaw, 0, 0]).create_box().coords[:-1])
            obs = LinearRing(obs_coords + 0.5 * random(obs_coords.shape))
            if obs.intersects(other_obstacle_range):
                continue
            obst_invalid = False
            for other_obs in other_obstcales:
                if obs.intersects(other_obs):
                    obst_invalid = True
                    break
            if obst_invalid:
                continue
            other_obstcales.append(obs)
    obstacles.extend(other_obstcales)
    if DEBUG:
        for obs in obstacles:
            ax.add_patch(plt.Polygon(xy=list(obs.coords), color='gray'))
    start_box_valid = False
    valid_start_x_range = (origin[0] - bay_half_len / 2, origin[0] + bay_half_len / 2)
    valid_start_y_range = (max_obstacle_y + 1, bay_PARK_WALL_DIST + max_obstacle_y - 1)
    while not start_box_valid:
        start_box_valid = True
        start_x = random_uniform_num(*valid_start_x_range)
        start_y = random_uniform_num(*valid_start_y_range)
        start_yaw = random_gaussian_num(0, pi / 6, -pi / 2, pi / 2)
        start_yaw = start_yaw + pi if random() < 0.5 else start_yaw
        start_box = State([start_x, start_y, start_yaw, 0, 0]).create_box()

        for obst in obstacles:
            if obst.intersects(start_box):
                if DEBUG:
                    start_box_valid = False
        if dest_box.intersects(start_box):
            if DEBUG:
                start_box_valid = False
    for obs in obstacles:
        if random() < DROUP_OUT_OBST:
            obstacles.remove(obs)
    if DEBUG:
        ax.add_patch(plt.Polygon(xy=list(State([start_x, start_y, start_yaw, 0, 0]).create_box().coords), color='g'))
        if generate_success:
            path = './log/figure/'
            num_files = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
            fig = plt.gcf()
            fig.savefig(path + f'image_{num_files}.png')
        plt.clf()
    if generate_success:
        obstacles.clear()
        start_x = -0.6564523656668699+3
        start_y = 8.335082157585665
        start_yaw = 3.007639913783333
        dest_x = 0.0 
        dest_y = 1.0679935193454844
        dest_yaw = 1.5707963267948966

        car_rbs, car_rfs, car_lfs, car_lbs  = list(State([start_x, start_y, start_yaw, 0, 0]).create_box().coords)[:-1]
        car_rbe, car_rfe, car_lfe, car_lbe  = list(State([dest_x, dest_y, dest_yaw, 0, 0]).create_box().coords)[:-1]
        ref_left_x = min(car_rbs[0], car_rfs[0], car_lfs[0], car_lbs[0], car_rbe[0], car_rfe[0], car_lfe[0], car_lbe[0])
        ref_right_x = max(car_rbs[0], car_rfs[0], car_lfs[0], car_lbs[0], car_rbe[0], car_rfe[0], car_lfe[0], car_lbe[0])
        park_left_x = min(car_rbe[0], car_rfe[0], car_lfe[0], car_lbe[0])-0.25
        park_right_x = max(car_rbe[0], car_rfe[0], car_lfe[0], car_lbe[0])+0.25
        park_up_y = max(car_rbe[1], car_rfe[1], car_lfe[1], car_lbe[1])-0.2
        car_up_y = min(car_rbs[1], car_rfs[1], car_lfs[1], car_lbs[1])
        ref_up_y = min(park_up_y, car_up_y)-0.2
        park_down_y = min(car_rbe[1], car_rfe[1], car_lfe[1], car_lbe[1])
        if start_yaw > -pi/2 and start_yaw < pi/2:
            ref_left_x = ref_left_x-15
            ref_right_x = ref_right_x+1
        else :
            ref_left_x = ref_left_x-1
            ref_right_x = ref_right_x+15
        add_park_up_obstacle = [LinearRing(( 
        (park_left_x-2, park_down_y-4),
        (park_right_x+2, park_down_y-4), 
        (park_right_x+2, park_down_y), 
        (park_left_x-2, park_down_y)))]
        add_park_down_left_obstacle = [LinearRing(( 
        (ref_left_x-2, ref_up_y-2),
        (park_left_x-2, ref_up_y-2), 
        (park_left_x-2, ref_up_y), 
        (ref_left_x-2, ref_up_y)))]
        add_park_down_right_obstacle = [LinearRing(( 
        (park_right_x+2, ref_up_y-2),
        (ref_right_x, ref_up_y-2), 
        (ref_right_x, ref_up_y), 
        (park_right_x+2, ref_up_y)))]
        add_park_left_obstacle = [LinearRing(( 
        (park_left_x-2, park_up_y),
        (park_left_x, park_up_y), 
        (park_left_x, park_down_y), 
        (park_left_x-2, park_down_y)))]
        add_park_right_obstacle = [LinearRing(( 
        (park_right_x, park_up_y),
        (park_right_x+2, park_up_y), 
        (park_right_x+2, park_down_y), 
        (park_right_x, park_down_y)))]
        add_left_obstacle = [LinearRing(( 
        (ref_left_x-2, start_y+8),
        (ref_left_x+1, start_y+8), 
        (ref_left_x+1, start_y-8), 
        (ref_left_x-2, start_y-8)))]
        add_right_obstacle = [LinearRing((
        (ref_right_x, start_y+8),
        (ref_right_x+2, start_y+8), 
        (ref_right_x+2, start_y-8), 
        (ref_right_x, start_y-8)))]
        add_plus_obstacle = [LinearRing((
        (ref_left_x-2, start_y+1.8),
        (park_left_x+9, start_y+1.8), 
        (park_left_x+9, start_y+8), 
        (ref_left_x-2, start_y+8)))]
        add_down_obstacle = [LinearRing((
        (ref_left_x-2, start_y+8),
        (ref_right_x+2, start_y+8), 
        (ref_right_x+2, start_y+9), 
        (ref_left_x-2, start_y+9)))]
        obstacles.extend(add_plus_obstacle)
        obstacles.extend(add_park_down_left_obstacle)
        obstacles.extend(add_park_down_right_obstacle)
        obstacles.extend(add_park_up_obstacle)
        obstacles.extend(add_park_left_obstacle)
        obstacles.extend(add_park_right_obstacle)
        obstacles.extend(add_right_obstacle)
        obstacles.extend(add_left_obstacle)
        obstacles.extend(add_down_obstacle)
        start_x = 15
        start_y= 10
        start_yaw = 0
        dest_x = 0.0 
        dest_y = 1.0679935193454844
        dest_yaw = 1.5707963267948966

        visualize_parking_map(start_x, start_y, start_yaw, dest_x, dest_y, dest_yaw, obstacles)
        return ([start_x, start_y, start_yaw], [dest_x, dest_y, dest_yaw],obstacles)
    else:
        return generate_bay_parking_case(map_level)

def generate_parallel_parking_case(map_level):
    '\n    Generate the parameters that a parallel parking case need.\n    \n    Returns\n    ----------\n        `start` (list): [x, y, yaw]\n        `dest` (list): [x, y, yaw]\n        `obstacles` (list): [ obstacle (`LinearRing`) , ...]\n    '
    origin = (0.0, 0.0)
    bay_half_len = 18.0
    max_PARA_PARK_LOT_LEN = MAX_PARK_LOT_LEN_DICT[map_level]
    min_PARA_PARK_LOT_LEN = MIN_PARK_LOT_LEN_DICT[map_level]
    para_PARK_WALL_DIST = PARA_PARK_WALL_DIST_DICT[map_level]
    n_obst = N_OBSTACLE_DICT[map_level]
    max_longitude_space = max_PARA_PARK_LOT_LEN - LENGTH
    min_longitude_space = min_PARA_PARK_LOT_LEN - LENGTH
    generate_success = True
    obstacle_back = LinearRing(((origin[0] + bay_half_len, origin[1]), (origin[0] + bay_half_len, origin[1] - 1), (origin[0] - bay_half_len, origin[1] - 1), (origin[0] - bay_half_len, origin[1])))
    if DEBUG:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        plt.axis('off')
    dest_yaw = random_gaussian_num(0, pi / 36, -pi / 12, pi / 12)
    rb, rf, _, _ = list(State([origin[0], origin[1], dest_yaw, 0, 0]).create_box().coords)[:-1]
    min_dest_y = -min(rb[1], rf[1]) + MIN_DIST_TO_OBST
    dest_x = origin[0]
    dest_y = random_gaussian_num(min_dest_y + 0.4, 0.2, min_dest_y, min_dest_y + 0.8)
    car_rb, car_rf, car_lf, car_lb = list(State([dest_x, dest_y, dest_yaw, 0, 0]).create_box().coords)[:-1]
    dest_box = LinearRing((car_rb, car_rf, car_lf, car_lb))
    if DEBUG:
        ax.add_patch(plt.Polygon(xy=list([car_rb, car_rf, car_lf, car_lb]), color='b'))
    non_critical_vehicle = []
    if random() < prob_huge_obst:
        max_dist_to_obst = max_longitude_space / 5 * 4
        min_dist_to_obst = min_longitude_space / 5 * 1
        left_obst_rf = get_rand_pos(*car_lb, pi * 11 / 12, pi * 13 / 12, min_dist_to_obst, max_dist_to_obst)
        left_obst_rb = get_rand_pos(*car_rb, pi * 11 / 12, pi * 13 / 12, min_dist_to_obst, max_dist_to_obst)
        obstacle_left = LinearRing((left_obst_rf, left_obst_rb, (origin[0] - bay_half_len, origin[1]), (origin[0] - bay_half_len, left_obst_rf[1])))
    else:
        max_dist_to_obst = max_longitude_space / 5 * 4
        min_dist_to_obst = min_longitude_space / 5 * 1
        left_car_x = origin[0] - (LENGTH + random_uniform_num(min_dist_to_obst, max_dist_to_obst))
        left_car_yaw = random_gaussian_num(0, pi / 36, -pi / 12, pi / 12)
        rb, rf, _, _ = list(State([left_car_x, origin[1], left_car_yaw, 0, 0]).create_box().coords)[:-1]
        min_left_car_y = -min(rb[1], rf[1]) + MIN_DIST_TO_OBST
        left_car_y = random_gaussian_num(min_left_car_y + 0.4, 0.2, min_left_car_y, min_left_car_y + 0.8)
        obstacle_left = State([left_car_x, left_car_y, left_car_yaw, 0, 0]).create_box()
        for _ in range(n_non_critical_car - 1):
            left_car_x -= LENGTH + MIN_DIST_TO_OBST + random_uniform_num(min_dist_to_obst, max_dist_to_obst)
            left_car_y += random_gaussian_num(0, 0.05, -0.1, 0.1)
            left_car_yaw = random_gaussian_num(0, pi / 36, -pi / 12, pi / 12)
            obstacle_left_ = State([left_car_x, left_car_y, left_car_yaw, 0, 0]).create_box()
            if random() < prob_non_critical_car:
                non_critical_vehicle.append(obstacle_left_)
    dist_dest_to_left_obst = dest_box.distance(obstacle_left)
    min_dist_to_obst = max(min_longitude_space - dist_dest_to_left_obst, 0) + MIN_DIST_TO_OBST
    max_dist_to_obst = max(max_longitude_space - dist_dest_to_left_obst, 0) + MIN_DIST_TO_OBST
    if random() < 0.5:
        right_obst_lf = get_rand_pos(*car_lf, -pi / 12, pi / 12, min_dist_to_obst, max_dist_to_obst)
        right_obst_lb = get_rand_pos(*car_rf, -pi / 12, pi / 12, min_dist_to_obst, max_dist_to_obst)
        obstacle_right = LinearRing(((origin[0] + bay_half_len, right_obst_lf[1]), (origin[0] + bay_half_len, origin[1]), right_obst_lb, right_obst_lf))
    else:
        right_car_x = origin[0] + (LENGTH + random_uniform_num(min_dist_to_obst, max_dist_to_obst))
        right_car_yaw = random_gaussian_num(0, pi / 36, -pi / 12, pi / 12)
        rb, rf, _, _ = list(State([right_car_x, origin[1], right_car_yaw, 0, 0]).create_box().coords)[:-1]
        min_right_car_y = -min(rb[1], rf[1]) + MIN_DIST_TO_OBST
        right_car_y = random_gaussian_num(min_right_car_y + 0.4, 0.2, min_right_car_y, min_right_car_y + 0.8)
        obstacle_right = State([right_car_x, right_car_y, right_car_yaw, 0, 0]).create_box()
        for _ in range(n_non_critical_car - 1):
            right_car_x += LENGTH + MIN_DIST_TO_OBST + random_uniform_num(min_dist_to_obst, max_dist_to_obst)
            right_car_y += random_gaussian_num(0, 0.05, -0.1, 0.1)
            right_car_yaw = random_gaussian_num(0, pi / 36, -pi / 12, pi / 12)
            obstacle_right_ = State([right_car_x, right_car_y, right_car_yaw, 0, 0]).create_box()
            if random() < prob_non_critical_car:
                non_critical_vehicle.append(obstacle_right_)
    dist_dest_to_right_obst = dest_box.distance(obstacle_right)
    if dist_dest_to_right_obst + dist_dest_to_left_obst < min_longitude_space or dist_dest_to_right_obst + dist_dest_to_left_obst > max_longitude_space or dist_dest_to_left_obst < MIN_DIST_TO_OBST or (dist_dest_to_right_obst < MIN_DIST_TO_OBST):
        generate_success = False
    obstacles = [obstacle_back, obstacle_left, obstacle_right]
    obstacles.extend(non_critical_vehicle)
    for obst in obstacles:
        if obst.intersects(dest_box):
            generate_success = False
    max_obstacle_y = max([np.max(np.array(obs.coords)[:, 1]) for obs in obstacles]) + MIN_DIST_TO_OBST
    other_obstcales = []
    if random() < 0.2:
        other_obstcales = [LinearRing(((origin[0] - bay_half_len, para_PARK_WALL_DIST + max_obstacle_y + MIN_DIST_TO_OBST), (origin[0] + bay_half_len, para_PARK_WALL_DIST + max_obstacle_y + MIN_DIST_TO_OBST), (origin[0] + bay_half_len, para_PARK_WALL_DIST + max_obstacle_y + MIN_DIST_TO_OBST + 0.1), (origin[0] - bay_half_len, para_PARK_WALL_DIST + max_obstacle_y + MIN_DIST_TO_OBST + 0.1)))]
    else:
        other_obstacle_range = LinearRing(((origin[0] - bay_half_len, para_PARK_WALL_DIST + max_obstacle_y), (origin[0] + bay_half_len, para_PARK_WALL_DIST + max_obstacle_y), (origin[0] + bay_half_len, para_PARK_WALL_DIST + max_obstacle_y + 8), (origin[0] - bay_half_len, para_PARK_WALL_DIST + max_obstacle_y + 8)))
        valid_obst_x_range = (origin[0] - bay_half_len + 2, origin[0] + bay_half_len - 2)
        valid_obst_y_range = (para_PARK_WALL_DIST + max_obstacle_y + 2, para_PARK_WALL_DIST + max_obstacle_y + 6)
        for _ in range(n_obst):
            obs_x = random_uniform_num(*valid_obst_x_range)
            obs_y = random_uniform_num(*valid_obst_y_range)
            obs_yaw = random() * pi * 2
            obs_coords = np.array(State([obs_x, obs_y, obs_yaw, 0, 0]).create_box().coords[:-1])
            obs = LinearRing(obs_coords + 0.5 * random(obs_coords.shape))
            if obs.intersects(other_obstacle_range):
                continue
            obst_invalid = False
            for other_obs in other_obstcales:
                if obs.intersects(other_obs):
                    obst_invalid = True
                    break
            if obst_invalid:
                continue
            other_obstcales.append(obs)
    obstacles.extend(other_obstcales)
    if DEBUG:
        for obs in obstacles:
            ax.add_patch(plt.Polygon(xy=list(obs.coords), color='gray'))
    start_box_valid = False
    valid_start_x_range = (origin[0] - bay_half_len / 2, origin[0] + bay_half_len / 2)
    valid_start_y_range = (max_obstacle_y + 1, para_PARK_WALL_DIST + max_obstacle_y - 1)
    while not start_box_valid:
        start_box_valid = True
        start_x = random_uniform_num(*valid_start_x_range)
        start_y = random_uniform_num(*valid_start_y_range)
        start_yaw = random_gaussian_num(0, pi / 6, -pi / 2, pi / 2)
        start_yaw = start_yaw + pi if random() < 0.5 else start_yaw
        start_box = State([start_x, start_y, start_yaw, 0, 0]).create_box()
        for obst in obstacles:
            if obst.intersects(start_box):
                if DEBUG:
                    start_box_valid = False
        if dest_box.intersects(start_box):
            if DEBUG:
                start_box_valid = False
    if cos(start_yaw) < 0:
        dest_box_center = np.mean(np.array(dest_box.coords[:-1]), axis=0)
        dest_x = 2 * dest_box_center[0] - dest_x
        dest_y = 2 * dest_box_center[1] - dest_y
        dest_yaw += pi
    for obs in obstacles:
        if random() < DROUP_OUT_OBST:
            obstacles.remove(obs)
    if DEBUG:
        ax.add_patch(plt.Polygon(xy=list(State([start_x, start_y, start_yaw, 0, 0]).create_box().coords), color='g'))
        plt.show()
    if generate_success:
        return ([start_x, start_y, start_yaw], [dest_x, dest_y, dest_yaw], obstacles)
    else:
        return generate_parallel_parking_case(map_level)

def plot_scene_components(start, dest, obstacles, title='场景组件'):
    '\n    绘制场景组件：起点、终点车位以及障碍物\n    '
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from env.vehicle import State
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, obstacle in enumerate(obstacles):
        coords = list(obstacle.coords)
        x_coords = [coord[0] for coord in coords]
        y_coords = [coord[1] for coord in coords]
        polygon = Polygon(list(zip(x_coords, y_coords)), facecolor='gray', alpha=0.7, edgecolor='black', linewidth=1, label=f'障碍物{i + 1}' if i == 0 else '')
        ax.add_patch(polygon)
    start_state = State(start + [0, 0])
    start_box = start_state.create_box()
    start_coords = list(start_box.coords)
    start_x_coords = [coord[0] for coord in start_coords]
    start_y_coords = [coord[1] for coord in start_coords]
    start_polygon = Polygon(list(zip(start_x_coords, start_y_coords)), facecolor='green', alpha=0.7, edgecolor='darkgreen', linewidth=2, label='起始位置')
    ax.add_patch(start_polygon)
    dest_state = State(dest + [0, 0])
    dest_box = dest_state.create_box()
    dest_coords = list(dest_box.coords)
    dest_x_coords = [coord[0] for coord in dest_coords]
    dest_y_coords = [coord[1] for coord in dest_coords]
    dest_polygon = Polygon(list(zip(dest_x_coords, dest_y_coords)), facecolor='red', alpha=0.7, edgecolor='darkred', linewidth=2, label='目标位置')
    ax.add_patch(dest_polygon)
    start_center = start_state.loc
    dest_center = dest_state.loc
    start_heading = start_state.heading
    dest_heading = dest_state.heading
    arrow_length = 2.0
    start_arrow_x = start_center.x + arrow_length * np.cos(start_heading)
    start_arrow_y = start_center.y + arrow_length * np.sin(start_heading)
    ax.arrow(start_center.x, start_center.y, start_arrow_x - start_center.x, start_arrow_y - start_center.y, head_width=0.3, head_length=0.5, fc='green', ec='green', linewidth=2)
    dest_arrow_x = dest_center.x + arrow_length * np.cos(dest_heading)
    dest_arrow_y = dest_center.y + arrow_length * np.sin(dest_heading)
    ax.arrow(dest_center.x, dest_center.y, dest_arrow_x - dest_center.x, dest_arrow_y - dest_center.y, head_width=0.3, head_length=0.5, fc='red', ec='red', linewidth=2)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    plt.tight_layout()
    plt.show()


def generate_bay_dead_end_parking_case(map_level):
    '\n    生成断头路停车场景 - 用于Extrem级别\n    特点：\n    1. 车位位于道路尽头，前方有墙壁\n    2. 左右两侧有障碍物，形成狭窄通道\n    3. 需要精确的倒车入库操作\n    '
    origin = (0.0, 0.0)
    road_length = 20.0
    road_width = random_uniform_num(16, 17)
    #road_width = 15.5
    wall_thickness = 0.5
    obstacles = []
    right_wall = LinearRing([(origin[0] + road_length, origin[1] - road_width / 2), (origin[0] + road_length, origin[1] + road_width / 2), (origin[0] + road_length + wall_thickness, origin[1] + road_width / 2), (origin[0] + road_length + wall_thickness, origin[1] - road_width / 2)])
    obstacles.append(right_wall)
    right_wall = LinearRing([(origin[0], origin[1] - road_width / 2), (origin[0], origin[1] + road_width / 2), (origin[0] - wall_thickness, origin[1] + road_width / 2), (origin[0] - wall_thickness, origin[1] - road_width / 2)])
    obstacles.append(right_wall)
    bottom_wall = LinearRing([(origin[0], origin[1] - road_width / 2), (origin[0] + road_length, origin[1] - road_width / 2), (origin[0] + road_length, origin[1] - road_width / 2 - wall_thickness), (origin[0], origin[1] - road_width / 2 - wall_thickness)])
    obstacles.append(bottom_wall)
    top_wall = LinearRing([(origin[0], origin[1] + road_width / 2), (origin[0] + road_length, origin[1] + road_width / 2), (origin[0] + road_length, origin[1] + road_width / 2 + wall_thickness), (origin[0], origin[1] + road_width / 2 + wall_thickness)])
    obstacles.append(top_wall)

    # 存储全局变量用于碰撞检测
    start_x_global = None
    start_y_global = None
    start_yaw_global = None
    dest_x_global = None
    dest_y_global = None
    dest_yaw_global = None
    
    if random() < 0.5:
        dest_yaw = pi / 2
        dest_y = origin[1] - road_width / 2 + REAR_HANG + 0.2
    else:
        dest_yaw = 3 * pi / 2
        dest_y = origin[1] + road_width / 2 - REAR_HANG - 0.2
    if random() < 0.5:
        dest_x = origin[0] + WIDTH + 0.2
    else:
        dest_x = origin[0] + road_length - WIDTH / 2 - 0.2
    '''dest_yaw = pi / 2
    dest_y = origin[1] - road_width / 2 + REAR_HANG + 0.5
    dest_x = origin[0] + WIDTH + 0.2'''

    # 保存最终的dest位置信息
    dest_x_global = dest_x
    dest_y_global = dest_y
    dest_yaw_global = dest_yaw

    dest_state = State([dest_x, dest_y, dest_yaw, 0, 0])
    dest_box = dest_state.create_box()
    obstacle_car_distance_max = 0.4
    obstacle_car_distance_min = 0.2
    wall_distance_min = 0.3
    wall_distance_max = 0.6
    if dest_yaw == pi / 2 and dest_x < road_length / 2:
        obstacle_car_x = dest_x + WIDTH + random_uniform_num(obstacle_car_distance_min, obstacle_car_distance_max)
        obstacle_car_y = origin[1] - road_width / 2 + REAR_HANG + random_uniform_num(wall_distance_min, wall_distance_max)
    elif dest_yaw == pi / 2 and dest_x >= road_length / 2:
        obstacle_car_x = dest_x - WIDTH - random_uniform_num(obstacle_car_distance_min, obstacle_car_distance_max)
        obstacle_car_y = origin[1] - road_width / 2 + REAR_HANG + random_uniform_num(wall_distance_min, wall_distance_max)
    elif dest_yaw == pi * 3 / 2 and dest_x < road_length / 2:
        obstacle_car_x = dest_x + WIDTH + random_uniform_num(obstacle_car_distance_min, obstacle_car_distance_max)
        obstacle_car_y = origin[1] + road_width / 2 - REAR_HANG - random_uniform_num(wall_distance_min, wall_distance_max)
    else:
        obstacle_car_x = dest_x - WIDTH - random_uniform_num(obstacle_car_distance_min, obstacle_car_distance_max)
        obstacle_car_y = origin[1] + road_width / 2 - REAR_HANG - random_uniform_num(wall_distance_min, wall_distance_max)
    
    obstacle_car_state = State([obstacle_car_x, obstacle_car_y, random_uniform_num(dest_yaw - 0, dest_yaw + 0), 0, 0])
    obstacle_car_box = obstacle_car_state.create_box()
    obstacles.append(obstacle_car_box)
    from random import randint
    num_opposite_cars = randint(3, 6)
    #num_opposite_cars = 5
    opposite_wall_x = origin[0]
    for i in range(num_opposite_cars):
        valid_position = False
        max_attempts = 50
        for attempt in range(max_attempts):
            lateral_distance = random_uniform_num(0, 12)
            #lateral_distance = 8
            if dest_x > road_length / 2:
                opposite_car_x = dest_x - lateral_distance
            else:
                opposite_car_x = dest_x + lateral_distance
            if dest_y > 0:
                #opposite_car_y = origin[1] - road_width / 2 + REAR_HANG + random_uniform_num(0, 0.1)
                opposite_car_y = origin[1] - road_width / 2 + REAR_HANG + 0.1
            else:
                #opposite_car_y = origin[1] + road_width / 2 - REAR_HANG - random_uniform_num(0, 0.1)
                opposite_car_y = origin[1] + road_width / 2 - REAR_HANG - 0.1
            if dest_yaw == pi / 2:
                opposite_car_yaw = random_uniform_num(pi * 3 / 2 , pi * 3 / 2 + 0.01)
                #opposite_car_yaw = pi * 3 / 2
            else:
                opposite_car_yaw = random_uniform_num(pi / 2 , pi / 2 + 0.01)
                #opposite_car_yaw = pi / 2
            opposite_car_state = State([opposite_car_x, opposite_car_y, opposite_car_yaw, 0, 0])
            opposite_car_box = opposite_car_state.create_box()
            collision = False
            for existing_obs in obstacles:
                if existing_obs.intersects(opposite_car_box):
                    collision = True
                    break
            if not collision:
                obstacles.append(opposite_car_box)
                valid_position = True
                break
    from random import randint
    num_opposite_cars_op = randint(1, 5)
    opposite_wall_x = origin[0]
    for i in range(num_opposite_cars_op):
        valid_position = False
        max_attempts = 50
        for attempt in range(max_attempts):
            lateral_distance = random_uniform_num(4, 18)
            #lateral_distance = 10
            if dest_x > road_length / 2:
                opposite_car_x = dest_x - lateral_distance
            else:
                opposite_car_x = dest_x + lateral_distance
            if dest_y > 0:
                #opposite_car_y = origin[1] + road_width / 2 - REAR_HANG - random_uniform_num(0, 0.1)
                opposite_car_y = origin[1] + road_width / 2 - REAR_HANG - 0.1
            else:
                opposite_car_y = origin[1] - road_width / 2 + REAR_HANG + 0.1
                #opposite_car_y = origin[1] - road_width / 2 + REAR_HANG + random_uniform_num(0, 0.1)
            opposite_car_yaw = random_uniform_num(dest_yaw - 0.02, dest_yaw + 0.02)
            opposite_car_state = State([opposite_car_x, opposite_car_y, opposite_car_yaw, 0, 0])
            opposite_car_box = opposite_car_state.create_box()
            collision = False
            for existing_obs in obstacles:
                if existing_obs.intersects(opposite_car_box):
                    collision = True
                    break
            if not collision:
                obstacles.append(opposite_car_box)
                valid_position = True
                break
    start_box_valid = False
    max_attempts = 100
    while not start_box_valid and max_attempts > 0:
        max_attempts -= 1
        start_y = random_uniform_num(origin[1] + 2, origin[1] - 2)
        if dest_x > road_length / 2:
            start_x = random_uniform_num(dest_x - 8, dest_x - 2)
            #start_x = dest_x - 8
            start_yaw = random_uniform_num(-pi / 4, pi /4)
        else:
            start_x = random_uniform_num(dest_x + 2, dest_x + 8)
            #start_x = dest_x + 8
            start_yaw = random_uniform_num(3 * pi / 4 , 5 * pi /4)
        start_state = State([start_x, start_y, start_yaw, 0, 0])
        start_box = start_state.create_box()
        start_box_valid = True
        for obst in obstacles:
            if obst.intersects(start_box):
                start_box_valid = False
                break
        if dest_box.intersects(start_box):
            start_box_valid = False
    if not start_box_valid:
        start_x = origin[0] + 3
        start_y = origin[1]
        start_yaw = pi / 2
    start_pos = [start_x, start_y, start_yaw]
    dest_pos = [dest_x, dest_y, dest_yaw]
    
    # 检测并删除与start或dest碰撞的障碍物
    remove_colliding_obstacles(start_x, start_y, start_yaw, dest_x, dest_y, dest_yaw, obstacles)

    # 调用可视化函数
    #visualize_parking_map(start_x, start_y, start_yaw, dest_x, dest_y, dest_yaw, obstacles)

    return ([start_x, start_y, start_yaw],[dest_x, dest_y, dest_yaw],  obstacles)


def generate_bay_1_parking_case(map_level):
    '\n    生成断头路停车场景 - 用于Extrem级别\n    特点：\n    1. 车位位于道路尽头，前方有墙壁\n    2. 左右两侧有障碍物，形成狭窄通道\n    3. 需要精确的倒车入库操作\n    '
    origin = (0.0, 0.0)
    road_length = 20.0
    road_width = random_uniform_num(16, 17)
    #road_width = 15.5
    wall_thickness = 0.5
    obstacles = []
    right_wall = LinearRing([(origin[0] + road_length, origin[1] - road_width / 2), (origin[0] + road_length, origin[1] + road_width / 2), (origin[0] + road_length + wall_thickness, origin[1] + road_width / 2), (origin[0] + road_length + wall_thickness, origin[1] - road_width / 2)])
    obstacles.append(right_wall)
    right_wall = LinearRing([(origin[0], origin[1] - road_width / 2), (origin[0], origin[1] + road_width / 2), (origin[0] - wall_thickness, origin[1] + road_width / 2), (origin[0] - wall_thickness, origin[1] - road_width / 2)])
    obstacles.append(right_wall)
    bottom_wall = LinearRing([(origin[0], origin[1] - road_width / 2), (origin[0] + road_length, origin[1] - road_width / 2), (origin[0] + road_length, origin[1] - road_width / 2 - wall_thickness), (origin[0], origin[1] - road_width / 2 - wall_thickness)])
    obstacles.append(bottom_wall)
    top_wall = LinearRing([(origin[0], origin[1] + road_width / 2), (origin[0] + road_length, origin[1] + road_width / 2), (origin[0] + road_length, origin[1] + road_width / 2 + wall_thickness), (origin[0], origin[1] + road_width / 2 + wall_thickness)])
    obstacles.append(top_wall)

    # 存储全局变量用于碰撞检测
    start_x_global = None
    start_y_global = None
    start_yaw_global = None
    dest_x_global = None
    dest_y_global = None
    dest_yaw_global = None
    
    if random() < 0.5:
        dest_yaw = pi / 2
        dest_y = origin[1] - road_width / 2 + REAR_HANG + 0.2
    else:
        dest_yaw = 3 * pi / 2
        dest_y = origin[1] + road_width / 2 - REAR_HANG - 0.2
    if random() < 0.5:
        dest_x = random_uniform_num(3,18)
    else:
        dest_x = random_uniform_num(3, 18)
    '''dest_yaw = pi / 2
    dest_y = origin[1] - road_width / 2 + REAR_HANG + 0.5
    dest_x = origin[0] + WIDTH + 0.2'''

    dest_state = State([dest_x, dest_y, dest_yaw, 0, 0])
    dest_box = dest_state.create_box()
    obstacle_car_distance_max = 0.4
    obstacle_car_distance_min = 0.2
    wall_distance_min = 0.1
    wall_distance_max = 0.1
    if dest_yaw == pi / 2 and dest_x < road_length / 2:
        obstacle_car_x = dest_x + WIDTH + random_uniform_num(obstacle_car_distance_min, obstacle_car_distance_max)
        obstacle_car_y = origin[1] - road_width / 2 + REAR_HANG + random_uniform_num(wall_distance_min, wall_distance_max)
    elif dest_yaw == pi / 2 and dest_x >= road_length / 2:
        obstacle_car_x = dest_x - WIDTH - random_uniform_num(obstacle_car_distance_min, obstacle_car_distance_max)
        obstacle_car_y = origin[1] - road_width / 2 + REAR_HANG + random_uniform_num(wall_distance_min, wall_distance_max)
    elif dest_yaw == pi * 3 / 2 and dest_x < road_length / 2:
        obstacle_car_x = dest_x + WIDTH + random_uniform_num(obstacle_car_distance_min, obstacle_car_distance_max)
        obstacle_car_y = origin[1] + road_width / 2 - REAR_HANG - random_uniform_num(wall_distance_min, wall_distance_max)
    else:
        obstacle_car_x = dest_x - WIDTH - random_uniform_num(obstacle_car_distance_min, obstacle_car_distance_max)
        obstacle_car_y = origin[1] + road_width / 2 - REAR_HANG - random_uniform_num(wall_distance_min, wall_distance_max)
    
    obstacle_car_state = State([obstacle_car_x, obstacle_car_y, random_uniform_num(dest_yaw - 0, dest_yaw + 0), 0, 0])
    obstacle_car_box = obstacle_car_state.create_box()
    obstacles.append(obstacle_car_box)
    from random import randint
    num_opposite_cars = randint(3, 8)
    #num_opposite_cars = 5
    opposite_wall_x = origin[0]
    for i in range(num_opposite_cars):
        valid_position = False
        max_attempts = 50
        for attempt in range(max_attempts):
            lateral_distance = random_uniform_num(0, 12)
            #lateral_distance = 8
            if dest_x > road_length / 2:
                opposite_car_x = dest_x - lateral_distance
            else:
                opposite_car_x = dest_x + lateral_distance
            opposite_car_x = randint(0, 20)
            if dest_y > 0:
                #opposite_car_y = origin[1] - road_width / 2 + REAR_HANG + random_uniform_num(0, 0.01)
                opposite_car_y = origin[1] - road_width / 2 + REAR_HANG + 0.1
            else:
                #opposite_car_y = origin[1] + road_width / 2 - REAR_HANG - random_uniform_num(0, 0.01)
                opposite_car_y = origin[1] + road_width / 2 - REAR_HANG - 0.1
            if dest_yaw == pi / 2:
                opposite_car_yaw = random_uniform_num(pi * 3 / 2 , pi * 3 / 2 + 0.01)
                #opposite_car_yaw = pi * 3 / 2
            else:
                opposite_car_yaw = random_uniform_num(pi / 2, pi / 2 + 0.01)
                #opposite_car_yaw = pi / 2
            opposite_car_state = State([opposite_car_x, opposite_car_y, opposite_car_yaw, 0, 0])
            opposite_car_box = opposite_car_state.create_box()
            collision = False
            for existing_obs in obstacles:
                if existing_obs.intersects(opposite_car_box):
                    collision = True
                    break
            if not collision:
                obstacles.append(opposite_car_box)
                valid_position = True
                break
    from random import randint
    num_opposite_cars_op = randint(3, 8)
    opposite_wall_x = origin[0]
    for i in range(num_opposite_cars_op):
        valid_position = False
        max_attempts = 50
        for attempt in range(max_attempts):
            lateral_distance = random_uniform_num(4, 18)
            #lateral_distance = 10
            '''if dest_x > road_length / 2:
                opposite_car_x = dest_x - lateral_distance
            else:
                opposite_car_x = dest_x + lateral_distance'''
            opposite_car_x = randint(0, 20)
            if dest_y > 0:
                #opposite_car_y = origin[1] + road_width / 2 - REAR_HANG - random_uniform_num(0, 0.1)
                opposite_car_y = origin[1] + road_width / 2 - REAR_HANG - 0.1
            else:
                #opposite_car_y = origin[1] - road_width / 2 + REAR_HANG + random_uniform_num(0, 0.1)
                opposite_car_y = origin[1] - road_width / 2 + REAR_HANG + 0.1
            opposite_car_yaw = random_uniform_num(dest_yaw , dest_yaw + 0.01)
            opposite_car_state = State([opposite_car_x, opposite_car_y, opposite_car_yaw, 0, 0])
            opposite_car_box = opposite_car_state.create_box()
            collision = False
            for existing_obs in obstacles:
                if existing_obs.intersects(opposite_car_box):
                    collision = True
                    break
            if not collision:
                obstacles.append(opposite_car_box)
                valid_position = True
                break
    start_box_valid = False
    max_attempts = 100
    while not start_box_valid and max_attempts > 0:
        max_attempts -= 1
        start_y = random_uniform_num(origin[1] + 2, origin[1] - 2)
        if dest_x > road_length / 2:
            start_x = random_uniform_num(dest_x - 8, dest_x - 2)
            #start_x = dest_x - 8
            start_yaw = random_uniform_num(-pi / 4, pi / 4)
        else:
            start_x = random_uniform_num(dest_x + 2, dest_x + 8)
            #start_x = dest_x + 8
            start_yaw = random_uniform_num(3 * pi / 4, 5 * pi / 4)
        start_state = State([start_x, start_y, start_yaw, 0, 0])
        start_box = start_state.create_box()
        start_box_valid = True
        for obst in obstacles:
            if obst.intersects(start_box):
                start_box_valid = False
                break
        if dest_box.intersects(start_box):
            start_box_valid = False
    if not start_box_valid:
        start_x = origin[0] + 3
        start_y = origin[1]
        start_yaw = pi / 2
    start_pos = [start_x, start_y, start_yaw]
    dest_pos = [dest_x, dest_y, dest_yaw]

    remove_colliding_obstacles(start_x, start_y, start_yaw, dest_x, dest_y, dest_yaw, obstacles)
    #visualize_parking_map(start_x, start_y, start_yaw, dest_x, dest_y, dest_yaw, obstacles)
    return ([start_x, start_y, start_yaw], [dest_x, dest_y, dest_yaw], obstacles)


def remove_colliding_obstacles(start_x, start_y, start_yaw, dest_x, dest_y, dest_yaw, obstacles):
    start_state = State([start_x, start_y, start_yaw, 0, 0])
    start_box = start_state.create_box()
    dest_state = State([dest_x, dest_y, dest_yaw, 0, 0])
    dest_box = dest_state.create_box()

    # 过滤掉与start或dest碰撞的障碍物
    filtered_obstacles = []
    for obst in obstacles:
        if not (obst.intersects(start_box) or obst.intersects(dest_box)):
            filtered_obstacles.append(obst)

    obstacles[:] = filtered_obstacles  # 更新原列表


class ParkingMapNormal(object):

    def __init__(self, map_level=MAP_LEVEL):
        self.case_id: int = None
        self.map_level = map_level
        self.start: State = None
        self.dest: State = None
        self.start_box: LinearRing = None
        self.dest_box: LinearRing = None
        self.xmin, self.xmax = (0, 0)
        self.ymin, self.ymax = (0, 0)
        self.n_obstacle = 0
        self.obstacles: List[Area] = []
        self.is_dead_end_scenario: bool = False  # 标记是否为断头路场景

    def reset(self, case_id: int=None, path: str=None) -> State:
        self.map_level = 'Complex'
        manual = 0
        if self.map_level == 'Complex':
            if random() < 0.5 and manual == 0:
                start, dest, obstacles = generate_bay_dead_end_parking_case(self.map_level)
                self.case_id = 2
            elif random() < 0.25:
                start, dest, obstacles = generate_bay_1_parking_case(self.map_level)
                self.case_id = 0
            else:
                start, dest, obstacles = generate_bay_1_parking_case(self.map_level)
                self.case_id = 1
        elif (case_id == 0 or (random() > 0.5 and case_id != 1)) and self.map_level in ['Normal', 'Complex']:
            start, dest, obstacles = generate_bay_parking_case(self.map_level)
            self.case_id = 0
        else:
            start, dest, obstacles = generate_bay_parking_case(self.map_level)
            self.case_id = 1
        '''if  self.map_level in ['Normal', 'Complex']:
            start, dest, obstacles = generate_bay_parking_case(self.map_level)
            self.case_id = 0
        else:
            start, dest, obstacles = generate_parallel_parking_case(self.map_level)
            self.case_id = 1'''
        self.start = State(start + [0, 0])
        self.start_box = self.start.create_box()
        self.dest = State(dest + [0, 0])
        self.dest_box = self.dest.create_box()
        self.xmin = np.floor(min(self.start.loc.x, self.dest.loc.x) - 10)
        self.xmax = np.ceil(max(self.start.loc.x, self.dest.loc.x) + 10)
        self.ymin = np.floor(min(self.start.loc.y, self.dest.loc.y) - 10)
        self.ymax = np.ceil(max(self.start.loc.y, self.dest.loc.y) + 10)
        self.obstacles = list([Area(shape=obs, subtype='obstacle', color=(150, 150, 150, 255)) for obs in obstacles])
        self.n_obstacle = len(self.obstacles)
        
        self.is_dead_end_scenario = self._is_dead_end_scenario()
        if self.is_dead_end_scenario:
            print("检测到断头路场景，交换起点和终点位置")
            temp_start = self.start
            temp_start_box = self.start_box
            self.start = self.dest
            self.start_box = self.dest_box
            self.dest = temp_start
            self.dest_box = temp_start_box
            
            self.xmin = np.floor(min(self.start.loc.x, self.dest.loc.x) - 10)
            self.xmax = np.ceil(max(self.start.loc.x, self.dest.loc.x) + 10)
            self.ymin = np.floor(min(self.start.loc.y, self.dest.loc.y) - 10)
            self.ymax = np.ceil(max(self.start.loc.y, self.dest.loc.y) + 10)
        return self.start


    def _is_dead_end_scenario(self):
        """
        检查当前地图配置是否为断头路场景
        结合角度判断和障碍物覆盖情况判断
        """
        angle_check = self._check_start_facing_dest_by_angle()
        is_dead_end_cov = self._is_dead_end_by_bilateral_coverage()
        is_dead_end = angle_check and is_dead_end_cov
        
        return is_dead_end
    
    def _check_start_facing_dest_by_angle(self):
        """
        检查起点车尾是否朝向终点
        """
        sx, sy = self.start.loc.x, self.start.loc.y
        dx, dy = self.dest.loc.x, self.dest.loc.y
        
        def _normalize_angle(a):
            return (a + np.pi) % (2 * np.pi) - np.pi  # (-π, π]
        def _angle_distance(a, b):
            return abs(_normalize_angle(a - b))      # [0, π]

        line_dir = np.arctan2(dy - sy, dx - sx)
        tail_dir_start = self.start.heading + np.pi
        tail_line_angle_diff = _angle_distance(tail_dir_start, line_dir)
        
        return tail_line_angle_diff > np.pi / 2
    
    def _is_dead_end_by_bilateral_coverage(self):
        """
        基于终点车位两侧障碍物覆盖情况判断断头路
        核心思想：分析车位左右两侧的障碍物分布和覆盖度
        使用y轴占用率检测，提高对长形小面积障碍物的检测准确性
        """
        dest_x = self.dest.loc.x
        dest_y = self.dest.loc.y
        dest_heading = self.dest.heading
        
        left_check_area = self._check_parking_side_coverage(
            dest_x, dest_y, dest_heading, side='left'
        )
        right_check_area = self._check_parking_side_coverage(
            dest_x, dest_y, dest_heading, side='right'
        )
        
        left_y_occupancy_check = self._check_y_axis_occupancy(
            left_check_area['check_area'], step_size=0.05, occupancy_threshold=0.8
        )
        right_y_occupancy_check = self._check_y_axis_occupancy(
            right_check_area['check_area'], step_size=0.05, occupancy_threshold=0.8
        )
        
        
        return left_y_occupancy_check or right_y_occupancy_check
        
    
    def _check_parking_side_coverage(self, dest_x, dest_y, dest_heading, side='left'):
        
        from configs import WIDTH, LENGTH, REAR_HANG, WHEEL_BASE
        
        rear_to_center_distance = WHEEL_BASE / 2 
        check_length = LENGTH * 1.2  
        check_width = 1.0           
        vehicle_center_x = dest_x + rear_to_center_distance * np.cos(dest_heading)
        vehicle_center_y = dest_y + rear_to_center_distance * np.sin(dest_heading)
        

        front_offset = WHEEL_BASE + FRONT_HANG
        rear_offset = LENGTH / 2 + check_length/2
        side_offset = WIDTH / 2 + 1.0  
        

        if side == 'left':
            if dest_heading >= -np.pi/2 and dest_heading < np.pi/2:
                corner_angle = dest_heading + np.pi/2  
                check_center_x = vehicle_center_x + front_offset * np.cos(dest_heading) + side_offset * np.cos(corner_angle)
                check_center_y = vehicle_center_y + front_offset * np.sin(dest_heading) + side_offset * np.sin(corner_angle)
            else:
                corner_angle = dest_heading + np.pi/2  
                check_center_x = vehicle_center_x + rear_offset * np.cos(dest_heading) + side_offset * np.cos(corner_angle)
                check_center_y = vehicle_center_y + rear_offset * np.sin(dest_heading) + side_offset * np.sin(corner_angle)
        else:
            if dest_heading >= -np.pi/2 and dest_heading < np.pi/2:
                corner_angle = dest_heading - np.pi/2  
                check_center_x = vehicle_center_x + front_offset * np.cos(dest_heading) + side_offset * np.cos(corner_angle)
                check_center_y = vehicle_center_y + front_offset * np.sin(dest_heading) + side_offset * np.sin(corner_angle)
            else:
                corner_angle = dest_heading - np.pi/2  
                check_center_x = vehicle_center_x + rear_offset * np.cos(dest_heading) + side_offset * np.cos(corner_angle)
                check_center_y = vehicle_center_y + rear_offset * np.sin(dest_heading) + side_offset * np.sin(corner_angle)
            
        check_area = self._create_side_check_area(
            check_center_x, check_center_y, dest_heading, 
            check_length, check_width
        )
        
        return {
            'check_area': check_area,
        }
    
    def _create_side_check_area(self, center_x, center_y, heading, length, width):

        from shapely.geometry import Polygon
        
        half_length = length / 2
        half_width = width / 2
        
        vertices = [
            (-half_length, -half_width),  # 左下
            (half_length, -half_width),   # 右下
            (half_length, half_width),    # 右上
            (-half_length, half_width)    # 左上
        ]
        
        rotated_vertices = []
        for vx, vy in vertices:
            rx = vx * np.cos(heading) - vy * np.sin(heading)
            ry = vx * np.sin(heading) + vy * np.cos(heading)
            rotated_vertices.append((center_x + rx, center_y + ry))
        
        return Polygon(rotated_vertices)

    def _check_y_axis_occupancy(self, check_area, step_size=0.1, occupancy_threshold=0.8):
        """
        检测检查区域内y轴占用率
        """
        from shapely.geometry import LineString, Point
        import numpy as np
        
        bounds = check_area.bounds
        minx, miny, maxx, maxy = bounds
        x_positions = np.arange(minx, maxx, step_size)
        y_occupancy_rates = []
        high_occupancy_positions = 0
        
        for i, x in enumerate(x_positions):
            y_occupancy = self._get_y_axis_occupancy_at_x(check_area, x, miny, maxy)
            y_occupancy_rates.append(y_occupancy)
            
            if y_occupancy > occupancy_threshold:
                high_occupancy_positions += 1
        
        max_occupancy = np.max(y_occupancy_rates) if y_occupancy_rates else 0
        
        is_high_occupancy = max_occupancy > occupancy_threshold  
        
        if is_high_occupancy:
            return True
        else:
            return False
    def _get_y_axis_occupancy_at_x(self, check_area, x, miny, maxy):
        """
        计算指定x位置y轴方向的障碍物占用率
        """
        from shapely.geometry import LineString, Point
        import numpy as np
        
        y_line = LineString([(x, miny), (x, maxy)])
        
        if not check_area.intersects(y_line):
            return 0.0
        
        intersection = check_area.intersection(y_line)
        if intersection.is_empty:
            return 0.0
        
        if hasattr(intersection, 'length'):
            total_y_length = intersection.length
        else:
            total_y_length = 0
            if hasattr(intersection, 'geoms'):
                for geom in intersection.geoms:
                    if hasattr(geom, 'length'):
                        total_y_length += geom.length
            else:
                total_y_length = 0
        
        if total_y_length == 0:
            return 0.0
        
        from shapely.geometry import MultiLineString
        occupied_segments = []
        obstacle_count = 0
        intersecting_obstacles = 0
        
        for i, obstacle in enumerate(self.obstacles):
            if hasattr(obstacle, 'shape'):
                try:
                    obstacle_shape = obstacle.shape
                    if hasattr(obstacle_shape, 'coords') and not hasattr(obstacle_shape, 'exterior'):
                        from shapely.geometry import Polygon
                        try:
                            obstacle_shape = Polygon(list(obstacle_shape.coords))
                        except Exception:
                            continue
                    obstacle_count += 1
                    if y_line.intersects(obstacle_shape):
                        intersecting_obstacles += 1
                        line_obstacle_intersection = y_line.intersection(obstacle_shape)
                        if not line_obstacle_intersection.is_empty:
                            if hasattr(line_obstacle_intersection, 'geoms'):
                                for geom in line_obstacle_intersection.geoms:
                                    if hasattr(geom, 'length') and geom.length > 0:
                                        occupied_segments.append(geom)
                            elif hasattr(line_obstacle_intersection, 'length') and line_obstacle_intersection.length > 0:
                                occupied_segments.append(line_obstacle_intersection)
                except Exception as e:
                    continue
        
        if occupied_segments:
            try:
                from shapely.ops import unary_union
                merged_segments = unary_union(occupied_segments)
                if hasattr(merged_segments, 'length'):
                    occupied_length = merged_segments.length
                else:
                    occupied_length = 0
                    if hasattr(merged_segments, 'geoms'):
                        for geom in merged_segments.geoms:
                            if hasattr(geom, 'length'):
                                occupied_length += geom.length
            except Exception:
                occupied_length = 0
                for geom in occupied_segments:
                    if hasattr(geom, 'length'):
                        occupied_length += geom.length
        else:
            occupied_length = 0.0
        
        if total_y_length > 0:
            occupancy_rate = min(occupied_length / total_y_length, 1.0)
        else:
            occupancy_rate = 0.0
        
        
        return occupancy_rate

