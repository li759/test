import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class AngleRewardCalculator:
    def normalize_angle(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

    def angle_distance(a: float, b: float) -> float:
        return abs(AngleRewardCalculator.normalize_angle(a - b))

    def get_angle_diff(angle1: float, angle2: float) -> float:
        angle_dif = math.acos(math.cos(angle1 - angle2))
        return angle_dif if angle_dif < math.pi / 2 else math.pi - angle_dif

    def get_direction_weight(distance_to_goal: float, angle_diff: float, M0: float=None, epsilon: float=None, k: float=None) -> float:
        if M0 is None:
            M0 = 5
        if epsilon is None:
            epsilon = 0.1
        if k is None:
            k = 2
        
        abs_distance = abs(distance_to_goal)
        abs_angle = np.abs(angle_diff)
        
        # 当距离为0时，直接返回最大权重
        if abs_distance < 1e-6:
            return 1.0
        
        # 当角度接近0时，使用线性插值避免凹坑
        if abs_angle < 1e-6:
            # 当角度为0时，权重应该很高
            return 1.0
        
        M = abs_distance / (np.sin(abs_angle) + epsilon)
        g = 1.0 / (1.0 + (M / M0) ** k)
        
        return g

    def compute_angle_reward(is_dead_end: bool,
                             prev_heading: float,
                             curr_heading: float,
                             dest_heading: float,
                             distance_to_goal: float,
                             angle_diff: float) -> float:
        angle_norm_ratio = math.pi
        if is_dead_end:
            prev_angle_diff = AngleRewardCalculator.angle_distance(prev_heading, dest_heading)
            curr_angle_diff = AngleRewardCalculator.angle_distance(curr_heading, dest_heading)
            return (prev_angle_diff / angle_norm_ratio) - (curr_angle_diff / angle_norm_ratio)
        else:
            prev_angle_diff = AngleRewardCalculator.get_angle_diff(prev_heading, dest_heading)
            base_angle_reward = prev_angle_diff / angle_norm_ratio - angle_diff / angle_norm_ratio
            direction_weight = AngleRewardCalculator.get_direction_weight(distance_to_goal, angle_diff)
            return base_angle_reward * direction_weight


def test_compute_angle_reward():
    distances_to_goal = np.linspace(0.0, 10.0, 50)  # 从0开始，包含距离为0的情况
    angles_deg = np.linspace(-90, 90, 50)   
    angles_rad = np.deg2rad(angles_deg)    
    
    D, A = np.meshgrid(distances_to_goal, angles_rad)
    
    weights = np.zeros_like(D)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            weights[i, j] = AngleRewardCalculator.get_direction_weight(
                D[i, j], A[i, j]
            )
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(D, A, weights, cmap='viridis', alpha=0.8)
    
    ax.set_xlabel('Distance to Goal (m)')
    ax.set_ylabel('Heading Angle (rad)')
    ax.set_zlabel('Direction Weight')
    ax.set_title('Direction Weight vs Distance to Goal and Heading Angle\n(Non-Dead-End Scenario, Distance to Goal)')
    
    yticks_rad = np.linspace(-np.pi/2, np.pi/2, 7)
    yticks_deg = np.rad2deg(yticks_rad)
    ax.set_yticks(yticks_rad)
    ax.set_yticklabels([f'{deg:.0f}°' for deg in yticks_deg])
    
    fig.colorbar(surf, shrink=0.5, aspect=20)
    
 
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.show()
    
    # 打印一些关键点的值
    print("Distance to Goal Direction Weight Analysis:")
    print("=" * 60)
    print("Smaller values = Closer to goal, Larger values = Farther from goal")
    print("-" * 60)
    
    # 测试几个关键点
    test_cases = [
        (0.0, 0),      # 距离目标0m, 0° (应该返回最大权重)
        (0.5, 0),      # 距离目标0.5m, 0°
        (1.0, 0),      # 距离目标1m, 0°
        (2.0, 0),      # 距离目标2m, 0°
        (5.0, 0),      # 距离目标5m, 0°
        (10.0, 0),     # 距离目标10m, 0°
        (0.0, 45),     # 距离目标0m, 45° (应该返回最大权重)
        (1.0, 45),     # 距离目标1m, 45°
        (2.0, 45),     # 距离目标2m, 45°
        (5.0, 45),     # 距离目标5m, 45°
        (0.0, 90),     # 距离目标0m, 90° (应该返回最大权重)
        (1.0, 90),     # 距离目标1m, 90°
        (2.0, 90),     # 距离目标2m, 90°
        (5.0, 90),     # 距离目标5m, 90°
    ]

    curr_heading1 = 65/57.3
    curr_heading2 = 70/57.3
    dest_heading = 90/57.3
    dest_diff = 1
    angle_diff1 = get_angle_diff(curr_heading1, dest_heading)
    angle_diff2 = get_angle_diff(curr_heading2, dest_heading)
    print("angle_diff1:",angle_diff1)
    print("angle_diff2:",angle_diff2)
    weight1 = AngleRewardCalculator.get_direction_weight(0, 1)
    weight2 = AngleRewardCalculator.get_direction_weight(dest_diff, angle_diff2)
    print("weight1:",weight1)
    print("weight2:",weight2)
    prev_heading = 60/57.3
    curr_heading = 55/57.3
    dest_heading = 90/57.3
    distance_to_goal = 2.0  # 距离目标2米
    angle_diff = get_angle_diff(curr_heading, dest_heading)
    angle_reward1 = AngleRewardCalculator.compute_angle_reward(False, prev_heading, curr_heading, dest_heading, distance_to_goal, angle_diff)
    print("angle_reward1:",angle_reward1)


    for distance_to_goal, angle_deg in test_cases:
        angle_rad = np.deg2rad(angle_deg)
        weight = AngleRewardCalculator.get_direction_weight(distance_to_goal, angle_rad)
        proximity = "Close" if distance_to_goal < 2 else "Medium" if distance_to_goal < 5 else "Far"
        print(f"Distance: {distance_to_goal:4.1f}m ({proximity:6}), Angle: {angle_deg:3.0f}° → Weight: {weight:.4f}")
    
    return fig, ax
def get_angle_diff(angle1, angle2):
    angle_dif = math.acos(math.cos(angle1 - angle2))
    return angle_dif if angle_dif < math.pi / 2 else math.pi - angle_dif

if __name__ == "__main__":
    print("Running distance to goal angle reward test...")
    test_compute_angle_reward()
