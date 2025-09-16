import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class AngleRewardCalculator:
    @staticmethod
    def normalize_angle(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

    @staticmethod
    def angle_distance(a: float, b: float) -> float:
        return abs(AngleRewardCalculator.normalize_angle(a - b))

    @staticmethod
    def get_angle_diff(angle1: float, angle2: float) -> float:
        angle_dif = math.acos(math.cos(angle1 - angle2))
        return angle_dif if angle_dif < math.pi / 2 else math.pi - angle_dif

    @staticmethod
    def get_direction_weight(lateral_distance: float, angle_diff: float, M0: float=None, epsilon: float=None, k: float=None) -> float:
        if M0 is None:
            M0 = 5
        if epsilon is None:
            epsilon = 0.1
        if k is None:
            k = 2
        
        abs_lateral_dist = abs(lateral_distance)
        abs_angle = np.abs(angle_diff)
        
        M = abs_lateral_dist / (np.sin(abs_angle) + epsilon)
        g = 1.0 / (1.0 + (M / M0) ** k)
        
        if lateral_distance > 0:
            g *= 0.9  
        
        return g

    @staticmethod
    def compute_angle_reward(is_dead_end: bool,
                             prev_heading: float,
                             curr_heading: float,
                             dest_heading: float,
                             lateral_distance: float,
                             angle_diff: float) -> float:
        angle_norm_ratio = math.pi
        if is_dead_end:
            prev_angle_diff = AngleRewardCalculator.angle_distance(prev_heading, dest_heading)
            curr_angle_diff = AngleRewardCalculator.angle_distance(curr_heading, dest_heading)
            return (prev_angle_diff / angle_norm_ratio) - (curr_angle_diff / angle_norm_ratio)
        else:
            prev_angle_diff = AngleRewardCalculator.get_angle_diff(prev_heading, dest_heading)
            base_angle_reward = prev_angle_diff / angle_norm_ratio - angle_diff / angle_norm_ratio
            direction_weight = AngleRewardCalculator.get_direction_weight(lateral_distance, angle_diff)
            return base_angle_reward * direction_weight


def test_compute_angle_reward():
    lateral_distances = np.linspace(-5.0, 5.0, 50) 
    angles_deg = np.linspace(-90, 90, 50)   
    angles_rad = np.deg2rad(angles_deg)    
    
    L, A = np.meshgrid(lateral_distances, angles_rad)
    
    weights = np.zeros_like(L)
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            weights[i, j] = AngleRewardCalculator.get_direction_weight(
                L[i, j], A[i, j]
            )
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(L, A, weights, cmap='viridis', alpha=0.8)
    
    ax.set_xlabel('Lateral Distance (m)')
    ax.set_ylabel('Heading Angle (rad)')
    ax.set_zlabel('Direction Weight')
    ax.set_title('Direction Weight vs Lateral Distance and Heading Angle\n(Non-Dead-End Scenario, Lateral Distance)')
    
    yticks_rad = np.linspace(-np.pi/2, np.pi/2, 7)
    yticks_deg = np.rad2deg(yticks_rad)
    ax.set_yticks(yticks_rad)
    ax.set_yticklabels([f'{deg:.0f}°' for deg in yticks_deg])
    
    fig.colorbar(surf, shrink=0.5, aspect=20)
    
 
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.show()
    
    # 打印一些关键点的值
    print("Lateral Distance Direction Weight Analysis:")
    print("=" * 60)
    print("Negative values = Left side, Positive values = Right side")
    print("-" * 60)
    
    # 测试几个关键点
    test_cases = [
        (-3.0, 0),     # 左侧3m, 0°
        (-1.0, 0),     # 左侧1m, 0°
        (0.0, 0),      # 中心, 0°
        (1.0, 0),      # 右侧1m, 0°
        (3.0, 0),      # 右侧3m, 0°
        (-2.0, 45),    # 左侧2m, 45°
        (0.0, 45),     # 中心, 45°
        (2.0, 45),     # 右侧2m, 45°
        (-1.0, 90),    # 左侧1m, 90°
        (0.0, 90),     # 中心, 90°
        (1.0, 90),     # 右侧1m, 90°
    ]

    curr_heading1 = 65/57.3
    curr_heading2 = 70/57.3
    dest_heading = 90/57.3
    dest_diff = 1
    angle_diff1 = get_angle_diff(curr_heading1, dest_heading)
    angle_diff2 = get_angle_diff(curr_heading2, dest_heading)
    print("angle_diff1:",angle_diff1)
    print("angle_diff2:",angle_diff2)
    weight1 = AngleRewardCalculator.get_direction_weight(dest_diff, angle_diff1)
    weight2 = AngleRewardCalculator.get_direction_weight(dest_diff, angle_diff2)
    print("weight1:",weight1)
    print("weight2:",weight2)
    prev_heading = 60/57.3
    curr_heading = 55/57.3
    dest_heading = 90/57.3
    lateral_distance = -1
    angle_diff = get_angle_diff(curr_heading, dest_heading)
    angle_reward1 = AngleRewardCalculator.compute_angle_reward(False, prev_heading, curr_heading, dest_heading, lateral_distance, angle_diff)
    print("angle_reward1:",angle_reward1)


    for lateral_dist, angle_deg in test_cases:
        angle_rad = np.deg2rad(angle_deg)
        weight = AngleRewardCalculator.get_direction_weight(lateral_dist, angle_rad)
        side = "Left" if lateral_dist < 0 else "Center" if lateral_dist == 0 else "Right"
        print(f"Lateral: {lateral_dist:4.1f}m ({side:6}), Angle: {angle_deg:3.0f}° → Weight: {weight:.4f}")
    
    return fig, ax
def get_angle_diff(angle1, angle2):
    angle_dif = math.acos(math.cos(angle1 - angle2))
    return angle_dif if angle_dif < math.pi / 2 else math.pi - angle_dif

if __name__ == "__main__":
    print("Running lateral distance angle reward test...")
    test_compute_angle_reward()
