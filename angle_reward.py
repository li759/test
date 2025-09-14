import math
import numpy as np
from configs import ADAPTIVE_ANGLE_M0, ADAPTIVE_ANGLE_EPSILON, ADAPTIVE_ANGLE_K

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
    def get_direction_weight(distance_to_goal: float, angle_diff: float, M0: float=None, epsilon: float=None, k: float=None) -> float:
        if M0 is None:
            M0 = ADAPTIVE_ANGLE_M0
        if epsilon is None:
            epsilon = ADAPTIVE_ANGLE_EPSILON
        if k is None:
            k = ADAPTIVE_ANGLE_K
        abs_angle = np.abs(angle_diff)
        M = distance_to_goal / (np.sin(abs_angle) + epsilon)
        g = 1.0 / (1.0 + (M / M0) ** k)
        return g

    @staticmethod
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
