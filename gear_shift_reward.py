import numpy as np

class GearShiftRewardCalculator:
    """
    Compute gear shift penalty with short-term and historical components.
    Maintains an internal history buffer for recent shift frequency.
    """
    def __init__(self, history_window: int = 100, recent_window: int = 20):
        self.history_window = history_window
        self.recent_window = recent_window
        self._gear_shift_history = []  # 1 for shift, 0 for no shift

    def reset(self):
        self._gear_shift_history = []

    def _base_cost(self, prev_action: np.ndarray, curr_action: np.ndarray) -> float:
        prev_speed = prev_action[1]
        curr_speed = curr_action[1]
        # prev_steer = prev_action[0]
        # curr_steer = curr_action[0]
        cost = 0.0
        # Penalize direction change (gear change)
        if prev_speed * curr_speed < 0:
            cost -= 1.0
        return cost

    def calculate(self, prev_action: np.ndarray, curr_action: np.ndarray) -> float:
        base_cost = self._base_cost(prev_action, curr_action)
        if base_cost < 0:
            self._gear_shift_history.append(1)
        else:
            self._gear_shift_history.append(0)
        if len(self._gear_shift_history) > self.history_window:
            self._gear_shift_history = self._gear_shift_history[-self.history_window:]
        recent = self._gear_shift_history[-self.recent_window:] if self.recent_window > 0 else self._gear_shift_history
        recent_shift_freq = sum(recent) / max(len(recent), 1)
        if recent_shift_freq > 0.7:
            multiplier = 1.5
        elif recent_shift_freq < 0.2:
            multiplier = 0.5
        else:
            multiplier = 1.0
        return base_cost * multiplier
