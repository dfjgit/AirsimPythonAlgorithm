"""
Crazyflie实体无人机训练奖励配置
"""
import json
import os


class CrazyflieRewardConfig:
    """实体无人机训练奖励配置类"""

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__),
                "crazyflie_reward_config.json"
            )
        self.config_path = config_path
        self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            print(f"[警告] 配置文件不存在: {self.config_path}")
            self._set_defaults()
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            rewards = config.get("reward_coefficients", {})
            self.speed_reward = rewards.get("speed_reward", 1.0)
            self.speed_penalty_threshold = rewards.get("speed_penalty_threshold", 1.5)
            self.speed_penalty = rewards.get("speed_penalty", 1.0)
            self.accel_penalty = rewards.get("accel_penalty", 0.1)
            self.angular_rate_penalty = rewards.get("angular_rate_penalty", 0.05)
            self.scan_reward = rewards.get("scan_reward", 2.0)
            self.out_of_range_penalty = rewards.get("out_of_range_penalty", 2.0)
            self.action_change_penalty = rewards.get("action_change_penalty", 0.05)
            self.action_magnitude_penalty = rewards.get("action_magnitude_penalty", 0.01)
            self.battery_optimal_reward = rewards.get("battery_optimal_reward", 0.5)
            self.battery_low_penalty = rewards.get("battery_low_penalty", 1.0)

            thresholds = config.get("thresholds", {})
            self.scan_entropy_threshold = thresholds.get("scan_entropy_threshold", 30)
            self.leader_range_buffer = thresholds.get("leader_range_buffer", 0.0)
            self.battery_optimal_min = thresholds.get("battery_optimal_min", 3.7)
            self.battery_optimal_max = thresholds.get("battery_optimal_max", 4.0)
            self.battery_low_threshold = thresholds.get("battery_low_threshold", 3.5)

            episode = config.get("episode", {})
            self.max_steps = episode.get("max_steps", 200)

            action_space = config.get("action_space", {})
            self.weight_min = action_space.get("weight_min", 0.5)
            self.weight_max = action_space.get("weight_max", 5.0)

            print("[OK] Crazyflie奖励配置加载成功")
        except Exception as e:
            print(f"[错误] Crazyflie奖励配置加载失败: {str(e)}")
            self._set_defaults()

    def _set_defaults(self):
        self.speed_reward = 1.0
        self.speed_penalty_threshold = 1.5
        self.speed_penalty = 1.0
        self.accel_penalty = 0.1
        self.angular_rate_penalty = 0.05
        self.scan_reward = 2.0
        self.out_of_range_penalty = 2.0
        self.action_change_penalty = 0.05
        self.action_magnitude_penalty = 0.01
        self.battery_optimal_reward = 0.5
        self.battery_low_penalty = 1.0

        self.scan_entropy_threshold = 30
        self.leader_range_buffer = 0.0
        self.battery_optimal_min = 3.7
        self.battery_optimal_max = 4.0
        self.battery_low_threshold = 3.5

        self.max_steps = 200
        self.weight_min = 0.5
        self.weight_max = 5.0

    def to_dict(self):
        return {
            "reward_coefficients": {
                "speed_reward": self.speed_reward,
                "speed_penalty_threshold": self.speed_penalty_threshold,
                "speed_penalty": self.speed_penalty,
                "accel_penalty": self.accel_penalty,
                "angular_rate_penalty": self.angular_rate_penalty,
                "scan_reward": self.scan_reward,
                "out_of_range_penalty": self.out_of_range_penalty,
                "action_change_penalty": self.action_change_penalty,
                "action_magnitude_penalty": self.action_magnitude_penalty,
                "battery_optimal_reward": self.battery_optimal_reward,
                "battery_low_penalty": self.battery_low_penalty
            },
            "thresholds": {
                "scan_entropy_threshold": self.scan_entropy_threshold,
                "leader_range_buffer": self.leader_range_buffer,
                "battery_optimal_min": self.battery_optimal_min,
                "battery_optimal_max": self.battery_optimal_max,
                "battery_low_threshold": self.battery_low_threshold
            },
            "episode": {
                "max_steps": self.max_steps
            },
            "action_space": {
                "weight_min": self.weight_min,
                "weight_max": self.weight_max
            }
        }

    def __str__(self):
        return f"CrazyflieRewardConfig(max_steps={self.max_steps})"

    def __repr__(self):
        return self.__str__()
