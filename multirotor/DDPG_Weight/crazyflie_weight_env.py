"""
Crazyflie实体无人机训练环境（在线/离线）
"""
import csv
import json
import math
import os
import sys
from typing import List, Optional

import gym
import numpy as np
from gym import spaces

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from crazyflie_reward_config import CrazyflieRewardConfig
from multirotor.Algorithm.Vector3 import Vector3
from multirotor.Crazyswarm.crazyflie_logging_data import CrazyflieLoggingData


def _safe_norm3(x: float, y: float, z: float) -> float:
    return math.sqrt(x * x + y * y + z * z)


def _normalize_direction(x: float, y: float, z: float, min_speed: float = 0.05) -> List[float]:
    speed = _safe_norm3(x, y, z)
    if speed < min_speed:
        return [1.0, 0.0, 0.0]
    return [x / speed, y / speed, z / speed]


def _get_cell_center(cell) -> Vector3:
    if cell is None:
        return Vector3()
    if isinstance(cell, Vector3):
        return cell
    center = getattr(cell, "center", None)
    if center is None and isinstance(cell, dict):
        center = cell.get("center")
    if isinstance(center, Vector3):
        return center
    if isinstance(center, dict) or hasattr(center, "x"):
        return Vector3.from_dict(center)
    if isinstance(center, (list, tuple)) and len(center) >= 3:
        try:
            return Vector3(float(center[0]), float(center[1]), float(center[2]))
        except (TypeError, ValueError):
            return Vector3()
    return Vector3()


def _get_cell_entropy(cell, default: float = 0.0) -> float:
    if cell is None:
        return default
    if isinstance(cell, dict):
        try:
            return float(cell.get("entropy", default))
        except (TypeError, ValueError):
            return default
    if hasattr(cell, "entropy"):
        try:
            return float(cell.entropy)
        except (TypeError, ValueError):
            return default
    return default


def _load_crazyflie_logs(log_path: str) -> List[CrazyflieLoggingData]:
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"日志文件不存在: {log_path}")

    if log_path.lower().endswith(".json"):
        with open(log_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        def _convert_item(item: dict) -> CrazyflieLoggingData:
            normalized = {k.strip().lower(): v for k, v in item.items()}
            converted = {
                "Id": int(normalized.get("id", 0)),
                "X": float(normalized.get("x", 0)),
                "Y": float(normalized.get("y", 0)),
                "Z": float(normalized.get("z", 0)),
                "Time": float(normalized.get("time", 0)),
                "Qx": float(normalized.get("qx", 0)),
                "Qy": float(normalized.get("qy", 0)),
                "Qz": float(normalized.get("qz", 0)),
                "Qw": float(normalized.get("qw", 1)),
                "Speed": float(normalized.get("speed", 0)),
                "XSpeed": float(normalized.get("xspeed", normalized.get("x_speed", 0))),
                "YSpeed": float(normalized.get("yspeed", normalized.get("y_speed", 0))),
                "ZSpeed": float(normalized.get("zspeed", normalized.get("z_speed", 0))),
                "AcceleratedSpeed": float(normalized.get("acceleratedspeed", normalized.get("accelerated_speed", 0))),
                "XAcceleratedSpeed": float(normalized.get("xacceleratedspeed", normalized.get("x_accelerated_speed", 0))),
                "YAcceleratedSpeed": float(normalized.get("yacceleratedspeed", normalized.get("y_accelerated_speed", 0))),
                "ZAcceleratedSpeed": float(normalized.get("zacceleratedspeed", normalized.get("z_accelerated_speed", 0))),
                "XEulerAngle": float(normalized.get("xeulerangle", normalized.get("x_euler_angle", 0))),
                "YEulerAngle": float(normalized.get("yeulerangle", normalized.get("y_euler_angle", 0))),
                "ZEulerAngle": float(normalized.get("zeulerangle", normalized.get("z_euler_angle", 0))),
                "XPalstance": float(normalized.get("xpalstance", normalized.get("x_palstance", 0))),
                "YPalstance": float(normalized.get("ypalstance", normalized.get("y_palstance", 0))),
                "ZPalstance": float(normalized.get("zpalstance", normalized.get("z_palstance", 0))),
                "XAccfPalstance": float(normalized.get("xaccfpalstance", normalized.get("x_accf_palstance", 0))),
                "YAccfPalstance": float(normalized.get("yaccfpalstance", normalized.get("y_accf_palstance", 0))),
                "ZAccfPalstance": float(normalized.get("zaccfpalstance", normalized.get("z_accf_palstance", 0))),
                "Battery": float(normalized.get("battery", 0))
            }
            return CrazyflieLoggingData.from_dict(converted)

        if isinstance(raw, list):
            return [_convert_item(item) for item in raw if isinstance(item, dict)]
        if isinstance(raw, dict):
            return [_convert_item(raw)]
        raise ValueError("JSON格式不支持，需为对象或数组")

    if log_path.lower().endswith(".csv"):
        logs: List[CrazyflieLoggingData] = []
        with open(log_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                normalized = {k.strip().lower(): v for k, v in row.items()}
                converted = {
                    "Id": int(normalized.get("id", 0)),
                    "X": float(normalized.get("x", 0)),
                    "Y": float(normalized.get("y", 0)),
                    "Z": float(normalized.get("z", 0)),
                    "Time": float(normalized.get("time", 0)),
                    "Qx": float(normalized.get("qx", 0)),
                    "Qy": float(normalized.get("qy", 0)),
                    "Qz": float(normalized.get("qz", 0)),
                    "Qw": float(normalized.get("qw", 1)),
                    "Speed": float(normalized.get("speed", 0)),
                    "XSpeed": float(normalized.get("xspeed", 0)),
                    "YSpeed": float(normalized.get("yspeed", 0)),
                    "ZSpeed": float(normalized.get("zspeed", 0)),
                    "AcceleratedSpeed": float(normalized.get("acceleratedspeed", 0)),
                    "XAcceleratedSpeed": float(normalized.get("xacceleratedspeed", 0)),
                    "YAcceleratedSpeed": float(normalized.get("yacceleratedspeed", 0)),
                    "ZAcceleratedSpeed": float(normalized.get("zacceleratedspeed", 0)),
                    "XEulerAngle": float(normalized.get("xeulerangle", 0)),
                    "YEulerAngle": float(normalized.get("yeulerangle", 0)),
                    "ZEulerAngle": float(normalized.get("zeulerangle", 0)),
                    "XPalstance": float(normalized.get("xpalstance", 0)),
                    "YPalstance": float(normalized.get("ypalstance", 0)),
                    "ZPalstance": float(normalized.get("zpalstance", 0)),
                    "XAccfPalstance": float(normalized.get("xaccfpalstance", 0)),
                    "YAccfPalstance": float(normalized.get("yaccfpalstance", 0)),
                    "ZAccfPalstance": float(normalized.get("zaccfpalstance", 0)),
                    "Battery": float(normalized.get("battery", 0))
                }
                logs.append(CrazyflieLoggingData.from_dict(converted))
        return logs

    raise ValueError("仅支持.json或.csv日志")


class CrazyflieLogEnv(gym.Env):
    """离线日志训练环境（动作不影响状态转移）"""

    def __init__(
        self,
        log_path: str,
        reward_config_path: Optional[str] = None,
        max_steps: Optional[int] = None,
        random_start: bool = False,
        step_stride: int = 1
    ):
        super().__init__()
        self.logs = _load_crazyflie_logs(log_path)
        if len(self.logs) < 2:
            raise ValueError("日志数据太短，至少需要2条记录")

        self.config = CrazyflieRewardConfig(reward_config_path)
        self.max_steps = max_steps or self.config.max_steps
        self.random_start = random_start
        self.step_stride = max(1, step_stride)

        self.observation_space = spaces.Box(
            low=-100.0,
            high=100.0,
            shape=(18,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=self.config.weight_min,
            high=self.config.weight_max,
            shape=(5,),
            dtype=np.float32
        )

        self.index = 0
        self.step_count = 0
        self.last_action = np.zeros(5, dtype=np.float32)

    def reset(self):
        if self.random_start:
            self.index = np.random.randint(0, len(self.logs) - 1)
        else:
            self.index = 0
        self.step_count = 0
        self.last_action = np.zeros(5, dtype=np.float32)
        return self._build_state(self.logs[self.index])

    def step(self, action):
        action = np.clip(action, self.config.weight_min, self.config.weight_max)
        reward = self._calculate_reward(self.logs[self.index], action)

        self.last_action = action.copy()
        self.step_count += 1
        self.index += self.step_stride

        done = self.index >= len(self.logs) - 1 or self.step_count >= self.max_steps
        next_state = self._build_state(self.logs[min(self.index, len(self.logs) - 1)])

        info = {"index": self.index}
        return next_state, reward, done, info

    def _build_state(self, log: CrazyflieLoggingData) -> np.ndarray:
        position = [log.X, log.Y, log.Z]
        velocity = [log.XSpeed, log.YSpeed, log.ZSpeed]
        direction = _normalize_direction(log.XSpeed, log.YSpeed, log.ZSpeed)
        entropy_info = [0.0, 0.0, 0.0]
        leader_rel = [0.0, 0.0, 0.0]
        scan_info = [0.0, 0.0, 0.0]
        state = position + velocity + direction + entropy_info + leader_rel + scan_info
        return np.array(state, dtype=np.float32)

    def _calculate_reward(self, log: CrazyflieLoggingData, action: np.ndarray) -> float:
        reward = 0.0
        speed = max(log.Speed, _safe_norm3(log.XSpeed, log.YSpeed, log.ZSpeed))
        reward += self.config.speed_reward * speed

        if speed > self.config.speed_penalty_threshold:
            reward -= self.config.speed_penalty

        accel_mag = max(log.AcceleratedSpeed, _safe_norm3(log.XAcceleratedSpeed, log.YAcceleratedSpeed, log.ZAcceleratedSpeed))
        reward -= self.config.accel_penalty * accel_mag

        ang_rate = _safe_norm3(log.XPalstance, log.YPalstance, log.ZPalstance)
        reward -= self.config.angular_rate_penalty * ang_rate

        if self.config.battery_optimal_min <= log.Battery <= self.config.battery_optimal_max:
            reward += self.config.battery_optimal_reward
        elif log.Battery < self.config.battery_low_threshold:
            reward -= self.config.battery_low_penalty

        action_delta = np.linalg.norm(action - self.last_action)
        reward -= self.config.action_change_penalty * action_delta
        reward -= self.config.action_magnitude_penalty * np.linalg.norm(action)

        return reward


class CrazyflieOnlineWeightEnv(gym.Env):
    """在线实体无人机训练环境（使用实时日志数据）"""

    def __init__(
        self,
        server,
        drone_name: str = "UAV1",
        reward_config_path: Optional[str] = None,
        step_duration: float = 5.0,
        reset_unity: bool = False,
        safety_limit: bool = True,
        max_weight_delta: float = 0.5
    ):
        super().__init__()
        self.server = server
        self.drone_name = drone_name
        self.step_duration = step_duration
        self.reset_unity = reset_unity
        self.safety_limit = safety_limit
        self.max_weight_delta = max_weight_delta

        self.config = CrazyflieRewardConfig(reward_config_path)

        self.observation_space = spaces.Box(
            low=-100.0,
            high=100.0,
            shape=(18,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=self.config.weight_min,
            high=self.config.weight_max,
            shape=(5,),
            dtype=np.float32
        )

        self.step_count = 0
        self.prev_scanned_cells = 0
        self.last_action = np.zeros(5, dtype=np.float32)

    def reset(self):
        self.step_count = 0
        self.last_action = np.zeros(5, dtype=np.float32)

        if self.reset_unity and self.server:
            self.server.reset_environment()

        if self.server and self.server.grid_data and self.server.grid_data.cells:
            self.prev_scanned_cells = self._count_scanned_cells()
        else:
            self.prev_scanned_cells = 0

        return self._get_state()

    def step(self, action):
        action = np.clip(action, self.config.weight_min, self.config.weight_max)
        if self.safety_limit and self.step_count > 0:
            action = np.clip(
                action,
                self.last_action - self.max_weight_delta,
                self.last_action + self.max_weight_delta
            )
            action = np.clip(action, self.config.weight_min, self.config.weight_max)

        weights = {
            "repulsionCoefficient": float(action[0]),
            "entropyCoefficient": float(action[1]),
            "distanceCoefficient": float(action[2]),
            "leaderRangeCoefficient": float(action[3]),
            "directionRetentionCoefficient": float(action[4])
        }

        if self.server:
            self.server.algorithms[self.drone_name].set_coefficients(weights)

        self.last_action = action.copy()
        self.step_count += 1

        if self.step_duration > 0:
            import time
            time.sleep(self.step_duration)

        next_state = self._get_state()
        reward = self._calculate_reward(action)
        done = self.step_count >= self.config.max_steps

        info = {"weights": weights}
        return next_state, reward, done, info

    def _get_state(self) -> np.ndarray:
        if not self.server:
            return np.zeros(18, dtype=np.float32)

        with self.server.data_lock:
            logging_data = self.server.crazyswarm.get_loggingData_by_droneName(self.drone_name)
            runtime_data = self.server.unity_runtime_data.get(self.drone_name)
            grid_data = self.server.grid_data

        if logging_data is None:
            return np.zeros(18, dtype=np.float32)

        pos = Vector3(logging_data.X, logging_data.Y, logging_data.Z)
        position = [pos.x, pos.y, pos.z]
        velocity = [logging_data.XSpeed, logging_data.YSpeed, logging_data.ZSpeed]
        direction = _normalize_direction(logging_data.XSpeed, logging_data.YSpeed, logging_data.ZSpeed)

        entropy_info = [0.0, 0.0, 0.0]
        if grid_data and getattr(grid_data, "cells", None):
            nearby_cells = [
                c for c in grid_data.cells[:50]
                if (_get_cell_center(c) - pos).magnitude() < 10.0
            ]
            if nearby_cells:
                entropies = [_get_cell_entropy(c) for c in nearby_cells]
                entropy_info = [
                    float(np.mean(entropies)),
                    float(np.max(entropies)),
                    float(np.std(entropies))
                ]

        leader_rel = [0.0, 0.0, 0.0]
        if runtime_data and runtime_data.leader_position:
            leader_rel = [
                runtime_data.leader_position.x - pos.x,
                runtime_data.leader_position.y - pos.y,
                runtime_data.leader_position.z - pos.z
            ]

        scan_info = [0.0, 0.0, 0.0]
        if grid_data and getattr(grid_data, "cells", None):
            total = len(grid_data.cells)
            scanned = sum(
                1 for c in grid_data.cells
                if _get_cell_entropy(c) < self.config.scan_entropy_threshold
            )
            scan_info = [scanned / max(total, 1), float(scanned), float(total - scanned)]

        state = position + velocity + direction + entropy_info + leader_rel + scan_info
        return np.array(state, dtype=np.float32)

    def _calculate_reward(self, action: np.ndarray) -> float:
        reward = 0.0
        if not self.server:
            return reward

        with self.server.data_lock:
            logging_data = self.server.crazyswarm.get_loggingData_by_droneName(self.drone_name)
            runtime_data = self.server.unity_runtime_data.get(self.drone_name)
            grid_data = self.server.grid_data

        if logging_data is None:
            return reward

        speed = max(logging_data.Speed, _safe_norm3(logging_data.XSpeed, logging_data.YSpeed, logging_data.ZSpeed))
        reward += self.config.speed_reward * speed
        if speed > self.config.speed_penalty_threshold:
            reward -= self.config.speed_penalty

        accel_mag = max(logging_data.AcceleratedSpeed, _safe_norm3(
            logging_data.XAcceleratedSpeed,
            logging_data.YAcceleratedSpeed,
            logging_data.ZAcceleratedSpeed
        ))
        reward -= self.config.accel_penalty * accel_mag

        ang_rate = _safe_norm3(logging_data.XPalstance, logging_data.YPalstance, logging_data.ZPalstance)
        reward -= self.config.angular_rate_penalty * ang_rate

        if self.config.battery_optimal_min <= logging_data.Battery <= self.config.battery_optimal_max:
            reward += self.config.battery_optimal_reward
        elif logging_data.Battery < self.config.battery_low_threshold:
            reward -= self.config.battery_low_penalty

        if grid_data and getattr(grid_data, "cells", None):
            current_scanned = sum(
                1 for c in grid_data.cells
                if _get_cell_entropy(c) < self.config.scan_entropy_threshold
            )
            new_scanned = current_scanned - self.prev_scanned_cells
            if new_scanned > 0:
                reward += self.config.scan_reward * new_scanned
            self.prev_scanned_cells = current_scanned

        if runtime_data and runtime_data.leader_position:
            dist_to_leader = (runtime_data.position - runtime_data.leader_position).magnitude()
            leader_radius = runtime_data.leader_scan_radius + self.config.leader_range_buffer
            if runtime_data.leader_scan_radius > 0 and dist_to_leader > leader_radius:
                reward -= self.config.out_of_range_penalty

        action_delta = np.linalg.norm(action - self.last_action)
        reward -= self.config.action_change_penalty * action_delta
        reward -= self.config.action_magnitude_penalty * np.linalg.norm(action)

        return reward

    def _count_scanned_cells(self) -> int:
        if not self.server or not self.server.grid_data:
            return 0
        return sum(
            1 for cell in self.server.grid_data.cells
            if _get_cell_entropy(cell) < self.config.scan_entropy_threshold
        )
