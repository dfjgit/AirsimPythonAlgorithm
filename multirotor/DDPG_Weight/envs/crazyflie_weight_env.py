"""
Crazyflie实体无人机训练环境（在线/离线）

本模块提供了两种训练环境：
1. CrazyflieLogEnv: 离线日志训练环境，使用历史日志数据进行训练，动作不影响状态转移
2. CrazyflieOnlineWeightEnv: 在线实体无人机训练环境，使用实时日志数据，动作会影响实际飞行状态

主要用于DDPG算法的权重参数训练，通过调整5个权重系数来优化无人机的飞行行为。
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

# 获取项目根目录并添加到系统路径中，以便导入项目内的其他模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.crazyflie_reward_config import CrazyflieRewardConfig
from multirotor.Algorithm.Vector3 import Vector3
from multirotor.Crazyswarm.crazyflie_logging_data import CrazyflieLoggingData


def _safe_norm3(x: float, y: float, z: float) -> float:
    """
    计算三维向量的欧几里得范数（模长）
    
    参数:
        x: X方向的分量
        y: Y方向的分量
        z: Z方向的分量
    
    返回:
        三维向量的模长（标量）
    """
    return math.sqrt(x * x + y * y + z * z)


def _normalize_direction(x: float, y: float, z: float, min_speed: float = 0.05) -> List[float]:
    """
    将三维向量归一化为单位方向向量
    
    如果向量模长小于最小速度阈值，则返回默认方向向量 [1, 0, 0]（X轴正方向）
    
    参数:
        x: X方向的分量
        y: Y方向的分量
        z: Z方向的分量
        min_speed: 最小速度阈值，低于此值则返回默认方向（默认0.05）
    
    返回:
        归一化后的单位方向向量 [x_norm, y_norm, z_norm]
    """
    speed = _safe_norm3(x, y, z)
    if speed < min_speed:
        return [1.0, 0.0, 0.0]
    return [x / speed, y / speed, z / speed]


def _get_cell_center(cell) -> Vector3:
    """
    从网格单元对象中提取中心点坐标
    
    支持多种数据格式：
    - Vector3对象
    - 字典格式（包含center键）
    - 对象属性（包含center属性）
    - 列表/元组格式 [x, y, z]
    
    参数:
        cell: 网格单元对象，可以是多种格式
    
    返回:
        网格单元的中心点坐标（Vector3对象），如果提取失败则返回零向量
    """
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
    """
    从网格单元对象中提取熵值
    
    熵值用于表示网格单元的信息不确定性，值越小表示该区域已被充分扫描
    
    参数:
        cell: 网格单元对象，可以是字典或包含entropy属性的对象
        default: 默认熵值，当无法提取时返回此值（默认0.0）
    
    返回:
        网格单元的熵值（浮点数）
    """
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
    """
    从文件加载Crazyflie无人机的日志数据
    
    支持JSON和CSV两种格式的日志文件。日志数据包含无人机的位置、速度、加速度、
    姿态、角速度、电池等信息。
    
    参数:
        log_path: 日志文件路径，支持.json或.csv格式
    
    返回:
        CrazyflieLoggingData对象列表，包含所有日志记录
    
    异常:
        FileNotFoundError: 当日志文件不存在时抛出
        ValueError: 当文件格式不支持或数据格式错误时抛出
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"日志文件不存在: {log_path}")

    # 处理JSON格式的日志文件
    if log_path.lower().endswith(".json"):
        with open(log_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        def _convert_item(item: dict) -> CrazyflieLoggingData:
            """
            将字典格式的日志项转换为CrazyflieLoggingData对象
            
            支持多种字段命名格式（驼峰、下划线等），自动归一化处理
            """
            # 将键名转换为小写并去除空格，以支持不同的命名格式
            normalized = {k.strip().lower(): v for k, v in item.items()}
            converted = {
                "Id": int(normalized.get("id", 0)),
                "X": float(normalized.get("x", 0)),
                "Y": float(normalized.get("y", 0)),
                "Z": float(normalized.get("z", 0)),
                "Time": float(normalized.get("time", 0)),
                "Qx": float(normalized.get("qx", 0)),  # 四元数X分量
                "Qy": float(normalized.get("qy", 0)),  # 四元数Y分量
                "Qz": float(normalized.get("qz", 0)),  # 四元数Z分量
                "Qw": float(normalized.get("qw", 1)),  # 四元数W分量
                "Speed": float(normalized.get("speed", 0)),  # 总速度
                "XSpeed": float(normalized.get("xspeed", normalized.get("x_speed", 0))),  # X方向速度
                "YSpeed": float(normalized.get("yspeed", normalized.get("y_speed", 0))),  # Y方向速度
                "ZSpeed": float(normalized.get("zspeed", normalized.get("z_speed", 0))),  # Z方向速度
                "AcceleratedSpeed": float(normalized.get("acceleratedspeed", normalized.get("accelerated_speed", 0))),  # 总加速度
                "XAcceleratedSpeed": float(normalized.get("xacceleratedspeed", normalized.get("x_accelerated_speed", 0))),  # X方向加速度
                "YAcceleratedSpeed": float(normalized.get("yacceleratedspeed", normalized.get("y_accelerated_speed", 0))),  # Y方向加速度
                "ZAcceleratedSpeed": float(normalized.get("zacceleratedspeed", normalized.get("z_accelerated_speed", 0))),  # Z方向加速度
                "XEulerAngle": float(normalized.get("xeulerangle", normalized.get("x_euler_angle", 0))),  # X轴欧拉角（滚转）
                "YEulerAngle": float(normalized.get("yeulerangle", normalized.get("y_euler_angle", 0))),  # Y轴欧拉角（俯仰）
                "ZEulerAngle": float(normalized.get("zeulerangle", normalized.get("z_euler_angle", 0))),  # Z轴欧拉角（偏航）
                "XPalstance": float(normalized.get("xpalstance", normalized.get("x_palstance", 0))),  # X轴角速度
                "YPalstance": float(normalized.get("ypalstance", normalized.get("y_palstance", 0))),  # Y轴角速度
                "ZPalstance": float(normalized.get("zpalstance", normalized.get("z_palstance", 0))),  # Z轴角速度
                "XAccfPalstance": float(normalized.get("xaccfpalstance", normalized.get("x_accf_palstance", 0))),  # X轴角加速度
                "YAccfPalstance": float(normalized.get("yaccfpalstance", normalized.get("y_accf_palstance", 0))),  # Y轴角加速度
                "ZAccfPalstance": float(normalized.get("zaccfpalstance", normalized.get("z_accf_palstance", 0))),  # Z轴角加速度
                "Battery": float(normalized.get("battery", 0))  # 电池电量
            }
            return CrazyflieLoggingData.from_dict(converted)

        # 处理JSON数组或单个对象
        if isinstance(raw, list):
            return [_convert_item(item) for item in raw if isinstance(item, dict)]
        if isinstance(raw, dict):
            return [_convert_item(raw)]
        raise ValueError("JSON格式不支持，需为对象或数组")

    # 处理CSV格式的日志文件
    if log_path.lower().endswith(".csv"):
        logs: List[CrazyflieLoggingData] = []
        # 使用utf-8-sig编码以处理BOM标记
        with open(log_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 归一化字段名（转小写、去空格）
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
    """
    离线日志训练环境（动作不影响状态转移）
    
    这是一个基于历史日志数据的训练环境，用于离线训练DDPG算法。
    环境从日志文件中读取无人机状态数据，智能体的动作（权重调整）不会影响
    状态转移，只用于计算奖励。这种设计允许快速迭代训练，无需实际飞行。
    
    观察空间（18维）：
        - 位置: [x, y, z] (3维)
        - 速度: [vx, vy, vz] (3维)
        - 方向: [dir_x, dir_y, dir_z] (归一化方向向量，3维)
        - 熵信息: [mean_entropy, max_entropy, std_entropy] (3维，离线环境为0)
        - 领机相对位置: [rel_x, rel_y, rel_z] (3维，离线环境为0)
        - 扫描信息: [scan_ratio, scanned_count, unscanned_count] (3维，离线环境为0)
    
    动作空间（5维）：
        - repulsionCoefficient: 排斥力系数
        - entropyCoefficient: 熵系数
        - distanceCoefficient: 距离系数
        - leaderRangeCoefficient: 领机范围系数
        - directionRetentionCoefficient: 方向保持系数
    """

    def __init__(
        self,
        log_path: str,
        reward_config_path: Optional[str] = None,
        max_steps: Optional[int] = None,
        random_start: bool = False,
        step_stride: int = 1
    ):
        """
        初始化离线日志训练环境
        
        参数:
            log_path: 日志文件路径（支持.json或.csv格式）
            reward_config_path: 奖励配置文件路径，如果为None则使用默认配置
            max_steps: 每个episode的最大步数，如果为None则使用配置文件中的值
            random_start: 是否随机选择起始位置，True则从日志中随机位置开始
            step_stride: 每次步进的跨度，1表示逐条读取，大于1表示跳跃读取
        """
        super().__init__()
        # 加载日志数据
        self.logs = _load_crazyflie_logs(log_path)
        if len(self.logs) < 2:
            raise ValueError("日志数据太短，至少需要2条记录")

        # 加载奖励配置
        self.config = CrazyflieRewardConfig(reward_config_path)
        self.max_steps = max_steps or self.config.max_steps
        self.random_start = random_start
        self.step_stride = max(1, step_stride)  # 确保步进跨度至少为1

        # 定义观察空间：18维向量，值范围[-100, 100]
        self.observation_space = spaces.Box(
            low=-100.0,
            high=100.0,
            shape=(18,),
            dtype=np.float32
        )
        # 定义动作空间：5维权重向量，值范围由配置文件定义
        self.action_space = spaces.Box(
            low=self.config.weight_min,
            high=self.config.weight_max,
            shape=(5,),
            dtype=np.float32
        )

        # 环境状态变量
        self.index = 0  # 当前日志索引
        self.step_count = 0  # 当前步数计数
        self.last_action = np.zeros(5, dtype=np.float32)  # 上一次的动作（权重）

    def reset(self):
        """
        重置环境到初始状态
        
        根据random_start参数决定从日志开头或随机位置开始。
        重置步数计数和上一次动作。
        
        返回:
            初始观察状态（18维numpy数组）
        """
        # 根据配置选择起始位置
        if self.random_start:
            self.index = np.random.randint(0, len(self.logs) - 1)
        else:
            self.index = 0
        # 重置计数器
        self.step_count = 0
        self.last_action = np.zeros(5, dtype=np.float32)
        # 返回初始状态
        return self._build_state(self.logs[self.index])

    def step(self, action):
        """
        执行一步动作
        
        在离线环境中，动作不会影响状态转移，只用于计算奖励。
        环境按照step_stride的跨度在日志中前进。
        
        参数:
            action: 5维权重向量，表示5个权重系数
        
        返回:
            next_state: 下一个观察状态（18维numpy数组）
            reward: 当前步的奖励值（浮点数）
            done: 是否结束episode（布尔值）
            info: 包含额外信息的字典（如当前索引）
        """
        # 将动作裁剪到有效范围内
        action = np.clip(action, self.config.weight_min, self.config.weight_max)
        # 计算奖励（基于当前日志数据和动作）
        reward = self._calculate_reward(self.logs[self.index], action)

        # 更新状态
        self.last_action = action.copy()
        self.step_count += 1
        self.index += self.step_stride  # 按步进跨度前进

        # 判断是否结束：到达日志末尾或超过最大步数
        done = self.index >= len(self.logs) - 1 or self.step_count >= self.max_steps
        # 获取下一个状态（确保索引不越界）
        next_state = self._build_state(self.logs[min(self.index, len(self.logs) - 1)])

        info = {"index": self.index}
        return next_state, reward, done, info

    def _build_state(self, log: CrazyflieLoggingData) -> np.ndarray:
        """
        从日志数据构建观察状态向量
        
        状态向量包含18个维度，分为6个部分：
        1. 位置（3维）
        2. 速度（3维）
        3. 方向（3维，归一化）
        4. 熵信息（3维，离线环境为0）
        5. 领机相对位置（3维，离线环境为0）
        6. 扫描信息（3维，离线环境为0）
        
        参数:
            log: CrazyflieLoggingData对象，包含无人机状态数据
        
        返回:
            18维观察状态向量（numpy数组）
        """
        # 位置信息（3维）
        position = [log.X, log.Y, log.Z]
        # 速度信息（3维）
        velocity = [log.XSpeed, log.YSpeed, log.ZSpeed]
        # 归一化方向向量（3维）
        direction = _normalize_direction(log.XSpeed, log.YSpeed, log.ZSpeed)
        # 熵信息（离线环境无网格数据，设为0）
        entropy_info = [0.0, 0.0, 0.0]
        # 领机相对位置（离线环境无领机信息，设为0）
        leader_rel = [0.0, 0.0, 0.0]
        # 扫描信息（离线环境无扫描数据，设为0）
        scan_info = [0.0, 0.0, 0.0]
        # 拼接所有状态分量
        state = position + velocity + direction + entropy_info + leader_rel + scan_info
        return np.array(state, dtype=np.float32)

    def _calculate_reward(self, log: CrazyflieLoggingData, action: np.ndarray) -> float:
        """
        计算当前步的奖励值
        
        奖励函数综合考虑多个因素：
        1. 速度奖励：鼓励适当的飞行速度
        2. 速度惩罚：惩罚过高的速度
        3. 加速度惩罚：惩罚过大的加速度（影响稳定性）
        4. 角速度惩罚：惩罚过大的角速度（影响稳定性）
        5. 电池奖励/惩罚：鼓励电池在最优范围内，惩罚电量过低
        6. 动作变化惩罚：惩罚权重变化过大（鼓励平滑调整）
        7. 动作幅度惩罚：惩罚权重绝对值过大（鼓励适度调整）
        
        参数:
            log: 当前日志数据，包含无人机状态
            action: 当前动作（5维权重向量）
        
        返回:
            奖励值（浮点数），可能为正数（奖励）或负数（惩罚）
        """
        reward = 0.0
        
        # 计算速度（取总速度和分量速度的最大值）
        speed = max(log.Speed, _safe_norm3(log.XSpeed, log.YSpeed, log.ZSpeed))
        # 速度奖励：鼓励保持适当速度
        reward += self.config.speed_reward * speed

        # 速度惩罚：如果速度超过阈值，给予惩罚
        if speed > self.config.speed_penalty_threshold:
            reward -= self.config.speed_penalty

        # 计算加速度大小
        accel_mag = max(log.AcceleratedSpeed, _safe_norm3(log.XAcceleratedSpeed, log.YAcceleratedSpeed, log.ZAcceleratedSpeed))
        # 加速度惩罚：过大的加速度影响飞行稳定性
        reward -= self.config.accel_penalty * accel_mag

        # 计算角速度大小
        ang_rate = _safe_norm3(log.XPalstance, log.YPalstance, log.ZPalstance)
        # 角速度惩罚：过大的角速度影响飞行稳定性
        reward -= self.config.angular_rate_penalty * ang_rate

        # 电池状态奖励/惩罚
        if self.config.battery_optimal_min <= log.Battery <= self.config.battery_optimal_max:
            # 电池在最优范围内，给予奖励
            reward += self.config.battery_optimal_reward
        elif log.Battery < self.config.battery_low_threshold:
            # 电池电量过低，给予惩罚
            reward -= self.config.battery_low_penalty

        # 动作变化惩罚：计算当前动作与上次动作的差异
        action_delta = np.linalg.norm(action - self.last_action)
        reward -= self.config.action_change_penalty * action_delta
        # 动作幅度惩罚：惩罚权重绝对值过大
        reward -= self.config.action_magnitude_penalty * np.linalg.norm(action)

        return reward


class CrazyflieOnlineWeightEnv(gym.Env):
    """
    在线实体无人机训练环境（使用实时日志数据）
    
    这是一个基于实时数据的训练环境，用于在线训练DDPG算法。
    环境从Unity仿真服务器获取实时无人机状态，智能体的动作（权重调整）
    会直接影响无人机的飞行行为。这种设计允许在实际飞行中优化权重参数。
    
    观察空间（18维）：
        - 位置: [x, y, z] (3维)
        - 速度: [vx, vy, vz] (3维)
        - 方向: [dir_x, dir_y, dir_z] (归一化方向向量，3维)
        - 熵信息: [mean_entropy, max_entropy, std_entropy] (3维，来自附近网格单元)
        - 领机相对位置: [rel_x, rel_y, rel_z] (3维，相对于领机的位置)
        - 扫描信息: [scan_ratio, scanned_count, unscanned_count] (3维，扫描进度)
    
    动作空间（5维）：
        - repulsionCoefficient: 排斥力系数
        - entropyCoefficient: 熵系数
        - distanceCoefficient: 距离系数
        - leaderRangeCoefficient: 领机范围系数
        - directionRetentionCoefficient: 方向保持系数
    
    安全特性：
        - 支持权重变化限制（max_weight_delta），防止权重突变导致飞行不稳定
        - 支持Unity环境重置
    """

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
        """
        初始化在线实体无人机训练环境
        
        参数:
            server: Unity仿真服务器对象，用于获取实时数据和设置权重
            drone_name: 无人机名称，用于标识要控制的无人机（默认"UAV1"）
            reward_config_path: 奖励配置文件路径，如果为None则使用默认配置
            step_duration: 每步的持续时间（秒），用于控制训练节奏（默认5.0秒）
            reset_unity: 是否在reset时重置Unity环境（默认False）
            safety_limit: 是否启用安全限制，限制权重变化幅度（默认True）
            max_weight_delta: 单步权重变化的最大允许值，用于安全限制（默认0.5）
        """
        super().__init__()
        self.server = server  # Unity服务器对象
        self.drone_name = drone_name  # 无人机名称
        self.step_duration = step_duration  # 每步持续时间
        self.reset_unity = reset_unity  # 是否重置Unity环境
        self.safety_limit = safety_limit  # 是否启用安全限制
        self.max_weight_delta = max_weight_delta  # 最大权重变化量

        # 加载奖励配置
        self.config = CrazyflieRewardConfig(reward_config_path)

        # 定义观察空间：18维向量，值范围[-100, 100]
        self.observation_space = spaces.Box(
            low=-100.0,
            high=100.0,
            shape=(18,),
            dtype=np.float32
        )
        # 定义动作空间：5维权重向量，值范围由配置文件定义
        self.action_space = spaces.Box(
            low=self.config.weight_min,
            high=self.config.weight_max,
            shape=(5,),
            dtype=np.float32
        )

        # 环境状态变量
        self.step_count = 0  # 当前步数计数
        self.episode_count = 0  # Episode 计数
        self.total_episode_reward = 0.0  # 当前 Episode 总奖励
        self.prev_scanned_cells = 0  # 上一次扫描的网格单元数量
        self.last_action = np.zeros(5, dtype=np.float32)  # 上一次的动作（权重）
        self._has_initial_action = False  # 是否已设置初始动作（用于安全限制）
        
    def reset(self):
        """
        重置环境到初始状态
        
        重置步数计数、动作历史，可选择重置Unity环境。
        更新扫描单元计数，用于计算扫描奖励。
        
        返回:
            初始观察状态（18维numpy数组）
        """
        # Episode 计数
        self.episode_count += 1
        
        # 重置计数器
        self.step_count = 0
        self.total_episode_reward = 0.0
        self.last_action = np.zeros(5, dtype=np.float32)
        self._has_initial_action = False
                
        # 如果配置了重置Unity环境，则执行重置
        if self.reset_unity and self.server:
            print(f"[DDPG环境] 🔄 Episode结束，执行环境重置...")
            self.server.reset_environment()
                
        # 更新已扫描网格单元数量（用于计算扫描奖励）
        if self.server and self.server.grid_data and self.server.grid_data.cells:
            self.prev_scanned_cells = self._count_scanned_cells()
        else:
            self.prev_scanned_cells = 0

        # 返回当前状态
        return self._get_state()

    def step(self, action):
        """
        执行一步动作
        
        在在线环境中，动作会直接影响无人机的飞行行为。
        首先应用安全限制（如果启用），然后将权重设置到Unity服务器，
        等待指定时间后获取新状态并计算奖励。
        
        参数:
            action: 5维权重向量，表示5个权重系数
        
        返回:
            next_state: 下一个观察状态（18维numpy数组）
            reward: 当前步的奖励值（浮点数）
            done: 是否结束episode（布尔值）
            info: 包含额外信息的字典（如当前权重）
        """
        # 将动作裁剪到有效范围内
        action = np.clip(action, self.config.weight_min, self.config.weight_max)
        
        # 应用安全限制：限制权重变化幅度，防止突变导致飞行不稳定
        if self.safety_limit and (self.step_count > 0 or self._has_initial_action):
            # 限制动作变化在[-max_weight_delta, +max_weight_delta]范围内
            action = np.clip(
                action,
                self.last_action - self.max_weight_delta,
                self.last_action + self.max_weight_delta
            )
            # 再次裁剪到有效范围（防止安全限制导致越界）
            action = np.clip(action, self.config.weight_min, self.config.weight_max)
        self._has_initial_action = False

        # 将动作向量转换为权重字典
        weights = {
            "repulsionCoefficient": float(action[0]),  # 排斥力系数
            "entropyCoefficient": float(action[1]),  # 熵系数
            "distanceCoefficient": float(action[2]),  # 距离系数
            "leaderRangeCoefficient": float(action[3]),  # 领机范围系数
            "directionRetentionCoefficient": float(action[4])  # 方向保持系数
        }

        # 将权重设置到Unity服务器的算法中，影响无人机行为
        if self.server:
            self.server.algorithms[self.drone_name].set_coefficients(weights)

        # 更新状态
        self.last_action = action.copy()
        self.step_count += 1

        # 等待指定时间，让无人机执行动作并产生状态变化
        if self.step_duration > 0:
            import time
            time.sleep(self.step_duration)

        # 获取新状态并计算奖励
        next_state = self._get_state()
        reward = self._calculate_reward(action)
        self.total_episode_reward += reward
        
        # 将训练统计信息传递给服务器（用于数据采集到 CSV）
        if self.server:
            self.server.set_training_stats(
                episode=self.episode_count,
                step=self.step_count,
                reward=float(reward),
                total_reward=float(self.total_episode_reward)
            )

        # 判断是否结束：达到最大步数
        done = self.step_count >= self.config.max_steps

        info = {"weights": weights}
        return next_state, reward, done, info

    def _get_state(self) -> np.ndarray:
        """
        从Unity服务器获取当前观察状态
        
        状态向量包含18个维度，分为6个部分：
        1. 位置（3维）：无人机当前位置
        2. 速度（3维）：无人机当前速度
        3. 方向（3维）：归一化的速度方向向量
        4. 熵信息（3维）：附近网格单元熵值的均值、最大值、标准差
        5. 领机相对位置（3维）：相对于领机的位置
        6. 扫描信息（3维）：扫描比例、已扫描数量、未扫描数量
        
        返回:
            18维观察状态向量（numpy数组），如果无法获取数据则返回零向量
        """
        # 如果服务器不存在，返回零向量
        if not self.server:
            return np.zeros(18, dtype=np.float32)

        # 使用锁保护数据访问，防止并发问题
        with self.server.data_lock:
            # 获取无人机的日志数据（位置、速度等）
            logging_data = self.server.crazyswarm.get_loggingData_by_droneName(self.drone_name)
            # 获取运行时数据（领机位置等）
            runtime_data = self.server.unity_runtime_data.get(self.drone_name)
            # 获取网格数据（用于计算熵和扫描信息）
            grid_data = self.server.grid_data

        # 如果日志数据不存在，返回零向量
        if logging_data is None:
            return np.zeros(18, dtype=np.float32)

        # 构建位置信息（3维）
        pos = Vector3(logging_data.X, logging_data.Y, logging_data.Z)
        position = [pos.x, pos.y, pos.z]
        # 构建速度信息（3维）
        velocity = [logging_data.XSpeed, logging_data.YSpeed, logging_data.ZSpeed]
        # 构建归一化方向向量（3维）
        direction = _normalize_direction(logging_data.XSpeed, logging_data.YSpeed, logging_data.ZSpeed)

        # 计算熵信息（3维）：附近网格单元的熵值统计
        entropy_info = [0.0, 0.0, 0.0]
        if grid_data and getattr(grid_data, "cells", None):
            # 查找附近10米内的网格单元（最多检查前50个）
            nearby_cells = [
                c for c in grid_data.cells[:50]
                if (_get_cell_center(c) - pos).magnitude() < 10.0
            ]
            if nearby_cells:
                # 提取所有附近单元的熵值
                entropies = [_get_cell_entropy(c) for c in nearby_cells]
                # 计算熵值的均值、最大值、标准差
                entropy_info = [
                    float(np.mean(entropies)),  # 平均熵值
                    float(np.max(entropies)),  # 最大熵值
                    float(np.std(entropies))  # 熵值标准差
                ]

        # 计算领机相对位置（3维）
        leader_rel = [0.0, 0.0, 0.0]
        if runtime_data and runtime_data.leader_position:
            leader_rel = [
                runtime_data.leader_position.x - pos.x,  # X方向相对位置
                runtime_data.leader_position.y - pos.y,  # Y方向相对位置
                runtime_data.leader_position.z - pos.z  # Z方向相对位置
            ]

        # 计算扫描信息（3维）
        scan_info = [0.0, 0.0, 0.0]
        if grid_data and getattr(grid_data, "cells", None):
            total = len(grid_data.cells)  # 总网格单元数
            # 统计已扫描的单元数（熵值低于阈值的单元）
            scanned = sum(
                1 for c in grid_data.cells
                if _get_cell_entropy(c) < self.config.scan_entropy_threshold
            )
            # 扫描比例、已扫描数量、未扫描数量
            scan_info = [
                scanned / max(total, 1),  # 扫描比例
                float(scanned),  # 已扫描数量
                float(total - scanned)  # 未扫描数量
            ]

        # 拼接所有状态分量
        state = position + velocity + direction + entropy_info + leader_rel + scan_info
        return np.array(state, dtype=np.float32)

    def _calculate_reward(self, action: np.ndarray) -> float:
        """
        计算当前步的奖励值
        
        奖励函数综合考虑多个因素：
        1. 速度奖励：鼓励适当的飞行速度
        2. 速度惩罚：惩罚过高的速度
        3. 加速度惩罚：惩罚过大的加速度（影响稳定性）
        4. 角速度惩罚：惩罚过大的角速度（影响稳定性）
        5. 电池奖励/惩罚：鼓励电池在最优范围内，惩罚电量过低
        6. 扫描奖励：奖励新扫描的网格单元（在线环境特有）
        7. 范围惩罚：惩罚超出领机扫描范围（在线环境特有）
        8. 动作变化惩罚：惩罚权重变化过大（鼓励平滑调整）
        9. 动作幅度惩罚：惩罚权重绝对值过大（鼓励适度调整）
        
        参数:
            action: 当前动作（5维权重向量）
        
        返回:
            奖励值（浮点数），可能为正数（奖励）或负数（惩罚）
        """
        reward = 0.0
        # 如果服务器不存在，返回零奖励
        if not self.server:
            return reward

        # 使用锁保护数据访问
        with self.server.data_lock:
            logging_data = self.server.crazyswarm.get_loggingData_by_droneName(self.drone_name)
            runtime_data = self.server.unity_runtime_data.get(self.drone_name)
            grid_data = self.server.grid_data

        # 如果日志数据不存在，返回零奖励
        if logging_data is None:
            return reward

        # 计算速度（取总速度和分量速度的最大值）
        speed = max(logging_data.Speed, _safe_norm3(logging_data.XSpeed, logging_data.YSpeed, logging_data.ZSpeed))
        # 速度奖励：鼓励保持适当速度
        reward += self.config.speed_reward * speed
        # 速度惩罚：如果速度超过阈值，给予惩罚
        if speed > self.config.speed_penalty_threshold:
            reward -= self.config.speed_penalty

        # 计算加速度大小
        accel_mag = max(logging_data.AcceleratedSpeed, _safe_norm3(
            logging_data.XAcceleratedSpeed,
            logging_data.YAcceleratedSpeed,
            logging_data.ZAcceleratedSpeed
        ))
        # 加速度惩罚：过大的加速度影响飞行稳定性
        reward -= self.config.accel_penalty * accel_mag

        # 计算角速度大小
        ang_rate = _safe_norm3(logging_data.XPalstance, logging_data.YPalstance, logging_data.ZPalstance)
        # 角速度惩罚：过大的角速度影响飞行稳定性
        reward -= self.config.angular_rate_penalty * ang_rate

        # 电池状态奖励/惩罚
        if self.config.battery_optimal_min <= logging_data.Battery <= self.config.battery_optimal_max:
            # 电池在最优范围内，给予奖励
            reward += self.config.battery_optimal_reward
        elif logging_data.Battery < self.config.battery_low_threshold:
            # 电池电量过低，给予惩罚
            reward -= self.config.battery_low_penalty

        # 扫描奖励（在线环境特有）：奖励新扫描的网格单元
        if grid_data and getattr(grid_data, "cells", None):
            # 统计当前已扫描的单元数
            current_scanned = sum(
                1 for c in grid_data.cells
                if _get_cell_entropy(c) < self.config.scan_entropy_threshold
            )
            # 计算新扫描的单元数
            new_scanned = current_scanned - self.prev_scanned_cells
            if new_scanned > 0:
                # 奖励新扫描的单元
                reward += self.config.scan_reward * new_scanned
            # 更新已扫描单元计数
            self.prev_scanned_cells = current_scanned

        # 范围惩罚（在线环境特有）：惩罚超出领机扫描范围
        if runtime_data and runtime_data.leader_position:
            # 计算到领机的距离
            dist_to_leader = (runtime_data.position - runtime_data.leader_position).magnitude()
            # 计算有效范围（领机扫描半径 + 缓冲距离）
            leader_radius = runtime_data.leader_scan_radius + self.config.leader_range_buffer
            # 如果超出范围，给予惩罚
            if runtime_data.leader_scan_radius > 0 and dist_to_leader > leader_radius:
                reward -= self.config.out_of_range_penalty

        # 动作变化惩罚：计算当前动作与上次动作的差异
        action_delta = np.linalg.norm(action - self.last_action)
        reward -= self.config.action_change_penalty * action_delta
        # 动作幅度惩罚：惩罚权重绝对值过大
        reward -= self.config.action_magnitude_penalty * np.linalg.norm(action)

        return reward

    def _count_scanned_cells(self) -> int:
        """
        统计已扫描的网格单元数量
        
        已扫描的单元定义为熵值低于阈值的单元。
        
        返回:
            已扫描的网格单元数量（整数）
        """
        if not self.server or not self.server.grid_data:
            return 0
        # 统计熵值低于阈值的单元数
        return sum(
            1 for cell in self.server.grid_data.cells
            if _get_cell_entropy(cell) < self.config.scan_entropy_threshold
        )

    def set_initial_action(self, weights: np.ndarray) -> None:
        """
        设置初始动作权重，用于与虚拟训练对齐安全裁剪
        
        这个方法允许在reset后、第一次step前设置初始权重。
        这对于从虚拟训练环境迁移到在线环境时很有用，可以确保
        安全限制（max_weight_delta）基于正确的初始值进行计算。
        
        参数:
            weights: 5维权重向量，如果为None或格式不正确则忽略
        """
        if weights is None:
            return
        # 转换为numpy数组
        weights = np.array(weights, dtype=np.float32)
        # 验证维度
        if weights.shape[0] != 5:
            return
        # 裁剪到有效范围
        weights = np.clip(weights, self.config.weight_min, self.config.weight_max)
        # 设置初始动作并标记
        self.last_action = weights.copy()
        self._has_initial_action = True
