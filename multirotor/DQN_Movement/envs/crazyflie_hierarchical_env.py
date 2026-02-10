"""
Crazyflie 实体无人机分层 DQN 训练环境

功能说明：
    - 将双层 DQN 的高层动作映射为 APF 的权重系数
    - 对齐 DDPG 的实机训练流程：通过 server.algorithms.set_coefficients 更新权重
    - 状态来源：Crazyswarm Logging + Unity Runtime + Grid Data
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import json
import logging
import time
from typing import Dict, Any, Optional

from multirotor.Algorithm.Vector3 import Vector3

logger = logging.getLogger("CrazyflieHierarchicalEnv")

class CrazyflieHierarchicalEnv(gym.Env):
    def __init__(
        self, 
        server, 
        drone_name: str = "UAV1", 
        config_path: Optional[str] = None,
        step_duration: float = 2.0  # 权重更新周期
    ):
        super(CrazyflieHierarchicalEnv, self).__init__()
        self.server = server
        self.drone_name = drone_name
        self.step_duration = step_duration
        
        # 1. 加载配置
        self.config = self._load_config(config_path)
        
        # 2. 动作空间: 对齐 HierarchicalMovementEnv (0-24 离散网格)
        # 在实机权重训练模式下，我们可以将动作映射为预设的权重组合
        self.action_space = spaces.Discrete(25)
        
        # 3. 观察空间: 27维 (对齐 HierarchicalMovementEnv)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32
        )
        
        # 4. 权重映射表 (简单示例：将动作索引映射为不同的权重组合)
        # 实际应用中可以根据 hl_action 映射到特定区域的 APF 参数优化
        self.weight_map = self._init_weight_map()
        
        self.step_count = 0
        self.episode_reward = 0
        self.episode_start_time = time.time()
        self.prev_scanned_cells = 0

    def _load_config(self, config_path):
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"thresholds": {"scanned_entropy": 30.0}}

    def _init_weight_map(self):
        """初始化动作到权重的映射 (重点关注 repulsionCoefficient)"""
        weights = {}
        for i in range(25):
            # 方案一：只根据动作索引动态调整避障权重 (repulsionCoefficient)
            # 这里的逻辑可以根据实际需求调整，例如：
            # 动作索引对应的区域如果障碍物多，则增大避障权重
            repulsion = 1.0 + (i % 5) * 0.5  # 范围 [1.0, 3.0]
            
            # 其他系数保持默认或固定值
            weights[i] = {
                "repulsionCoefficient": float(repulsion),
                "entropyCoefficient": 2.0,
                "distanceCoefficient": 1.0,
                "leaderRangeCoefficient": 1.0,
                "directionRetentionCoefficient": 0.5
            }
        return weights

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.step_count = 0
        self.episode_reward = 0
        self.episode_start_time = time.time()
        
        # 初始权重同步 (可选)
        if self.server:
            # 获取当前 server 的权重作为基准，只改 repulsion
            current_weights = self.server.algorithms[self.drone_name].get_current_coefficients()
            # 记录初始已扫描单元
            self.prev_scanned_cells = self._count_scanned_cells()
        
        return self._get_observation(), {}

    def step(self, action):
        action = int(action)
        weight_config = self.weight_map[action]
        
        # 对齐 DDPG 流程：下发权重到 Server 的算法模块
        if self.server:
            # 安全检查：确保有实机数据才更新，防止下发到无效状态
            logging_data = self.server.crazyswarm.get_loggingData_by_droneName(self.drone_name)
            if logging_data and logging_data.Battery > 3.0: # 简单的安全检查
                self.server.algorithms[self.drone_name].set_coefficients(weight_config)
            else:
                logger.warning(f"实机 {self.drone_name} 数据异常或电量不足，跳过权重更新")
            
        # 等待物理执行周期 (给 APF 算法时间驱动实机)
        time.sleep(self.step_duration)
        
        next_obs = self._get_observation()
        reward = self._calculate_reward(next_obs)
        self.episode_reward += reward
        self.step_count += 1
        
        done = self._check_done()
        
        # 注入诊断信息
        info = {
            "hl_action": action,
            "repulsion": weight_config["repulsionCoefficient"],
            "step": self.step_count
        }
        
        return next_obs, reward, done, False, info

    def _get_observation(self):
        """对齐 HierarchicalMovementEnv 的 27 维观察空间"""
        if not self.server:
            return np.zeros(27, dtype=np.float32)
            
        with self.server.data_lock:
            logging_data = self.server.crazyswarm.get_loggingData_by_droneName(self.drone_name)
            runtime_data = self.server.unity_runtime_data.get(self.drone_name)
            
        if not logging_data:
            return np.zeros(27, dtype=np.float32)
            
        # 位置 (3)
        pos = np.array([logging_data.X / 20.0, logging_data.Y / 20.0, logging_data.Z / 20.0])
        
        # Leader 相对位置 (3)
        leader_rel = np.zeros(3)
        if runtime_data and runtime_data.leader_position:
            leader_rel = np.array([
                (runtime_data.leader_position.x - pos[0]),
                (runtime_data.leader_position.y - pos[1]),
                (runtime_data.leader_position.z - pos[2])
            ])
            
        # 粗略网格熵值 (16) - 简化处理
        coarse_entropy = np.full(16, 50.0)
        
        # 扫描进度 (3)
        total = len(self.server.grid_data.cells) if self.server.grid_data else 1
        scanned = self._count_scanned_cells()
        scan_info = np.array([scanned / max(total, 1), float(scanned), float(total - scanned)])
        
        # 电量信息 (2)
        battery_info = np.array([logging_data.Battery, 100.0])
        
        return np.concatenate([pos, leader_rel, coarse_entropy, scan_info, battery_info]).astype(np.float32)

    def _calculate_reward(self, obs):
        reward = -0.1 # 步惩罚
        current_scanned = self._count_scanned_cells()
        if current_scanned > self.prev_scanned_cells:
            reward += (current_scanned - self.prev_scanned_cells) * 10.0
        self.prev_scanned_cells = current_scanned
        return reward

    def _count_scanned_cells(self):
        if not self.server or not self.server.grid_data:
            return 0
        thresh = self.config["thresholds"]["scanned_entropy"]
        return sum(1 for c in self.server.grid_data.cells if c.entropy < thresh)

    def _check_done(self):
        elapsed = time.time() - self.episode_start_time
        return elapsed > 300 or self.step_count > 100
