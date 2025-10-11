import sys
import os
import numpy as np
from typing import Dict, List, Tuple, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from multirotor.Algorithm.Vector3 import Vector3
from multirotor.Algorithm.scanner_runtime_data import ScannerRuntimeData
from multirotor.Algorithm.HexGridDataModel import HexGridDataModel


class DroneLearningEnv:
    """无人机强化学习环境，提供OpenAI Gym风格的接口"""

    def __init__(self, server, drone_name):
        """
        初始化无人机学习环境
        :param server: MultiDroneAlgorithmServer实例
        :param drone_name: 无人机名称
        """
        self.server = server
        self.drone_name = drone_name

        # 状态空间维度：无人机位置(3) + 速度(3) + 方向向量(3) + 各方向权重(5) + Leader相对位置(3) + 扫描效率(1)
        self.state_dim = 3 + 3 + 3 + 5 + 3 + 1

        # 动作空间维度：5个权重系数，每个系数有5个离散取值（-2, -1, 0, 1, 2）的调整步长
        self.action_dim = 5 * 5  # 每个权重有5种调整方式，共5个权重

        # 从配置中读取参数
        config = server.config_data
        
        # 权重调整步长
        self.coefficient_step = config.learning_env_coefficient_step

        # 权重范围限制
        self.coefficient_ranges = config.learning_env_coefficient_ranges.copy()

        # 奖励权重配置
        self.reward_config = {
            'exploration_weight': config.reward_exploration_weight,
            'efficiency_weight': config.reward_efficiency_weight,
            'collision_penalty': config.reward_collision_penalty,
            'boundary_penalty': config.reward_boundary_penalty,
            'energy_penalty': config.reward_energy_penalty,
            'completion_reward': config.reward_completion_reward
        }

        # 记录上一步的状态，用于计算奖励
        self.prev_state = None
        self.prev_scanned_area = 0.0

        # 重置环境
        self.reset()

    def reset(self):
        """重置环境并返回初始状态"""
        # 重置内部状态记录
        self.prev_state = None
        self.prev_scanned_area = self._calculate_scanned_area()

        # 返回初始状态
        return self.get_state()

    def step(self, action):
        """
        执行动作并返回状态、奖励、完成标志和信息
        :param action: 选择的动作索引
        :return: (next_state, reward, done, info)
        """
        # 1. 根据动作调整权重系数
        coefficients_adjustment = self._action_to_coefficients(action)
        current_coefficients = self.server.algorithms[self.drone_name].get_current_coefficients()

        # 应用权重调整
        new_coefficients = {}
        for name, value in current_coefficients.items():
            if name in coefficients_adjustment:
                new_value = value + coefficients_adjustment[name] * self.coefficient_step
                # 确保新值在有效范围内
                min_val, max_val = self.coefficient_ranges[name]
                new_coefficients[name] = max(min_val, min(max_val, new_value))
            else:
                new_coefficients[name] = value

        # 设置新的权重系数
        self.server.algorithms[self.drone_name].set_coefficients(new_coefficients)

        # 2. 获取当前状态
        current_state = self.get_state()

        # 3. 计算奖励
        reward = self.calculate_reward(self.prev_state, current_state)

        # 4. 检查是否完成
        done = self._check_done()

        # 5. 保存当前状态作为下一步的前状态
        self.prev_state = current_state
        self.prev_scanned_area = self._calculate_scanned_area()

        # 6. 准备信息
        info = {
            'coefficients': new_coefficients,
            'reward_components': self._reward_components  # 记录奖励的各个组成部分
        }

        return current_state, reward, done, info

    def get_state(self):
        """获取当前环境状态"""
        with self.server.data_lock:
            runtime_data = self.server.unity_runtime_data[self.drone_name]
            algorithm = self.server.algorithms[self.drone_name]

            # 获取无人机位置
            position = [runtime_data.position.x, runtime_data.position.y, runtime_data.position.z]

            # 计算无人机速度（简化为方向向量乘以移动速度）
            velocity = [runtime_data.finalMoveDir.x * self.server.config_data.moveSpeed,
                        runtime_data.finalMoveDir.y * self.server.config_data.moveSpeed,
                        runtime_data.finalMoveDir.z * self.server.config_data.moveSpeed]

            # 获取无人机方向
            direction = [runtime_data.forward.x, runtime_data.forward.y, runtime_data.forward.z]

            # 获取当前权重系数
            coefficients = algorithm.get_current_coefficients()
            coefficient_values = [
                coefficients['repulsionCoefficient'],
                coefficients['entropyCoefficient'],
                coefficients['distanceCoefficient'],
                coefficients['leaderRangeCoefficient'],
                coefficients['directionRetentionCoefficient']
            ]

            # 获取Leader相对位置
            if runtime_data.leader_position is not None:
                leader_relative_pos = [
                    runtime_data.leader_position.x - runtime_data.position.x,
                    runtime_data.leader_position.y - runtime_data.position.y,
                    runtime_data.leader_position.z - runtime_data.position.z
                ]
            else:
                leader_relative_pos = [0.0, 0.0, 0.0]

            # 计算扫描效率（已扫描区域占总区域的比例）
            total_area = self._estimate_total_area()
            if total_area > 0:
                scanned_area_ratio = min(1.0, self._calculate_scanned_area() / total_area)
            else:
                scanned_area_ratio = 0.0

        # 组合所有状态特征
        state = (position + velocity + direction + coefficient_values +
                 leader_relative_pos + [scanned_area_ratio])

        return np.array(state, dtype=np.float32)

    def calculate_reward(self, prev_state, current_state):
        """计算奖励值"""
        # 初始化奖励组件
        self._reward_components = {
            'exploration': 0.0,
            'efficiency': 0.0,
            'collision': 0.0,
            'boundary': 0.0,
            'energy': 0.0,
            'completion': 0.0
        }

        # 计算探索奖励（基于新增扫描区域）
        current_scanned_area = self._calculate_scanned_area()
        exploration_reward = (current_scanned_area - self.prev_scanned_area) * self.reward_config['exploration_weight']
        self._reward_components['exploration'] = exploration_reward

        # 计算效率奖励（基于单位时间扫描面积）
        # 注意：这里简化处理，实际应考虑时间因素
        self._reward_components['efficiency'] = exploration_reward * self.reward_config['efficiency_weight']

        # 计算碰撞惩罚（检查与其他无人机的距离）
        with self.server.data_lock:
            runtime_data = self.server.unity_runtime_data[self.drone_name]
            current_pos = runtime_data.position
            other_positions = runtime_data.otherScannerPositions

            min_distance = float('inf')
            for other_pos in other_positions:
                distance = (current_pos - other_pos).magnitude()
                min_distance = min(min_distance, distance)

            # 如果距离过近，给予惩罚
            if min_distance < self.server.config_data.minSafeDistance and self.server.config_data.minSafeDistance > 0:
                collision_penalty = self.reward_config['collision_penalty'] * (
                            self.server.config_data.minSafeDistance - min_distance) / self.server.config_data.minSafeDistance
                self._reward_components['collision'] = collision_penalty

        # 计算越界惩罚（检查是否超出Leader范围）
        with self.server.data_lock:
            runtime_data = self.server.unity_runtime_data[self.drone_name]
            if runtime_data.leader_position is not None:
                distance_to_leader = (runtime_data.position - runtime_data.leader_position).magnitude()
                if distance_to_leader > runtime_data.leader_scan_radius and runtime_data.leader_scan_radius > 0:
                    boundary_penalty = self.reward_config['boundary_penalty'] * (
                                distance_to_leader - runtime_data.leader_scan_radius) / runtime_data.leader_scan_radius
                    self._reward_components['boundary'] = boundary_penalty

        # 计算能耗惩罚（基于动作幅度）
        # 简化处理：根据方向变化幅度计算能耗
        if prev_state is not None:
            # 提取前一状态和当前状态中的方向向量
            prev_direction = prev_state[6:9]  # 方向在状态中的位置
            curr_direction = current_state[6:9]

            # 计算方向变化的角度（余弦相似度）
            direction_change = 1 - np.dot(prev_direction, curr_direction) / (
                        np.linalg.norm(prev_direction) * np.linalg.norm(curr_direction) + 1e-8)

            # 应用能耗惩罚
            self._reward_components['energy'] = direction_change * self.reward_config['energy_penalty']

        # 计算完成奖励（如果扫描区域达到一定比例）
        total_area = self._estimate_total_area()
        if total_area > 0 and current_scanned_area / total_area > 0.95:
            self._reward_components['completion'] = self.reward_config['completion_reward']

        # 计算总奖励
        total_reward = sum(self._reward_components.values())

        return total_reward

    def _action_to_coefficients(self, action):
        """将动作索引转换为权重系数调整"""
        # 动作分解：前3位表示要调整的权重，后2位表示调整步长
        coefficient_index = action // 5
        step_index = action % 5 - 2  # -2, -1, 0, 1, 2

        # 权重名称映射
        coefficient_names = [
            'repulsionCoefficient',
            'entropyCoefficient',
            'distanceCoefficient',
            'leaderRangeCoefficient',
            'directionRetentionCoefficient'
        ]

        # 创建调整字典
        adjustment = {}
        if 0 <= coefficient_index < len(coefficient_names):
            adjustment[coefficient_names[coefficient_index]] = step_index

        return adjustment

    def _calculate_scanned_area(self):
        """计算已扫描的区域面积"""
        # 简化处理：基于网格中的低熵单元格数量（不使用不存在的visited属性）
        with self.server.grid_lock:
            grid_data = self.server.grid_data
            scanned_cells = sum(1 for cell in grid_data.cells if cell.entropy < 0.5)

            # 假设每个单元格的面积为固定值
            cell_area = 1.0  # 实际应根据网格配置计算

            return scanned_cells * cell_area

    def _estimate_total_area(self):
        """估计总区域面积"""
        with self.server.grid_lock:
            return len(self.server.grid_data.cells) * 1.0  # 简化处理

    def _check_done(self):
        """检查是否完成任务"""
        total_area = self._estimate_total_area()
        # 避免除零错误
        if total_area == 0:
            return False

        # 任务完成条件：扫描区域达到95%以上
        scanned_ratio = self._calculate_scanned_area() / total_area
        return scanned_ratio > 0.95
