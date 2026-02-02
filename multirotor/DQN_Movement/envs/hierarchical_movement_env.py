import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import sys
import json
import logging
import time
import traceback
from typing import Dict, Any, List, Optional, Tuple

# 添加项目路径以支持 Algorithm 模块导入
current_dir = os.path.dirname(os.path.abspath(__file__))
multirotor_dir = os.path.dirname(current_dir)
if multirotor_dir not in sys.path:
    sys.path.insert(0, multirotor_dir)

from Algorithm.Vector3 import Vector3
from Algorithm.scanner_algorithm import ScannerAlgorithm
from Algorithm.scanner_config_data import ScannerConfigData

logger = logging.getLogger("HierarchicalMovementEnv")

class HierarchicalMovementEnv(gym.Env):
    """
    分层强化学习移动环境 (Hierarchical Reinforcement Learning for Drone Movement)
    
    高层 (HL): Global Planner (DQN)
        - 决策周期: 5.0s
        - 动作空间: 全局选点 (5x5 区域选择)
        - 目标: 引导无人机前往高熵或未扫描区域
    
    底层 (LL): Local Controller (DQN + APF)
        - 决策周期: 0.5s
        - 动作空间: 6个离散移动方向
        - 目标: 快速准确到达高层指定目标，同时通过 APF 避障
    """
    
    def __init__(self, server=None, drone_name="UAV1", config_path=None):
        super(HierarchicalMovementEnv, self).__init__()
        
        self.server = server
        self.drone_name = drone_name
        
        # 加载配置
        self.config = self._load_config(config_path)
        self.term_cfg = self.config.get('termination_config', {
            "target_scan_ratio": 0.95,
            "max_collision_count": 1,
            "max_elapsed_time_sec": 300.0,
            "stagnation_timeout_sec": 30.0
        })
        
        # 初始化底层 APF 算法
        scanner_cfg = ScannerConfigData()
        if self.server and hasattr(self.server, 'config_data'):
            # 复制服务器配置
            for attr in vars(self.server.config_data):
                setattr(scanner_cfg, attr, getattr(self.server.config_data, attr))
        self.apf_algo = ScannerAlgorithm(scanner_cfg)
        
        # --- 高层 (HL) 定义 ---
        # 动作空间: 5x5 离散网格选择 (0-24)
        self.action_space = spaces.Discrete(25)  # Gymnasium 标准属性
        self.hl_action_space = self.action_space  # 别名，便于理解
        
        # 观察空间: 27维
        # - 自身位置 (3)
        # - Leader 相对位置 (3)
        # - 粗略网格熵值 (4x4 = 16)
        # - 扫描进度 (3)
        # - 电量信息 (2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32
        )
        
        # --- 底层 (LL) 参数 ---
        self.ll_steps_per_hl = 30  # 最大 15s / 0.5s = 30步 (按需触发)
        self.ll_step_duration = 0.5
        self.arrival_threshold = 2.0 # 到达阈值 2米
        
        # LL 动作映射 (修正映射：0/1对应高度Y，2/3对应左右Z，4/5对应前后X)
        self.ll_action_step = self.config['movement']['step_size']
        self.ll_action_map = {
            0: np.array([0, self.ll_action_step, 0]),      # 上 (Y+)
            1: np.array([0, -self.ll_action_step, 0]),     # 下 (Y-)
            2: np.array([0, 0, -self.ll_action_step]),     # 左 (Z-)
            3: np.array([0, 0, self.ll_action_step]),      # 右 (Z+)
            4: np.array([self.ll_action_step, 0, 0]),      # 前 (X+)
            5: np.array([-self.ll_action_step, 0, 0])      # 后 (X-)
        }
        
        # 内部状态
        self.current_hl_goal = None
        self.step_count = 0
        self.episode_reward = 0
        self.collision_count = 0
        self.out_of_range_count = 0
        self.prev_scanned_cells = 0
        self.prev_entropy_sum = 0
        
        self.episode_index = 0  # 新增：用于 DataCollector 的 Episode 计数
        self._first_reset = True
        
        # 底层策略 (在训练 HL 时需要，如果是协同训练则由外部管理)
        self.ll_policy = None 

    def _load_config(self, config_path):
        if config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, "..", "configs", "movement_dqn_config.json")
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return self._default_config()

    def _default_config(self):
        return {
            "movement": {"step_size": 1.0, "max_steps": 100}, # HL steps
            "rewards": {
                "exploration": 10.0, "collision": -50.0, "out_of_range": -30.0,
                "goal_reached": 50.0, "step_penalty": -1.0,
                "battery_low_penalty": 10.0, "battery_optimal_reward": 2.0
            },
            "thresholds": {
                "collision_distance": 2.0, "scanned_entropy": 10.0,
                "success_scan_ratio": 0.95,
                "battery_low_threshold": 3.5,
                "battery_optimal_min": 3.7,
                "battery_optimal_max": 4.1
            }
        }

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        if self._first_reset:
            self._first_reset = False
            if self.server and hasattr(self.server, 'reset_battery_voltage'):
                self.server.reset_battery_voltage(self.drone_name)
        else:
            if self.server:
                self.server.reset_environment()
                if hasattr(self.server, 'reset_battery_voltage'):
                    self.server.reset_battery_voltage(self.drone_name)
                time.sleep(1.0)
                
        self.step_count = 0
        self.episode_reward = 0
        self.collision_count = 0
        self.out_of_range_count = 0
        self.prev_scanned_cells = self._count_scanned_cells()
        self.prev_entropy_sum = self._get_total_entropy()
        self.current_hl_goal = None
        self.episode_index += 1  # 增加 Episode 计数
        
        return self._get_hl_observation(), {}

    def step(self, hl_action):
        """
        高层决策一步 (基于条件触发：到达、超时、起步)
        """
        # 3. 起步阶段：如果当前没有设定目标，或者收到新指令，则立即开始执行
        # (在 RL step 中，hl_action 的到来本身就是一种触发)
        
        # 1. 将 hl_action 映射为物理目标点
        self.current_hl_goal = self._map_hl_action_to_goal(hl_action)
        
        total_ll_reward = 0
        terminated = False
        actual_steps = 0
        
        # 2. 执行底层步，直到满足触发条件
        # 条件 1: 目标达成 (Arrival) -> 在循环内检查
        # 条件 2: 超时 (Timeout) -> range(self.ll_steps_per_hl) 限制了最大 15s
        for i in range(self.ll_steps_per_hl):
            actual_steps += 1
            ll_obs = self._get_ll_observation()
            
            # 2.1 获取 LL 动作
            if self.ll_policy:
                ll_action, _ = self.ll_policy.predict(ll_obs, deterministic=True)
            else:
                ll_action = self._simple_ll_planner(ll_obs)
            
            # 2.2 执行底层移动
            ll_next_obs, ll_reward, ll_done, info = self.ll_step(ll_action)
            total_ll_reward += ll_reward
            
            if ll_done:
                terminated = True
                break
                
            # --- 核心修改：检查是否到达目标 (Arrival) ---
            goal_rel = ll_next_obs[0:3]
            dist = np.linalg.norm(goal_rel)
            if dist < self.arrival_threshold:
                # print(f"[到达] {self.drone_name} 已到达目标范围 (2m)，触发下一决策")
                break
        
        # 3. 高层统计与奖励
        self.step_count += 1
        hl_reward = self._calculate_hl_reward(hl_action, total_ll_reward)
        self.episode_reward += hl_reward
        
        # --- 新增：记录训练统计到 DataCollector ---
        if self.server and hasattr(self.server, 'set_training_stats'):
            self.server.set_training_stats(
                episode=self.episode_index,
                step=self.step_count,
                reward=float(hl_reward),
                total_reward=float(self.episode_reward)
            )
            
        # --- 新增：记录高层动作和目标到 DataCollector (用于 scan_data.csv) ---
        if self.server and hasattr(self.server, 'data_collector'):
            self.server.data_collector.set_external_data('hl_action', int(hl_action))
            if self.current_hl_goal:
                self.server.data_collector.set_external_data('hl_goal_x', float(self.current_hl_goal.x))
                self.server.data_collector.set_external_data('hl_goal_y', float(self.current_hl_goal.y))
                self.server.data_collector.set_external_data('hl_goal_z', float(self.current_hl_goal.z))
        
        # 检查是否结束
        if not terminated:
            terminated = self._check_done()
            
        return self._get_hl_observation(), hl_reward, terminated, False, {}

    def ll_step(self, action):
        """
        底层执行一步 (0.5s)，集成 APF 避障
        """
        # 转换动作为位移
        dqn_displacement = self.ll_action_map[int(action)]
        
        # 集成 APF 避障得到最终位移
        displacement = self._calculate_apf_displacement(dqn_displacement)

        # 执行移动
        if self.server:
            self._apply_movement(displacement)
            time.sleep(0.05) # 基础等待
            
            # 更新电量消耗
            if hasattr(self.server, "update_battery_voltage"):
                # 计算动作强度（基于位移大小）
                step_norm = float(np.linalg.norm(displacement))
                base_step = max(self.ll_action_step, 1e-6)
                action_intensity = min(1.0, step_norm / base_step)
                self.server.update_battery_voltage(self.drone_name, action_intensity)
            
        next_ll_obs = self._get_ll_observation()
        reward = self._calculate_ll_reward(action, next_ll_obs)
        
        # 检查底层是否发生碰撞或越界
        done = self._is_ll_done()
        
        return next_ll_obs, reward, done, {}

    def _calculate_apf_displacement(self, dqn_displacement):
        """计算融合 APF 后的位移向量"""
        if not self.server:
            return dqn_displacement
            
        with self.server.data_lock:
            runtime_data = self.server.unity_runtime_data.get(self.drone_name)
            if not runtime_data:
                return dqn_displacement
                
            # 计算排斥力
            apf_repulsion_dir = self.apf_algo.calculate_collide_direction(runtime_data)
            
            # Veto 逻辑: 如果排斥力足够大，则覆盖 DQN 指令或与之融合
            if apf_repulsion_dir.magnitude() > 0.1:
                # 融合因子 (排斥力越大，占比越高)
                alpha = min(1.0, apf_repulsion_dir.magnitude() * 2.0)
                
                # DQN 归一化方向
                dqn_dir_vec = Vector3(dqn_displacement[0], dqn_displacement[1], dqn_displacement[2]).normalized()
                
                # 最终合成方向
                final_dir_vec = (dqn_dir_vec * (1 - alpha) + apf_repulsion_dir * alpha).normalized()
                return np.array([final_dir_vec.x, final_dir_vec.y, final_dir_vec.z]) * self.ll_action_step
            else:
                return dqn_displacement

    def _is_ll_done(self):
        """检查底层是否达到终止条件（碰撞等）"""
        if not self.server:
            return False
        with self.server.data_lock:
            rd = self.server.unity_runtime_data.get(self.drone_name)
            if rd:
                min_dist = self._get_min_distance_to_others(rd)
                # 放宽碰撞阈值，避免APF正常避障时误触发
                collision_threshold = self.config['thresholds']['collision_distance']
                if min_dist < collision_threshold * 0.8:  # 更严格的碰撞判定（距离更小才算真正碰撞）
                    self.collision_count += 1
                    # 增加容忍次数，避免过早终止（从5次增加到15次）
                    if self.collision_count >= 15: 
                        print(f"[终止] {self.drone_name} 碰撞次数过多: {self.collision_count}次")
                        return True
                else:
                    # 距离恢复安全时重置计数器（给予训练纠错机会）
                    if self.collision_count > 0 and min_dist > collision_threshold * 1.5:
                        self.collision_count = max(0, self.collision_count - 1)  # 缓慢恢复
        return False

    def _get_hl_observation(self):
        if not self.server:
            return np.zeros(27, dtype=np.float32)
            
        try:
            with self.server.data_lock:
                rd = self.server.unity_runtime_data.get(self.drone_name)
                pos = rd.position
                position = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
                
                if rd.leader_position:
                    leader_rel = np.array([
                        rd.leader_position.x - pos.x,
                        rd.leader_position.y - pos.y,
                        rd.leader_position.z - pos.z
                    ], dtype=np.float32)
                else:
                    leader_rel = np.zeros(3, dtype=np.float32)
                    
            with self.server.grid_lock:
                grid = self.server.grid_data
                # 粗略网格熵值 (4x4)
                coarse_entropy = self._get_coarse_grid_entropy(grid, rd.leader_position, rd.leader_scan_radius)
                # 扫描进度
                scan_info = self._get_scan_info(grid)
                
            battery_info = self._get_battery_info()
            
            return np.concatenate([position, leader_rel, coarse_entropy, scan_info, battery_info]).astype(np.float32)
        except:
            return np.zeros(27, dtype=np.float32)

    def _get_ll_observation(self):
        """底层观察: 13维"""
        if not self.server:
            return np.zeros(13, dtype=np.float32)
        try:
            with self.server.data_lock:
                rd = self.server.unity_runtime_data.get(self.drone_name)
                pos = rd.position
                
                # 1. 相对 HL 目标的距离
                if self.current_hl_goal:
                    goal_rel = np.array([
                        self.current_hl_goal.x - pos.x,
                        self.current_hl_goal.y - pos.y,
                        self.current_hl_goal.z - pos.z
                    ], dtype=np.float32)
                else:
                    goal_rel = np.zeros(3, dtype=np.float32)
                
                # 2. 速度 (3)
                vel = rd.velocity
                velocity = np.array([vel.x, vel.y, vel.z], dtype=np.float32)
                
                # 3. Leader 相对位置 (3)
                if rd.leader_position:
                    leader_rel = np.array([
                        rd.leader_position.x - pos.x,
                        rd.leader_position.y - pos.y,
                        rd.leader_position.z - pos.z
                    ], dtype=np.float32)
                else:
                    leader_rel = np.zeros(3, dtype=np.float32)
                    
                # 4. 最近物体距离 (1)
                min_dist = self._get_min_distance_to_others(rd)
                
            with self.server.grid_lock:
                # 5. 局部平均熵 (3: mean, max, std)
                entropy_info = self._get_local_entropy_info(self.server.grid_data, pos)
                
            return np.concatenate([goal_rel, velocity, leader_rel, [min_dist], entropy_info]).astype(np.float32)
        except:
            return np.zeros(13, dtype=np.float32)

    def _map_hl_action_to_goal(self, action) -> Vector3:
        """
        将 0-24 映射为以 Leader 为中心或在 AOI 内的 5x5 网格点
        """
        if not self.server: return Vector3(0, 8, 0)
        
        with self.server.data_lock:
            rd = self.server.unity_runtime_data.get(self.drone_name)
            center = rd.leader_position if rd.leader_position else Vector3(0, 0, 0)
            radius = rd.leader_scan_radius if rd.leader_scan_radius > 0 else 50.0
            
        # 在 2*radius 的范围内划分 5x5 网格
        row = action // 5
        col = action % 5
        
        grid_size = (2 * radius) / 5
        offset_x = (col - 2) * grid_size
        offset_z = (row - 2) * grid_size
        
        # 维持在最佳扫描高度 (Y轴)
        target_y = self.config['thresholds'].get('optimal_scan_height', 8.0)
        
        return Vector3(center.x + offset_x, target_y, center.z + offset_z)

    def _calculate_hl_reward(self, hl_action, total_ll_reward):
        # 高层奖励基于全局扫描增量
        current_scanned = self._count_scanned_cells()
        new_cells = current_scanned - self.prev_scanned_cells
        self.prev_scanned_cells = current_scanned
        
        # 1. 扫描增量奖励
        reward = new_cells * self.config['rewards']['exploration']
        
        # 2. 累加底层奖励 (体现效率)
        ll_weight = self.config['rewards'].get('ll_efficiency_weight', 0.1)
        reward += total_ll_reward * ll_weight
        
        # 3. 惩罚步数
        reward += self.config['rewards']['step_penalty']
        
        # 4. 熵值降低奖励
        current_entropy = self._get_total_entropy()
        entropy_reduced = self.prev_entropy_sum - current_entropy
        if entropy_reduced > 0:
            entropy_weight = self.config['rewards'].get('entropy_reduction_weight', 2.0)
            reward += entropy_reduced * entropy_weight
        self.prev_entropy_sum = current_entropy
        
        # 5. 电量奖励与惩罚
        if self.server and hasattr(self.server, "get_battery_voltage"):
            try:
                current_voltage = self.server.get_battery_voltage(self.drone_name)
                cfg_rewards = self.config.get("rewards", {})
                cfg_thresholds = self.config.get("thresholds", {})
                
                # 电量过低惩罚
                if current_voltage < cfg_thresholds.get("battery_low_threshold", 3.5):
                    reward -= cfg_rewards.get("battery_low_penalty", 10.0)
                # 电量最优范围奖励
                elif cfg_thresholds.get("battery_optimal_min", 3.7) <= current_voltage <= cfg_thresholds.get("battery_optimal_max", 4.1):
                    reward += cfg_rewards.get("battery_optimal_reward", 2.0)
            except:
                pass
        
        return reward

    def _calculate_ll_reward(self, action, next_obs):
        # 底层奖励基于是否接近 HL 目标
        goal_dist = np.linalg.norm(next_obs[0:3])
        reward = -goal_dist * 0.1  # 距离惩罚
        
        if goal_dist < 2.0:
            reward += 10.0  # 接近目标奖励 (增加一点)
            
        # --- 新增：高度稳定性奖励与惩罚 ---
        # 获取当前高度 (Y轴对应 obs 的 [0,1,2] 中的目标相对位置，但我们需要绝对高度)
        # obs 结构：[goal_rel(3), velocity(3), leader_rel(3), min_dist(1), entropy_info(3)]
        # 我们可以通过 leader_rel 和 server 数据计算，或者直接从 server 获取
        
        if self.server:
            with self.server.data_lock:
                rd = self.server.unity_runtime_data.get(self.drone_name)
                if rd:
                    current_height = rd.position.y
                    
                    min_height = self.config['thresholds'].get('min_scan_height', 2.0)
                    max_height = self.config['thresholds'].get('max_scan_height', 15.0)
                    optimal_height = self.config['thresholds'].get('optimal_scan_height', 8.0)
                    height_penalty = self.config['rewards'].get('height_penalty', -5.0)
                    
                    if current_height < min_height:
                        reward += height_penalty * (min_height - current_height) # 飞得越低惩罚越大
                    elif current_height > max_height:
                        reward += height_penalty * (current_height - max_height) # 飞得越高惩罚越大
                    
                    # 最佳高度奖励
                    if abs(current_height - optimal_height) < 1.5:
                        reward += self.config['rewards'].get('optimal_height_bonus', 1.0)
                        
                    # 【新增】超出领导者范围的返回奖励
                    if rd.leader_position:
                        dist_to_leader = (rd.position - rd.leader_position).magnitude()
                        radius = rd.leader_scan_radius if rd.leader_scan_radius > 0 else 50.0
                        
                        if dist_to_leader > radius:
                            # 如果超出范围，检查是否在往回飞(通过leader_rel判断)
                            leader_rel = next_obs[6:9]  # leader_rel位置
                            goal_rel = next_obs[0:3]     # 目标相对位置
                            
                            # 如果目标在领导者附近(比当前位置更接近),给予奖励
                            dist_goal_to_leader = np.linalg.norm(goal_rel - leader_rel)
                            if dist_goal_to_leader < dist_to_leader:
                                return_bonus = self.config['rewards'].get('return_to_range_bonus', 8.0)
                                reward += return_bonus
        
        return reward

    def _simple_ll_planner(self, obs):
        """简单趋向目标的逻辑 (修正后的轴对应关系)"""
        goal_rel = obs[0:3] # [dx, dy, dz]
        # X: 前后 (4/5), Y: 高度 (0/1), Z: 左右 (2/3)
        if np.abs(goal_rel[1]) > 2.0: # 优先调整高度误差
            return 0 if goal_rel[1] > 0 else 1
            
        if np.abs(goal_rel[0]) > np.abs(goal_rel[2]):
            return 4 if goal_rel[0] > 0 else 5
        else:
            return 3 if goal_rel[2] > 0 else 2

    def _get_coarse_grid_entropy(self, grid_data, leader_pos, leader_radius):
        """获取 4x4 粗略网格的熵值统计"""
        if not grid_data or not grid_data.cells or not leader_pos:
            return np.full(16, 50.0, dtype=np.float32)
        
        # 将 leader 范围内的区域划分为 4x4
        # 范围为 [leader_pos.x - radius, leader_pos.x + radius] x [leader_pos.y - radius, leader_pos.y + radius]
        entropy_grid = np.zeros((4, 4))
        count_grid = np.zeros((4, 4))
        
        radius = leader_radius if leader_radius > 0 else 50.0
        grid_size = (2 * radius) / 4
        
        for cell in grid_data.cells:
            # 计算相对于 leader 的偏移
            dx = cell.center.x - leader_pos.x
            dy = cell.center.y - leader_pos.y
            
            # 检查是否在 4x4 范围内
            if -radius <= dx < radius and -radius <= dy < radius:
                col = int((dx + radius) // grid_size)
                row = int((dy + radius) // grid_size)
                
                # 边界处理
                col = min(3, max(0, col))
                row = min(3, max(0, row))
                
                entropy_grid[row, col] += cell.entropy
                count_grid[row, col] += 1
        
        # 计算平均值
        for r in range(4):
            for c in range(4):
                if count_grid[r, c] > 0:
                    entropy_grid[r, c] /= count_grid[r, c]
                else:
                    entropy_grid[r, c] = 0.0 # 无数据区域视为已扫描或不可达
                    
        return entropy_grid.flatten().astype(np.float32)

    def _get_local_entropy_info(self, grid_data, position):
        if not grid_data or not grid_data.cells:
            return np.array([50.0, 50.0, 0.0], dtype=np.float32)
        # 借用 MovementEnv 的逻辑
        dist_thresh = 10.0
        nearby = [c for c in grid_data.cells[:100] if (c.center - position).magnitude() < dist_thresh]
        if not nearby: return np.array([50.0, 50.0, 0.0], dtype=np.float32)
        entropies = [c.entropy for c in nearby]
        return np.array([np.mean(entropies), np.max(entropies), np.std(entropies)], dtype=np.float32)

    def _get_scan_info(self, grid_data):
        if not grid_data or not grid_data.cells: return np.array([0, 0, 0], dtype=np.float32)
        total = len(grid_data.cells)
        scanned = sum(1 for c in grid_data.cells if c.entropy < 10.0)
        return np.array([scanned/total, scanned, total-scanned], dtype=np.float32)

    def _count_scanned_cells(self):
        if not self.server or not self.server.grid_data: return 0
        with self.server.grid_lock:
            return sum(1 for c in self.server.grid_data.cells if c.entropy < 10.0)

    def _get_total_entropy(self):
        if not self.server or not self.server.grid_data: return 0.0
        with self.server.grid_lock:
            return sum(c.entropy for c in self.server.grid_data.cells)

    def _get_min_distance_to_others(self, rd):
        if not rd.otherScannerPositions: return 100.0
        pos = rd.position
        dists = [(pos - op).magnitude() for op in rd.otherScannerPositions]
        return min(dists)

    def _get_battery_info(self):
        if not self.server or not hasattr(self.server, 'get_battery_voltage'):
            return np.array([4.2, 100.0], dtype=np.float32)
        
        try:
            voltage = self.server.get_battery_voltage(self.drone_name)
            percentage = 100.0
            
            if hasattr(self.server, "battery_manager"):
                battery_info = self.server.battery_manager.get_battery_info(self.drone_name)
                if battery_info:
                    percentage = battery_info.get_remaining_percentage()
            
            return np.array([voltage, percentage], dtype=np.float32)
        except:
            return np.array([4.2, 100.0], dtype=np.float32)

    def _apply_movement(self, displacement):
        if not self.server: return
        from Algorithm.Vector3 import Vector3
        mag = np.linalg.norm(displacement)
        if mag > 1e-6:
            direction = displacement / mag
            self.server.set_dqn_movement(self.drone_name, Vector3(direction[0], direction[1], direction[2]))

    def _check_done(self):
        """判断Episode是否结束 (统一终止逻辑)"""
        # 计算高层对应的累计物理时间
        # self.step_count 是高层步数，每步包含 ll_steps_per_hl 个底层步
        elapsed_time = self.step_count * self.ll_steps_per_hl * self.ll_step_duration
        
        # 1. 达到最大物理仿真时间
        if elapsed_time >= self.term_cfg['max_elapsed_time_sec']:
            print(f"[终止] 达到最大仿真时间: {elapsed_time:.1f}s / {self.term_cfg['max_elapsed_time_sec']}s")
            return True
        
        # 2. 达到目标扫描比例
        with self.server.grid_lock:
            total = len(self.server.grid_data.cells)
            if total > 0:
                scanned = sum(1 for c in self.server.grid_data.cells if c.entropy < 10.0)
                scan_ratio = scanned / total
                if scan_ratio >= self.term_cfg['target_scan_ratio']:
                    print(f"[终止] 任务成功：覆盖率 {scan_ratio:.2%} >= {self.term_cfg['target_scan_ratio']:.2%}")
                    return True
        
        # 3. 碰撞次数达到阈值
        if self.collision_count >= self.term_cfg['max_collision_count']:
            print(f"[终止] 发生碰撞或超过上限: {self.collision_count} / {self.term_cfg['max_collision_count']}")
            return True
            
        return False

class MultiDroneHierarchicalMovementEnv(gym.Env):
    """
    多无人机分层强化学习环境
    采用轮流决策机制，共享高层规划模型
    """
    def __init__(self, server=None, drone_names=None, config_path=None):
        super(MultiDroneHierarchicalMovementEnv, self).__init__()
        self.server = server
        self.drone_names = drone_names if drone_names else ["UAV1"]
        self.num_drones = len(self.drone_names)
        self.current_drone_idx = 0
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 动作空间 (HL): 0-24
        self.action_space = spaces.Discrete(25)
        
        # 观察空间 (HL): 27维
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32
        )
        
        # 初始化单机环境字典，用于复用逻辑
        self.envs = {
            name: HierarchicalMovementEnv(server, name, config_path)
            for name in self.drone_names
        }
        
        self.step_count = 0
        self.total_episode_reward = 0
        self.episode_index = 0  # 新增：用于 DataCollector 的 Episode 计数
        self._first_reset = True

    def _load_config(self, config_path):
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        # 尝试默认路径
        default_path = os.path.join(os.path.dirname(__file__), "..", "configs", "hierarchical_dqn_config.json")
        if os.path.exists(default_path):
            with open(default_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def reset(self, seed=None, options=None):
        if seed is not None: np.random.seed(seed)
        
        if self._first_reset:
            self._first_reset = False
            if self.server and hasattr(self.server, 'reset_battery_voltage'):
                for name in self.drone_names:
                    self.server.reset_battery_voltage(name)
        else:
            if self.server:
                self.server.reset_environment()
                if hasattr(self.server, 'reset_battery_voltage'):
                    for name in self.drone_names:
                        self.server.reset_battery_voltage(name)
                time.sleep(1.0)
        
        for name in self.drone_names:
            self.envs[name].reset()
            
        self.step_count = 0
        self.total_episode_reward = 0
        self.episode_index += 1  # 增加 Episode 计数
        self.current_drone_idx = 0
        
        return self.envs[self.drone_names[0]]._get_hl_observation(), {}

    def step(self, action):
        current_drone = self.drone_names[self.current_drone_idx]
        current_env = self.envs[current_drone]
        
        # 1. 更新当前无人机的高层目标
        current_env.current_hl_goal = current_env._map_hl_action_to_goal(action)
        
        total_ll_reward = 0
        terminated = False
        
        # 2. 执行底层步，直到当前无人机到达或超时
        # 使用当前环境配置的步数 (默认 30 步 = 15s)
        for i in range(current_env.ll_steps_per_hl):
            # 为每个无人机决定位移并应用
            for name in self.drone_names:
                env = self.envs[name]
                ll_obs = env._get_ll_observation()
                
                # 获取 LL 动作
                if env.ll_policy:
                    ll_action, _ = env.ll_policy.predict(ll_obs, deterministic=True)
                else:
                    ll_action = env._simple_ll_planner(ll_obs)
                
                dqn_displacement = env.ll_action_map[int(ll_action)]
                displacement = env._calculate_apf_displacement(dqn_displacement)
                
                # 应用移动
                env._apply_movement(displacement)
                
                # 更新每个无人机的电量消耗
                if self.server and hasattr(self.server, "update_battery_voltage"):
                    step_norm = float(np.linalg.norm(displacement))
                    base_step = max(env.ll_action_step, 1e-6)
                    action_intensity = min(1.0, step_norm / base_step)
                    self.server.update_battery_voltage(name, action_intensity)
            
            # 统一等待环境更新
            if self.server: time.sleep(0.05)
            
            # 计算当前决策无人机的底层奖励和状态
            next_ll_obs = current_env._get_ll_observation()
            total_ll_reward += current_env._calculate_ll_reward(0, next_ll_obs)
            
            # --- 核心修改：检查当前无人机是否到达目标 (Arrival) ---
            goal_rel = next_ll_obs[0:3]
            dist = np.linalg.norm(goal_rel)
            if dist < current_env.arrival_threshold:
                # print(f"[同步到达] {current_drone} 已到达目标，结束本轮 HL step")
                break
                
            if current_env._is_ll_done():
                terminated = True
                break
        
        # 3. 高层统计与奖励 (针对当前无人机)
        current_env.step_count += 1
        hl_reward = current_env._calculate_hl_reward(action, total_ll_reward)
        self.total_episode_reward += hl_reward
        self.step_count += 1
        
        # --- 新增：记录训练统计到 DataCollector (多机模式) ---
        if self.server and hasattr(self.server, 'set_training_stats'):
            self.server.set_training_stats(
                episode=self.episode_index,
                step=self.step_count,
                reward=float(hl_reward),
                total_reward=float(self.total_episode_reward)
            )
            
        # --- 新增：记录高层动作和目标到 DataCollector ---
        if self.server and hasattr(self.server, 'data_collector'):
            self.server.data_collector.set_external_data('hl_action', int(action))
            self.server.data_collector.set_external_data('drone_name', current_drone)
            if current_env.current_hl_goal:
                self.server.data_collector.set_external_data('hl_goal_x', float(current_env.current_hl_goal.x))
                self.server.data_collector.set_external_data('hl_goal_y', float(current_env.current_hl_goal.y))
                self.server.data_collector.set_external_data('hl_goal_z', float(current_env.current_hl_goal.z))
        
        # 切换到下一个无人机
        self.current_drone_idx = (self.current_drone_idx + 1) % self.num_drones
        next_drone = self.drone_names[self.current_drone_idx]
        
        # 检查整体是否结束
        done = terminated or self.step_count >= self.config['movement']['max_steps'] * self.num_drones
        
        next_obs = self.envs[next_drone]._get_hl_observation()
        
        # 更新 info
        info = {
            'drone_name': current_drone,
            'total_reward': self.total_episode_reward,
            'step_count': self.step_count
        }
        
        return next_obs, hl_reward, done, False, info

    def set_ll_policy(self, policy):
        """为所有无人机设置底层策略"""
        for env in self.envs.values():
            env.ll_policy = policy
