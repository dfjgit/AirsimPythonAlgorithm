"""
简单的权重学习环境
使用Stable-Baselines3训练APF权重系数
"""
import numpy as np
import gym
from gym import spaces
import os
from dqn_reward_config_data import DQNRewardConfig


class SimpleWeightEnv(gym.Env):
    """
    简单的APF权重学习环境
    
    目标: 学习5个权重系数 (α1, α2, α3, α4, α5)
    """
    
    def __init__(self, server=None, drone_name="UAV1", reward_config_path=None):
        super(SimpleWeightEnv, self).__init__()
        
        self.server = server
        self.drone_name = drone_name
        
        # 加载奖励配置
        if reward_config_path is None:
            # 使用默认路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            reward_config_path = os.path.join(current_dir, "dqn_reward_config.json")
        
        self.reward_config = DQNRewardConfig(reward_config_path)
        print(f"[OK] DQN环境已加载奖励配置")
        
        # 状态空间: 18维
        # [位置(3) + 速度(3) + 方向(3) + 熵值(3) + Leader(3) + 扫描(3)]
        self.observation_space = spaces.Box(
            low=-100.0,
            high=100.0,
            shape=(18,),
            dtype=np.float32
        )
        
        # 动作空间: 5维连续（5个权重系数）
        # 使用配置文件中的范围
        self.action_space = spaces.Box(
            low=self.reward_config.weight_min,
            high=self.reward_config.weight_max,
            shape=(5,),
            dtype=np.float32
        )
        
        # 记录上一步的状态
        self.prev_scanned_cells = 0
        self.prev_position = None
        self.step_count = 0
        
    def reset(self):
        """重置环境"""
        self.prev_scanned_cells = self._count_scanned_cells()
        self.prev_position = None
        self.step_count = 0
        
        return self._get_state()
    
    def step(self, action):
        """
        执行一步
        
        :param action: [α1, α2, α3, α4, α5] - 5个权重系数
        :return: observation, reward, done, info
        """
        # 确保action在有效范围内（使用配置）
        action = np.clip(action, self.reward_config.weight_min, self.reward_config.weight_max)
        
        # 权重归一化：避免某个权重过高导致行为失衡
        # 方法1: 软归一化（保持相对比例，但限制最大差异）
        action_mean = np.mean(action)
        action_std = np.std(action)
        
        # 如果标准差过大，进行平滑（使用配置）
        if action_std > self.reward_config.std_threshold:
            # 将极端值拉回到均值附近
            action = action_mean + (action - action_mean) * self.reward_config.std_smoothing
            action = np.clip(action, self.reward_config.weight_min, self.reward_config.weight_max)
        
        # 方法2: 确保所有权重在合理范围内
        # 最大值不超过最小值的N倍（使用配置）
        min_weight = np.min(action)
        max_weight = np.max(action)
        if max_weight > min_weight * self.reward_config.max_min_ratio:
            # 缩放权重
            scale = (min_weight * self.reward_config.max_min_ratio) / max_weight
            action = action * scale
            action = np.clip(action, self.reward_config.weight_min, self.reward_config.weight_max)
        
        # 将权重设置到APF算法
        weights = {
            'repulsionCoefficient': float(action[0]),
            'entropyCoefficient': float(action[1]),
            'distanceCoefficient': float(action[2]),
            'leaderRangeCoefficient': float(action[3]),
            'directionRetentionCoefficient': float(action[4])
        }
        
        if self.server:
            self.server.algorithms[self.drone_name].set_coefficients(weights)
        
        # 获取新状态
        next_state = self._get_state()
        
        # 计算奖励
        reward = self._calculate_reward()
        
        # 判断是否结束（使用配置）
        self.step_count += 1
        done = self.step_count >= self.reward_config.max_steps
        
        # 额外信息
        info = {
            'weights': weights,
            'scanned_cells': self._count_scanned_cells()
        }
        
        return next_state, reward, done, info
    
    def _get_state(self):
        """获取当前状态（18维）"""
        if not self.server:
            # 如果没有server，返回随机状态（用于测试）
            return np.random.randn(18).astype(np.float32)
        
        try:
            with self.server.data_lock:
                runtime_data = self.server.unity_runtime_data[self.drone_name]
                grid_data = self.server.grid_data
                
                # 1. 位置 (3)
                pos = runtime_data.position
                position = [pos.x, pos.y, pos.z]
                
                # 2. 速度 (3)
                vel = runtime_data.finalMoveDir
                velocity = [
                    vel.x * self.server.config_data.moveSpeed,
                    vel.y * self.server.config_data.moveSpeed,
                    vel.z * self.server.config_data.moveSpeed
                ]
                
                # 3. 方向 (3)
                fwd = runtime_data.forward
                direction = [fwd.x, fwd.y, fwd.z]
                
                # 4. 附近熵值 (3)
                entropy_info = self._get_entropy_info(grid_data, pos)
                
                # 5. Leader相对位置 (3)
                if runtime_data.leader_position:
                    leader_rel = [
                        runtime_data.leader_position.x - pos.x,
                        runtime_data.leader_position.y - pos.y,
                        runtime_data.leader_position.z - pos.z
                    ]
                else:
                    leader_rel = [0.0, 0.0, 0.0]
                
                # 6. 扫描进度 (3)
                scan_info = self._get_scan_info(grid_data)
                
                # 组合状态
                state = position + velocity + direction + entropy_info + leader_rel + scan_info
                
                return np.array(state, dtype=np.float32)
                
        except Exception as e:
            print(f"获取状态失败: {str(e)}")
            return np.zeros(18, dtype=np.float32)
    
    def _calculate_reward(self):
        """计算奖励（使用配置文件中的系数）"""
        if not self.server:
            return 0.0
        
        reward = 0.0
        
        try:
            with self.server.data_lock:
                runtime_data = self.server.unity_runtime_data[self.drone_name]
                
                # 1. 探索奖励：新扫描的单元格（使用配置）
                current_scanned = self._count_scanned_cells()
                new_scanned = current_scanned - self.prev_scanned_cells
                reward += new_scanned * self.reward_config.exploration_reward
                self.prev_scanned_cells = current_scanned
                
                # 2. 碰撞惩罚（使用配置）
                min_dist = self._get_min_distance_to_others(runtime_data)
                if min_dist < self.reward_config.collision_distance:
                    reward -= self.reward_config.collision_penalty
                
                # 3. 越界惩罚（使用配置）
                if runtime_data.leader_position:
                    dist_to_leader = (runtime_data.position - runtime_data.leader_position).magnitude()
                    if runtime_data.leader_scan_radius > 0 and dist_to_leader > runtime_data.leader_scan_radius:
                        reward -= self.reward_config.out_of_range_penalty
                
                # 4. 平滑运动奖励（使用配置）
                if self.prev_position:
                    movement = (runtime_data.position - self.prev_position).magnitude()
                    if self.reward_config.movement_min < movement < self.reward_config.movement_max:
                        reward += self.reward_config.smooth_movement_reward
                
                self.prev_position = runtime_data.position
                
        except Exception as e:
            print(f"计算奖励失败: {str(e)}")
        
        return reward
    
    def _get_entropy_info(self, grid_data, position):
        """获取附近熵值信息（使用配置）"""
        if not grid_data or not grid_data.cells:
            return [50.0, 50.0, 0.0]
        
        # 找附近N米内的单元格（使用配置）
        nearby_cells = [
            cell for cell in grid_data.cells[:100]  # 限制数量避免卡顿
            if (cell.center - position).magnitude() < self.reward_config.nearby_entropy_distance
        ]
        
        if not nearby_cells:
            return [50.0, 50.0, 0.0]
        
        entropies = [cell.entropy for cell in nearby_cells]
        return [
            float(np.mean(entropies)),
            float(np.max(entropies)),
            float(np.std(entropies))
        ]
    
    def _get_scan_info(self, grid_data):
        """获取扫描进度（使用配置）"""
        if not grid_data or not grid_data.cells:
            return [0.0, 0.0, 0.0]
        
        total = len(grid_data.cells)
        scanned = sum(1 for cell in grid_data.cells if cell.entropy < self.reward_config.scanned_entropy_threshold)
        
        return [
            scanned / max(total, 1),
            float(scanned),
            float(total - scanned)
        ]
    
    def _count_scanned_cells(self):
        """统计已扫描单元格（使用配置）"""
        if not self.server or not self.server.grid_data:
            return 0
        
        try:
            with self.server.data_lock:
                return sum(1 for cell in self.server.grid_data.cells if cell.entropy < self.reward_config.scanned_entropy_threshold)
        except:
            return 0
    
    def _get_min_distance_to_others(self, runtime_data):
        """获取到其他无人机的最小距离"""
        if not runtime_data.otherScannerPositions:
            return 999.0
        
        distances = [
            (runtime_data.position - other_pos).magnitude()
            for other_pos in runtime_data.otherScannerPositions
        ]
        return min(distances) if distances else 999.0


# 测试代码
if __name__ == "__main__":
    print("测试SimpleWeightEnv...")
    
    # 创建环境（无server，用于测试）
    env = SimpleWeightEnv(server=None, drone_name="UAV1")
    
    print(f"观察空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    
    # 重置环境
    state = env.reset()
    print(f"初始状态shape: {state.shape}")
    print(f"初始状态: {state[:5]}...")
    
    # 执行几步
    for i in range(5):
        # 随机动作（5个权重）
        action = env.action_space.sample()
        print(f"\n步骤 {i+1}:")
        print(f"  动作(权重): {action}")
        
        state, reward, done, info = env.step(action)
        print(f"  奖励: {reward:.2f}")
        print(f"  完成: {done}")
    
    print("\n[OK] 环境测试通过！")

