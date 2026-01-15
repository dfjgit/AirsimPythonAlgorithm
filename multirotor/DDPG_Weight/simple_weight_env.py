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
    
    def __init__(self, server=None, drone_name="UAV1", reward_config_path=None, reset_unity=True, step_duration=5.0):
        super(SimpleWeightEnv, self).__init__()
        
        self.server = server
        self.drone_name = drone_name
        self.reset_unity = reset_unity  # 是否每次episode重置Unity环境
        self.step_duration = step_duration  # 每步飞行时长（秒）
        
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
        self.step_count = 0
        self.episode_count = 0  # 记录Episode编号
        self.last_action = np.zeros(5)  # 记录上一步的动作，用于电量消耗计算
        
    def reset(self):
        """重置环境"""
        import time
        import sys
        
        # Episode计数
        self.episode_count += 1
        
        print(f"\n{'='*60}")
        print(f"🔄 重置环境 - Episode #{self.episode_count}")
        print(f"{'='*60}")
        
        # 如果有server
        if self.server:
            # 重置电量数据
            if self.reset_unity:
                print(f"🔋 重置电量数据...")
                self.server.reset_battery_voltage(self.drone_name)
                print(f"  ✅ 电量已重置为4.2V")
            
            # 模式A：标准episode训练（重置Unity环境）
            if self.reset_unity:
                print(f"🎮 正在重置Unity环境...")
                self.server.reset_environment()
                
                # 等待重置完成
                for i in range(3):
                    sys.stdout.write(f"\r  ⏳ 等待重置... {'.' * (i+1)}   ")
                    sys.stdout.flush()
                    time.sleep(1)
                print(f"\r  ✅ Unity重置完成!     ")
            
            # 等待数据就绪
            print(f"\n📡 等待数据同步...")
            max_wait = 10
            wait_time = 0
            while wait_time < max_wait:
                has_grid = bool(self.server.grid_data.cells)
                has_runtime = bool(self.server.unity_runtime_data.get(self.drone_name))
                
                if has_grid and has_runtime:
                    grid_count = len(self.server.grid_data.cells)
                    print(f"✅ 数据就绪！")
                    print(f"  🗺️  网格单元: {grid_count} 个")
                    print(f"  🚁 无人机: {self.drone_name}")
                    break
                
                dots = '.' * (int(wait_time * 2) % 4)
                sys.stdout.write(f"\r  等待数据{dots}    ")
                sys.stdout.flush()
                time.sleep(0.5)
                wait_time += 0.5
            
            if wait_time >= max_wait:
                print(f"\r  ⚠️  等待数据超时     ")
        
        # 重置内部状态
        if self.reset_unity:
            self.prev_scanned_cells = 0
        else:
            if self.server:
                with self.server.data_lock:
                    self.prev_scanned_cells = self._count_scanned_cells()
            else:
                self.prev_scanned_cells = 0
        
        self.step_count = 0
        self.last_action = np.zeros(5)
        
        state = self._get_state()
        
        # 显示电量信息
        if self.server:
            current_voltage = self.server.get_battery_voltage(self.drone_name)
            print(f"🔋 当前电量: {current_voltage:.2f}V")
        
        print(f"\n{'='*60}")
        print(f"🎯 开始 Episode #{self.episode_count}")
        print(f"{'='*60}")
        print(f"📊 配置:")
        print(f"  • Episode编号: #{self.episode_count}")
        print(f"  • 最大步数: {self.reward_config.max_steps}")
        print(f"  • 每步时长: {self.step_duration}秒")
        print(f"  • 预计时长: {self.reward_config.max_steps * self.step_duration / 60:.1f}分钟")
        print(f"{'='*60}\n")
        
        return state
    
    def step(self, action):
        """
        执行一步
        
        :param action: [α1, α2, α3, α4, α5] - 5个权重系数
        :return: observation, reward, done, info
        """
        import time
        import sys
        
        # 确保action在有效范围内
        action = np.clip(action, self.reward_config.weight_min, self.reward_config.weight_max)
        
        # 将权重设置到APF算法
        weights = {
            'repulsionCoefficient': float(action[0]),
            'entropyCoefficient': float(action[1]),
            'distanceCoefficient': float(action[2]),
            'leaderRangeCoefficient': float(action[3]),
            'directionRetentionCoefficient': float(action[4])
        }
        
        # 打印当前步骤信息
        self.step_count += 1
        progress_percent = (self.step_count / self.reward_config.max_steps) * 100
        
        print(f"\n{'─'*60}")
        print(f"🔄 步骤 {self.step_count}/{self.reward_config.max_steps} ({progress_percent:.1f}%)")
        print(f"{'─'*60}")
        print(f"📊 设置权重:")
        print(f"  • 斥力系数: {weights['repulsionCoefficient']:.3f}")
        print(f"  • 熵系数:   {weights['entropyCoefficient']:.3f}")
        print(f"  • 距离系数: {weights['distanceCoefficient']:.3f}")
        print(f"  • Leader:   {weights['leaderRangeCoefficient']:.3f}")
        print(f"  • 方向保持: {weights['directionRetentionCoefficient']:.3f}")
        
        # 在 step() 方法中
        if self.server:
            # 更新电量消耗（使用新的电量模块）
            if self.step_count > 1:
                action_intensity = np.linalg.norm(action - self.last_action)
                self.server.battery_manager.update_voltage(self.drone_name, action_intensity)
            
            # 显示当前电量
            battery_info = self.server.battery_manager.get_battery_info(self.drone_name)
            current_voltage = battery_info.voltage
            print(f"🔋 当前电量: {current_voltage:.2f}V ({battery_info.get_remaining_percentage():.1f}%)")
            
            # 设置权重（算法线程会使用新权重飞行）
            self.server.algorithms[self.drone_name].set_coefficients(weights)
            
            # 倒计时等待无人机飞行
            print(f"\n⏱️  等待无人机飞行 {self.step_duration:.0f} 秒...")
            
            # 使用倒计时显示
            for remaining in range(int(self.step_duration), 0, -1):
                elapsed = self.step_duration - remaining
                bar_length = 40
                filled = int((elapsed / self.step_duration) * bar_length)
                bar = '█' * filled + '░' * (bar_length - filled)
                
                sys.stdout.write(f"\r  [{bar}] {remaining:2d}秒剩余  ")
                sys.stdout.flush()
                time.sleep(1)
            
            print(f"\r  [{'█'*40}] ✅ 完成!     ")
        else:
            time.sleep(0.1)  # 测试模式快速跳过
        
        # 记录当前动作
        self.last_action = action.copy()
        
        # 获取新状态
        next_state = self._get_state()
        
        # 计算奖励
        reward = self._calculate_reward()
        
        # 判断是否结束
        done = self.step_count >= self.reward_config.max_steps
        
        # 显示奖励信息
        print(f"\n📈 本步奖励: {reward:+.2f}")
        
        if self.server:
            with self.server.data_lock:
                grid_data = self.server.grid_data
                if grid_data and grid_data.cells:
                    total_cells = len(grid_data.cells)
                    scanned_cells = sum(1 for cell in grid_data.cells if cell.entropy < 30)
                    scan_progress = (scanned_cells / total_cells) * 100
                    print(f"🗺️  扫描进度: {scanned_cells}/{total_cells} ({scan_progress:.1f}%)")
        
        if done:
            print(f"\n{'='*60}")
            print(f"✅ Episode #{self.episode_count} 完成！共 {self.step_count} 步")
            print(f"{'='*60}")
            print(f"🔄 即将自动重置环境，开始下一个Episode...")
            print(f"{'='*60}\n")
        
        # 额外信息
        info = {
            'weights': weights,
            'scanned_cells': self.prev_scanned_cells
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
        """计算奖励（包含电量奖励机制）"""
        if not self.server:
            return 0.0
        
        reward = 0.0
        
        try:
            with self.server.data_lock:
                runtime_data = self.server.unity_runtime_data[self.drone_name]
                grid_data = self.server.grid_data
                
                # 1. 探索奖励：新扫描的单元格
                current_scanned = sum(1 for cell in grid_data.cells if cell.entropy < 30) if grid_data.cells else 0
                new_scanned = current_scanned - self.prev_scanned_cells
                reward += new_scanned * self.reward_config.exploration_reward
                self.prev_scanned_cells = current_scanned
                
                # 2. 越界惩罚
                if runtime_data.leader_position:
                    dist_to_leader = (runtime_data.position - runtime_data.leader_position).magnitude()
                    if runtime_data.leader_scan_radius > 0 and dist_to_leader > runtime_data.leader_scan_radius:
                        reward -= self.reward_config.out_of_range_penalty
                
                # 3. 电量奖励机制
                current_voltage = self.server.get_battery_voltage(self.drone_name)
                
                # 最优电量范围奖励 (3.7V - 4.0V)
                if 3.7 <= current_voltage <= 4.0:
                    reward += self.reward_config.battery_optimal_reward
                    print(f"🔋 电量奖励: +{self.reward_config.battery_optimal_reward:.2f} (电量{current_voltage:.2f}V在最优范围)")
                
                # 低电量惩罚 (低于3.5V)
                elif current_voltage < 3.5:
                    reward -= self.reward_config.battery_low_penalty
                    print(f"🔋 电量惩罚: -{self.reward_config.battery_low_penalty:.2f} (电量{current_voltage:.2f}V过低)")
                
                # 4. 动作平稳奖励（减少剧烈动作）
                if self.step_count > 1:
                    action_intensity = np.linalg.norm(self.last_action)
                    if action_intensity < 0.5:  # 动作强度小于0.5时给予奖励
                        reward += self.reward_config.action_smooth_reward
                        print(f"🔄 平稳动作奖励: +{self.reward_config.action_smooth_reward:.2f}")
                
        except Exception as e:
            print(f"[错误] 计算奖励失败: {str(e)}")
        
        return reward
    
    def _get_entropy_info(self, grid_data, position):
        """获取附近熵值信息"""
        if not grid_data or not grid_data.cells:
            return [50.0, 50.0, 0.0]
        
        # 找附近10米内的单元格
        nearby_cells = [
            cell for cell in grid_data.cells[:100]
            if (cell.center - position).magnitude() < 10.0
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
        """获取扫描进度"""
        if not grid_data or not grid_data.cells:
            return [0.0, 0.0, 0.0]
        
        total = len(grid_data.cells)
        scanned = sum(1 for cell in grid_data.cells if cell.entropy < 30)
        
        return [
            scanned / max(total, 1),
            float(scanned),
            float(total - scanned)
        ]
    
    def _count_scanned_cells(self):
        """统计已扫描单元格（不加锁版本，由调用者加锁）"""
        if not self.server or not self.server.grid_data:
            return 0
        
        try:
            # 注意：不在这里加锁，避免嵌套锁
            # 调用者应该已经持有data_lock
            return sum(1 for cell in self.server.grid_data.cells if cell.entropy < 30)
        except:
            return 0


# 测试代码
if __name__ == "__main__":
    print("测试SimpleWeightEnv...")
    
    # 测试两种模式
    print("\n[模式A] 标准episode训练:")
    env_a = SimpleWeightEnv(server=None, drone_name="UAV1", reset_unity=True)
    print(f"  观察空间: {env_a.observation_space.shape}")
    print(f"  动作空间: {env_a.action_space.shape}")
    
    print("\n[模式B] 连续学习:")
    env_b = SimpleWeightEnv(server=None, drone_name="UAV1", reset_unity=False)
    print(f"  观察空间: {env_b.observation_space.shape}")
    print(f"  动作空间: {env_b.action_space.shape}")
    
    print("\n[OK] 两种模式都可用！")