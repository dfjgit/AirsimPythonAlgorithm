"""
简单的权重学习环境
使用Stable-Baselines3训练APF权重系数
"""
import numpy as np
import gym
from gym import spaces
import os

try:
    from configs.crazyflie_reward_config import CrazyflieRewardConfig
except ImportError:
    try:
        from ..configs.crazyflie_reward_config import CrazyflieRewardConfig
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from configs.crazyflie_reward_config import CrazyflieRewardConfig


class SimpleWeightEnv(gym.Env):
    """
    简单的APF权重学习环境
    
    目标: 学习5个权重系数 (α1, α2, α3, α4, α5)
    """
    
    def __init__(
        self,
        server=None,
        drone_name="UAV1",
        reward_config_path=None,
        reset_unity=True,
        step_duration=5.0,
        safety_limit=True,
        max_weight_delta=0.5
    ):
        super(SimpleWeightEnv, self).__init__()
        
        self.server = server
        self.drone_name = drone_name
        self.reset_unity = reset_unity  # 是否每次episode重置Unity环境
        self.step_duration = step_duration  # 每步飞行时长（秒）
        
        # 加载奖励配置（与实体训练一致）
        if reward_config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            reward_config_path = os.path.join(current_dir, "..", "configs", "crazyflie_reward_config.json")

        self.reward_config = CrazyflieRewardConfig(reward_config_path)
        
        # 统一终止配置
        self.term_cfg = {
            "target_scan_ratio": 0.95,
            "max_collision_count": 1,
            "max_elapsed_time_sec": 300.0
        }
        
        # 应用统一环境配置（方案 B：解耦物理规则与训练参数）
        self._apply_unified_config()
            
        print("[OK] 训练环境已加载奖励配置和终止配置")
        
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
        self.collision_count = 0  # 新增碰撞计数
        self.episode_count = 0  # 记录Episode编号
        self.total_episode_reward = 0.0  # 记录当前Episode的总奖励
        self.last_action = np.zeros(5)  # 记录上一步的动作，用于电量消耗计算
        self.prev_velocity = np.zeros(3, dtype=np.float32)
        self.prev_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.safety_limit = safety_limit
        self.max_weight_delta = max_weight_delta
        self._has_initial_action = False
        
        # 首次重置标志（用于跳过启动时的物理重置）
        self._first_reset = True
        
    def _first_reset_logic(self):
        """处理首次重置逻辑"""
        pass # Placeholder if needed, or just keep as is in reset()

    def _apply_unified_config(self):
        """应用统一环境配置，确保物理规则一致性（方案 B）"""
        unified_env_cfg = None
        
        # 1. 优先从 server 获取 (AlgorithmServer 持有最新的 ScannerConfigData)
        if self.server and hasattr(self.server, 'config_data') and hasattr(self.server.config_data, 'env_config'):
            unified_env_cfg = self.server.config_data.env_config
        
        # 2. 如果没有 server，尝试从本地 scanner_config.json 加载
        if unified_env_cfg is None:
            try:
                # 寻找根目录下的 scanner_config.json
                # 当前文件在 multirotor/DDPG_Weight/envs/，根目录在 multirotor/
                config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scanner_config.json")
                if os.path.exists(config_path):
                    import json
                    with open(config_path, 'r', encoding='utf-8') as f:
                        full_cfg = json.load(f)
                        unified_env_cfg = full_cfg.get('env_config')
            except Exception as e:
                print(f"[Warning] 加载本地统一配置失败: {e}")

        if unified_env_cfg:
            print(f"[UnifiedConfig] 正在应用统一环境配置于 {self.__class__.__name__}...")
            
            # 覆盖终止条件
            if 'termination' in unified_env_cfg:
                self.term_cfg.update(unified_env_cfg['termination'])
                # DDPG 环境可能没有 time_limit，我们将其映射到 step_count
                # max_steps = max_elapsed_time_sec / step_duration
                if 'max_elapsed_time_sec' in unified_env_cfg['termination'] and self.step_duration > 0:
                    calculated_steps = int(unified_env_cfg['termination']['max_elapsed_time_sec'] / self.step_duration)
                    self.reward_config.max_steps = calculated_steps
                print(f"  • 终止条件已更新 (MaxSteps={self.reward_config.max_steps})")

            # 覆盖电量阈值
            if 'battery' in unified_env_cfg:
                b_cfg = unified_env_cfg['battery']
                self.reward_config.battery_low_threshold = float(b_cfg.get('low_threshold', self.reward_config.battery_low_threshold))
                self.reward_config.battery_optimal_min = float(b_cfg.get('optimal_min', self.reward_config.battery_optimal_min))
                self.reward_config.battery_optimal_max = float(b_cfg.get('optimal_max', self.reward_config.battery_optimal_max))
                print(f"  • 电量阈值已对齐: Low<{self.reward_config.battery_low_threshold}V")

            # 覆盖基础奖励系数
            if 'base_rewards' in unified_env_cfg:
                base_rewards = unified_env_cfg['base_rewards']
                reward_map = {
                    'scan_reward': 'scan_reward',
                    'out_of_range_penalty': 'out_of_range_penalty',
                    'battery_low_penalty': 'battery_low_penalty',
                    'battery_optimal_reward': 'battery_optimal_reward'
                }
                for u_key, local_attr in reward_map.items():
                    if u_key in base_rewards:
                        val = abs(float(base_rewards[u_key]))
                        setattr(self.reward_config, local_attr, val)
                print(f"  • 奖励系数已对齐: Scan={self.reward_config.scan_reward}")

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
            # 重置所有虚拟无人机的电量数据（每个 Episode 都需要）
            print(f"🔋 重置电量数据...")
            for drone_name in self.server.drone_names:
                self.server.reset_battery_voltage(drone_name)
            print(f"  ✅ 所有无人机电量已重置为4.2V")
            
            # 首次重置：跳过物理重置（因为无人机已通过 start_mission() 起飞）
            if self._first_reset:
                self._first_reset = False
                print(f"🚀 首次reset，跳过Unity物理重置，直接初始化状态")
                print(f"💡 无人机已通过 start_mission() 启动，继续使用当前飞行状态")
            else:
                # 后续 Episode：执行完整的物理重置（如果启用）
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
        # 物理 reset 后扫描进度从零开始，否则继承当前扫描进度（连续训练模式）
        # 注意：首次 reset 时 _first_reset 已经变为 False，但 episode_count=1，视为物理重置
        if self.episode_count == 1 or (self.reset_unity and not self._first_reset):
            self.prev_scanned_cells = 0
        else:
            if self.server:
                with self.server.data_lock:
                    self.prev_scanned_cells = self._count_scanned_cells()
            else:
                self.prev_scanned_cells = 0
        
        self.step_count = 0
        self.collision_count = 0 # 重置碰撞计数
        self.total_episode_reward = 0.0
        self.last_action = np.zeros(5)
        self.prev_velocity = np.zeros(3, dtype=np.float32)
        self.prev_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self._has_initial_action = False
        
        state = self._get_state()
        
        # 显示所有无人机的电量信息
        if self.server:
            print(f"🔋 电量状态:")
            for drone_name in self.server.drone_names:
                current_voltage = self.server.get_battery_voltage(drone_name)
                battery_info = self.server.battery_manager.get_battery_info(drone_name)
                if battery_info:
                    print(f"  • {drone_name}: {current_voltage:.2f}V ({battery_info.get_remaining_percentage():.1f}%)")
                else:
                    print(f"  • {drone_name}: {current_voltage:.2f}V")
        
        print(f"\n{'='*60}")
        print(f"🎯 开始 Episode #{self.episode_count}")
        print(f"{'='*60}")
        print(f"📊 配置:")
        print(f"  • Episode编号: #{self.episode_count}")
        print(f"  • 最大步数: {self.reward_config.max_steps}")
        print(f"  • 每步时长: {self.step_duration}秒")
        print(f"  • 预计时长: {self.reward_config.max_steps * self.step_duration / 60:.1f}分钟")
        print(f"{'='*60}\n")
        
        # 通知服务器 Episode 切换 (用于数据采集及时记录上一个 Episode)
        if self.server:
            self.server.set_training_stats(
                episode=self.episode_count,
                step=0,
                reward=0.0,
                total_reward=0.0
            )
        
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
        if self.safety_limit and (self.step_count > 0 or self._has_initial_action):
            action = np.clip(
                action,
                self.last_action - self.max_weight_delta,
                self.last_action + self.max_weight_delta
            )
            action = np.clip(action, self.reward_config.weight_min, self.reward_config.weight_max)
        self._has_initial_action = False
        
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
            # 更新所有虚拟无人机的电量消耗
            if self.step_count > 1:
                with self.server.data_lock:
                    for drone_name in self.server.drone_names:
                        # 获取无人机的运行时数据
                        runtime_data = self.server.unity_runtime_data.get(drone_name)
                        if runtime_data:
                            # 使用实际速度大小作为动作强度
                            # finalMoveDir 是方向向量，乘以 moveSpeed 得到实际速度
                            move_dir = runtime_data.finalMoveDir
                            move_speed = self.server.config_data.moveSpeed
                            speed_magnitude = np.sqrt(move_dir.x**2 + move_dir.y**2 + move_dir.z**2) * move_speed
                            
                            # 归一化到 0-1 范围（假设最大速度为 moveSpeed）
                            action_intensity = min(1.0, speed_magnitude / max(move_speed, 0.1))
                            self.server.battery_manager.update_voltage(drone_name, action_intensity)
            
            # 显示所有无人机的当前电量
            print(f"🔋 电量状态:")
            for drone_name in self.server.drone_names:
                battery_info = self.server.battery_manager.get_battery_info(drone_name)
                if battery_info:
                    current_voltage = battery_info.voltage
                    print(f"  • {drone_name}: {current_voltage:.2f}V ({battery_info.get_remaining_percentage():.1f}%)")
            
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
        
        # 获取新状态
        next_state = self._get_state()
        
        # 计算奖励
        reward = self._calculate_reward(action)
        self.total_episode_reward += reward
        
        # 更新碰撞计数（基于状态中的距离或服务器数据）
        if self.server:
            with self.server.data_lock:
                rd = self.server.unity_runtime_data.get(self.drone_name)
                if rd:
                    min_dist = self._get_min_distance_to_others(rd)
                    if min_dist < 2.0: # 碰撞阈值
                        self.collision_count += 1

        # 将训练统计信息传递给服务器（用于数据采集）
        if self.server:
            self.server.set_training_stats(
                episode=self.episode_count,
                step=self.step_count,
                reward=float(reward),
                total_reward=float(self.total_episode_reward)
            )
        
        # 记录当前动作
        self.last_action = action.copy()
        
        # 判断是否结束 (统一终止逻辑)
        elapsed_time = self.step_count * self.step_duration
        done = False
        
        if elapsed_time >= self.term_cfg['max_elapsed_time_sec']:
            print(f"[终止] 达到最大仿真时间: {elapsed_time:.1f}s")
            done = True
        elif self.collision_count >= self.term_cfg['max_collision_count']:
            print(f"[终止] 发生碰撞: {self.collision_count}")
            done = True
        else:
            # 检查覆盖率
            if self.server:
                with self.server.data_lock:
                    total_cells = len(self.server.grid_data.cells)
                    if total_cells > 0:
                        scanned_cells = sum(1 for cell in self.server.grid_data.cells 
                                          if cell.entropy < self.reward_config.scan_entropy_threshold)
                        scan_ratio = scanned_cells / total_cells
                        if scan_ratio >= self.term_cfg['target_scan_ratio']:
                            print(f"[终止] 覆盖率达成: {scan_ratio:.2%}")
                            done = True
        
        # 显示奖励信息
        print(f"\n📈 本步奖励: {reward:+.2f}")
        
        if self.server:
            with self.server.data_lock:
                grid_data = self.server.grid_data
                if grid_data and grid_data.cells:
                    total_cells = len(grid_data.cells)
                    scanned_cells = sum(
                        1 for cell in grid_data.cells
                        if cell.entropy < self.reward_config.scan_entropy_threshold
                    )
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
    
    def _get_min_distance_to_others(self, rd):
        """获取到其他无人机的最小距离"""
        if not rd or not rd.otherScannerPositions:
            return 999.0
        pos = rd.position
        dists = [np.sqrt((pos.x-op.x)**2 + (pos.y-op.y)**2 + (pos.z-op.z)**2) for op in rd.otherScannerPositions]
        return min(dists) if dists else 999.0

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
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """计算奖励（尽量与实体奖励结构一致）"""
        if not self.server:
            return 0.0

        reward = 0.0

        try:
            with self.server.data_lock:
                runtime_data = self.server.unity_runtime_data[self.drone_name]
                grid_data = self.server.grid_data

            # 1. 速度奖励与超速惩罚
            vel = runtime_data.finalMoveDir
            current_velocity = np.array(
                [vel.x, vel.y, vel.z],
                dtype=np.float32
            ) * float(self.server.config_data.moveSpeed)
            speed = float(np.linalg.norm(current_velocity))
            reward += self.reward_config.speed_reward * speed
            if speed > self.reward_config.speed_penalty_threshold:
                reward -= self.reward_config.speed_penalty

            # 2. 加速度惩罚（速度变化近似）
            if self.step_duration > 0:
                accel_mag = float(np.linalg.norm(current_velocity - self.prev_velocity) / self.step_duration)
            else:
                accel_mag = float(np.linalg.norm(current_velocity - self.prev_velocity))
            reward -= self.reward_config.accel_penalty * accel_mag

            # 3. 角速度惩罚（方向变化近似）
            fwd = runtime_data.forward
            current_direction = np.array([fwd.x, fwd.y, fwd.z], dtype=np.float32)
            current_norm = np.linalg.norm(current_direction)
            prev_norm = np.linalg.norm(self.prev_direction)
            if current_norm > 1e-6 and prev_norm > 1e-6:
                dot = float(np.clip(np.dot(current_direction, self.prev_direction) / (current_norm * prev_norm), -1.0, 1.0))
                angle = float(np.arccos(dot))
                angular_rate = angle / self.step_duration if self.step_duration > 0 else angle
                reward -= self.reward_config.angular_rate_penalty * angular_rate

            # 4. 扫描奖励
            current_scanned = 0
            if grid_data and grid_data.cells:
                current_scanned = sum(
                    1 for cell in grid_data.cells
                    if cell.entropy < self.reward_config.scan_entropy_threshold
                )
            new_scanned = current_scanned - self.prev_scanned_cells
            if new_scanned > 0:
                reward += self.reward_config.scan_reward * new_scanned
            self.prev_scanned_cells = current_scanned

            # 5. 越界惩罚（Leader范围）
            if runtime_data.leader_position and runtime_data.leader_scan_radius > 0:
                dist_to_leader = (runtime_data.position - runtime_data.leader_position).magnitude()
                leader_radius = runtime_data.leader_scan_radius + self.reward_config.leader_range_buffer
                if dist_to_leader > leader_radius:
                    reward -= self.reward_config.out_of_range_penalty

            # 6. 动作变化与幅度惩罚
            action_delta = float(np.linalg.norm(action - self.last_action))
            reward -= self.reward_config.action_change_penalty * action_delta
            reward -= self.reward_config.action_magnitude_penalty * float(np.linalg.norm(action))

            # 7. 电量奖励机制
            current_voltage = self.server.get_battery_voltage(self.drone_name)
            if self.reward_config.battery_optimal_min <= current_voltage <= self.reward_config.battery_optimal_max:
                reward += self.reward_config.battery_optimal_reward
                print(f"🔋 电量奖励: +{self.reward_config.battery_optimal_reward:.2f} (电量{current_voltage:.2f}V在最优范围)")
            elif current_voltage < self.reward_config.battery_low_threshold:
                reward -= self.reward_config.battery_low_penalty
                print(f"🔋 电量惩罚: -{self.reward_config.battery_low_penalty:.2f} (电量{current_voltage:.2f}V过低)")

            # 更新历史速度/方向
            self.prev_velocity = current_velocity
            if current_norm > 1e-6:
                self.prev_direction = current_direction

        except Exception as e:
            print(f"[错误] 计算奖励失败: {str(e)}")

        return reward
    
    def _get_entropy_info(self, grid_data, position):
        """获取附近熵值信息"""
        if not grid_data or not grid_data.cells:
            return [0.0, 0.0, 0.0]
        
        # 找附近10米内的单元格
        nearby_cells = [
            cell for cell in grid_data.cells[:100]
            if (cell.center - position).magnitude() < 10.0
        ]
        
        if not nearby_cells:
            return [0.0, 0.0, 0.0]
        
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
        scanned = sum(
            1 for cell in grid_data.cells
            if cell.entropy < self.reward_config.scan_entropy_threshold
        )
        
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
            return sum(
                1 for cell in self.server.grid_data.cells
                if cell.entropy < self.reward_config.scan_entropy_threshold
            )
        except:
            return 0

    def set_initial_action(self, weights: np.ndarray) -> None:
        """设置初始动作权重，用于与实体训练对齐安全裁剪"""
        if weights is None:
            return
        weights = np.array(weights, dtype=np.float32)
        if weights.shape[0] != 5:
            return
        weights = np.clip(weights, self.reward_config.weight_min, self.reward_config.weight_max)
        self.last_action = weights.copy()
        self._has_initial_action = True


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