"""
无人机移动环境 - DQN训练
使用离散动作空间（6方向位移）直接控制无人机移动
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import json
import logging

# 配置日志
logger = logging.getLogger("MovementEnv")


class MovementEnv(gym.Env):
    """
    无人机移动学习环境
    
    动作空间: 6个离散动作（上/下/左/右/前/后）
    观察空间: 位置、速度、熵值、leader位置等
    """
    
    def __init__(self, server=None, drone_name="UAV1", config_path=None):
        super(MovementEnv, self).__init__()
        
        self.server = server
        self.drone_name = drone_name
        
        # 加载配置
        self.config = self._load_config(config_path)
        print(f"[OK] 移动DQN环境已加载配置")
        
        # 动作空间: 6个离散动作
        # 0: 向上, 1: 向下, 2: 向左, 3: 向右, 4: 向前, 5: 向后
        self.action_space = spaces.Discrete(6)
        
        # 观察空间维度说明：
        # - 位置(3): x, y, z
        # - 速度(3): vx, vy, vz
        # - 朝向(3): forward_x, forward_y, forward_z
        # - 局部熵值统计(3): 平均熵, 最大熵, 熵标准差
        # - Leader相对位置(3): dx, dy, dz
        # - Leader范围信息(2): 距离, 是否越界
        # - 扫描进度(3): 已扫描比例, 已扫描数量, 未扫描数量
        # - 其他无人机最近距离(1)
        # - 电量信息(2): 当前电压, 剩余电量百分比
        # 总计: 23维
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(23,),
            dtype=np.float32
        )
        
        # 动作到位移的映射（单位：米）
        self.action_step = self.config['movement']['step_size']
        self.action_map = {
            0: np.array([0, 0, self.action_step]),      # 上
            1: np.array([0, 0, -self.action_step]),     # 下
            2: np.array([-self.action_step, 0, 0]),     # 左
            3: np.array([self.action_step, 0, 0]),      # 右
            4: np.array([0, self.action_step, 0]),      # 前
            5: np.array([0, -self.action_step, 0])      # 后
        }
        
        # 状态记录
        self.prev_scanned_cells = 0
        self.prev_position = None
        self.prev_entropy_sum = 0
        self.step_count = 0
        self.episode_reward = 0
        
        self.collision_count = 0
        self.out_of_range_count = 0
        
        # 首次重置标志（用于跳过启动时的重置）
        self._first_reset = True
    
    def _load_config(self, config_path):
        """加载配置文件"""
        if config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, "movement_dqn_config.json")
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 返回默认配置
            return self._default_config()
    
    def _default_config(self):
        """默认配置"""
        return {
            "movement": {
                "step_size": 1.0,
                "max_steps": 500
            },
            "rewards": {
                "exploration": 10.0,
                "collision": -50.0,
                "out_of_range": -30.0,
                "smooth_movement": 1.0,
                "entropy_reduction": 5.0,
                "high_entropy_exploration": 5.0,
                "entropy_gradient_bonus": 2.0,
                "step_penalty": -0.1,
                "success": 100.0,
                "height_penalty": -5.0,
                "optimal_height_bonus": 1.0
            },
            "thresholds": {
                "collision_distance": 2.0,
                "scanned_entropy": 30.0,
                "nearby_entropy_distance": 10.0,
                "success_scan_ratio": 0.95,
                "high_entropy_threshold": 40.0,
                "min_scan_height": 2.0,
                "max_scan_height": 15.0,
                "optimal_scan_height": 8.0
            }
        }
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        print(f"[DQN环境] reset() 被调用")
        if seed is not None:
            np.random.seed(seed)
        
        # 首次重置：跳过环境重置（因为无人机刚起飞，领导者刚开始移动）
        if self._first_reset:
            self._first_reset = False
            print(f"[DQN环境] 🚀 首次reset，跳过环境重置，直接初始化状态")
            # 仅重置电量（确保电量从满电开始）
            if self.server and hasattr(self.server, 'reset_battery_voltage'):
                self.server.reset_battery_voltage(self.drone_name)
        else:
            # 后续重置：执行完整的环境重置（Episode结束）
            if self.server:
                print(f"[DQN环境] 🔄 Episode结束，执行完整环境重置...")
                self.server.reset_environment()
                # 重置电量
                if hasattr(self.server, 'reset_battery_voltage'):
                    self.server.reset_battery_voltage(self.drone_name)
                import time
                time.sleep(1.0)  # 等待Unity完成重置
                print(f"[DQN环境] ✅ 环境重置完成")
        
        print(f"[DQN环境] 初始化状态...")
        self.prev_scanned_cells = self._count_scanned_cells()
        self.prev_entropy_sum = self._get_total_entropy()
        self.prev_position = None
        self.step_count = 0
        self.episode_reward = 0
        self.collision_count = 0
        self.out_of_range_count = 0
        
        print(f"[DQN环境] 初始化信息:")
        print(f"  - 初始扫描数: {self.prev_scanned_cells}")
        print(f"  - 初始总熄值: {self.prev_entropy_sum:.2f}")
        
        print(f"[DQN环境] 获取初始状态...")
        state = self._get_state()
        print(f"[DQN环境] reset() 完成，状态shape: {state.shape}")
        return state, {}
    
    def step(self, action):
        """
        执行一步动作
        
        :param action: 0-5的整数，表示6个移动方向
        :return: observation, reward, terminated, truncated, info
        """
        # 确保action是整数（从numpy数组转换）
        if hasattr(action, 'item'):
            action = action.item()
        action = int(action)
        
        print(f"[DQN环境] step({action}) 被调用")
        
        # 记录当前位置
        print(f"[DQN环境] 获取当前状态...")
        current_state = self._get_state()
        print(f"[DQN环境] 当前状态获取完成")
        
        # 将动作转换为位移向量
        displacement = self.action_map[action]
        
        # 发送移动指令到server（如果连接）
        if self.server:
            print(f"[DQN环境] 发送移动指令: {displacement}")
            self._apply_movement(displacement)
            print(f"[DQN环境] 移动指令已发送")
        
        # 等待一小段时间让环境更新
        if self.server:
            import time
            time.sleep(0.05)  # 50ms
            print(f"[DQN环境] 等待完成")
        
        # 获取新状态
        print(f"[DQN环境] 获取新状态...")
        next_state = self._get_state()
        print(f"[DQN环境] 新状态获取完成")
        
        # 计算奖励
        print(f"[DQN环境] 计算奖励...")
        reward = self._calculate_reward(action, current_state, next_state)
        self.episode_reward += reward
        print(f"[DQN环境] 奖励计算完成: {reward:.2f}")
        
        # 判断是否结束
        self.step_count += 1
        terminated = self._check_done()  # episode自然结束
        truncated = False  # 不使用截断
        
        # 额外信息
        info = {
            'action': action,
            'displacement': displacement.tolist(),
            'scanned_cells': self._count_scanned_cells(),
            'collision_count': self.collision_count,
            'out_of_range_count': self.out_of_range_count,
            'episode_reward': self.episode_reward
        }
        
        if self.step_count % 10 == 0:
            print(f"[DQN环境] 步骤 {self.step_count}, 奖励: {reward:.2f}, episode总奖励: {self.episode_reward:.2f}")
        
        return next_state, reward, terminated, truncated, info
    
    def _get_state(self):
        """获取当前观察状态（23维：包含电量信息）"""
        if not self.server:
            # 测试模式：返回随机状态
            return np.random.randn(23).astype(np.float32)
            
        try:
            # 分离锁的获取，避免死锁
            # 第一步：从 runtime_data 获取无人机状态
            with self.server.data_lock:
                runtime_data = self.server.unity_runtime_data.get(self.drone_name)
                if not runtime_data:
                    print(f"[DQN环境] 警告: 无人机 {self.drone_name} 的runtime_data不存在，返回零状态")
                    return np.zeros(23, dtype=np.float32)
                    
                # 1. 位置 (3)
                pos = runtime_data.position
                position = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
                    
                # 2. 速度 (3)
                vel = runtime_data.finalMoveDir
                velocity = np.array([
                    vel.x * self.server.config_data.moveSpeed,
                    vel.y * self.server.config_data.moveSpeed,
                    vel.z * self.server.config_data.moveSpeed
                ], dtype=np.float32)
                    
                # 3. 朝向 (3)
                fwd = runtime_data.forward
                direction = np.array([fwd.x, fwd.y, fwd.z], dtype=np.float32)
                    
                # 5. Leader相对位置 (3)
                if runtime_data.leader_position:
                    leader_rel = np.array([
                        runtime_data.leader_position.x - pos.x,
                        runtime_data.leader_position.y - pos.y,
                        runtime_data.leader_position.z - pos.z
                    ], dtype=np.float32)
                else:
                    leader_rel = np.zeros(3, dtype=np.float32)
                    
                # 6. Leader范围信息 (2)
                if runtime_data.leader_position and runtime_data.leader_scan_radius > 0:
                    dist_to_leader = np.linalg.norm(leader_rel)
                    is_out_of_range = 1.0 if dist_to_leader > runtime_data.leader_scan_radius else 0.0
                    leader_range = np.array([dist_to_leader, is_out_of_range], dtype=np.float32)
                else:
                    leader_range = np.zeros(2, dtype=np.float32)
                
            # 第二步：从 grid_data 获取网格信息
            with self.server.grid_lock:
                grid_data = self.server.grid_data
                if not grid_data or not grid_data.cells:
                    print(f"[DQN环境] 警告: grid_data 为空或没有cells，返回零状态")
                    # 返回带有基本信息的状态
                    entropy_info = np.array([50.0, 50.0, 0.0], dtype=np.float32)
                    scan_info = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    min_dist_array = np.array([100.0], dtype=np.float32)
                else:
                    # 4. 局部熙值统计 (3)
                    entropy_info = self._get_entropy_info(grid_data, pos)
                        
                    # 7. 扫描进度 (3)
                    scan_info = self._get_scan_info(grid_data)
                        
                    # 8. 最近无人机距离 (1) - 需要 runtime_data，稍后单独获取
                    min_dist_array = np.array([100.0], dtype=np.float32)  # 默认值
                
            # 第三步：获取其他无人机距离（需要重新获取 data_lock）
            with self.server.data_lock:
                runtime_data = self.server.unity_runtime_data.get(self.drone_name)
                if runtime_data:
                    min_dist = self._get_min_distance_to_others(runtime_data)
                    min_dist_array = np.array([min_dist], dtype=np.float32)
                
            # 获取电量信息 (2)
            battery_info = self._get_battery_info()
            
            # 组合状态向量
            state = np.concatenate([
                position,           # 3
                velocity,           # 3
                direction,          # 3
                entropy_info,       # 3
                leader_rel,         # 3
                leader_range,       # 2
                scan_info,          # 3
                min_dist_array,     # 1
                battery_info        # 2
            ])
                
            return state
                
        except Exception as e:
            print(f"获取状态失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return np.zeros(23, dtype=np.float32)
    
    def _calculate_reward(self, action, prev_state, next_state):
        """计算奖励"""
        if not self.server:
            return 0.0
        
        reward = 0.0
        cfg_reward = self.config['rewards']
        cfg_thresh = self.config['thresholds']
        
        try:
            with self.server.data_lock:
                runtime_data = self.server.unity_runtime_data[self.drone_name]
                pos = runtime_data.position
                
                # 0. 计算稳定性系数 (基于到 Leader 的距离)
                stability_factor = 1.0
                dist_to_leader = 0.0
                if runtime_data.leader_position and runtime_data.leader_scan_radius > 0:
                    dist_to_leader = np.sqrt(
                        (pos.x - runtime_data.leader_position.x) ** 2 +
                        (pos.y - runtime_data.leader_position.y) ** 2 +
                        (pos.z - runtime_data.leader_position.z) ** 2
                    )
                    radius = runtime_data.leader_scan_radius
                    dist_ratio = dist_to_leader / radius
                    
                    # 从配置获取比例
                    safe_ratio = cfg_thresh.get('stability_safe_ratio', 0.7)
                    penalty_ratio = cfg_thresh.get('stability_penalty_ratio', 0.8)
                    
                    if dist_ratio > 1.0:
                        stability_factor = 0.0  # 越界后完全取消探索奖励，强制其返回
                    elif dist_ratio > safe_ratio:
                        # 在 safe_ratio - 1.0 之间线性衰减，从 1.0 降至 0.1
                        stability_factor = 1.0 - (dist_ratio - safe_ratio) / (1.0 - safe_ratio) * 0.9
                    
                    # 额外稳定性惩罚：靠近边界或越界时给予负反馈
                    if dist_ratio > penalty_ratio:
                        # 引导无人机远离边界
                        penalty_weight = cfg_reward.get('stability_penalty_weight', 20.0)
                        reward -= (dist_ratio - penalty_ratio) * penalty_weight

                # 1. 探索奖励：新扫描的单元格 (受稳定性系数影响)
                current_scanned = self._count_scanned_cells()
                new_scanned = current_scanned - self.prev_scanned_cells
                if new_scanned > 0:
                    reward += new_scanned * cfg_reward['exploration'] * stability_factor
                self.prev_scanned_cells = current_scanned
                
                # 2. 熵值降低奖励 (受稳定性系数影响)
                current_entropy = self._get_total_entropy()
                entropy_reduced = self.prev_entropy_sum - current_entropy
                if entropy_reduced > 0:
                    reward += entropy_reduced * cfg_reward['entropy_reduction'] * stability_factor
                self.prev_entropy_sum = current_entropy
                
                # 3. 【优化】局部高熵探索奖励 - 仅在稳定时引导无人机寻找高熵区域
                # 修正索引：0-2位置, 3-5速度, 6-8方向, 9-11熵信息
                local_avg_entropy = next_state[9]  # 局部平均熵
                local_max_entropy = next_state[10] # 局部最大熵
                
                # 奖励进入高熵区域的行为 (受稳定性系数影响)
                if local_max_entropy > cfg_thresh.get('high_entropy_threshold', 40.0):
                    entropy_exploration_bonus = cfg_reward.get('high_entropy_exploration', 5.0)
                    reward += entropy_exploration_bonus * stability_factor
                    
                # 奖励向高熵方向移动 (受稳定性系数影响)
                if self.prev_position:
                    prev_local_avg_entropy = prev_state[9]
                    entropy_increase = local_avg_entropy - prev_local_avg_entropy
                    if entropy_increase > 0:
                        entropy_gradient_reward = entropy_increase * cfg_reward.get('entropy_gradient_bonus', 2.0)
                        reward += entropy_gradient_reward * stability_factor
                
                # 4. 高度控制奖励/惩罚
                current_height = pos.z
                
                # 检查是否在合理的扫描高度范围内
                min_scan_height = cfg_thresh.get('min_scan_height', 2.0)
                max_scan_height = cfg_thresh.get('max_scan_height', 15.0)
                optimal_height = cfg_thresh.get('optimal_scan_height', 8.0)
                
                if current_height < min_scan_height:
                    # 飞得太低
                    reward += cfg_reward.get('height_penalty', -5.0)
                elif current_height > max_scan_height:
                    # 飞得太高
                    reward += cfg_reward.get('height_penalty', -5.0)
                elif abs(current_height - optimal_height) < 2.0:
                    # 在最佳扫描高度附近
                    reward += cfg_reward.get('optimal_height_bonus', 1.0)
                
                # 5. 碰撞惩罚
                min_dist = self._get_min_distance_to_others(runtime_data)
                if min_dist < cfg_thresh['collision_distance']:
                    reward += cfg_reward['collision']
                    self.collision_count += 1
                
                # 6. 越界惩罚
                if runtime_data.leader_position and runtime_data.leader_scan_radius > 0:
                    if dist_to_leader > runtime_data.leader_scan_radius:
                        reward += cfg_reward['out_of_range']
                        self.out_of_range_count += 1
                
                # 7. 平滑运动奖励
                if self.prev_position:
                    current_pos = runtime_data.position
                    movement = np.sqrt(
                        (current_pos.x - self.prev_position.x) ** 2 +
                        (current_pos.y - self.prev_position.y) ** 2 +
                        (current_pos.z - self.prev_position.z) ** 2
                    )
                    # 鼓励适度移动
                    if 0.5 < movement < 5.0:
                        reward += cfg_reward['smooth_movement']
                
                self.prev_position = runtime_data.position
                
                # 8. 每步小惩罚（鼓励快速完成）
                reward += cfg_reward['step_penalty']
                
                # 9. 成功奖励
                scan_ratio = self._get_scan_ratio()
                if scan_ratio >= cfg_thresh['success_scan_ratio']:
                    reward += cfg_reward['success']
                
                # 10. 电量奖励与惩罚
                if hasattr(self.server, 'get_battery_voltage'):
                    current_voltage = self.server.get_battery_voltage(self.drone_name)
                    battery_info = self.server.battery_manager.get_battery_info(self.drone_name)
                    if battery_info:
                        remaining_pct = battery_info.get_remaining_percentage()
                        
                        # 电量过低惩罚
                        if 'battery_low_threshold' in cfg_thresh and current_voltage < cfg_thresh['battery_low_threshold']:
                            penalty = cfg_reward.get('battery_low_penalty', 10.0)
                            reward -= penalty
                        
                        # 电量最优范围奖励
                        if 'battery_optimal_min' in cfg_thresh and 'battery_optimal_max' in cfg_thresh:
                            if cfg_thresh['battery_optimal_min'] <= current_voltage <= cfg_thresh['battery_optimal_max']:
                                bonus = cfg_reward.get('battery_optimal_reward', 2.0)
                                reward += bonus
                
                # 每个动作都更新电量消耗
                if hasattr(self.server, 'update_battery_voltage'):
                    action_intensity = 0.5  # 动作强度，可根据实际动作调整
                    self.server.update_battery_voltage(self.drone_name, action_intensity)
                
        except Exception as e:
            print(f"计算奖励失败: {str(e)}")
        
        return reward
    
    def _check_done(self):
        """判断episode是否结束"""
        # 达到最大步数
        if self.step_count >= self.config['movement']['max_steps']:
            print(f"[DQN环境] Episode 结束: 达到最大步数 {self.step_count}/{self.config['movement']['max_steps']}")
            return True
        
        # 扫描完成
        scan_ratio = self._get_scan_ratio()
        if scan_ratio >= self.config['thresholds']['success_scan_ratio']:
            print(f"[DQN环境] Episode 结束: 扫描完成 {scan_ratio:.2%} >= {self.config['thresholds']['success_scan_ratio']:.2%}")
            return True
        
        # 碰撞次数过多
        if self.collision_count >= 10:
            print(f"[DQN环境] Episode 结束: 碰撞次数过多 {self.collision_count}/10")
            return True
        
        # [DEBUG] 打印当前状态（仅前10步）
        if self.step_count <= 10:
            print(f"[DQN环境] _check_done() - 步骤 {self.step_count}: scan_ratio={scan_ratio:.2%}, collision={self.collision_count}, out_of_range={self.out_of_range_count}")
        
        return False
    
    def _apply_movement(self, displacement):
        """应用移动到无人机（通过AlgorithmServer的DQN控制模式）"""
        if not self.server:
            return  # 没有server连接，无法控制
        
        try:
            # 检查server是否处亊DQN控制模式
            if not hasattr(self.server, 'control_mode') or self.server.control_mode != 'dqn':
                logger.warning("警告: AlgorithmServer未处于DQN控制模式，移动指令可能不会生效")
                return
            
            # 将位移转换为Unity坐标系的方向向量
            # displacement是NumPy数组: [dx, dy, dz]
            # 需要转换为Vector3对象（Unity坐标系）
            from Algorithm.Vector3 import Vector3
            
            # 计算归一化的方向向量
            magnitude = np.linalg.norm(displacement)
            if magnitude > 1e-6:
                # 归一化方向
                direction = displacement / magnitude
                # 转换为Vector3（Unity坐标系：X=前后，Y=高度，Z=左右）
                move_direction = Vector3(direction[0], direction[1], direction[2])
            else:
                # 位移过小，不移动
                move_direction = Vector3(0, 0, 0)
            
            # 通过AlgorithmServer设置DQN移动指令
            self.server.set_dqn_movement(self.drone_name, move_direction)
            
        except Exception as e:
            import traceback
            logger.error(f"应用移动失败: {str(e)}")
            logger.debug(traceback.format_exc())
    
    def _get_entropy_info(self, grid_data, position):
        """获取局部熵值统计"""
        if not grid_data or not grid_data.cells:
            return np.array([50.0, 50.0, 0.0], dtype=np.float32)
        
        nearby_distance = self.config['thresholds']['nearby_entropy_distance']
        
        # 找附近单元格
        nearby_cells = [
            cell for cell in grid_data.cells[:100]
            if np.sqrt(
                (cell.center.x - position.x) ** 2 +
                (cell.center.y - position.y) ** 2 +
                (cell.center.z - position.z) ** 2
            ) < nearby_distance
        ]
        
        if not nearby_cells:
            return np.array([50.0, 50.0, 0.0], dtype=np.float32)
        
        entropies = [cell.entropy for cell in nearby_cells]
        return np.array([
            np.mean(entropies),
            np.max(entropies),
            np.std(entropies)
        ], dtype=np.float32)
    
    def _get_scan_info(self, grid_data):
        """获取扫描进度信息"""
        if not grid_data or not grid_data.cells:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        total = len(grid_data.cells)
        scanned = sum(
            1 for cell in grid_data.cells
            if cell.entropy < self.config['thresholds']['scanned_entropy']
        )
        
        return np.array([
            scanned / max(total, 1),
            float(scanned),
            float(total - scanned)
        ], dtype=np.float32)
    
    def _count_scanned_cells(self):
        """统计已扫描单元格数量"""
        if not self.server or not self.server.grid_data:
            return 0
            
        try:
            with self.server.grid_lock:  # 使用 grid_lock 而不是 data_lock
                return sum(
                    1 for cell in self.server.grid_data.cells
                    if cell.entropy < self.config['thresholds']['scanned_entropy']
                )
        except:
            return 0
        
    def _get_total_entropy(self):
        """获取总熙值"""
        if not self.server or not self.server.grid_data:
            return 0.0
            
        try:
            with self.server.grid_lock:  # 使用 grid_lock 而不是 data_lock
                return sum(cell.entropy for cell in self.server.grid_data.cells)
        except:
            return 0.0
        
    def _get_scan_ratio(self):
        """获取扫描完成比例"""
        if not self.server or not self.server.grid_data:
            return 0.0
            
        try:
            with self.server.grid_lock:  # 使用 grid_lock 而不是 data_lock
                total = len(self.server.grid_data.cells)
                if total == 0:
                    return 0.0
                scanned = sum(
                    1 for cell in self.server.grid_data.cells
                    if cell.entropy < self.config['thresholds']['scanned_entropy']
                )
                return scanned / total
        except:
            return 0.0
    
    def _get_min_distance_to_others(self, runtime_data):
        """获取到其他无人机的最小距离"""
        if not runtime_data.otherScannerPositions:
            return 999.0
        
        pos = runtime_data.position
        distances = [
            np.sqrt(
                (pos.x - other_pos.x) ** 2 +
                (pos.y - other_pos.y) ** 2 +
                (pos.z - other_pos.z) ** 2
            )
            for other_pos in runtime_data.otherScannerPositions
        ]
        return min(distances) if distances else 999.0
    
    def _get_battery_info(self):
        """获取电量信息：[电压, 剩余百分比]"""
        if not self.server or not hasattr(self.server, 'get_battery_voltage'):
            return np.array([4.2, 100.0], dtype=np.float32)  # 默认值：满电
        
        try:
            voltage = self.server.get_battery_voltage(self.drone_name)
            battery_info = self.server.battery_manager.get_battery_info(self.drone_name)
            if battery_info:
                percentage = battery_info.get_remaining_percentage()
                return np.array([voltage, percentage], dtype=np.float32)
            else:
                return np.array([voltage, 100.0], dtype=np.float32)
        except:
            return np.array([4.2, 100.0], dtype=np.float32)
    
    def render(self, mode='human'):
        """可视化（可选）"""
        pass
    
    def close(self):
        """关闭环境"""
        pass


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试 MovementEnv - 无人机移动DQN环境")
    print("=" * 60)
    
    # 创建环境（无server，测试模式）
    env = MovementEnv(server=None, drone_name="UAV1")
    
    print(f"\n观察空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    print(f"动作映射:")
    for action, displacement in env.action_map.items():
        action_name = ['上', '下', '左', '右', '前', '后'][action]
        print(f"  {action}: {action_name} -> {displacement}")
    
    # 重置环境
    state, info = env.reset()
    print(f"\n初始状态shape: {state.shape}")
    print(f"初始状态前5维: {state[:5]}")
    
    # 执行几步测试
    print("\n执行动作测试:")
    for i in range(6):
        action = i  # 测试所有6个动作
        action_name = ['上', '下', '左', '右', '前', '后'][action]
        
        print(f"\n步骤 {i+1}: 动作={action} ({action_name})")
        state, reward, done, info = env.step(action)
        print(f"  奖励: {reward:.2f}")
        print(f"  完成: {done}")
        print(f"  信息: {info}")
        
        if done:
            break
    
    print("\n" + "=" * 60)
    print("[OK] 环境测试通过！")
    print("=" * 60)


class MultiDroneMovementEnv(gym.Env):
    """
    多无人机移动学习环境（参数共享）
    
    多个无人机轮流执行动作，共享同一个 DQN 模型
    动作空间: 6个离散动作（上/下/左/右/前/后）
    观察空间: 位置、速度、熙值、leader位置等
    """
    
    def __init__(self, server=None, drone_names=None, config_path=None):
        super(MultiDroneMovementEnv, self).__init__()
        
        self.server = server
        self.drone_names = drone_names if drone_names else ["UAV1"]
        self.num_drones = len(self.drone_names)
        
        # 当前控制的无人机索引（轮流控制）
        self.current_drone_idx = 0
        
        # 加载配置
        self.config = self._load_config(config_path)
        print(f"[OK] 多无人机 DQN 环境已加载配置")
        print(f"  无人机数量: {self.num_drones}")
        print(f"  无人机列表: {self.drone_names}")
        
        # 动作空间: 6个离散动作（所有无人机共享）
        self.action_space = spaces.Discrete(6)
        
        # 观察空间: 23维（所有无人机共享相同结构）
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(23,),
            dtype=np.float32
        )
        
        # 动作到位移的映射
        self.action_step = self.config['movement']['step_size']
        self.action_map = {
            0: np.array([0, 0, self.action_step]),      # 上
            1: np.array([0, 0, -self.action_step]),     # 下
            2: np.array([-self.action_step, 0, 0]),     # 左
            3: np.array([self.action_step, 0, 0]),      # 右
            4: np.array([0, self.action_step, 0]),      # 前
            5: np.array([0, -self.action_step, 0])      # 后
        }
        
        # 为每个无人机维护独立的状态记录
        self.drone_states = {}
        for drone_name in self.drone_names:
            self.drone_states[drone_name] = {
                'prev_scanned_cells': 0,
                'prev_position': None,
                'prev_entropy_sum': 0,
                'collision_count': 0,
                'out_of_range_count': 0,
                'episode_reward': 0
            }
        
        # 全局状态
        self.step_count = 0
        self.total_episode_reward = 0
        self.episode_index = 0  # Episode 计数器（用于 DataCollector）
        
        # 首次重置标志（用于跳过启动时的重置）
        self._first_reset = True
        
    def _load_config(self, config_path):
        """加载配置文件"""
        if config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, "movement_dqn_config.json")
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return self._default_config()
    
    def _default_config(self):
        """默认配置"""
        return {
            "movement": {
                "step_size": 1.0,
                "max_steps": 500
            },
            "rewards": {
                "exploration": 10.0,
                "collision": -50.0,
                "out_of_range": -30.0,
                "smooth_movement": 1.0,
                "entropy_reduction": 5.0,
                "high_entropy_exploration": 5.0,
                "entropy_gradient_bonus": 2.0,
                "step_penalty": -0.1,
                "success": 100.0,
                "height_penalty": -5.0,
                "optimal_height_bonus": 1.0
            },
            "thresholds": {
                "collision_distance": 2.0,
                "scanned_entropy": 30.0,
                "nearby_entropy_distance": 10.0,
                "success_scan_ratio": 0.95,
                "high_entropy_threshold": 40.0,
                "min_scan_height": 2.0,
                "max_scan_height": 15.0,
                "optimal_scan_height": 8.0
            }
        }
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        print(f"[DQN多机环境] reset() 被调用")
        if seed is not None:
            np.random.seed(seed)
        
        # 首次重置：跳过环境重置（因为无人机刚起飞，领导者刚开始移动）
        if self._first_reset:
            self._first_reset = False
            print(f"[DQN多机环境] 🚀 首次reset，跳过环境重置，直接初始化状态")
            # 仅重置所有无人机的电量
            if self.server and hasattr(self.server, 'reset_battery_voltage'):
                for drone_name in self.drone_names:
                    self.server.reset_battery_voltage(drone_name)
        else:
            # 后续重置：执行完整的环境重置（Episode结束）
            if self.server:
                print(f"[DQN多机环境] 🔄 Episode结束，执行完整环境重置...")
                self.server.reset_environment()
                # 重置所有无人机的电量
                if hasattr(self.server, 'reset_battery_voltage'):
                    for drone_name in self.drone_names:
                        self.server.reset_battery_voltage(drone_name)
                import time
                time.sleep(1.0)
                print(f"[DQN多机环境] ✅ 环境重置完成")
        
        # 重置每个无人机的状态
        print(f"[DQN多机环境] 重置 {self.num_drones} 个无人机状态...")
        for drone_name in self.drone_names:
            self.drone_states[drone_name] = {
                'prev_scanned_cells': self._count_scanned_cells(),
                'prev_position': None,
                'prev_entropy_sum': self._get_total_entropy(),
                'collision_count': 0,
                'out_of_range_count': 0,
                'episode_reward': 0
            }
        
        self.step_count = 0
        self.total_episode_reward = 0
        self.current_drone_idx = 0
        
        # Episode 计数器递增
        self.episode_index += 1
        
        # 返回第一个无人机的状态
        print(f"[DQN多机环境] 获取初始状态...")
        state = self._get_state(self.drone_names[0])
        print(f"[DQN多机环境] reset() 完成，状态shape: {state.shape}")
        return state, {}
    
    def step(self, action):
        """
        执行一步动作（当前无人机）
        
        :param action: 0-5的整数，表示6个移动方向
        :return: observation, reward, terminated, truncated, info
        """
        print(f"[DQN多机环境] step({action}) 被调用")
        
        # 确保 action 是整数
        if hasattr(action, 'item'):
            action = action.item()
        action = int(action)
        
        # 当前控制的无人机
        current_drone = self.drone_names[self.current_drone_idx]
        print(f"[DQN多机环境] 当前控制无人机: {current_drone}")
        
        # 获取当前状态
        print(f"[DQN多机环境] 获取当前状态...")
        current_state = self._get_state(current_drone)
        print(f"[DQN多机环境] 当前状态获取完成")
        
        # 执行动作
        print(f"[DQN多机环境] 执行动作 {action}...")
        displacement = self.action_map[action]
        if self.server:
            self._apply_movement(current_drone, displacement)
            import time
            time.sleep(0.05)
        print(f"[DQN多机环境] 动作执行完成")
        
        # 获取新状态
        print(f"[DQN多机环境] 获取新状态...")
        next_state = self._get_state(current_drone)
        print(f"[DQN多机环境] 新状态获取完成")
        
        # 计算奖励
        print(f"[DQN多机环境] 计算奖励...")
        reward = self._calculate_reward(current_drone, action, current_state, next_state)
        print(f"[DQN多机环境] 奖励计算完成: {reward:.2f}")
        
        self.drone_states[current_drone]['episode_reward'] += reward
        self.total_episode_reward += reward
        
        # 更新计数器
        self.step_count += 1
        
        # 将训练统计信息传递给服务器（用于 DataCollector）
        if self.server and hasattr(self.server, 'set_training_stats'):
            self.server.set_training_stats(
                episode=self.episode_index,
                step=self.step_count,
                reward=float(reward),
                total_reward=float(self.total_episode_reward)
            )
        
        # 轮流到下一个无人机
        self.current_drone_idx = (self.current_drone_idx + 1) % self.num_drones
        
        # 判断是否结束
        print(f"[DQN多机环境] 检查是否结束...")
        terminated = self._check_done()
        print(f"[DQN多机环境] 结束检查完成: {terminated}")
        truncated = False
        
        # 额外信息
        info = {
            'drone_name': current_drone,
            'action': action,
            'displacement': displacement.tolist(),
            'scanned_cells': self._count_scanned_cells(),
            'total_reward': self.total_episode_reward,
            'step_count': self.step_count,
            'current_drone_idx': self.current_drone_idx
        }
        
        # 返回下一个无人机的状态
        next_drone = self.drone_names[self.current_drone_idx]
        print(f"[DQN多机环境] 获取下一个无人机状态: {next_drone}")
        next_observation = self._get_state(next_drone)
        print(f"[DQN多机环境] step() 完成")
        
        return next_observation, reward, terminated, truncated, info
    
    def _get_state(self, drone_name):
        """获取指定无人机的观察状态（21维）"""
        if not self.server:
            return np.random.randn(21).astype(np.float32)
        
        try:
            # 分离锁的获取，避免死锁
            # 第一步：从 runtime_data 获取无人机状态
            with self.server.data_lock:
                runtime_data = self.server.unity_runtime_data.get(drone_name)
                if not runtime_data:
                    print(f"[DQN多机环境] 警告: 无人机 {drone_name} 的runtime_data不存在")
                    return np.zeros(21, dtype=np.float32)
                
                # 1. 位置 (3)
                pos = runtime_data.position
                position = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
                
                # 2. 速度 (3)
                vel = runtime_data.velocity
                velocity = np.array([vel.x, vel.y, vel.z], dtype=np.float32)
                
                # 3. 朝向 (3)
                fwd = runtime_data.forward
                forward = np.array([fwd.x, fwd.y, fwd.z], dtype=np.float32)
                
                # 5. Leader相对位置 (3)
                if runtime_data.leader_position:
                    leader_rel = np.array([
                        runtime_data.leader_position.x - pos.x,
                        runtime_data.leader_position.y - pos.y,
                        runtime_data.leader_position.z - pos.z
                    ], dtype=np.float32)
                else:
                    leader_rel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                
                # 6. Leader范围信息 (2)
                leader_distance = float(np.linalg.norm(leader_rel))
                is_out_of_range = 1.0 if leader_distance > runtime_data.leader_scan_radius else 0.0
                leader_info = np.array([leader_distance, is_out_of_range], dtype=np.float32)
            
            # 第二步：从 grid_data 获取网格信息
            with self.server.grid_lock:
                grid_data = self.server.grid_data
                if not grid_data or not grid_data.cells:
                    print(f"[DQN多机环境] 警告: grid_data 为空或没有cells")
                    # 返回带有基本信息的状态
                    entropy_info = np.array([50.0, 50.0, 0.0], dtype=np.float32)
                    scan_info = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    min_dist_info = np.array([100.0], dtype=np.float32)
                else:
                    # 4. 局部熙值统计 (3)
                    nearby_distance = self.config['thresholds']['nearby_entropy_distance']
                    nearby_cells = [c for c in grid_data.cells if (c.center - pos).magnitude() < nearby_distance]
                    if nearby_cells:
                        entropies = [c.entropy for c in nearby_cells]
                        entropy_info = np.array([
                            float(np.mean(entropies)),
                            float(np.max(entropies)),
                            float(np.std(entropies))
                        ], dtype=np.float32)
                    else:
                        entropy_info = np.array([50.0, 50.0, 0.0], dtype=np.float32)
                    
                    # 7. 扫描进度 (3)
                    scanned_threshold = self.config['thresholds']['scanned_entropy']
                    scanned_count = sum(1 for cell in grid_data.cells if cell.entropy < scanned_threshold)
                    total_cells = len(grid_data.cells)
                    unscanned_count = total_cells - scanned_count
                    scan_ratio = scanned_count / total_cells if total_cells > 0 else 0.0
                    scan_info = np.array([scan_ratio, float(scanned_count), float(unscanned_count)], dtype=np.float32)
                    
                    # 8. 其他无人机最近距离 (1) - 需要重新获取 data_lock
                    min_dist_info = np.array([100.0], dtype=np.float32)  # 默认值
            
            # 第三步：获取其他无人机距离
            min_distance = self._get_min_distance_to_others(drone_name)
            min_dist_info = np.array([min_distance], dtype=np.float32)
            
            # 第四步：获取电量信息
            battery_info = self._get_battery_info_for_drone(drone_name)
            
            # 拼接所有特征
            state = np.concatenate([
                position,
                velocity,
                forward,
                entropy_info,
                leader_rel,
                leader_info,
                scan_info,
                min_dist_info,
                battery_info
            ])
            
            return state.astype(np.float32)
            
        except Exception as e:
            logger.error(f"获取状态失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return np.zeros(23, dtype=np.float32)
    
    def _get_min_distance_to_others(self, drone_name):
        """获取到其他无人机的最小距离"""
        try:
            with self.server.data_lock:
                current_pos = self.server.unity_runtime_data[drone_name].position
                min_distance = float('inf')
                
                for other_drone in self.drone_names:
                    if other_drone != drone_name:
                        other_pos = self.server.unity_runtime_data[other_drone].position
                        distance = (current_pos - other_pos).magnitude()
                        min_distance = min(min_distance, distance)
                
                return min_distance if min_distance != float('inf') else 100.0
        except:
            return 100.0
    
    def _get_battery_info_for_drone(self, drone_name):
        """获取指定无人机的电量信息：[电压, 剩余百分比]"""
        if not self.server or not hasattr(self.server, 'get_battery_voltage'):
            return np.array([4.2, 100.0], dtype=np.float32)  # 默认值：满电
        
        try:
            voltage = self.server.get_battery_voltage(drone_name)
            battery_info = self.server.battery_manager.get_battery_info(drone_name)
            if battery_info:
                percentage = battery_info.get_remaining_percentage()
                return np.array([voltage, percentage], dtype=np.float32)
            else:
                return np.array([voltage, 100.0], dtype=np.float32)
        except:
            return np.array([4.2, 100.0], dtype=np.float32)
    
    def _apply_movement(self, drone_name, displacement):
        """应用移动到无人机（通过AlgorithmServer的DQN控制模式）"""
        if not self.server:
            return
        
        try:
            if not hasattr(self.server, 'control_mode') or self.server.control_mode != 'dqn':
                logger.warning("警告: AlgorithmServer未处于DQN控制模式")
                return
            
            from Algorithm.Vector3 import Vector3
            magnitude = np.linalg.norm(displacement)
            if magnitude > 1e-6:
                direction = displacement / magnitude
                move_direction = Vector3(direction[0], direction[1], direction[2])
            else:
                move_direction = Vector3(0, 0, 0)
            
            self.server.set_dqn_movement(drone_name, move_direction)
            
        except Exception as e:
            logger.error(f"应用移动失败: {str(e)}")
    
    def _calculate_reward(self, drone_name, action, current_state, next_state):
        """计算奖励"""
        reward = 0.0
        drone_state = self.drone_states[drone_name]
        
        # 0. 计算稳定性系数 (基于到 Leader 的距离)
        stability_factor = 1.0
        dist_to_leader = next_state[15]  # 观察状态中的 Leader 距离
        is_out_of_range = next_state[16] > 0.5
        
        cfg_reward = self.config['rewards']
        cfg_thresh = self.config['thresholds']
        
        try:
            with self.server.data_lock:
                runtime_data = self.server.unity_runtime_data[drone_name]
                radius = runtime_data.leader_scan_radius
                if radius > 0:
                    dist_ratio = dist_to_leader / radius
                    
                    # 从配置获取比例
                    safe_ratio = cfg_thresh.get('stability_safe_ratio', 0.7)
                    penalty_ratio = cfg_thresh.get('stability_penalty_ratio', 0.8)
                    
                    if dist_ratio > 1.0:
                        stability_factor = 0.0
                    elif dist_ratio > safe_ratio:
                        # safe_ratio - 1.0 之间线性衰减
                        stability_factor = 1.0 - (dist_ratio - safe_ratio) / (1.0 - safe_ratio) * 0.9
                    
                    # 额外稳定性惩罚
                    if dist_ratio > penalty_ratio:
                        penalty_weight = cfg_reward.get('stability_penalty_weight', 20.0)
                        reward -= (dist_ratio - penalty_ratio) * penalty_weight
        except:
            pass
            
        # 1. 探索奖励 (受稳定性系数影响)
        current_scanned = self._count_scanned_cells()
        new_cells = current_scanned - drone_state['prev_scanned_cells']
        if new_cells > 0:
            reward += new_cells * cfg_reward['exploration'] * stability_factor
        drone_state['prev_scanned_cells'] = current_scanned
            
        # 2. 熵值降低奖励 (受稳定性系数影响)
        current_entropy = self._get_total_entropy()
        entropy_reduction = drone_state['prev_entropy_sum'] - current_entropy
        if entropy_reduction > 0:
            reward += entropy_reduction * cfg_reward['entropy_reduction'] * 0.01 * stability_factor
        drone_state['prev_entropy_sum'] = current_entropy
            
        # 3. 【优化】局部高熵探索奖励
        # 修正索引：9-平均熵, 10-最大熵
        local_avg_entropy = next_state[9]
        local_max_entropy = next_state[10]
            
        if local_max_entropy > cfg_thresh.get('high_entropy_threshold', 40.0):
            entropy_exploration_bonus = cfg_reward.get('high_entropy_exploration', 5.0)
            reward += entropy_exploration_bonus * stability_factor
            
        if drone_state['prev_position']:
            prev_local_avg_entropy = current_state[9]
            entropy_increase = local_avg_entropy - prev_local_avg_entropy
            if entropy_increase > 0:
                entropy_gradient_reward = entropy_increase * cfg_reward.get('entropy_gradient_bonus', 2.0)
                reward += entropy_gradient_reward * stability_factor
            
        # 4. 【新增】高度控制奖励/惩罚
        try:
            with self.server.data_lock:
                runtime_data = self.server.unity_runtime_data[drone_name]
                pos = runtime_data.position
                current_height = pos.z
                    
                min_scan_height = cfg_thresh.get('min_scan_height', 2.0)
                max_scan_height = cfg_thresh.get('max_scan_height', 15.0)
                optimal_height = cfg_thresh.get('optimal_scan_height', 8.0)
                    
                if current_height < min_scan_height:
                    reward += cfg_reward.get('height_penalty', -5.0)
                elif current_height > max_scan_height:
                    reward += cfg_reward.get('height_penalty', -5.0)
                elif abs(current_height - optimal_height) < 2.0:
                    reward += cfg_reward.get('optimal_height_bonus', 1.0)
                    
                drone_state['prev_position'] = pos
        except Exception as e:
            logger.debug(f"高度奖励计算失败: {str(e)}")
            
        # 5. 碰撞惩罚
        min_distance = self._get_min_distance_to_others(drone_name)
        if min_distance < cfg_thresh['collision_distance']:
            reward += cfg_reward['collision']
            drone_state['collision_count'] += 1
        
        # 6. 超出Leader范围惩罚
        if is_out_of_range:
            reward += cfg_reward['out_of_range']
            drone_state['out_of_range_count'] += 1
        
        # 7. 步骤惩罚
        reward += cfg_reward['step_penalty']
        
        # 8. 电量奖励与惩罚
        if self.server and hasattr(self.server, 'get_battery_voltage'):
            try:
                current_voltage = self.server.get_battery_voltage(drone_name)
                battery_info = self.server.battery_manager.get_battery_info(drone_name)
                if battery_info:
                    # 电量过低惩罚
                    if 'battery_low_threshold' in cfg_thresh:
                        if current_voltage < cfg_thresh['battery_low_threshold']:
                            penalty = cfg_reward.get('battery_low_penalty', 10.0)
                            reward -= penalty
                    
                    # 电量最优范围奖励
                    if 'battery_optimal_min' in cfg_thresh and 'battery_optimal_max' in cfg_thresh:
                        opt_min = cfg_thresh['battery_optimal_min']
                        opt_max = cfg_thresh['battery_optimal_max']
                        if opt_min <= current_voltage <= opt_max:
                            bonus = cfg_reward.get('battery_optimal_reward', 2.0)
                            reward += bonus
                
                # 更新电量消耗
                if hasattr(self.server, 'update_battery_voltage'):
                    action_intensity = 0.5
                    self.server.update_battery_voltage(drone_name, action_intensity)
            except Exception as e:
                logger.debug(f"电量奖励计算失败: {str(e)}")
        
        return reward
    
    def _check_done(self):
        """检查episode是否结束"""
        # 超过最大步数
        max_steps = self.config['movement']['max_steps'] * self.num_drones
        if self.step_count >= max_steps:
            return True
        
        # 扫描完成
        scan_ratio = self._get_scan_ratio()
        if scan_ratio >= self.config['thresholds']['success_scan_ratio']:
            return True
        
        return False
    
    def _count_scanned_cells(self):
        """统计已扫描单元格数量"""
        if not self.server:
            return 0
        try:
            with self.server.grid_lock:
                scanned_threshold = self.config['thresholds']['scanned_entropy']
                return sum(1 for cell in self.server.grid_data.cells if cell.entropy < scanned_threshold)
        except:
            return 0
    
    def _get_total_entropy(self):
        """获取总熙值"""
        if not self.server:
            return 0.0
        try:
            with self.server.grid_lock:
                return sum(cell.entropy for cell in self.server.grid_data.cells)
        except:
            return 0.0
    
    def _get_scan_ratio(self):
        """获取扫描比例"""
        if not self.server:
            return 0.0
        try:
            with self.server.grid_lock:
                total = len(self.server.grid_data.cells)
                if total == 0:
                    return 0.0
                # 直接在这里计算，而不是调用 _count_scanned_cells()（避免重复获取锁）
                scanned_threshold = self.config['thresholds']['scanned_entropy']
                scanned = sum(1 for cell in self.server.grid_data.cells if cell.entropy < scanned_threshold)
                return scanned / total
        except:
            return 0.0
