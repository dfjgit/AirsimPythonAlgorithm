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
import time
from Algorithm.battery_data import BatteryStatus

# 配置日志
logger = logging.getLogger("MovementEnv")


def _wait_for_server_ready(server, drone_names, timeout_sec: float = 12.0) -> bool:
    """Wait until runtime and grid data are both available after a reset."""
    if server is None:
        return False

    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            has_grid = bool(server.grid_data and server.grid_data.cells)
            has_runtime = all(
                server.unity_runtime_data.get(name) is not None
                and server.unity_runtime_data[name].position is not None
                for name in drone_names
            )
            if has_grid and has_runtime:
                return True
        except Exception:
            pass
        time.sleep(0.2)
    return False


def _apply_low_altitude_guard(server, drone_name, displacement, config, action_step):
    """Prevent DQN from repeatedly pushing a drone into the floor right after takeoff."""
    if server is None:
        return displacement

    try:
        with server.data_lock:
            runtime_data = server.unity_runtime_data.get(drone_name)
            if not runtime_data or runtime_data.position is None:
                return displacement
            current_height = float(runtime_data.position.y)
    except Exception:
        return displacement

    thresholds = config.get("thresholds", {})
    low_altitude_recovery_height = float(
        thresholds.get("low_altitude_recovery_height", 1.2)
    )

    # If a drone is close to the floor, force a climb instead of continuing to descend
    # or trying to skim horizontally while effectively landed.
    if current_height < low_altitude_recovery_height:
        guarded = np.array(displacement, dtype=np.float32, copy=True)
        guarded[0] = 0.0
        guarded[2] = 0.0
        guarded[1] = abs(float(action_step))
        return guarded

    return displacement


class MovementEnv(gym.Env):
    """
    无人机移动学习环境
    
    动作空间: 6个离散动作（上/下/左/右/前/后）
    观察空间: 位置、速度、熵值、leader位置等
    """
    
    def __init__(self, server=None, drone_name="UAV1", config_path=None, step_duration=0.5):
        self._lock_timeout_sec = 0.2
        self._state_timeout_sec = 2.0
        self._warned_lock_timeout = False
        self._warned_state_timeout = False
        super(MovementEnv, self).__init__()
        
        self.server = server
        self.drone_name = drone_name
        self.step_duration = step_duration  # 物理步长（秒）
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 统一环境配置加载逻辑 (方案B: 解耦物理规则)
        self._apply_unified_config()
        
        print(f"[OK] 移动DQN环境已加载配置并应用统一环境规则")
        self.verbose_step_logs = False
        
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
        # 修正映射：0/1对应高度Y，2/3对应左右Z，4/5对应前后X (与Unity/AlgorithmServer一致)
        self.action_step = self.config['movement']['step_size']
        self.action_repeat = max(1, int(self.config['movement'].get('action_repeat', 3)))
        self.action_map = {
            0: np.array([0, self.action_step, 0]),      # 上 (Y+)
            1: np.array([0, -self.action_step, 0]),     # 下 (Y-)
            2: np.array([0, 0, -self.action_step]),     # 左 (Z-)
            3: np.array([0, 0, self.action_step]),      # 右 (Z+)
            4: np.array([self.action_step, 0, 0]),      # 前 (X+)
            5: np.array([-self.action_step, 0, 0])      # 后 (X-)
        }

        # 状态记录
        self.prev_scanned_cells = 0
        self.last_oob_diag = {}
        self.out_of_range_steps = 0
        self.prev_position = None
        self.prev_entropy_sum = 0
        self.step_count = 0
        self.episode_reward = 0
        self.collision_count = 0
        self.out_of_range_count = 0
        self.last_done_reason = None
        self.episode_start_time = time.time()
        self._first_reset = True

    def _step_log(self, message: str) -> None:
        if self.verbose_step_logs:
            print(message)
    
    def _apply_unified_config(self):
        """从统一源加载环境规则（终止阈值、电量参数、基础奖励）"""
        unified_env_cfg = None
        
        # 1. 尝试从 server 获取 (最优先)
        if self.server and hasattr(self.server, 'config_data') and hasattr(self.server.config_data, 'env_config'):
            unified_env_cfg = self.server.config_data.env_config
        else:
            # 2. 尝试从本地 apf_algorithm_config.json 加载
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                scanner_cfg_path = os.path.join(current_dir, "..", "..", "..", "apf_algorithm_config.json")
                if os.path.exists(scanner_cfg_path):
                    with open(scanner_cfg_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        unified_env_cfg = data.get('env_config')
            except Exception as e:
                logger.warning(f"无法加载统一环境配置: {e}")

        if not unified_env_cfg:
            # 使用默认终止配置
            self.term_cfg = self.config.get('termination_config', {
                "target_scan_ratio": 0.25,
                "max_collision_count": 15,
                "max_elapsed_time_sec": 300.0,
                "stagnation_timeout_sec": 30.0
            })
            return

        # DQN 本地配置优先，统一环境配置只作为缺省值。
        merged_term_cfg = dict(unified_env_cfg.get('termination', {}))
        merged_term_cfg.update(self.config.get('termination_config', {}))
        self.term_cfg = merged_term_cfg
        
        # --- B. 应用电量阈值 ---
        battery_cfg = unified_env_cfg.get('battery', {})
        if 'thresholds' not in self.config: self.config['thresholds'] = {}
        
        self.config['thresholds']['battery_low_threshold'] = battery_cfg.get('low_threshold', 3.5)
        self.config['thresholds']['battery_optimal_min'] = battery_cfg.get('optimal_min', 3.7)
        self.config['thresholds']['battery_optimal_max'] = battery_cfg.get('optimal_max', 4.1)
        
        # --- C. 应用基础奖励系数 ---
        base_rewards = unified_env_cfg.get('base_rewards', {})
        if 'rewards' not in self.config: self.config['rewards'] = {}
        
        # 映射统一奖励到本地配置
        reward_map = {
            'scan_reward': 'exploration',           # 新扫描奖励
            'out_of_range_penalty': 'out_of_range', # 越界惩罚
            'battery_low_penalty': 'battery_low_penalty',
            'battery_optimal_reward': 'battery_optimal_reward',
            'collision_penalty': 'collision',
            'step_penalty': 'step_penalty'
        }
        
        for u_key, local_key in reward_map.items():
            if u_key in base_rewards and local_key not in self.config['rewards']:
                self.config['rewards'][local_key] = float(base_rewards[u_key])

    def _load_config(self, config_path):
        """加载配置文件"""
        if config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, "..", "configs", "movement_dqn_config.json")
        
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
                "success_scan_ratio": 0.25,
                "high_entropy_threshold": 40.0,
                "min_scan_height": 2.0,
                "max_scan_height": 15.0,
                "optimal_scan_height": 8.0
            }
        }
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        # 获取并清除上一次结束的原因
        reason = getattr(self, 'last_done_reason', 'None')
        print(f"\n[DQN环境] reset() 被调用，上一轮结束原因: {reason}")
        self.last_done_reason = None
        
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
                reason = getattr(self, 'last_done_reason', 'manual')
                print(f"[DQN环境] 🔄 Episode结束，执行完整环境重置... (原因: {reason})")
                self.server.reset_environment(reason=f"MovementEnv_{reason}", reset_grid=True)
                # 重置电量
                if hasattr(self.server, 'reset_battery_voltage'):
                    self.server.reset_battery_voltage(self.drone_name)
                ready = _wait_for_server_ready(self.server, [self.drone_name], timeout_sec=12.0)
                if not ready:
                    print(f"[DQN环境] ⚠️ 重置后数据未完全就绪，继续尝试使用当前状态")
                print(f"[DQN环境] ✅ 环境重置完成")
        
        print(f"[DQN环境] 初始化状态...")
        self.prev_scanned_cells = self._count_scanned_cells()
        self.prev_entropy_sum = self._get_total_entropy()
        self.prev_position = None
        self.step_count = 0
        self.episode_reward = 0
        self.collision_count = 0
        self.out_of_range_count = 0
        self.out_of_range_steps = 0  # 重置越界步数统计
        
        print(f"[DQN环境] 初始化信息:")
        print(f"  - 初始扫描数: {self.prev_scanned_cells}")
        print(f"  - 初始总熄值: {self.prev_entropy_sum:.2f}")
        
        self.episode_start_time = time.time()  # 重置 Episode 开始时间
        
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
        
        self._step_log(f"[DQN环境] step({action}) 被调用")
        
        # 记录当前位置
        self._step_log(f"[DQN环境] 获取当前状态...")
        current_state = self._get_state()
        self._step_log(f"[DQN环境] 当前状态获取完成")
        
        displacement = self.action_map[action]
        reward = 0.0
        next_state = current_state
        terminated = False

        for repeat_idx in range(self.action_repeat):
            if self.server:
                self._step_log(f"[DQN环境] 发送移动指令[{repeat_idx + 1}/{self.action_repeat}]: {displacement}")
                self._apply_movement(displacement)
                self._step_log(f"[DQN环境] 移动指令已发送")
                time.sleep(self.step_duration)
                self._step_log(f"[DQN环境] 物理步长等待完成 ({self.step_duration}s)")

            self._step_log(f"[DQN环境] 获取新状态...")
            next_state = self._get_state()
            self._step_log(f"[DQN环境] 新状态获取完成")

            self._step_log(f"[DQN环境] 计算奖励...")
            sub_reward = self._calculate_reward(action, current_state, next_state)
            reward += sub_reward
            self.episode_reward += sub_reward
            self._step_log(f"[DQN环境] 子步奖励计算完成: {sub_reward:.2f}")

            self.step_count += 1
            current_state = next_state
            if terminated:
                break

        self._step_log(f"[DQN环境] 奖励计算完成: {reward:.2f}")

        truncated = False  # 不使用截断
        
        # 额外信息 - 注入诊断数据
        info = {
            'action': action,
            'displacement': displacement.tolist(),
            'scanned_cells': self._count_scanned_cells(),
            'collision_count': self.collision_count,
            'out_of_range_count': self.out_of_range_count,
            'episode_reward': self.episode_reward,
            'oob_diag': self.last_oob_diag  # 包含越界时的决策诊断
        }
        
        if self.verbose_step_logs and self.step_count % 10 == 0:
            print(f"[DQN环境] 步骤 {self.step_count}, 奖励: {reward:.2f}, episode总奖励: {self.episode_reward:.2f}")
        
        return next_state, reward, terminated, truncated, info
    
    def _get_state(self):
        """获取当前观察状态（23维：包含电量信息）"""
        if not self.server:
            # 测试模式：返回随机状态
            return np.random.randn(23).astype(np.float32)

        deadline = time.time() + self._state_timeout_sec
        while True:
            if time.time() > deadline:
                if not self._warned_state_timeout:
                    logger.warning(f"获取状态超时({self._state_timeout_sec}s)，返回零状态")
                    self._warned_state_timeout = True
                return np.zeros(23, dtype=np.float32)

            acquired = self.server.data_lock.acquire(timeout=self._lock_timeout_sec)
            if not acquired:
                if not self._warned_lock_timeout:
                    logger.warning(f"获取 data_lock 超时({self._lock_timeout_sec}s)，将重试")
                    self._warned_lock_timeout = True
                continue

            try:
                runtime_data = self.server.unity_runtime_data.get(self.drone_name)
                if not runtime_data:
                    return np.zeros(23, dtype=np.float32)

                pos = runtime_data.position
                if not pos:
                    return np.zeros(23, dtype=np.float32)

                position = np.array([pos.x, pos.y, pos.z], dtype=np.float32)

                vel = runtime_data.finalMoveDir
                velocity = np.array([
                    vel.x * self.server.config_data.moveSpeed,
                    vel.y * self.server.config_data.moveSpeed,
                    vel.z * self.server.config_data.moveSpeed
                ], dtype=np.float32)

                fwd = runtime_data.forward
                direction = np.array([fwd.x, fwd.y, fwd.z], dtype=np.float32)

                if rd.leader_position:
                    # 修正：对相对位置进行归一化，除以扫描半径
                    # 这样 0-1 表示在圈内，>1 表示在圈外，信号更显著
                    radius = rd.leader_scan_radius if rd.leader_scan_radius > 0 else 50.0
                    leader_rel = np.array([
                        (rd.leader_position.x - pos.x) / radius,
                        (rd.leader_position.y - pos.y) / radius,
                        (rd.leader_position.z - pos.z) / radius
                    ], dtype=np.float32)
                else:
                    leader_rel = np.zeros(3, dtype=np.float32)

                if runtime_data.leader_position and runtime_data.leader_scan_radius > 0:
                    dist_to_leader = np.linalg.norm(leader_rel)
                    is_out_of_range = 1.0 if dist_to_leader > runtime_data.leader_scan_radius else 0.0
                    leader_range = np.array([dist_to_leader, is_out_of_range], dtype=np.float32)
                else:
                    leader_range = np.zeros(2, dtype=np.float32)
            finally:
                self.server.data_lock.release()

            acquired_grid = self.server.grid_lock.acquire(timeout=self._lock_timeout_sec)
            if not acquired_grid:
                if not self._warned_lock_timeout:
                    logger.warning(f"获取 grid_lock 超时({self._lock_timeout_sec}s)，将重试")
                    self._warned_lock_timeout = True
                continue

            try:
                grid_data = self.server.grid_data
                if not grid_data or not getattr(grid_data, 'cells', None):
                    entropy_info = np.array([50.0, 50.0, 0.0], dtype=np.float32)
                    scan_info = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    min_dist_array = np.array([100.0], dtype=np.float32)
                else:
                    entropy_info = self._get_entropy_info(grid_data, pos)
                    scan_info = self._get_scan_info(grid_data)
                    min_dist_array = np.array([100.0], dtype=np.float32)
            finally:
                self.server.grid_lock.release()

            acquired2 = self.server.data_lock.acquire(timeout=self._lock_timeout_sec)
            if acquired2:
                try:
                    runtime_data2 = self.server.unity_runtime_data.get(self.drone_name)
                    if runtime_data2:
                        min_dist = self._get_min_distance_to_others(runtime_data2)
                        min_dist_array = np.array([min_dist], dtype=np.float32)
                finally:
                    self.server.data_lock.release()

            battery_info = self._get_battery_info()

            state = np.concatenate([
                position,
                velocity,
                direction,
                entropy_info,
                leader_rel,
                leader_range,
                scan_info,
                min_dist_array,
                battery_info
            ])

            return state.astype(np.float32)

    
    def _calculate_reward(self, action, prev_state, next_state):
        """计算奖励"""
        if not self.server:
            return 0.0
        
        reward = 0.0
        cfg_reward = self.config['rewards']
        cfg_thresh = self.config['thresholds']
        
        # 初始化诊断信息
        diag_info = {
            'is_oob': False,
            'dist_ratio': 0.0,
            'delta_dist': 0.0,
            'alignment': 0.0,
            'action_name': ['上', '下', '左', '右', '前', '后'][action],
            'reward': 0.0
        }
        
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
                    diag_info['dist_ratio'] = float(dist_ratio)
                    
                    # 从配置获取比例
                    safe_ratio = cfg_thresh.get('stability_safe_ratio', 0.7)
                    penalty_ratio = cfg_thresh.get('stability_penalty_ratio', 0.8)
                    
                    if dist_ratio > 1.0:
                        diag_info['is_oob'] = True
                        # 越界后：将任务切换为“尽快回圈”，避免圈外出现可学习的“摆烂”局部最优
                        stability_factor = 0.0

                        # 距离 shaping：更靠近 leader 就奖励；远离则惩罚
                        if self.prev_position:
                            prev_dist = np.sqrt(
                                (self.prev_position.x - runtime_data.leader_position.x)**2 +
                                (self.prev_position.y - runtime_data.leader_position.y)**2 +
                                (self.prev_position.z - runtime_data.leader_position.z)**2
                            )
                            delta = prev_dist - dist_to_leader
                            diag_info['delta_dist'] = float(delta)
                        else:
                            delta = 0.0

                        # 基础圈外每步惩罚
                        reward += cfg_reward.get('out_of_range_step_penalty', -1.0)

                        if delta > 0:
                            # 回圈进度奖励
                            reward += delta * cfg_reward.get('return_progress_weight', 200.0)
                            reward += cfg_reward.get('return_to_range_bonus', 50.0)

                            # 动作方向对齐奖励
                            dir_to_leader = np.array([
                                runtime_data.leader_position.x - runtime_data.position.x,
                                runtime_data.leader_position.y - runtime_data.position.y,
                                runtime_data.leader_position.z - runtime_data.position.z
                            ])
                            norm = np.linalg.norm(dir_to_leader)
                            if norm > 1e-6:
                                ideal_dir = dir_to_leader / norm
                                actual_dir = self.action_map[action] / np.linalg.norm(self.action_map[action])
                                alignment = float(np.dot(ideal_dir, actual_dir))
                                diag_info['alignment'] = alignment
                                if alignment > 0:
                                    reward += alignment * cfg_reward.get('return_alignment_weight', 20.0)
                        else:
                            # 继续往外飞：额外惩罚
                            reward += cfg_reward.get('out_of_range', -30.0)

                        diag_info['reward'] = float(reward)
                        self.last_oob_diag = diag_info
                        # 越界阶段奖励单独结算
                        return reward

                    elif dist_ratio > safe_ratio:
                        # 在 safe_ratio - 1.0 之间线性衰减
                        stability_factor = 1.0 - (dist_ratio - safe_ratio) / (1.0 - safe_ratio) * 0.9
                    
                    if dist_ratio > penalty_ratio:
                        penalty_weight = cfg_reward.get('stability_penalty_weight', 20.0)
                        reward -= (dist_ratio - penalty_ratio) * penalty_weight

                # 正常情况下的奖励计算...
                # (此处省略后续代码，search_replace 会处理匹配)

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
                current_height = pos.y # 修正：Unity中Y轴是高度
                
                # 检查是否在合理的扫描高度范围内
                min_scan_height = cfg_thresh.get('min_scan_height', 2.0)
                max_scan_height = cfg_thresh.get('max_scan_height', 15.0)
                optimal_height = cfg_thresh.get('optimal_scan_height', 8.0)
                
                if current_height < min_scan_height:
                    # 飞得太低
                    reward += cfg_reward.get('height_penalty', -5.0) * (min_scan_height - current_height)
                elif current_height > max_scan_height:
                    # 飞得太高
                    reward += cfg_reward.get('height_penalty', -5.0) * (current_height - max_height)
                elif abs(current_height - optimal_height) < 1.5:
                    # 在最佳扫描高度附近
                    reward += cfg_reward.get('optimal_height_bonus', 1.0)
                
                # 5. 碰撞惩罚与容忍机制
                min_dist = self._get_min_distance_to_others(runtime_data)
                collision_threshold = cfg_thresh['collision_distance']
                
                if min_dist < collision_threshold * 0.8:  # 只有进入 80% 的安全距离才算碰撞
                    reward += cfg_reward['collision']
                    self.collision_count += 1
                else:
                    # 距离恢复安全时重置计数器（给予训练纠错机会）
                    if self.collision_count > 0 and min_dist > collision_threshold * 1.5:
                        self.collision_count = max(0, self.collision_count - 1)  # 缓慢恢复
                
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
                    # 计算动作强度（位移长度归一化）
                    # 在 step 函数中已经根据 action 计算好了 displacement
                    # displacement 是 NumPy 数组: [dx, dy, dz]
                    step_norm = float(np.linalg.norm(self.action_map[action]))
                    base_step = max(self.action_step, 1e-6)
                    action_intensity = min(1.0, max(0.0, step_norm / base_step))
                    self.server.update_battery_voltage(self.drone_name, action_intensity)
                
        except Exception as e:
            print(f"计算奖励失败: {str(e)}")
        
        return reward
    
    def _check_done(self):
        """??episode???? (??????)"""
        elapsed_time = time.time() - self.episode_start_time
        cfg_thresh = self.config.get('thresholds', {})

        if elapsed_time >= self.term_cfg['max_elapsed_time_sec']:
            self.last_done_reason = f"Timeout ({elapsed_time:.1f}s >= {self.term_cfg['max_elapsed_time_sec']}s)"
            print(f"[??] {self.last_done_reason}")
            return True

        scan_ratio = self._get_scan_ratio()
        if scan_ratio >= self.term_cfg['target_scan_ratio']:
            self.last_done_reason = f"Target Scan Ratio Reached ({scan_ratio:.2%} >= {self.term_cfg['target_scan_ratio']:.2%})"
            print(f"[??] {self.last_done_reason}")
            return True

        total_collisions = sum(state['collision_count'] for state in self.drone_states.values())
        if total_collisions >= self.term_cfg['max_collision_count']:
            self.last_done_reason = f"Collision Limit Reached ({total_collisions} >= {self.term_cfg['max_collision_count']})"
            print(f"[??] {self.last_done_reason}")
            return True

        terminate_on_out_of_range = bool(cfg_thresh.get('terminate_on_out_of_range', True))
        max_oob_steps = max(1, int(cfg_thresh.get('max_out_of_range_steps', 1)))
        max_oob_duration_sec = float(
            cfg_thresh.get(
                'max_out_of_range_duration_sec',
                max_oob_steps * self.step_duration * self.action_repeat
            )
        )
        severe_ratio = float(cfg_thresh.get('severe_out_of_range_ratio', 1.05))

        if self.server:
            for drone_name in self.drone_names:
                try:
                    state = self.server.drone_controller.get_vehicle_state(drone_name)
                    if not state.get("flying", True):
                        self.last_done_reason = f"Drone {drone_name} Landed (Physics)"
                        print(f"[??] {self.last_done_reason}")
                        return True

                    drone_state = self.drone_states.get(drone_name, {})
                    current_oob_steps = int(drone_state.get('out_of_range_steps', 0))
                    if current_oob_steps > 0 and terminate_on_out_of_range:
                        self.last_done_reason = (
                            f"Drone {drone_name} Out of Range Reset "
                            f"(steps={current_oob_steps})"
                        )
                        print(f"[??] {self.last_done_reason}")
                        return True

                    if current_oob_steps >= max_oob_steps and terminate_on_out_of_range:
                        self.last_done_reason = (
                            f"Drone {drone_name} Out of Range Too Long "
                            f"({current_oob_steps} >= {max_oob_steps})"
                        )
                        print(f"[??] {self.last_done_reason}")
                        return True

                    with self.server.data_lock:
                        rd = self.server.unity_runtime_data.get(drone_name)
                        if rd and rd.position and rd.leader_position and rd.leader_scan_radius > 0:
                            dist = np.sqrt(
                                (rd.position.x - rd.leader_position.x) ** 2 +
                                (rd.position.y - rd.leader_position.y) ** 2 +
                                (rd.position.z - rd.leader_position.z) ** 2
                            )
                            threshold = rd.leader_scan_radius * severe_ratio
                            if dist > threshold:
                                self.last_done_reason = (
                                    f"Drone {drone_name} Severe Out of Range "
                                    f"({dist:.1f}m > {threshold:.1f}m)"
                                )
                                print(f"[??] {self.last_done_reason}")
                                return True

                    if hasattr(self.server, "battery_manager"):
                        battery_info = self.server.battery_manager.get_battery_info(drone_name)
                        if battery_info:
                            current_voltage = float(getattr(battery_info, "voltage", 4.2))
                            battery_status = getattr(battery_info, "status", None)
                            if current_voltage <= 3.2 + 1e-6 or battery_status == BatteryStatus.EMPTY:
                                self.last_done_reason = f"Drone {drone_name} Battery Empty ({current_voltage:.2f}V)"
                                print(f"[??] {self.last_done_reason}")
                                return True
                except Exception as exc:
                    logger.debug(f"Done check skipped for {drone_name}: {exc}")
                    continue

        return False

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
    
    def __init__(self, server=None, drone_names=None, config_path=None, step_duration=0.5):
        super(MultiDroneMovementEnv, self).__init__()
        
        self._lock_timeout_sec = 0.2
        self._state_timeout_sec = 2.0
        self._warned_lock_timeout = False
        self._warned_state_timeout = False

        self.server = server
        self.drone_names = drone_names if drone_names else ["UAV1"]
        self.num_drones = len(self.drone_names)
        self.step_duration = step_duration  # 物理步长（秒）
        
        # 当前控制的无人机索引（轮流控制）
        self.current_drone_idx = 0
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 统一环境配置加载逻辑 (方案B: 解耦物理规则)
        self._apply_unified_config()
        
        print(f"[OK] 多无人机 DQN 环境已加载配置并应用统一环境规则")
        print(f"  无人机数量: {self.num_drones}")
        print(f"  无人机列表: {self.drone_names}")
        self.verbose_step_logs = False
        
        # 动作空间: 6个离散动作（所有无人机共享）
        self.action_space = spaces.Discrete(6)
        
        # 观察空间: 24维（23维基础状态 + 1维无人机身份特征）
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(30,),
            dtype=np.float32
        )
        
        # 动作到位移的映射
        # 修正映射：0/1对应高度Y，2/3对应左右Z，4/5对应前后X
        self.action_step = self.config['movement']['step_size']
        self.action_repeat = max(1, int(self.config['movement'].get('action_repeat', 3)))
        self.action_map = {
            0: np.array([0, self.action_step, 0]),      # 上 (Y+)
            1: np.array([0, -self.action_step, 0]),     # 下 (Y-)
            2: np.array([0, 0, -self.action_step]),     # 左 (Z-)
            3: np.array([0, 0, self.action_step]),      # 右 (Z+)
            4: np.array([self.action_step, 0, 0]),      # 前 (X+)
            5: np.array([-self.action_step, 0, 0])      # 后 (X-)
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
                'out_of_range_steps': 0,
                'out_of_range_duration_sec': 0.0,
                'oob_started_at': None,
                'severe_out_of_range_hits': 0,
                'landed_hits': 0,
                'idle_hits': 0,
                'no_scan_hits': 0,
                'oob_no_return_hits': 0,
                'episode_reward': 0
            }

        self.last_done_reason = None
        self.step_count = 0
        self.total_episode_reward = 0
        self.episode_index = 0
        self.episode_start_time = time.time()
        self._first_reset = True

    def _step_log(self, message: str) -> None:
        if self.verbose_step_logs:
            print(message)
        
    def _apply_unified_config(self):
        """从统一源加载环境规则（终止阈值、电量参数、基础奖励）"""
        unified_env_cfg = None
        
        # 1. 尝试从 server 获取 (最优先)
        if self.server and hasattr(self.server, 'config_data') and hasattr(self.server.config_data, 'env_config'):
            unified_env_cfg = self.server.config_data.env_config
        else:
            # 2. 尝试从本地 apf_algorithm_config.json 加载
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                scanner_cfg_path = os.path.join(current_dir, "..", "..", "..", "apf_algorithm_config.json")
                if os.path.exists(scanner_cfg_path):
                    with open(scanner_cfg_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        unified_env_cfg = data.get('env_config')
            except Exception as e:
                logger.warning(f"无法加载统一环境配置: {e}")

        if not unified_env_cfg:
            # 使用默认终止配置
            self.term_cfg = self.config.get('termination_config', {
                "target_scan_ratio": 0.95,
                "max_collision_count": 1,
                "max_elapsed_time_sec": 300.0,
                "stagnation_timeout_sec": 30.0
            })
            return

        # DQN 本地配置优先，统一环境配置只作为缺省值。
        merged_term_cfg = dict(unified_env_cfg.get('termination', {}))
        merged_term_cfg.update(self.config.get('termination_config', {}))
        self.term_cfg = merged_term_cfg
        
        # --- B. 应用电量阈值 ---
        battery_cfg = unified_env_cfg.get('battery', {})
        if 'thresholds' not in self.config: self.config['thresholds'] = {}
        
        self.config['thresholds']['battery_low_threshold'] = battery_cfg.get('low_threshold', 3.5)
        self.config['thresholds']['battery_optimal_min'] = battery_cfg.get('optimal_min', 3.7)
        self.config['thresholds']['battery_optimal_max'] = battery_cfg.get('optimal_max', 4.1)
        
        # --- C. 应用基础奖励系数 ---
        base_rewards = unified_env_cfg.get('base_rewards', {})
        if 'rewards' not in self.config: self.config['rewards'] = {}
        
        # 映射统一奖励到本地配置
        reward_map = {
            'scan_reward': 'exploration',           # 新扫描奖励
            'out_of_range_penalty': 'out_of_range', # 越界惩罚
            'battery_low_penalty': 'battery_low_penalty',
            'battery_optimal_reward': 'battery_optimal_reward',
            'collision_penalty': 'collision',
            'step_penalty': 'step_penalty'
        }
        
        for u_key, local_key in reward_map.items():
            if u_key in base_rewards and local_key not in self.config['rewards']:
                self.config['rewards'][local_key] = float(base_rewards[u_key])

    def _load_config(self, config_path):
        """加载配置文件"""
        if config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, "..", "configs", "movement_dqn_config.json")
        
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
        """Reset the multi-drone episode state."""
        reason = getattr(self, 'last_done_reason', 'None')
        print(f"\n[DQN Multi] reset() called, previous done reason: {reason}")

        if seed is not None:
            np.random.seed(seed)

        if self._first_reset:
            self._first_reset = False
            print("[DQN Multi] First reset: skip world reset and only initialize state")
            if self.server and hasattr(self.server, 'reset_battery_voltage'):
                for drone_name in self.drone_names:
                    self.server.reset_battery_voltage(drone_name)
        else:
            if self.server:
                reset_reason = reason if reason not in (None, 'None') else 'manual'
                print(f"[DQN Multi] Episode finished, resetting environment (reason: {reset_reason})")
                self.server.reset_environment(reason=f"MultiDroneMovementEnv_{reset_reason}", reset_grid=True)
                if hasattr(self.server, 'reset_battery_voltage'):
                    for d_name in self.drone_names:
                        self.server.reset_battery_voltage(d_name)
                ready = _wait_for_server_ready(self.server, self.drone_names, timeout_sec=12.0)
                if not ready:
                    print("[DQN Multi] WARNING: runtime/grid data not fully ready after reset")
                print("[DQN Multi] Environment reset complete")

        print(f"[DQN Multi] Resetting state for {self.num_drones} drones")
        for drone_name in self.drone_names:
            self.drone_states[drone_name] = {
                'prev_scanned_cells': self._count_scanned_cells(),
                'prev_position': None,
                'prev_entropy_sum': self._get_total_entropy(),
                'collision_count': 0,
                'out_of_range_count': 0,
                'out_of_range_steps': 0,
                'out_of_range_duration_sec': 0.0,
                'oob_started_at': None,
                'severe_out_of_range_hits': 0,
                'landed_hits': 0,
                'idle_hits': 0,
                'no_scan_hits': 0,
                'oob_no_return_hits': 0,
                'episode_reward': 0
            }

        self.step_count = 0
        self.total_episode_reward = 0
        self.current_drone_idx = 0
        self.episode_start_time = time.time()
        self.episode_index += 1

        print("[DQN Multi] Fetching initial state")
        state = self._get_state(self.drone_names[0])
        print(f"[DQN Multi] reset() complete, state shape: {state.shape}")
        self.last_done_reason = None
        return state, {}

    def step(self, action):
        """Execute one action for the current drone in the multi-drone environment."""
        if hasattr(action, 'item'):
            action = action.item()
        action = int(action)

        self._step_log(f"[DQN Multi] step({action}) called")

        current_drone = self.drone_names[self.current_drone_idx]
        self._step_log(f"[DQN Multi] Current controlled drone: {current_drone}")

        self._step_log("[DQN Multi] Fetching current state...")
        current_state = self._get_state(current_drone)
        self._step_log("[DQN Multi] Current state fetched")

        self._step_log(f"[DQN Multi] Executing action {action}...")
        displacement = self.action_map[action]
        reward = 0.0
        next_state = current_state
        terminated = False
        step_out_of_range = False
        current_drone_state = self.drone_states[current_drone]
        current_drone_state['_step_collision_detected'] = False
        current_drone_state['_step_collision_clear'] = False
        current_drone_state['_collision_penalty_applied_this_step'] = False

        for repeat_idx in range(self.action_repeat):
            if self.server:
                self._apply_movement(current_drone, displacement)
                time.sleep(self.step_duration)
                self._step_log(
                    f"[DQN Multi] Physics step wait complete "
                    f"({self.step_duration}s, repeat {repeat_idx + 1}/{self.action_repeat})"
                )
            self._step_log("[DQN Multi] Action execution complete")

            self._step_log("[DQN Multi] Fetching next state...")
            next_state = self._get_state(current_drone)
            self._step_log("[DQN Multi] Next state fetched")

            self._step_log("[DQN Multi] Calculating reward...")
            sub_reward = self._calculate_reward(current_drone, action, current_state, next_state)
            reward += sub_reward
            self.drone_states[current_drone]['episode_reward'] += sub_reward
            self.total_episode_reward += sub_reward
            step_out_of_range = step_out_of_range or bool(next_state[16] > 0.5)
            self._step_log(f"[DQN Multi] Reward calculation complete: {sub_reward:.2f}")

            current_state = next_state
            if terminated:
                break

        next_step_index = self.step_count + 1
        oob_checks_active = self._oob_checks_active(next_step_index)
        if step_out_of_range and oob_checks_active:
            current_drone_state['out_of_range_count'] = int(
                current_drone_state.get('out_of_range_count', 0)
            ) + 1
            current_drone_state['out_of_range_steps'] = int(
                current_drone_state.get('out_of_range_steps', 0)
            ) + 1
        else:
            current_drone_state['out_of_range_steps'] = 0

        if current_drone_state.get('_step_collision_detected', False):
            current_drone_state['collision_count'] = int(
                current_drone_state.get('collision_count', 0)
            ) + 1
        elif (
            current_drone_state.get('_step_collision_clear', False)
            and current_drone_state.get('collision_count', 0) > 0
        ):
            current_drone_state['collision_count'] = max(
                0,
                int(current_drone_state.get('collision_count', 0)) - 1
            )

        current_drone_state.pop('_step_collision_detected', None)
        current_drone_state.pop('_step_collision_clear', None)
        current_drone_state.pop('_collision_penalty_applied_this_step', None)

        self.step_count = next_step_index
        if not terminated:
            terminated = self._check_done()

        if self.server and hasattr(self.server, 'set_training_stats'):
            self.server.set_training_stats(
                episode=self.episode_index,
                step=self.step_count,
                reward=float(reward),
                total_reward=float(self.total_episode_reward)
            )

        self.current_drone_idx = (self.current_drone_idx + 1) % self.num_drones

        self._step_log("[DQN Multi] Checking termination conditions...")
        if terminated and self.last_done_reason and 'Out of Range' in self.last_done_reason:
            terminal_penalty = float(self.config.get('rewards', {}).get('terminal_out_of_range_penalty', -120.0))
            reward += terminal_penalty
            self.drone_states[current_drone]['episode_reward'] += terminal_penalty
            self.total_episode_reward += terminal_penalty
            self._step_log(f"[DQN Multi] Applied terminal out-of-range penalty: {terminal_penalty:.2f}")
        self._step_log(f"[DQN Multi] Termination check result: {terminated}")
        truncated = False

        leader_distance = None
        is_out_of_range = False
        current_drone_state = self.drone_states.get(current_drone, {})
        try:
            if self.server:
                with self.server.data_lock:
                    rd = self.server.unity_runtime_data.get(current_drone)
                    if rd and rd.position and rd.leader_position and rd.leader_scan_radius > 0:
                        leader_distance = float(np.sqrt(
                            (rd.position.x - rd.leader_position.x) ** 2 +
                            (rd.position.y - rd.leader_position.y) ** 2 +
                            (rd.position.z - rd.leader_position.z) ** 2
                        ))
                        is_out_of_range = bool(leader_distance > rd.leader_scan_radius)
        except Exception:
            leader_distance = None

        info = {
            'drone_name': current_drone,
            'action': action,
            'displacement': displacement.tolist(),
            'scanned_cells': self._count_scanned_cells(),
            'total_reward': self.total_episode_reward,
            'step_count': self.step_count,
            'current_drone_idx': self.current_drone_idx,
            'leader_distance': leader_distance,
            'is_out_of_range': is_out_of_range,
            'out_of_range_steps': int(current_drone_state.get('out_of_range_steps', 0)),
            'out_of_range_count': int(current_drone_state.get('out_of_range_count', 0)),
            'out_of_range_duration_sec': float(current_drone_state.get('out_of_range_duration_sec', 0.0)),
            'current_drone_reward': float(current_drone_state.get('episode_reward', 0.0)),
            'last_done_reason': self.last_done_reason,
        }

        next_drone = self.drone_names[self.current_drone_idx]
        self._step_log(f"[DQN Multi] Fetching next drone state: {next_drone}")
        next_observation = self._get_state(next_drone)
        self._step_log("[DQN Multi] step() complete")

        return next_observation, reward, terminated, truncated, info

    def _oob_checks_active(self, step_index=None):
        """Return whether OOR counting/termination should be active for this decision."""
        cfg_thresh = self.config.get('thresholds', {})
        post_reset_grace_sec = float(cfg_thresh.get('post_reset_grace_sec', 6.0))
        min_steps_before_oob_checks = max(
            0, int(cfg_thresh.get('min_steps_before_oob_checks', 24))
        )
        elapsed_time = time.time() - self.episode_start_time
        effective_step_index = self.step_count if step_index is None else int(step_index)
        return (
            elapsed_time >= post_reset_grace_sec
            and effective_step_index >= min_steps_before_oob_checks
        )


    def _get_state(self, drone_name):
        """Return a 30D observation for the selected drone."""
        if not self.server:
            return np.random.randn(30).astype(np.float32)

        deadline = time.time() + self._state_timeout_sec
        while True:
            if time.time() > deadline:
                if not self._warned_state_timeout:
                    logger.warning(f"[DQN Multi] State fetch timed out ({self._state_timeout_sec}s); returning zero observation")
                    self._warned_state_timeout = True
                return np.zeros(30, dtype=np.float32)

            acquired = self.server.data_lock.acquire(timeout=self._lock_timeout_sec)
            if not acquired:
                if not self._warned_lock_timeout:
                    logger.warning(f"[DQN多机环境] 获取 data_lock 超时({self._lock_timeout_sec}s)，将重试")
                    self._warned_lock_timeout = True
                continue

            try:
                runtime_data = self.server.unity_runtime_data.get(drone_name)
                if not runtime_data:
                    return np.zeros(26, dtype=np.float32)

                pos = runtime_data.position
                position = np.array([pos.x, pos.y, pos.z], dtype=np.float32)

                vel = runtime_data.velocity
                velocity = np.array([vel.x, vel.y, vel.z], dtype=np.float32)

                fwd = runtime_data.forward
                forward = np.array([fwd.x, fwd.y, fwd.z], dtype=np.float32)

                if runtime_data.leader_position:
                    leader_rel = np.array([
                        runtime_data.leader_position.x - pos.x,
                        runtime_data.leader_position.y - pos.y,
                        runtime_data.leader_position.z - pos.z
                    ], dtype=np.float32)
                else:
                    leader_rel = np.zeros(3, dtype=np.float32)

                leader_distance = float(np.linalg.norm(leader_rel))
                leader_scan_radius = max(float(runtime_data.leader_scan_radius), 1e-6)
                horizontal_leader_distance = float(np.linalg.norm([leader_rel[0], leader_rel[2]]))
                is_out_of_range = 1.0 if leader_distance > leader_scan_radius else 0.0
                leader_info = np.array([leader_distance, is_out_of_range], dtype=np.float32)
                leader_recovery_info = np.array([
                    min(3.0, leader_distance / leader_scan_radius),
                    min(3.0, horizontal_leader_distance / leader_scan_radius),
                ], dtype=np.float32)
                if horizontal_leader_distance > 1e-6:
                    leader_horizontal_unit = np.array([
                        leader_rel[0] / horizontal_leader_distance,
                        leader_rel[2] / horizontal_leader_distance,
                    ], dtype=np.float32)
                else:
                    leader_horizontal_unit = np.zeros(2, dtype=np.float32)

                drone_state = self.drone_states.get(drone_name, {})
                max_oob_duration_sec = max(
                    float(self.config.get('thresholds', {}).get('max_out_of_range_duration_sec', 1.0)),
                    1e-6,
                )
                max_oob_steps = max(
                    1,
                    int(self.config.get('thresholds', {}).get('max_out_of_range_steps', 1))
                )
                oob_state_info = np.array([
                    min(
                        3.0,
                        float(drone_state.get('out_of_range_duration_sec', 0.0)) / max_oob_duration_sec,
                    ),
                    min(
                        3.0,
                        float(drone_state.get('out_of_range_steps', 0)) / float(max_oob_steps),
                    ),
                ], dtype=np.float32)
            finally:
                self.server.data_lock.release()

            acquired_grid = self.server.grid_lock.acquire(timeout=self._lock_timeout_sec)
            if not acquired_grid:
                if not self._warned_lock_timeout:
                    logger.warning(f"[DQN多机环境] 获取 grid_lock 超时({self._lock_timeout_sec}s)，将重试")
                    self._warned_lock_timeout = True
                continue

            try:
                grid_data = self.server.grid_data
                if not grid_data or not getattr(grid_data, 'cells', None):
                    entropy_info = np.array([50.0, 50.0, 0.0], dtype=np.float32)
                    scan_info = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    min_dist_info = np.array([100.0], dtype=np.float32)
                else:
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

                    scanned_threshold = self.config['thresholds']['scanned_entropy']
                    scanned_count = sum(1 for cell in grid_data.cells if cell.entropy < scanned_threshold)
                    total_cells = len(grid_data.cells)
                    unscanned_count = total_cells - scanned_count
                    scan_ratio = scanned_count / total_cells if total_cells > 0 else 0.0
                    scanned_ratio = float(scanned_count) / total_cells if total_cells > 0 else 0.0
                    unscanned_ratio = float(unscanned_count) / total_cells if total_cells > 0 else 0.0
                    scan_info = np.array([scan_ratio, scanned_ratio, unscanned_ratio], dtype=np.float32)

                    min_dist_info = np.array([100.0], dtype=np.float32)
            finally:
                self.server.grid_lock.release()

            min_distance = self._get_min_distance_to_others(drone_name)
            min_dist_info = np.array([min_distance], dtype=np.float32)

            battery_info = self._get_battery_info_for_drone(drone_name)
            drone_idx = self.drone_names.index(drone_name)
            if self.num_drones > 1:
                drone_identity = np.array(
                    [drone_idx / float(self.num_drones - 1)], dtype=np.float32
                )
            else:
                drone_identity = np.array([0.0], dtype=np.float32)

            state = np.concatenate([
                position,
                velocity,
                forward,
                entropy_info,
                leader_rel,
                leader_info,
                leader_recovery_info,
                leader_horizontal_unit,
                oob_state_info,
                scan_info,
                min_dist_info,
                battery_info,
                drone_identity,
            ])

            return state.astype(np.float32)

        try:
            return np.zeros(30, dtype=np.float32)
        except Exception:
            return np.zeros(30, dtype=np.float32)
    
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
            return np.array([4.2, 1.0], dtype=np.float32)  # 默认值：满电
        
        try:
            voltage = self.server.get_battery_voltage(drone_name)
            battery_info = self.server.battery_manager.get_battery_info(drone_name)
            if battery_info:
                percentage = battery_info.get_remaining_percentage()
                return np.array([voltage, percentage / 100.0], dtype=np.float32)
            else:
                return np.array([voltage, 1.0], dtype=np.float32)
        except:
            return np.array([4.2, 1.0], dtype=np.float32)
    
    def _apply_movement(self, drone_name, displacement):
        """应用移动到无人机（通过AlgorithmServer的DQN控制模式）"""
        if not self.server:
            return
        
        try:
            displacement = _apply_low_altitude_guard(
                self.server,
                drone_name,
                displacement,
                self.config,
                self.action_step,
            )
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
            
            update_interval = 0.5
            if (
                hasattr(self.server, 'config_data')
                and hasattr(self.server.config_data, 'updateInterval')
            ):
                update_interval = float(self.server.config_data.updateInterval)

            cycle_duration = (
                float(self.step_duration)
                * float(self.action_repeat)
                * float(max(1, self.num_drones))
            )
            pulse_duration = max(
                cycle_duration,
                update_interval * 1.05,
            )
            self.server.set_dqn_movement(
                drone_name,
                move_direction,
                duration_sec=pulse_duration,
            )
            
        except Exception as e:
            logger.error(f"应用移动失败: {str(e)}")
    
    def _calculate_reward(self, drone_name, action, current_state, next_state):
        """计算奖励"""
        reward = 0.0
        drone_state = self.drone_states[drone_name]
        applied_oob_penalty = False
        
        # 0. 计算稳定性系数 (基于到 Leader 的距离)
        stability_factor = 1.0
        dist_to_leader = next_state[15]  # 观察状态中的 Leader 距离
        is_out_of_range = next_state[16] > 0.5
        
        cfg_reward = self.config['rewards']
        cfg_thresh = self.config['thresholds']
        
        # 计算上一步到领导者的距离(用于判断是否在返回)
        prev_dist_to_leader = current_state[15] if current_state is not None else dist_to_leader
        is_returning = (dist_to_leader < prev_dist_to_leader) and is_out_of_range
        prev_leader_rel = current_state[12:15] if current_state is not None else next_state[12:15]
        next_leader_rel = next_state[12:15]
        prev_horizontal_dist = float(np.linalg.norm([prev_leader_rel[0], prev_leader_rel[2]]))
        horizontal_dist = float(np.linalg.norm([next_leader_rel[0], next_leader_rel[2]]))
        is_returning_horizontally = horizontal_dist < prev_horizontal_dist - 1e-4
        
        try:
            with self.server.data_lock:
                runtime_data = self.server.unity_runtime_data[drone_name]
                radius = runtime_data.leader_scan_radius
                if radius > 0:
                    dist_ratio = dist_to_leader / radius
                    
                    # 从配置获取比例
                    safe_ratio = cfg_thresh.get('stability_safe_ratio', 0.7)
                    penalty_ratio = cfg_thresh.get('stability_penalty_ratio', 0.8)
                    
                    # 圈外阶段只允许学习“尽快回圈”，避免边界来回刷分
                    if dist_ratio > 1.0:
                        stability_factor = 0.0
                        applied_oob_penalty = True
                        urgency_gain = float(cfg_reward.get('oob_urgency_gain', 2.5))
                        urgency_scale = 1.0 + max(0.0, dist_ratio - 1.0) * urgency_gain

                        oob_step_penalty = cfg_reward.get('out_of_range_step_penalty', -8.0)
                        return_bonus = cfg_reward.get('return_to_range_bonus', 5.0)
                        progress_weight = cfg_reward.get('return_progress_weight', 40.0)
                        alignment_weight = cfg_reward.get('return_alignment_weight', 8.0)
                        horizontal_alignment_weight = cfg_reward.get(
                            'return_horizontal_alignment_weight',
                            alignment_weight * 1.5,
                        )
                        outward_progress_penalty = cfg_reward.get('outward_progress_penalty', 10.0)
                        outward_alignment_penalty = cfg_reward.get('outward_alignment_penalty', 5.0)
                        vertical_action_penalty = float(cfg_reward.get('oob_vertical_action_penalty', 4.0))

                        reward += oob_step_penalty * urgency_scale

                        rd = self.server.unity_runtime_data[drone_name]
                        dir_to_leader = np.array([
                            rd.leader_position.x - rd.position.x,
                            rd.leader_position.y - rd.position.y,
                            rd.leader_position.z - rd.position.z
                        ])
                        norm = np.linalg.norm(dir_to_leader)
                        alignment = 0.0
                        if norm > 1e-6:
                            ideal_dir = dir_to_leader / norm
                            actual_dir = self.action_map[action] / np.linalg.norm(self.action_map[action])
                            alignment = float(np.dot(ideal_dir, actual_dir))

                        horizontal_dir = np.array([dir_to_leader[0], dir_to_leader[2]], dtype=np.float32)
                        horizontal_norm = np.linalg.norm(horizontal_dir)
                        horizontal_alignment = 0.0
                        action_horizontal = np.array([self.action_map[action][0], self.action_map[action][2]], dtype=np.float32)
                        action_horizontal_norm = np.linalg.norm(action_horizontal)
                        if horizontal_norm > 1e-6 and action_horizontal_norm > 1e-6:
                            horizontal_alignment = float(
                                np.dot(horizontal_dir / horizontal_norm, action_horizontal / action_horizontal_norm)
                            )

                        if is_returning or is_returning_horizontally:
                            drone_state['oob_no_return_hits'] = 0
                            progress_reward = max(0.0, prev_dist_to_leader - dist_to_leader) * progress_weight
                            horizontal_progress_reward = max(0.0, prev_horizontal_dist - horizontal_dist) * progress_weight
                            reward += (return_bonus * urgency_scale + progress_reward + horizontal_progress_reward)
                            if alignment > 0:
                                reward += alignment * alignment_weight * urgency_scale
                            if horizontal_alignment > 0:
                                reward += horizontal_alignment * horizontal_alignment_weight * urgency_scale
                        else:
                            drone_state['oob_no_return_hits'] = int(
                                drone_state.get('oob_no_return_hits', 0)
                            ) + 1
                            reward += cfg_reward.get('out_of_range', -30.0) * urgency_scale
                            outward_progress = max(0.0, dist_to_leader - prev_dist_to_leader)
                            outward_horizontal_progress = max(0.0, horizontal_dist - prev_horizontal_dist)
                            no_return_penalty = float(
                                cfg_reward.get('no_return_progress_penalty', 3.0)
                            )
                            no_return_hits_cap = max(
                                1,
                                int(cfg_reward.get('no_return_progress_hits_cap', 4))
                            )
                            reward -= min(
                                drone_state['oob_no_return_hits'],
                                no_return_hits_cap
                            ) * no_return_penalty * urgency_scale
                            if outward_progress > 0.0:
                                reward -= outward_progress * outward_progress_penalty * urgency_scale
                            if outward_horizontal_progress > 0.0:
                                reward -= outward_horizontal_progress * outward_progress_penalty * urgency_scale
                            if alignment < 0.0:
                                reward += alignment * outward_alignment_penalty * urgency_scale
                            if horizontal_alignment < 0.0:
                                reward += horizontal_alignment * (outward_alignment_penalty * 1.5) * urgency_scale

                        current_height = float(rd.position.y)
                        if abs(float(self.action_map[action][1])) > 1e-6 and current_height >= float(
                            cfg_thresh.get('low_altitude_recovery_height', 1.2)
                        ):
                            reward -= vertical_action_penalty * urgency_scale
                    elif dist_ratio > float(cfg_thresh.get('preemptive_return_ratio', 0.82)):
                        preemptive_progress_weight = float(
                            cfg_reward.get('preemptive_return_progress_weight', 6.0)
                        )
                        preemptive_alignment_weight = float(
                            cfg_reward.get('preemptive_return_alignment_weight', 2.0)
                        )
                        inward_progress = max(0.0, prev_horizontal_dist - horizontal_dist)
                        if inward_progress > 0.0:
                            reward += inward_progress * preemptive_progress_weight
                        action_horizontal = np.array([self.action_map[action][0], self.action_map[action][2]], dtype=np.float32)
                        action_horizontal_norm = np.linalg.norm(action_horizontal)
                        leader_horizontal = np.array([next_leader_rel[0], next_leader_rel[2]], dtype=np.float32)
                        leader_horizontal_norm = np.linalg.norm(leader_horizontal)
                        if action_horizontal_norm > 1e-6 and leader_horizontal_norm > 1e-6:
                            inward_alignment = float(
                                np.dot(leader_horizontal / leader_horizontal_norm, action_horizontal / action_horizontal_norm)
                            )
                            if inward_alignment > 0.0:
                                reward += inward_alignment * preemptive_alignment_weight
                        drone_state['oob_no_return_hits'] = 0
                    elif dist_ratio > safe_ratio:
                        # safe_ratio - 1.0 之间线性衰减
                        stability_factor = 1.0 - (dist_ratio - safe_ratio) / (1.0 - safe_ratio) * 0.5  # 从1.0衰减到0.5
                        drone_state['oob_no_return_hits'] = 0
                    else:
                        drone_state['oob_no_return_hits'] = 0
                    
                    # 【修改】稳定性惩罚降低强度,避免过度打压
                    if dist_ratio > penalty_ratio:
                        penalty_weight = cfg_reward.get('stability_penalty_weight', 20.0)
                        # 惩罚上限设置,避免单步惩罚过重
                        penalty = min((dist_ratio - penalty_ratio) * penalty_weight, 30.0)
                        reward -= penalty
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

        made_scan_progress = (new_cells > 0) or (entropy_reduction > 0.0)

        prev_is_out_of_range = bool(current_state[16] > 0.5) if current_state is not None else False
        if prev_is_out_of_range and not is_out_of_range:
            reward += float(
                cfg_reward.get(
                    'return_to_range_success_bonus',
                    cfg_reward.get('return_to_range_bonus', 5.0) * 1.5
                )
            )
            
        # 3. 【优化】局部高熵探索奖励
        # 修正索引：9-平均熵, 10-最大熵
        local_avg_entropy = next_state[9]
        local_max_entropy = next_state[10]
            
        if made_scan_progress and local_max_entropy > cfg_thresh.get('high_entropy_threshold', 40.0):
            entropy_exploration_bonus = cfg_reward.get('high_entropy_exploration', 5.0)
            reward += entropy_exploration_bonus * stability_factor
            
        if drone_state['prev_position']:
            prev_local_avg_entropy = current_state[9]
            entropy_increase = local_avg_entropy - prev_local_avg_entropy
            if made_scan_progress and entropy_increase > 0:
                entropy_gradient_reward = entropy_increase * cfg_reward.get('entropy_gradient_bonus', 2.0)
                reward += entropy_gradient_reward * stability_factor
            
        # 4. 【新增】高度控制奖励/惩罚
        try:
            with self.server.data_lock:
                runtime_data = self.server.unity_runtime_data[drone_name]
                pos = runtime_data.position
                current_height = pos.y # 修正：Unity中Y轴是高度
                    
                min_scan_height = cfg_thresh.get('min_scan_height', 2.0)
                max_scan_height = cfg_thresh.get('max_scan_height', 15.0)
                optimal_height = cfg_thresh.get('optimal_scan_height', 8.0)
                height_penalty_base = cfg_reward.get('height_penalty', -5.0)
                    
                if current_height < min_scan_height:
                    reward += height_penalty_base * (min_scan_height - current_height)
                elif current_height > max_scan_height:
                    reward += height_penalty_base * (current_height - max_scan_height)
                elif made_scan_progress and abs(current_height - optimal_height) < 1.5:
                    reward += cfg_reward.get('optimal_height_bonus', 1.0)
                    
                drone_state['prev_position'] = pos
        except Exception as e:
            logger.debug(f"高度奖励计算失败: {str(e)}")

        # 4.1. 连续静止惩罚
        idle_threshold = float(cfg_thresh.get('idle_distance_threshold', 0.08))
        idle_penalty = float(cfg_reward.get('idle_step_penalty', -4.0))
        post_reset_grace_sec = float(cfg_thresh.get('post_reset_grace_sec', 6.0))
        if current_state is not None and self.episode_start_time is not None:
            movement_distance = float(np.linalg.norm(next_state[:3] - current_state[:3]))
            if (
                movement_distance < idle_threshold
                and (time.time() - self.episode_start_time) >= post_reset_grace_sec
            ):
                drone_state['idle_hits'] = int(drone_state.get('idle_hits', 0)) + 1
                reward += idle_penalty * min(drone_state['idle_hits'], 3)
            else:
                drone_state['idle_hits'] = 0

        # 4.2. 连续没有扫描进展时增加惩罚，压制“靠形状奖励混时间”的策略。
        no_scan_penalty = float(cfg_reward.get('no_scan_progress_penalty', -2.0))
        no_scan_grace_sec = float(cfg_thresh.get('scan_progress_grace_sec', post_reset_grace_sec))
        if self.episode_start_time is not None and (time.time() - self.episode_start_time) >= no_scan_grace_sec:
            if made_scan_progress:
                drone_state['no_scan_hits'] = 0
            else:
                drone_state['no_scan_hits'] = int(drone_state.get('no_scan_hits', 0)) + 1
                reward += no_scan_penalty * min(drone_state['no_scan_hits'], 5)

        # 5. 碰撞惩罚与容忍机制
        min_distance = self._get_min_distance_to_others(drone_name)
        collision_threshold = cfg_thresh['collision_distance']

        if min_distance < collision_threshold * 0.8:
            drone_state['_step_collision_detected'] = True
            if not drone_state.get('_collision_penalty_applied_this_step', False):
                reward += cfg_reward['collision']
                drone_state['_collision_penalty_applied_this_step'] = True
        else:
            if min_distance > collision_threshold * 1.5:
                drone_state['_step_collision_clear'] = True

        # 6. 超出Leader范围惩罚
        # 仅施加单步惩罚；连续越界计数在 step() 里按“一个决策一步”统一维护，
        # 避免 action_repeat 的每个子步都被当成一次完整越界。
        if is_out_of_range and not applied_oob_penalty:
            reward += cfg_reward['out_of_range']

        # 7. 步骤惩罚
        reward += cfg_reward['step_penalty']
        
        # 8. ???????
        if self.server and hasattr(self.server, 'get_battery_voltage'):
            try:
                current_voltage = self.server.get_battery_voltage(drone_name)
                battery_info = self.server.battery_manager.get_battery_info(drone_name)
                if battery_info:
                    if 'battery_low_threshold' in cfg_thresh:
                        if current_voltage < cfg_thresh['battery_low_threshold']:
                            penalty = cfg_reward.get('battery_low_penalty', 10.0)
                            reward -= penalty

                    if 'battery_optimal_min' in cfg_thresh and 'battery_optimal_max' in cfg_thresh:
                        opt_min = cfg_thresh['battery_optimal_min']
                        opt_max = cfg_thresh['battery_optimal_max']
                        if made_scan_progress and opt_min <= current_voltage <= opt_max:
                            bonus = cfg_reward.get('battery_optimal_reward', 2.0)
                            reward += bonus

                if hasattr(self.server, 'update_battery_voltage'):
                    step_norm = float(np.linalg.norm(self.action_map[action]))
                    base_step = max(self.action_step, 1e-6)
                    action_intensity = min(1.0, max(0.0, step_norm / base_step))
                    self.server.update_battery_voltage(drone_name, action_intensity)
            except Exception as e:
                logger.debug(f"????????: {str(e)}")

        return reward

    def _check_done(self):
        """Check unified termination conditions for the multi-drone episode."""
        elapsed_time = time.time() - self.episode_start_time
        cfg_thresh = self.config.get('thresholds', {})
        post_reset_grace_sec = float(cfg_thresh.get('post_reset_grace_sec', 6.0))
        landed_grace_sec = float(cfg_thresh.get('landed_grace_sec', max(8.0, post_reset_grace_sec)))
        landed_confirm_steps = max(1, int(cfg_thresh.get('landed_confirm_steps', 3)))
        min_steps_before_oob_checks = max(
            0, int(cfg_thresh.get('min_steps_before_oob_checks', 24))
        )
        severe_oob_enabled = bool(cfg_thresh.get('severe_out_of_range_enabled', False))
        severe_confirm_steps = max(1, int(cfg_thresh.get('severe_out_of_range_confirm_steps', 3)))

        if elapsed_time >= self.term_cfg['max_elapsed_time_sec']:
            self.last_done_reason = f"Timeout ({elapsed_time:.1f}s >= {self.term_cfg['max_elapsed_time_sec']}s)"
            print(f"[Done] {self.last_done_reason}")
            return True

        scan_ratio = self._get_scan_ratio()
        if scan_ratio >= self.term_cfg['target_scan_ratio']:
            self.last_done_reason = f"Target Scan Ratio Reached ({scan_ratio:.2%} >= {self.term_cfg['target_scan_ratio']:.2%})"
            print(f"[Done] {self.last_done_reason}")
            return True

        total_collisions = sum(state['collision_count'] for state in self.drone_states.values())
        if total_collisions >= self.term_cfg['max_collision_count']:
            self.last_done_reason = f"Collision Limit Reached ({total_collisions} >= {self.term_cfg['max_collision_count']})"
            print(f"[Done] {self.last_done_reason}")
            return True

        terminate_on_out_of_range = bool(cfg_thresh.get('terminate_on_out_of_range', True))
        max_oob_steps = max(1, int(cfg_thresh.get('max_out_of_range_steps', 1)))
        max_oob_duration_sec = float(
            cfg_thresh.get('max_out_of_range_duration_sec', float('inf'))
        )
        severe_ratio = float(cfg_thresh.get('severe_out_of_range_ratio', 1.05))

        if self.server:
            for drone_name in self.drone_names:
                try:
                    state = self.server.drone_controller.get_vehicle_state(drone_name)
                    drone_state = self.drone_states.get(drone_name, {})
                    if not state.get("flying", True) and elapsed_time >= landed_grace_sec:
                        drone_state['landed_hits'] = int(drone_state.get('landed_hits', 0)) + 1
                        if drone_state['landed_hits'] >= landed_confirm_steps:
                            self.last_done_reason = f"Drone {drone_name} Landed (Physics)"
                            print(f"[Done] {self.last_done_reason}")
                            return True
                    else:
                        drone_state['landed_hits'] = 0

                    current_oob_steps = int(drone_state.get('out_of_range_steps', 0))
                    oob_checks_active = self._oob_checks_active()

                    with self.server.data_lock:
                        rd = self.server.unity_runtime_data.get(drone_name)
                        if rd and rd.position and rd.leader_position and rd.leader_scan_radius > 0:
                            dist = np.sqrt(
                                (rd.position.x - rd.leader_position.x) ** 2 +
                                (rd.position.y - rd.leader_position.y) ** 2 +
                                (rd.position.z - rd.leader_position.z) ** 2
                            )
                            threshold = rd.leader_scan_radius * severe_ratio
                            is_currently_oob = dist > rd.leader_scan_radius
                            if oob_checks_active and is_currently_oob:
                                started_at = drone_state.get('oob_started_at')
                                if started_at is None:
                                    drone_state['oob_started_at'] = elapsed_time
                                    drone_state['out_of_range_duration_sec'] = 0.0
                                else:
                                    drone_state['out_of_range_duration_sec'] = max(
                                        0.0, elapsed_time - float(started_at)
                                    )
                            else:
                                drone_state['oob_started_at'] = None
                                drone_state['out_of_range_duration_sec'] = 0.0

                            current_oob_duration = float(
                                drone_state.get('out_of_range_duration_sec', 0.0)
                            )
                            if (
                                terminate_on_out_of_range
                                and oob_checks_active
                                and (
                                    current_oob_steps >= max_oob_steps
                                    or current_oob_duration >= max_oob_duration_sec
                                )
                            ):
                                self.last_done_reason = (
                                    f"Drone {drone_name} Out of Range Too Long "
                                    f"({current_oob_duration:.1f}s / {current_oob_steps} steps)"
                                )
                                print(f"[Done] {self.last_done_reason}")
                                return True

                            if dist > threshold:
                                drone_state['severe_out_of_range_hits'] = int(
                                    drone_state.get('severe_out_of_range_hits', 0)
                                ) + 1
                                if (
                                    severe_oob_enabled
                                    and oob_checks_active
                                    and drone_state['severe_out_of_range_hits'] >= severe_confirm_steps
                                ):
                                    self.last_done_reason = f"Drone {drone_name} Severe Out of Range ({dist:.1f}m > {threshold:.1f}m)"
                                    print(f"[Done] {self.last_done_reason}")
                                    return True
                            else:
                                drone_state['severe_out_of_range_hits'] = 0

                    if hasattr(self.server, "battery_manager"):
                        battery_info = self.server.battery_manager.get_battery_info(drone_name)
                        if battery_info:
                            current_voltage = float(getattr(battery_info, "voltage", 4.2))
                            battery_status = getattr(battery_info, "status", None)
                            if current_voltage <= 3.2 + 1e-6 or battery_status == BatteryStatus.EMPTY:
                                self.last_done_reason = f"Drone {drone_name} Battery Empty ({current_voltage:.2f}V)"
                                print(f"[Done] {self.last_done_reason}")
                                return True
                except Exception as exc:
                    logger.debug(f"Done check skipped for {drone_name}: {exc}")
                    continue

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
