"""
简单的权重学习环境
使用Stable-Baselines3训练APF权重系数
"""

import numpy as np
import gym
from gym import spaces
import os
import time
import json
from collections import deque

try:
    from multirotor.Algorithm.battery_data import BatteryStatus
except ImportError:
    try:
        from Algorithm.battery_data import BatteryStatus
    except ImportError:
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from multirotor.Algorithm.battery_data import BatteryStatus

try:
    from configs.crazyflie_reward_config import CrazyflieRewardConfig
except ImportError:
    try:
        from ..configs.crazyflie_reward_config import CrazyflieRewardConfig
    except ImportError:
        import sys
        import os

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
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
        reset_grid_entropy=True,  # 新增：是否在episode重置时重置网格熵值
        step_duration=5.0,
        safety_limit=True,
        max_weight_delta=0.5,
        action_smoothing=0.35,
        weight_edge_margin=0.35,
        weight_edge_push=0.25,
        min_distance_weight=1.0,
        max_entropy_weight=4.5,
        max_leader_weight=4.5,
    ):
        super(SimpleWeightEnv, self).__init__()

        self.server = server
        self.drone_name = drone_name
        self.reset_unity = reset_unity  # 是否每次episode重置Unity环境
        self.reset_grid_entropy = reset_grid_entropy  # 是否重置网格熵值（默认True，每个Episode完整重置）
        self.step_duration = step_duration  # 每步飞行时长（秒）

        # 加载奖励配置（与实体训练一致）
        if reward_config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            reward_config_path = os.path.join(
                current_dir, "..", "configs", "crazyflie_reward_config.json"
            )

        self.reward_config = CrazyflieRewardConfig(reward_config_path)

        # 统一终止配置
        self.term_cfg = {
            "target_scan_ratio": 0.25,
            "max_collision_count": 6,   # 与DQN主实验口径对齐
            "max_elapsed_time_sec": 300.0,  # 与DQN主实验口径对齐
            "out_of_range_reset_enabled": True,
            "out_of_range_continuous_count": 12,  # 12 * 2s ≈ 24s，与DQN 24s OOR 时长对齐
        }

        # 边界控制配置
        self.boundary_cfg = {
            "warning_ratio": 0.7,  # 警告阈值比例（距离边界70%开始警告）
            "danger_ratio": 0.85,  # 危险阈值比例（距离边界85%开始危险惩罚）
            "warning_penalty": 2.0,  # 警告区渐进惩罚（每步）
            "danger_penalty": 5.0,  # 危险区渐进惩罚（每步）
            "center_reward": 3.0,  # 朝向中心飞行奖励
            "max_reset_penalty": 15.0,  # 出圈重置最大惩罚（降低避免过度惩罚）
        }
        self.reward_shaping_cfg = {
            "base_step_cost": 2.0,
            "no_progress_penalty": 3.0,
            "scan_complete_bonus": 120.0,
            "time_limit_completion_bonus": 30.0,
            "failure_base_penalty": 25.0,
            "poor_progress_scan_ratio": 5.0,
            "healthy_time_limit_scan_ratio": 8.0,
            "poor_progress_penalty": 80.0,
            "early_failure_steps": 3,
            "early_failure_penalty": 120.0,
            "collision_terminal_penalty": 40.0,
            "short_collision_penalty": 80.0,
            "short_collision_reward_cap": -30.0,
            "out_of_range_terminal_penalty": 35.0,
            "time_limit_low_progress_penalty": 20.0,
            "battery_reward_scale": 0.25,
            "center_reward_scale": 0.5,
            "obstacle_warning_distance": 4.0,
            "obstacle_danger_distance": 2.2,
            "obstacle_warning_penalty": 3.0,
            "obstacle_danger_penalty": 9.0,
            "hotspot_obstacle_warning_distance": 6.5,
            "hotspot_obstacle_danger_distance": 3.8,
            "hotspot_obstacle_warning_penalty": 8.0,
            "hotspot_obstacle_danger_penalty": 20.0,
            "collision_hotspot_center_x": -0.50,
            "collision_hotspot_center_z": -10.50,
            "collision_hotspot_warning_radius": 2.80,
            "collision_hotspot_danger_radius": 1.40,
            "collision_hotspot_warning_penalty": 6.0,
            "collision_hotspot_danger_penalty": 16.0,
            "collision_hotspot_corridor_half_width_x": 2.20,
            "collision_hotspot_corridor_half_width_z": 2.70,
            "collision_hotspot_corridor_penalty": 6.0,
        }

        # 出圈计数器
        self.out_of_range_count = 0
        self._out_of_range_continuous_count = 0  # 连续出圈步数计数器

        # 应用统一环境配置（方案 B：解耦物理规则与训练参数）
        self._apply_unified_config()

        print("[OK] 训练环境已加载奖励配置和终止配置")
        self.verbose_step_logs = False

        # 状态空间: 18维
        # [位置(3) + 速度(3) + 方向(3) + 熵值(3) + Leader(3) + 扫描(3)]
        self.observation_space = spaces.Box(
            low=-100.0, high=100.0, shape=(18,), dtype=np.float32
        )

        # 动作空间: 7维连续（5个APF权重系数 + 2个避障参数）
        # [α1-α5]: APF权重系数，范围 weight_min ~ weight_max
        # [α6]: 避障距离，范围 5.0 ~ 30.0
        # [α7]: 避障系数，范围 1.0 ~ 15.0
        self.action_space = spaces.Box(
            low=np.array(
                [self.reward_config.weight_min] * 5 + [5.0, 1.0], dtype=np.float32
            ),
            high=np.array(
                [self.reward_config.weight_max] * 5 + [30.0, 15.0], dtype=np.float32
            ),
            dtype=np.float32,
        )

        # 记录上一步的状态
        self.prev_scanned_cells = 0
        self.step_count = 0
        self.collision_count = 0  # 新增碰撞计数
        self.episode_count = 0  # 记录Episode编号
        self.total_episode_reward = 0.0  # 记录当前Episode的总奖励
        self.last_action = np.zeros(7)  # 记录上一步的动作（7维），用于电量消耗计算
        self._episode_max_global_scan_ratio = 0.0
        self._episode_min_global_entropy = 100.0
        self.prev_velocity = np.zeros(3, dtype=np.float32)
        self.prev_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.safety_limit = safety_limit
        self.max_weight_delta = max_weight_delta
        self.action_smoothing = float(np.clip(action_smoothing, 0.0, 1.0))
        self.weight_edge_margin = max(0.0, float(weight_edge_margin))
        self.weight_edge_push = float(np.clip(weight_edge_push, 0.0, 1.0))
        self.min_distance_weight = float(np.clip(min_distance_weight, self.reward_config.weight_min, self.reward_config.weight_max))
        self.max_entropy_weight = float(np.clip(max_entropy_weight, self.reward_config.weight_min, self.reward_config.weight_max))
        self.max_leader_weight = float(np.clip(max_leader_weight, self.reward_config.weight_min, self.reward_config.weight_max))
        self._apf_lower_bounds = np.array([
            self.reward_config.weight_min,
            self.reward_config.weight_min,
            self.min_distance_weight,
            self.reward_config.weight_min,
            self.reward_config.weight_min,
        ], dtype=np.float32)
        self._apf_upper_bounds = np.array([
            self.reward_config.weight_max,
            self.max_entropy_weight,
            self.reward_config.weight_max,
            self.max_leader_weight,
            self.reward_config.weight_max,
        ], dtype=np.float32)
        self._last_reset_reason = ""  # 记录上次重置原因
        self._out_of_range_count = 0  # 出圈计数器
        self._out_of_range_start_time = None  # 出圈开始时间
        self._has_initial_action = False
        # 碰撞事件追踪（防止“近距离但未接触”误判为碰撞）
        self._last_collision_timestamp = 0
        self._last_collision_wall_time = 0.0
        self._episode_wall_start_time = time.time()
        self.collision_cfg = {
            "penetration_threshold": 0.03,  # 仅把有明显穿透的事件记为碰撞
            "minor_penetration_threshold": 0.005,  # 非地面对象的轻微接触阈值
            "event_cooldown_sec": 0.8,  # 短时间内不重复记同类碰撞
            "episode_grace_sec": 4.0,  # reset后短暂忽略碰撞抖动
            "ground_episode_grace_sec": 6.0,  # 起飞稳定前更长时间忽略地面碰撞
            "fallback_proximity_distance": 0.35,  # 仅极近距离才用作兜底碰撞
            "ignored_object_penetration_threshold": 0.25,
            "ground_penetration_threshold": 0.18,
            "ground_safe_height": 1.2,
            "ground_collision_event_threshold": 3,
            "unnamed_object_penetration_threshold": 0.20,
            "min_steps_before_unnamed_collision": 3,
            "ignored_objects": ("ground", "landscape", "floor", "terrain"),
            "ground_aliases": (
                "diban",
                "floor",
                "ground",
                "terrain",
                "landscape",
                "groundplane",
                "plane",
                "room_diban",
            ),
        }
        self._last_collision_object_name = ""
        self._last_collision_penetration = 0.0
        self._last_collision_position = ""
        self._recent_trajectory = deque(maxlen=6)
        self._ground_collision_streak = 0

        # 首次重置标志（用于跳过启动时的物理重置）
        self._first_reset = True

    def _step_log(self, message: str) -> None:
        if self.verbose_step_logs:
            print(message)

    def _first_reset_logic(self):
        """处理首次重置逻辑"""
        pass  # Placeholder if needed, or just keep as is in reset()

    def _stabilize_apf_action(self, apf_action: np.ndarray) -> np.ndarray:
        """??????????????????????"""
        stabilized = np.array(apf_action, dtype=np.float32)

        if self.action_smoothing > 0.0 and (self.step_count > 0 or self._has_initial_action):
            last_apf = self.last_action[:5].astype(np.float32)
            stabilized = (1.0 - self.action_smoothing) * last_apf + self.action_smoothing * stabilized

        if self.weight_edge_margin > 0.0 and self.weight_edge_push > 0.0:
            centers = (self._apf_lower_bounds + self._apf_upper_bounds) / 2.0
            lower_trigger = np.minimum(self._apf_lower_bounds + self.weight_edge_margin, centers)
            upper_trigger = np.maximum(self._apf_upper_bounds - self.weight_edge_margin, centers)

            for idx in range(len(stabilized)):
                if stabilized[idx] < lower_trigger[idx]:
                    stabilized[idx] += (lower_trigger[idx] - stabilized[idx]) * self.weight_edge_push
                elif stabilized[idx] > upper_trigger[idx]:
                    stabilized[idx] -= (stabilized[idx] - upper_trigger[idx]) * self.weight_edge_push

        return np.clip(stabilized, self._apf_lower_bounds, self._apf_upper_bounds)

    def _apply_unified_config(self):
        """应用统一环境配置，确保物理规则一致性（方案 B）"""
        unified_env_cfg = None

        # 1. 优先从 server 获取 (AlgorithmServer 持有最新的 ScannerConfigData)
        if (
            self.server
            and hasattr(self.server, "config_data")
            and hasattr(self.server.config_data, "env_config")
        ):
            unified_env_cfg = self.server.config_data.env_config

        # 2. 如果没有 server，尝试从本地 system_config.json 加载
        if unified_env_cfg is None:
            try:
                # 寻找根目录下的 system_config.json
                # 当前文件在 multirotor/DDPG_Weight/envs/，根目录在 multirotor/
                config_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "..",
                    "..",
                    "system_config.json",
                )
                if os.path.exists(config_path):
                    import json

                    with open(config_path, "r", encoding="utf-8-sig") as f:
                        full_cfg = json.load(f)
                        unified_env_cfg = full_cfg.get("environment")
            except Exception as e:
                print(f"[Warning] 加载本地统一配置失败: {e}")

        if unified_env_cfg:
            print(
                f"[UnifiedConfig] 正在应用统一环境配置于 {self.__class__.__name__}..."
            )

            # 覆盖终止条件
            if "termination" in unified_env_cfg:
                self.term_cfg.update(unified_env_cfg["termination"])
                # DDPG 环境可能没有 time_limit，我们将其映射到 step_count
                # max_steps = max_elapsed_time_sec / step_duration
                if (
                    "max_elapsed_time_sec" in unified_env_cfg["termination"]
                    and self.step_duration > 0
                ):
                    calculated_steps = int(
                        unified_env_cfg["termination"]["max_elapsed_time_sec"]
                        / self.step_duration
                    )
                    self.reward_config.max_steps = calculated_steps
                print(f"  • 终止条件已更新 (MaxSteps={self.reward_config.max_steps})")

            # 覆盖电量阈值
            if "battery" in unified_env_cfg:
                b_cfg = unified_env_cfg["battery"]
                self.reward_config.battery_low_threshold = float(
                    b_cfg.get("low_threshold", self.reward_config.battery_low_threshold)
                )
                self.reward_config.battery_optimal_min = float(
                    b_cfg.get("optimal_min", self.reward_config.battery_optimal_min)
                )
                self.reward_config.battery_optimal_max = float(
                    b_cfg.get("optimal_max", self.reward_config.battery_optimal_max)
                )
                print(
                    f"  • 电量阈值已对齐: Low<{self.reward_config.battery_low_threshold}V"
                )

            # 覆盖基础奖励系数
            if "base_rewards" in unified_env_cfg:
                base_rewards = unified_env_cfg["base_rewards"]
                reward_map = {
                    "scan_reward": "scan_reward",
                    "out_of_range_penalty": "out_of_range_penalty",
                    "battery_low_penalty": "battery_low_penalty",
                    "battery_optimal_reward": "battery_optimal_reward",
                }
                for u_key, local_attr in reward_map.items():
                    if u_key in base_rewards:
                        val = abs(float(base_rewards[u_key]))
                        setattr(self.reward_config, local_attr, val)
                print(f"  • 奖励系数已对齐: Scan={self.reward_config.scan_reward}")

    def reset(self, seed=None, options=None):
        """重置环境"""
        import time
        import sys

        if seed is not None:
            np.random.seed(int(seed))

        # Episode计数
        self.episode_count += 1

        print(f"\n{'=' * 60}")
        print(f"🔄 重置环境 - Episode #{self.episode_count}")
        print(f"{'=' * 60}")

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
                    # 使用保存的重置原因，并传递是否重置熵值的标志
                    reason = getattr(self, "_last_reset_reason", "Episode结束")
                    self.server.reset_environment(reason=reason, reset_grid=self.reset_grid_entropy)

                    # 等待重置完成
                    for i in range(3):
                        sys.stdout.write(f"\r  ⏳ 等待重置... {'.' * (i + 1)}   ")
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

                dots = "." * (int(wait_time * 2) % 4)
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
        self.collision_count = 0  # 重置碰撞计数
        self.total_episode_reward = 0.0
        self.last_action = np.zeros(7)  # 7维动作空间
        self.prev_velocity = np.zeros(3, dtype=np.float32)
        self.prev_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self._has_initial_action = False
        self._out_of_range_count = 0  # 重置出圈计数
        self._out_of_range_start_time = None  # 重置出圈计时
        self._out_of_range_continuous_count = 0  # 重置连续出圈步数计数
        self._last_collision_timestamp = 0
        self._last_collision_wall_time = 0.0
        self._last_collision_object_name = ""
        self._last_collision_penetration = 0.0
        self._last_collision_position = ""
        self._recent_trajectory.clear()
        self._ground_collision_streak = 0
        self._episode_wall_start_time = time.time()
        self._episode_max_global_scan_ratio = 0.0
        self._episode_min_global_entropy = 100.0

        state = self._get_state()

        # 显示所有无人机的电量信息
        if self.server:
            print(f"🔋 电量状态:")
            for drone_name in self.server.drone_names:
                current_voltage = self.server.get_battery_voltage(drone_name)
                battery_info = self.server.battery_manager.get_battery_info(drone_name)
                if battery_info:
                    print(
                        f"  • {drone_name}: {current_voltage:.2f}V ({battery_info.get_remaining_percentage():.1f}%)"
                    )
                else:
                    print(f"  • {drone_name}: {current_voltage:.2f}V")

        print(f"\n{'=' * 60}")
        print(f"🎯 开始 Episode #{self.episode_count}")
        print(f"{'=' * 60}")
        print(f"📊 配置:")
        print(f"  • Episode编号: #{self.episode_count}")
        print(f"  • 最大步数: {self.reward_config.max_steps}")
        print(f"  • 每步时长: {self.step_duration}秒")
        print(
            f"  • 预计时长: {self.reward_config.max_steps * self.step_duration / 60:.1f}分钟"
        )
        print(
            "  • 终止条件: "
            f"碰撞>={self.term_cfg.get('max_collision_count')}, "
            f"时长>={self.term_cfg.get('max_elapsed_time_sec')}s, "
            f"出圈连续>={self.term_cfg.get('out_of_range_continuous_count')}, "
            f"目标扫描率>={self.term_cfg.get('target_scan_ratio'):.2f}"
        )
        print(f"{'=' * 60}\n")

        # 通知服务器 Episode 切换 (用于数据采集及时记录上一个 Episode)
        if self.server:
            if hasattr(self.server, "reset_episode_timer"):
                self.server.reset_episode_timer()
            self.server.set_training_stats(
                episode=self.episode_count, step=0, reward=0.0, total_reward=0.0
            )

        return state

    def step(self, action):
        """
        执行一步

        :param action: [α1, α2, α3, α4, α5, α6, α7] - 5个权重系数 + 2个避障参数
        :return: observation, reward, done, info
        """
        import time
        import sys

        try:
            # 分离APF权重和避障参数
            apf_action = action[:5]  # 前5个是APF权重
            obstacle_action = action[5:]  # 后2个是避障参数

            # 确保APF权重在有效范围内
            apf_action = np.clip(
                apf_action, self.reward_config.weight_min, self.reward_config.weight_max
            )

            # 确保避障参数在有效范围内
            obstacle_action = np.clip(obstacle_action, [5.0, 1.0], [30.0, 15.0])

            # 安全限制（仅对APF权重应用）
            if self.safety_limit and (self.step_count > 0 or self._has_initial_action):
                last_apf = self.last_action[:5]
                apf_action = np.clip(
                    apf_action,
                    last_apf - self.max_weight_delta,
                    last_apf + self.max_weight_delta,
                )
                apf_action = np.clip(
                    apf_action, self.reward_config.weight_min, self.reward_config.weight_max
                )
            apf_action = self._stabilize_apf_action(apf_action)
            self._has_initial_action = False

            # 合并动作
            action = np.concatenate([apf_action, obstacle_action])
            self.last_action = action.copy()

            # 将权重设置到APF算法
            weights = {
                "repulsionCoefficient": float(action[0]),
                "entropyCoefficient": float(action[1]),
                "distanceCoefficient": float(action[2]),
                "leaderRangeCoefficient": float(action[3]),
                "directionRetentionCoefficient": float(action[4]),
                "obstacleRepulsionDistance": float(action[5]),
                "obstacleRepulsionCoefficient": float(action[6]),
            }

            # 打印当前步骤信息
            self.step_count += 1
            progress_percent = (self.step_count / self.reward_config.max_steps) * 100

            if self.verbose_step_logs:
                print(f"\n{'─' * 60}")
                print(
                    f"🔄 步骤 {self.step_count}/{self.reward_config.max_steps} ({progress_percent:.1f}%)"
                )
                print(f"{'─' * 60}")
                print(f"📊 设置权重:")
                print(f"  • 斥力系数: {weights['repulsionCoefficient']:.3f}")
                print(f"  • 熵系数:   {weights['entropyCoefficient']:.3f}")
                print(f"  • 距离系数: {weights['distanceCoefficient']:.3f}")
                print(f"  • Leader:   {weights['leaderRangeCoefficient']:.3f}")
                print(f"  • 方向保持: {weights['directionRetentionCoefficient']:.3f}")
                print(f"🚧 避障参数:")
                print(f"  • 避障距离: {weights['obstacleRepulsionDistance']:.1f}")
                print(f"  • 避障系数: {weights['obstacleRepulsionCoefficient']:.1f}")

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
                                speed_magnitude = (
                                    np.sqrt(move_dir.x**2 + move_dir.y**2 + move_dir.z**2)
                                    * move_speed
                                )

                                # 归一化到 0-1 范围（假设最大速度为 moveSpeed）
                                action_intensity = min(
                                    1.0, speed_magnitude / max(move_speed, 0.1)
                                )
                                self.server.battery_manager.update_voltage(
                                    drone_name, action_intensity
                                )

                # 显示所有无人机的当前电量
                if self.verbose_step_logs:
                    print(f"🔋 电量状态:")
                    for drone_name in self.server.drone_names:
                        battery_info = self.server.battery_manager.get_battery_info(drone_name)
                        if battery_info:
                            current_voltage = battery_info.voltage
                            print(
                                f"  • {drone_name}: {current_voltage:.2f}V ({battery_info.get_remaining_percentage():.1f}%)"
                            )

                # 设置权重（算法线程会使用新权重飞行）
                # 为所有无人机设置相同的权重，确保多机协同训练时都能正常移动
                for drone_name in self.server.drone_names:
                    self.server.algorithms[drone_name].set_coefficients(weights)
                self._step_log(f"📊 已为所有无人机设置权重: {len(self.server.drone_names)}台")

                # 倒计时等待无人机飞行
                if self.verbose_step_logs:
                    print(f"\n⏱️  等待无人机飞行 {self.step_duration:.0f} 秒...")

                    # 使用倒计时显示
                    for remaining in range(int(self.step_duration), 0, -1):
                        elapsed = self.step_duration - remaining
                        bar_length = 40
                        filled = int((elapsed / self.step_duration) * bar_length)
                        bar = "█" * filled + "░" * (bar_length - filled)

                        sys.stdout.write(f"\r  [{bar}] {remaining:2d}秒剩余  ")
                        sys.stdout.flush()
                        time.sleep(1)

                    print(f"\r  [{'█' * 40}] ✅ 完成!     ")
                else:
                    time.sleep(self.step_duration)
            else:
                time.sleep(0.1)  # 测试模式快速跳过

            # 获取新状态
            next_state = self._get_state()

            # 计算奖励
            reward = self._calculate_reward(action)

            # 更新碰撞计数（使用真实碰撞事件，避免“近距离误判碰撞”）
            self._update_collision_count()
            episode_metrics = self._collect_episode_metrics()
            # 记录当前动作
            self.last_action = action.copy()

            # 判断是否结束 (统一终止逻辑)
            elapsed_time = self.step_count * self.step_duration
            done = False
            reset_reason = ""

            # 检查出圈状态
            is_out_of_range = False
            dist_to_leader = 0.0
            leader_radius = 0.0
            if self.server:
                with self.server.data_lock:
                    rd = self.server.unity_runtime_data.get(self.drone_name)
                    if rd and rd.leader_position and rd.leader_scan_radius > 0:
                        dist_to_leader = (rd.position - rd.leader_position).magnitude()
                        leader_radius = (
                            rd.leader_scan_radius + self.reward_config.leader_range_buffer
                        )
                        is_out_of_range = dist_to_leader > leader_radius

            # 出圈检测与重置逻辑（改为基于连续步数）
            if is_out_of_range:
                self._out_of_range_continuous_count += 1
                self._step_log(
                    f"⚠️  出圈警告: 距离Leader {dist_to_leader:.1f}m > 半径 {leader_radius:.1f}m "
                    f"(连续第{self._out_of_range_continuous_count}步)"
                )

                # 检查是否达到连续出圈阈值（从2步延长到3步，给予更多学习时间）
                if self.term_cfg.get("out_of_range_reset_enabled", True):
                    threshold = self.term_cfg.get("out_of_range_continuous_count", 3)
                    if self._out_of_range_continuous_count >= threshold:
                        self._out_of_range_count += 1
                        print(
                            f"🚨 [出圈重置] 连续出圈 {self._out_of_range_continuous_count} 步，"
                            f"累计出圈: {self._out_of_range_count} 次"
                        )
                        # 添加额外的出圈惩罚（使用配置的最大值，避免过度惩罚）
                        penalty = self.boundary_cfg["max_reset_penalty"]
                        reward -= penalty
                        print(f"💔 出圈重置惩罚: -{penalty:.2f}")
                        done = True
                        reset_reason = "出圈"
            else:
                # 回到圈内，重置连续计数
                if self._out_of_range_continuous_count > 0:
                    self._step_log(f"✅ 回到圈内，重置连续出圈计数")
                self._out_of_range_continuous_count = 0

            if not done:
                if elapsed_time >= self.term_cfg["max_elapsed_time_sec"]:
                    print(f"[终止] 达到最大仿真时间: {elapsed_time:.1f}s")
                    done = True
                    reset_reason = "达到时长上限"
                elif self.collision_count >= self.term_cfg["max_collision_count"]:
                    print(f"[终止] 发生碰撞: {self.collision_count}")
                    done = True
                    reset_reason = "碰撞"
                elif self.server and hasattr(self.server, "battery_manager"):
                    battery_info = self.server.battery_manager.get_battery_info(
                        self.drone_name
                    )
                    if battery_info:
                        current_voltage = float(
                            getattr(battery_info, "voltage", 4.2) or 4.2
                        )
                        battery_status = getattr(battery_info, "status", None)
                        if (
                            current_voltage <= 3.2 + 1e-6
                            or battery_status == BatteryStatus.EMPTY
                        ):
                            print(f"[终止] 电量耗尽: {current_voltage:.2f}V")
                            done = True
                            reset_reason = "电量耗尽"
                else:
                    # 检查覆盖率
                    if self.server:
                        with self.server.data_lock:
                            total_cells = len(self.server.grid_data.cells)
                            if total_cells > 0:
                                scanned_cells = sum(
                                    1
                                    for cell in self.server.grid_data.cells
                                    if cell.entropy
                                    < self.reward_config.scan_entropy_threshold
                                )
                                scan_ratio = scanned_cells / total_cells
                                if scan_ratio >= self.term_cfg["target_scan_ratio"]:
                                    print(f"[终止] 覆盖率达成: {scan_ratio:.2%}")
                                    done = True
                                    reset_reason = "扫描完成"

            if done:
                reward = self._apply_terminal_reward_adjustment(
                    reward, reset_reason, episode_metrics
                )

            self.total_episode_reward += reward

            # 将训练统计信息传递给服务器（用于数据采集）
            if self.server:
                self.server._last_collision_object_name = self._last_collision_object_name
                self.server._last_collision_penetration_depth = float(
                    self._last_collision_penetration
                )
                self.server.set_training_stats(
                    episode=self.episode_count,
                    step=self.step_count,
                    reward=float(reward),
                    total_reward=float(self.total_episode_reward),
                )

                self.server.data_collector.set_external_data(
                    "collision_count", int(self.collision_count)
                )
                self.server.data_collector.set_external_data(
                    "out_of_range_count", int(self._out_of_range_count)
                )
                self.server.data_collector.set_external_data(
                    "max_global_scan_ratio",
                    float(episode_metrics["episode_max_global_scan_ratio"]),
                )
                self.server.data_collector.set_external_data(
                    "min_global_avg_entropy",
                    float(episode_metrics["episode_min_global_entropy"]),
                )
                self.server.data_collector.set_external_data(
                    "global_scanned_count",
                    int(episode_metrics["global_scanned_count"]),
                )
                self.server.data_collector.set_external_data(
                    "global_total_count",
                    int(episode_metrics["global_total_count"]),
                )
                self.server.data_collector.set_external_data(
                    "collision_object_name", self._last_collision_object_name
                )
                self.server.data_collector.set_external_data(
                    "collision_penetration_depth",
                    float(self._last_collision_penetration),
                )
                self.server.data_collector.set_external_data(
                    "collision_position", self._last_collision_position
                )
                self.server.data_collector.set_external_data(
                    "recent_trajectory", self._get_recent_trajectory_json()
                )
                self.server.data_collector.set_external_data("reset_reason", "")

            # 显示奖励信息
            self._step_log(f"\n📈 本步奖励: {reward:+.2f}")

            if self.server:
                with self.server.data_lock:
                    grid_data = self.server.grid_data
                    if grid_data and grid_data.cells:
                        total_cells = len(grid_data.cells)
                        scanned_cells = sum(
                            1
                            for cell in grid_data.cells
                            if cell.entropy < self.reward_config.scan_entropy_threshold
                        )
                        scan_progress = (scanned_cells / total_cells) * 100
                        self._step_log(
                            f"🗺️  扫描进度: {scanned_cells}/{total_cells} ({scan_progress:.1f}%)"
                        )

            if done:
                # 保存重置原因，供下次 reset 使用
                self._last_reset_reason = reset_reason
                if self.server:
                    self.server._last_collision_object_name = (
                        self._last_collision_object_name
                    )
                    self.server._last_collision_penetration_depth = float(
                        self._last_collision_penetration
                    )
                    self.server.data_collector.set_external_data(
                        "reset_reason", reset_reason
                    )
                    self.server.data_collector.set_external_data(
                        "collision_count", int(self.collision_count)
                    )
                    self.server.data_collector.set_external_data(
                        "out_of_range_count", int(self._out_of_range_count)
                    )
                    self.server.data_collector.set_external_data(
                        "max_global_scan_ratio",
                        float(episode_metrics["episode_max_global_scan_ratio"]),
                    )
                    self.server.data_collector.set_external_data(
                        "min_global_avg_entropy",
                        float(episode_metrics["episode_min_global_entropy"]),
                    )
                    self.server.data_collector.set_external_data(
                        "global_scanned_count",
                        int(episode_metrics["global_scanned_count"]),
                    )
                    self.server.data_collector.set_external_data(
                        "global_total_count",
                        int(episode_metrics["global_total_count"]),
                    )
                    self.server.data_collector.set_external_data(
                        "collision_object_name", self._last_collision_object_name
                    )
                    self.server.data_collector.set_external_data(
                        "collision_penetration_depth",
                        float(self._last_collision_penetration),
                    )
                    self.server.data_collector.set_external_data(
                        "collision_position", self._last_collision_position
                    )
                    self.server.data_collector.set_external_data(
                        "recent_trajectory", self._get_recent_trajectory_json()
                    )
                    self.server.data_collector.set_external_data(
                        "terminal_episode", int(self.episode_count)
                    )
                    self.server.data_collector.set_external_data(
                        "terminal_reset_reason", reset_reason
                    )
                    self.server.data_collector.set_external_data(
                        "terminal_collision_count", int(self.collision_count)
                    )
                    self.server.data_collector.set_external_data(
                        "terminal_out_of_range_count", int(self._out_of_range_count)
                    )
                    self.server.data_collector.set_external_data(
                        "terminal_max_global_scan_ratio",
                        float(episode_metrics["episode_max_global_scan_ratio"]),
                    )
                    self.server.data_collector.set_external_data(
                        "terminal_min_global_avg_entropy",
                        float(episode_metrics["episode_min_global_entropy"]),
                    )
                    self.server.data_collector.set_external_data(
                        "terminal_collision_object_name", self._last_collision_object_name
                    )
                    self.server.data_collector.set_external_data(
                        "terminal_collision_penetration_depth",
                        float(self._last_collision_penetration),
                    )
                    self.server.data_collector.set_external_data(
                        "terminal_collision_position", self._last_collision_position
                    )
                    self.server.data_collector.set_external_data(
                        "terminal_recent_trajectory", self._get_recent_trajectory_json()
                    )
                print(f"\n{'=' * 60}")
                print(f"✅ Episode #{self.episode_count} 完成！共 {self.step_count} 步")
                print(f"📌 重置原因: {reset_reason}")
                print(f"{'=' * 60}")
                print(f"🔄 即将自动重置环境，开始下一个Episode...")
                print(f"{'=' * 60}\n")

            # 额外信息
            info = {
                "weights": weights,
                "scanned_cells": self.prev_scanned_cells,
                "reset_reason": reset_reason,
                "collision_count": int(self.collision_count),
                "out_of_range_count": int(self._out_of_range_count),
                "max_global_scan_ratio": float(episode_metrics["episode_max_global_scan_ratio"]),
                "min_global_avg_entropy": float(episode_metrics["episode_min_global_entropy"]),
                "global_scanned_count": int(episode_metrics["global_scanned_count"]),
                "global_total_count": int(episode_metrics["global_total_count"]),
                "collision_object_name": self._last_collision_object_name,
                "collision_penetration_depth": float(self._last_collision_penetration),
                "collision_position": self._last_collision_position,
                "recent_trajectory": self._get_recent_trajectory_json(),
            }

            return next_state, reward, done, info

        except Exception as e:
            # 捕获step中的所有未处理异常
            print(f"\n{'=' * 60}")
            print(f"[严重错误] 环境step方法中发生异常: {str(e)}")
            print(f"[步数] {self.step_count}")
            print(f"{'=' * 60}")
            import traceback
            traceback.print_exc()
            # 返回一个安全的默认值，避免训练崩溃
            return (
                np.zeros(self.observation_space.shape, dtype=np.float32),
                0.0,
                True,  # done=True，结束这个episode
                {"error": str(e)}
            )

    def _get_min_distance_to_others(self, rd):
        """获取到其他无人机的最小距离"""
        if not rd or not rd.otherScannerPositions:
            return 999.0
        pos = rd.position
        dists = []
        for op in rd.otherScannerPositions:
            dist = np.sqrt((pos.x - op.x) ** 2 + (pos.y - op.y) ** 2 + (pos.z - op.z) ** 2)
            # 过滤自身点/重复点（距离≈0）
            if dist > 1e-3:
                dists.append(dist)
        return min(dists) if dists else 999.0

    def _get_current_height(self) -> float:
        """读取当前无人机高度，优先使用 Unity 运行时高度。"""
        if not self.server:
            return 0.0
        try:
            with self.server.data_lock:
                rd = self.server.unity_runtime_data.get(self.drone_name)
            if rd and getattr(rd, "position", None) is not None:
                return float(rd.position.y)
        except Exception:
            pass
        return 0.0

    def _is_ground_object_name(self, object_name: str) -> bool:
        """识别场景中不同命名方式的地面对象。"""
        if not object_name:
            return False
        object_name_lower = object_name.strip().lower()
        if any(
            token in object_name_lower for token in self.collision_cfg["ignored_objects"]
        ):
            return True
        return any(
            token in object_name_lower for token in self.collision_cfg["ground_aliases"]
        )

    def _get_runtime_position(self):
        """Return current drone position from runtime data."""
        if not self.server:
            return None
        try:
            with self.server.data_lock:
                rd = self.server.unity_runtime_data.get(self.drone_name)
            if rd and getattr(rd, "position", None) is not None:
                return rd.position
        except Exception:
            pass
        return None

    def _format_position(self, position) -> str:
        if position is None:
            return ""
        try:
            return f"{float(position.x):.2f},{float(position.y):.2f},{float(position.z):.2f}"
        except Exception:
            return ""

    def _append_recent_trajectory(self, position) -> None:
        formatted = self._format_position(position)
        if not formatted:
            return
        if not self._recent_trajectory or self._recent_trajectory[-1] != formatted:
            self._recent_trajectory.append(formatted)

    def _get_recent_trajectory_json(self) -> str:
        return json.dumps(list(self._recent_trajectory), ensure_ascii=False)

    def _get_normal_obstacles(self):
        if not self.server:
            return []
        try:
            unity_socket = getattr(self.server, "unity_socket", None)
            obstacles = getattr(unity_socket, "received_obstacles", None)
            if not isinstance(obstacles, list):
                return []
            result = []
            for obstacle in obstacles:
                if not isinstance(obstacle, dict):
                    continue
                category = str(obstacle.get("category", "normal") or "normal").lower()
                if category == "restricted":
                    continue
                result.append(obstacle)
            return result
        except Exception:
            return []

    def _distance_to_obstacle_surface(self, position, obstacle):
        if position is None or not obstacle:
            return None

        try:
            pos = np.array(
                [float(position.x), float(position.y), float(position.z)],
                dtype=np.float32,
            )
            shape_type = obstacle.get("shapeType")
            center = obstacle.get("center") or {}
            vertices = obstacle.get("vertices") or []
            radius = float(obstacle.get("radius", 0.0) or 0.0)

            center_vec = None
            if center:
                center_vec = np.array(
                    [
                        float(center.get("x", 0.0) or 0.0),
                        float(center.get("y", 0.0) or 0.0),
                        float(center.get("z", 0.0) or 0.0),
                    ],
                    dtype=np.float32,
                )

            shape_name = str(shape_type).lower()
            if center_vec is not None and (
                shape_name in {"1", "3", "sphere", "circle"} or radius > 0.0
            ):
                return max(float(np.linalg.norm(pos - center_vec)) - radius, 0.0)

            if vertices:
                vertex_array = np.array(
                    [
                        [
                            float(vertex.get("x", 0.0) or 0.0),
                            float(vertex.get("y", 0.0) or 0.0),
                            float(vertex.get("z", 0.0) or 0.0),
                        ]
                        for vertex in vertices
                    ],
                    dtype=np.float32,
                )
                mins = vertex_array.min(axis=0)
                maxs = vertex_array.max(axis=0)
                clipped = np.minimum(np.maximum(pos, mins), maxs)
                if np.all(pos >= mins) and np.all(pos <= maxs):
                    return 0.0
                return float(np.linalg.norm(pos - clipped))

            if center_vec is not None:
                return float(np.linalg.norm(pos - center_vec))
        except Exception:
            return None

        return None

    def _apply_obstacle_proximity_penalty(self, runtime_data) -> float:
        position = getattr(runtime_data, "position", None)
        if position is None:
            return 0.0

        nearest_distance = None
        nearest_name = ""
        for obstacle in self._get_normal_obstacles():
            distance = self._distance_to_obstacle_surface(position, obstacle)
            if distance is None:
                continue
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_name = str(obstacle.get("name", "") or "")

        if nearest_distance is None:
            return 0.0

        warning_distance = max(
            float(self.reward_shaping_cfg["obstacle_warning_distance"]), 1e-6
        )
        danger_distance = min(
            float(self.reward_shaping_cfg["obstacle_danger_distance"]),
            warning_distance,
        )
        warning_penalty = float(self.reward_shaping_cfg["obstacle_warning_penalty"])
        danger_penalty = float(self.reward_shaping_cfg["obstacle_danger_penalty"])

        # Obstacle (2) is a confirmed hotspot in the current scene, so give it
        # a wider buffer and stronger penalty before actual contact.
        if nearest_name.strip().lower() == "obstacle (2)":
            warning_distance = max(
                float(self.reward_shaping_cfg["hotspot_obstacle_warning_distance"]),
                warning_distance,
            )
            danger_distance = min(
                float(self.reward_shaping_cfg["hotspot_obstacle_danger_distance"]),
                warning_distance,
            )
            warning_penalty = max(
                float(self.reward_shaping_cfg["hotspot_obstacle_warning_penalty"]),
                warning_penalty,
            )
            danger_penalty = max(
                float(self.reward_shaping_cfg["hotspot_obstacle_danger_penalty"]),
                danger_penalty,
            )

        if nearest_distance > warning_distance:
            return 0.0

        if nearest_distance <= danger_distance:
            penalty = danger_penalty
        else:
            ratio = (warning_distance - nearest_distance) / max(
                warning_distance - danger_distance, 1e-6
            )
            penalty = warning_penalty + ratio * (danger_penalty - warning_penalty)

        print(
            f"🚧 障碍物接近惩罚: name={nearest_name or 'Unknown'}, "
            f"distance={nearest_distance:.2f}m, penalty=-{penalty:.2f}"
        )
        return penalty

    def _apply_collision_hotspot_penalty(self, runtime_data) -> float:
        position = getattr(runtime_data, "position", None)
        if position is None:
            return 0.0

        try:
            dx = float(position.x) - float(
                self.reward_shaping_cfg["collision_hotspot_center_x"]
            )
            dz = float(position.z) - float(
                self.reward_shaping_cfg["collision_hotspot_center_z"]
            )
        except Exception:
            return 0.0

        horizontal_distance = float(np.sqrt(dx * dx + dz * dz))
        warning_radius = max(
            float(self.reward_shaping_cfg["collision_hotspot_warning_radius"]), 1e-6
        )
        danger_radius = min(
            float(self.reward_shaping_cfg["collision_hotspot_danger_radius"]),
            warning_radius,
        )

        if horizontal_distance > warning_radius:
            return 0.0

        if horizontal_distance <= danger_radius:
            penalty = float(self.reward_shaping_cfg["collision_hotspot_danger_penalty"])
        else:
            ratio = (warning_radius - horizontal_distance) / max(
                warning_radius - danger_radius, 1e-6
            )
            penalty = float(
                self.reward_shaping_cfg["collision_hotspot_warning_penalty"]
            ) + ratio * (
                float(self.reward_shaping_cfg["collision_hotspot_danger_penalty"])
                - float(self.reward_shaping_cfg["collision_hotspot_warning_penalty"])
            )

        print(
            f"📍 热点区避让惩罚: center=("
            f"{self.reward_shaping_cfg['collision_hotspot_center_x']:.2f},"
            f"{self.reward_shaping_cfg['collision_hotspot_center_z']:.2f}), "
            f"distance={horizontal_distance:.2f}m, penalty=-{penalty:.2f}"
        )
        return penalty


    def _apply_collision_hotspot_corridor_penalty(self, runtime_data) -> float:
        position = getattr(runtime_data, "position", None)
        if position is None:
            return 0.0

        try:
            dx = abs(
                float(position.x) - float(self.reward_shaping_cfg["collision_hotspot_center_x"])
            )
            dz = abs(
                float(position.z) - float(self.reward_shaping_cfg["collision_hotspot_center_z"])
            )
        except Exception:
            return 0.0

        half_width_x = max(
            float(self.reward_shaping_cfg["collision_hotspot_corridor_half_width_x"]),
            1e-6,
        )
        half_width_z = max(
            float(self.reward_shaping_cfg["collision_hotspot_corridor_half_width_z"]),
            1e-6,
        )

        if dx > half_width_x or dz > half_width_z:
            return 0.0

        ratio_x = 1.0 - min(dx / half_width_x, 1.0)
        ratio_z = 1.0 - min(dz / half_width_z, 1.0)
        penalty = (
            float(self.reward_shaping_cfg["collision_hotspot_corridor_penalty"])
            * ratio_x
            * ratio_z
        )
        if penalty <= 0.0:
            return 0.0

        print(
            f"🧱 热点通道惩罚: dx={dx:.2f}m, dz={dz:.2f}m, penalty=-{penalty:.2f}"
        )
        return penalty

    def _update_collision_count(self) -> None:
        """更新碰撞计数：优先使用AirSim真实碰撞事件，距离仅作兜底。"""
        if not self.server:
            return

        now = time.time()
        # reset后短暂忽略碰撞状态抖动，避免刚重置即误判
        if now - self._episode_wall_start_time < self.collision_cfg["episode_grace_sec"]:
            return

        # 1) 首选：AirSim真实碰撞事件（按time_stamp去重）
        try:
            collision = self.server.drone_controller.check_collision(self.drone_name)
            if collision and collision.get("has_collided", False):
                time_stamp = int(collision.get("time_stamp", 0) or 0)
                penetration = float(collision.get("penetration_depth", 0.0) or 0.0)
                object_name = str(collision.get("object_name", "") or "")
                object_name = object_name.strip()
                object_name_lower = object_name.lower()
                is_ignored_object = self._is_ground_object_name(object_name)
                has_named_object = bool(object_name)
                current_height = self._get_current_height()

                is_new_event = time_stamp > 0 and time_stamp != self._last_collision_timestamp
                named_hit = (
                    has_named_object
                    and not is_ignored_object
                    and penetration >= self.collision_cfg["minor_penetration_threshold"]
                )
                unnamed_hit = (
                    not has_named_object
                    and self.step_count
                    > self.collision_cfg["min_steps_before_unnamed_collision"]
                    and penetration
                    >= self.collision_cfg["unnamed_object_penetration_threshold"]
                )
                ground_grace_elapsed = (
                    now - self._episode_wall_start_time
                    >= self.collision_cfg["ground_episode_grace_sec"]
                )
                safe_height_reached = (
                    current_height >= self.collision_cfg["ground_safe_height"]
                )

                if is_new_event:
                    if is_ignored_object and penetration >= self.collision_cfg["ground_penetration_threshold"]:
                        self._ground_collision_streak += 1
                    else:
                        self._ground_collision_streak = 0

                ground_hit = (
                    has_named_object
                    and is_ignored_object
                    and penetration >= self.collision_cfg["ground_penetration_threshold"]
                    and ground_grace_elapsed
                    and self.step_count > self.collision_cfg["min_steps_before_unnamed_collision"]
                    and (
                        safe_height_reached
                        or self._ground_collision_streak
                        >= self.collision_cfg["ground_collision_event_threshold"]
                    )
                )
                strong_hit = named_hit or ground_hit or unnamed_hit
                cooldown_ok = (
                    now - self._last_collision_wall_time
                    >= self.collision_cfg["event_cooldown_sec"]
                )

                if is_new_event and strong_hit and cooldown_ok:
                    self.collision_count += 1
                    self._last_collision_wall_time = now
                    self._last_collision_object_name = object_name or "Unknown"
                    self._last_collision_penetration = penetration
                    self._last_collision_position = self._format_position(self._get_runtime_position())
                    print(
                        f"⚠️ 检测到真实碰撞事件: object={object_name or 'Unknown'}, "
                        f"penetration={penetration:.3f}m, count={self.collision_count}"
                    )
                elif is_new_event and not strong_hit:
                    if is_ignored_object:
                        print(
                            f"ℹ️ 忽略地面碰撞事件: object={object_name or 'Unknown'}, "
                            f"penetration={penetration:.3f}m, height={current_height:.2f}m, "
                            f"streak={self._ground_collision_streak}, step={self.step_count}"
                        )
                    else:
                        print(
                            f"ℹ️ 忽略碰撞事件: object={object_name or 'Unknown'}, "
                            f"penetration={penetration:.3f}m, step={self.step_count}"
                        )

                if time_stamp > 0:
                    self._last_collision_timestamp = time_stamp
                return
            self._ground_collision_streak = 0
        except Exception as e:
            print(f"[Warning] 读取AirSim碰撞状态失败，启用距离兜底: {e}")

        # 2) 兜底：仅在极近距离才记碰撞（避免把正常编队/避障当碰撞）
        try:
            with self.server.data_lock:
                rd = self.server.unity_runtime_data.get(self.drone_name)
            if rd:
                min_dist = self._get_min_distance_to_others(rd)
                if (
                    min_dist < self.collision_cfg["fallback_proximity_distance"]
                    and now - self._last_collision_wall_time
                    >= self.collision_cfg["event_cooldown_sec"]
                ):
                    self.collision_count += 1
                    self._last_collision_wall_time = now
                    self._last_collision_object_name = "NEAR_DRONE"
                    self._last_collision_penetration = 0.0
                    self._last_collision_position = self._format_position(self._get_runtime_position())
                    print(
                        f"⚠️ 极近距离兜底碰撞: min_dist={min_dist:.3f}m, "
                        f"count={self.collision_count}"
                    )
        except Exception as e:
            print(f"[Warning] 距离兜底碰撞检测失败: {e}")

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
                    vel.z * self.server.config_data.moveSpeed,
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
                        runtime_data.leader_position.z - pos.z,
                    ]
                else:
                    leader_rel = [0.0, 0.0, 0.0]

                # 6. 扫描进度 (3)
                scan_info = self._get_scan_info(grid_data)

                # 组合状态
                state = (
                    position
                    + velocity
                    + direction
                    + entropy_info
                    + leader_rel
                    + scan_info
                )

                return np.array(state, dtype=np.float32)

        except Exception as e:
            print(f"获取状态失败: {str(e)}")
            return np.zeros(18, dtype=np.float32)

    def _collect_episode_metrics(self):
        global_scan_ratio = 0.0
        global_avg_entropy = 100.0
        global_scanned_count = 0
        global_total_count = 0

        if self.server:
            with self.server.data_lock:
                grid_data = self.server.grid_data
                if grid_data and grid_data.cells:
                    global_total_count = len(grid_data.cells)
                    global_scanned_count = sum(
                        1
                        for cell in grid_data.cells
                        if cell.entropy < self.reward_config.scan_entropy_threshold
                    )
                    if global_total_count > 0:
                        global_scan_ratio = global_scanned_count / global_total_count * 100.0
                    entropy_values = [float(cell.entropy) for cell in grid_data.cells]
                    if entropy_values:
                        global_avg_entropy = float(np.mean(entropy_values))

        self._episode_max_global_scan_ratio = max(
            self._episode_max_global_scan_ratio, global_scan_ratio
        )
        self._episode_min_global_entropy = min(
            self._episode_min_global_entropy, global_avg_entropy
        )

        return {
            "global_scan_ratio": global_scan_ratio,
            "global_avg_entropy": global_avg_entropy,
            "global_scanned_count": global_scanned_count,
            "global_total_count": global_total_count,
            "episode_max_global_scan_ratio": self._episode_max_global_scan_ratio,
            "episode_min_global_entropy": self._episode_min_global_entropy,
        }

    def _apply_terminal_reward_adjustment(
        self, reward: float, reset_reason: str, episode_metrics: dict
    ) -> float:
        if reset_reason == "扫描完成":
            return reward + self.reward_shaping_cfg["scan_complete_bonus"]

        max_scan_ratio = float(
            episode_metrics.get("episode_max_global_scan_ratio", 0.0)
        )
        poor_progress_threshold = max(
            self.reward_shaping_cfg["poor_progress_scan_ratio"], 1e-6
        )
        healthy_time_limit_threshold = max(
            self.reward_shaping_cfg["healthy_time_limit_scan_ratio"], 1e-6
        )
        progress_scale = min(max_scan_ratio / poor_progress_threshold, 1.0)

        if reset_reason == "达到时长上限":
            if max_scan_ratio >= healthy_time_limit_threshold:
                bonus = self.reward_shaping_cfg["time_limit_completion_bonus"]
                print(
                    f"⚖️  终止奖励修正: 原因={reset_reason}, 最大全局扫描={max_scan_ratio:.2f}%, 奖励+{bonus:.2f}"
                )
                return reward + bonus
            penalty = self.reward_shaping_cfg["time_limit_low_progress_penalty"]
            if max_scan_ratio < self.reward_shaping_cfg["poor_progress_scan_ratio"]:
                penalty += self.reward_shaping_cfg["poor_progress_penalty"] * (
                    1.0 - progress_scale
                )
            print(
                f"⚖️  终止奖励修正: 原因={reset_reason}, 最大全局扫描={max_scan_ratio:.2f}%, 惩罚-{penalty:.2f}"
            )
            return reward - penalty

        penalty = self.reward_shaping_cfg["failure_base_penalty"]
        penalty += self.reward_shaping_cfg["poor_progress_penalty"] * (
            1.0 - progress_scale
        )

        if self.step_count <= self.reward_shaping_cfg["early_failure_steps"]:
            penalty += self.reward_shaping_cfg["early_failure_penalty"]

        if reset_reason == "碰撞":
            penalty += self.reward_shaping_cfg["collision_terminal_penalty"]
            if self.step_count <= self.reward_shaping_cfg["early_failure_steps"]:
                penalty += self.reward_shaping_cfg["short_collision_penalty"]
        elif reset_reason == "出圈":
            penalty += self.reward_shaping_cfg["out_of_range_terminal_penalty"]

        print(
            f"⚖️  终止奖励修正: 原因={reset_reason}, 最大全局扫描={max_scan_ratio:.2f}%, 惩罚-{penalty:.2f}"
        )
        adjusted_reward = reward - penalty
        if (
            reset_reason == "碰撞"
            and self.step_count <= self.reward_shaping_cfg["early_failure_steps"]
        ):
            adjusted_reward = min(
                adjusted_reward,
                self.reward_shaping_cfg["short_collision_reward_cap"],
            )
        return adjusted_reward

    def _calculate_reward(self, action: np.ndarray) -> float:
        """计算奖励（尽量与实体奖励结构一致）"""
        if not self.server:
            return 0.0

        reward = -self.reward_shaping_cfg["base_step_cost"]

        try:
            with self.server.data_lock:
                runtime_data = self.server.unity_runtime_data[self.drone_name]
                grid_data = self.server.grid_data

            self._append_recent_trajectory(getattr(runtime_data, "position", None))

            # 1. 速度奖励与超速惩罚
            vel = runtime_data.finalMoveDir
            current_velocity = np.array(
                [vel.x, vel.y, vel.z], dtype=np.float32
            ) * float(self.server.config_data.moveSpeed)
            speed = float(np.linalg.norm(current_velocity))
            reward += self.reward_config.speed_reward * speed
            if speed > self.reward_config.speed_penalty_threshold:
                reward -= self.reward_config.speed_penalty

            # 2. 加速度惩罚（速度变化近似）
            if self.step_duration > 0:
                accel_mag = float(
                    np.linalg.norm(current_velocity - self.prev_velocity)
                    / self.step_duration
                )
            else:
                accel_mag = float(np.linalg.norm(current_velocity - self.prev_velocity))
            reward -= self.reward_config.accel_penalty * accel_mag

            # 3. 角速度惩罚（方向变化近似）
            fwd = runtime_data.forward
            current_direction = np.array([fwd.x, fwd.y, fwd.z], dtype=np.float32)
            current_norm = np.linalg.norm(current_direction)
            prev_norm = np.linalg.norm(self.prev_direction)
            if current_norm > 1e-6 and prev_norm > 1e-6:
                dot = float(
                    np.clip(
                        np.dot(current_direction, self.prev_direction)
                        / (current_norm * prev_norm),
                        -1.0,
                        1.0,
                    )
                )
                angle = float(np.arccos(dot))
                angular_rate = (
                    angle / self.step_duration if self.step_duration > 0 else angle
                )
                reward -= self.reward_config.angular_rate_penalty * angular_rate

            # 4. 扫描奖励
            current_scanned = 0
            if grid_data and grid_data.cells:
                current_scanned = sum(
                    1
                    for cell in grid_data.cells
                    if cell.entropy < self.reward_config.scan_entropy_threshold
                )
            new_scanned = current_scanned - self.prev_scanned_cells
            if new_scanned > 0:
                reward += self.reward_config.scan_reward * new_scanned
            else:
                reward -= self.reward_shaping_cfg["no_progress_penalty"]
            self.prev_scanned_cells = current_scanned

            # 5. 边界控制奖励（渐进式边界惩罚 + 朝向中心奖励）
            if runtime_data.leader_position and runtime_data.leader_scan_radius > 0:
                dist_to_leader = (
                    runtime_data.position - runtime_data.leader_position
                ).magnitude()
                leader_radius = (
                    runtime_data.leader_scan_radius
                    + self.reward_config.leader_range_buffer
                )

                # 计算距离比例（0在中心，1在边界）
                distance_ratio = (
                    dist_to_leader / leader_radius if leader_radius > 0 else 0
                )

                if dist_to_leader > leader_radius:
                    # 出圈惩罚（已经在重置逻辑中处理，这里避免重复）
                    pass
                elif distance_ratio > self.boundary_cfg["danger_ratio"]:
                    # 危险区：距离边界85%-100%，高惩罚
                    danger_level = (
                        distance_ratio - self.boundary_cfg["danger_ratio"]
                    ) / (1.0 - self.boundary_cfg["danger_ratio"])
                    penalty = self.boundary_cfg["danger_penalty"] * danger_level
                    reward -= penalty
                    print(
                        f"⚠️  危险区警告: 距离边界{distance_ratio * 100:.0f}%, 惩罚-{penalty:.2f}"
                    )
                elif distance_ratio > self.boundary_cfg["warning_ratio"]:
                    # 警告区：距离边界70%-85%，渐进惩罚
                    warning_level = (
                        distance_ratio - self.boundary_cfg["warning_ratio"]
                    ) / (
                        self.boundary_cfg["danger_ratio"]
                        - self.boundary_cfg["warning_ratio"]
                    )
                    penalty = self.boundary_cfg["warning_penalty"] * warning_level
                    reward -= penalty
                    print(
                        f"💡 接近边界: 距离边界{distance_ratio * 100:.0f}%, 惩罚-{penalty:.2f}"
                    )
                else:
                    # 安全区：朝向中心飞行的奖励
                    # 计算速度方向与朝向Leader中心方向的夹角
                    to_center = runtime_data.leader_position - runtime_data.position
                    to_center_norm = np.linalg.norm(
                        [to_center.x, to_center.y, to_center.z]
                    )

                    if to_center_norm > 0.1:  # 避免除零
                        velocity_norm = np.linalg.norm(current_velocity)
                        if velocity_norm > 0.1:  # 有速度时才计算
                            # 计算速度朝向中心的分量
                            to_center_dir = (
                                np.array([to_center.x, to_center.y, to_center.z])
                                / to_center_norm
                            )
                            velocity_dir = current_velocity / velocity_norm

                            # 点积：1表示朝向中心，-1表示远离中心
                            alignment = np.dot(velocity_dir, to_center_dir)

                            # 奖励朝向中心飞行（只在靠近边界时才给奖励）
                            if alignment > 0.3 and distance_ratio > 0.5:
                                center_reward = (
                                    self.boundary_cfg["center_reward"]
                                    * self.reward_shaping_cfg["center_reward_scale"]
                                    * alignment
                                )
                                reward += center_reward
                                print(f"🎯 朝向中心飞行: 奖励+{center_reward:.2f}")

            # 6. Obstacle proximity penalty
            reward -= self._apply_obstacle_proximity_penalty(runtime_data)
            reward -= self._apply_collision_hotspot_penalty(runtime_data)
            reward -= self._apply_collision_hotspot_corridor_penalty(runtime_data)

            # 7. Action change and magnitude penalty
            action_delta = float(np.linalg.norm(action - self.last_action))
            reward -= self.reward_config.action_change_penalty * action_delta
            reward -= self.reward_config.action_magnitude_penalty * float(
                np.linalg.norm(action)
            )

            # 8. Battery reward logic
            current_voltage = self.server.get_battery_voltage(self.drone_name)
            if (
                self.reward_config.battery_optimal_min
                <= current_voltage
                <= self.reward_config.battery_optimal_max
            ):
                battery_reward = (
                    self.reward_config.battery_optimal_reward
                    * self.reward_shaping_cfg["battery_reward_scale"]
                )
                reward += battery_reward
                print(
                    f"🔋 电量奖励: +{battery_reward:.2f} (电量{current_voltage:.2f}V在最优范围)"
                )
            elif current_voltage < self.reward_config.battery_low_threshold:
                reward -= self.reward_config.battery_low_penalty
                print(
                    f"🔋 电量惩罚: -{self.reward_config.battery_low_penalty:.2f} (电量{current_voltage:.2f}V过低)"
                )

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
            cell
            for cell in grid_data.cells[:100]
            if (cell.center - position).magnitude() < 10.0
        ]

        if not nearby_cells:
            return [0.0, 0.0, 0.0]

        entropies = [cell.entropy for cell in nearby_cells]
        return [
            float(np.mean(entropies)),
            float(np.max(entropies)),
            float(np.std(entropies)),
        ]

    def _get_scan_info(self, grid_data):
        """获取扫描进度"""
        if not grid_data or not grid_data.cells:
            return [0.0, 0.0, 0.0]

        total = len(grid_data.cells)
        scanned = sum(
            1
            for cell in grid_data.cells
            if cell.entropy < self.reward_config.scan_entropy_threshold
        )

        return [scanned / max(total, 1), float(scanned), float(total - scanned)]

    def _count_scanned_cells(self):
        """统计已扫描单元格（不加锁版本，由调用者加锁）"""
        if not self.server or not self.server.grid_data:
            return 0

        try:
            # 注意：不在这里加锁，避免嵌套锁
            # 调用者应该已经持有data_lock
            return sum(
                1
                for cell in self.server.grid_data.cells
                if cell.entropy < self.reward_config.scan_entropy_threshold
            )
        except:
            return 0

    def set_initial_action(self, weights: np.ndarray) -> None:
        """设置初始动作权重，用于与实体训练对齐安全裁剪"""
        if weights is None:
            return
        weights = np.array(weights, dtype=np.float32)

        # 支持5维（旧版）和7维（新版）动作
        if weights.shape[0] == 5:
            # 旧版5维权重，补充默认避障参数
            default_obstacle_params = np.array(
                [15.0, 5.0], dtype=np.float32
            )  # 默认避障距离15，系数5
            weights = np.concatenate([weights, default_obstacle_params])
        elif weights.shape[0] != 7:
            return

        # 裁剪到有效范围
        apf_weights = np.clip(weights[:5], self._apf_lower_bounds, self._apf_upper_bounds)
        obstacle_params = np.clip(weights[5:], [5.0, 1.0], [30.0, 15.0])
        weights = np.concatenate([apf_weights, obstacle_params])

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




