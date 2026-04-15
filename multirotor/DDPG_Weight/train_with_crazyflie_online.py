"""
基于实体Crazyflie实时日志的在线训练脚本

功能说明：
    - 在实体Crazyflie无人机上使用DDPG算法进行在线训练
    - 实时与实体无人机交互，收集飞行数据并计算奖励
    - 集成训练可视化模块，实时显示训练进度和统计信息
    - 支持权重安全限制，防止训练过程中权重变化过大导致飞行不稳定
    - 支持从已有模型继续训练，支持加载初始权重

主要特性：
    - 在线训练：每一步都与实体无人机实时交互
    - 安全限制：限制权重变化幅度，确保飞行安全
    - 可视化支持：实时显示训练统计、奖励曲线、权重变化
    - 模型保存：自动保存最佳模型和检查点

训练环境：
    - 环境类型：CrazyflieOnlineWeightEnv（实体无人机在线环境）
    - 算法：DDPG（Deep Deterministic Policy Gradient）
    - 动作空间：5维连续空间（APF权重系数）
    - 状态空间：由环境自动定义

使用方法：
    python train_with_crazyflie_online.py --config config.json
    python train_with_crazyflie_online.py --drone-name UAV1 --total-timesteps 500

安全提示：
    - 训练前必须确认已连接实体无人机并确保安全
    - 建议在安全环境中进行训练
    - 训练过程中请密切监控无人机状态

日期：2026-01-23
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

# 添加项目根目录到Python路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
MODULE_ROOT = os.path.dirname(os.path.abspath(__file__))
if MODULE_ROOT not in sys.path:
    sys.path.insert(0, MODULE_ROOT)

# 训练回调基类（带fallback，便于测试环境导入）
try:
    from stable_baselines3.common.callbacks import BaseCallback
except Exception:  # pragma: no cover - minimal fallback for tests without SB3
    class BaseCallback:  # type: ignore[override]
        def __init__(self, verbose: int = 0):
            self.verbose = verbose
            self.locals = {}
            self.num_timesteps = 0
            self.model = None

        def init_callback(self, model) -> None:
            self.model = model
            if hasattr(self, "_init_callback"):
                self._init_callback()

        def update_locals(self, locals_) -> None:
            self.locals = locals_

        def _init_callback(self) -> None:
            return None

        def _on_training_start(self) -> None:
            return None

        def _on_training_end(self) -> None:
            return None

        def _on_step(self) -> bool:
            return True

# 导入训练环境和数据记录器模块（可视化模块改为延迟导入）
from envs.crazyflie_weight_env import CrazyflieOnlineWeightEnv  # 实体无人机在线训练环境
from envs.crazyflie_data_logger import CrazyflieDataLogger  # 实体无人机数据记录器
from online_training_callbacks import EpisodeAwareTrainingCallback
from real_flight_priority import (
    RealFlightPriorityTrainer,
    normalize_real_flight_weighting_config,
)
from weighted_online_runner import run_weighted_online_training


def _load_train_config(path: str) -> dict:
    """
    加载训练配置文件

    功能：
        从 JSON 文件读取训练配置参数
        支持两种格式：
        1. 传统格式：直接返回配置字典
        2. 统一格式：包含 common 和模式专用配置，自动合并

    参数：
        path: 配置文件路径（JSON格式）

    返回：
        dict: 配置参数字典

    异常：
        FileNotFoundError: 配置文件不存在
        ValueError: 配置文件格式无效
    """
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("配置文件必须为JSON对象")

    # 检查是否为统一配置格式（包含 common 和 crazyflie_online 键）
    if "common" in data and "crazyflie_online" in data:
        # 统一配置格式：合并 common 和 crazyflie_online 配置
        merged_config = {}
        merged_config.update(data.get("common", {}))
        merged_config.update(data.get("crazyflie_online", {}))
        return merged_config
    else:
        # 传统配置格式：直接返回
        return data


def _load_real_weighting_config(config: dict):
    """
    加载并规范化 real_weighting 配置块

    规则：
        - 若配置中不存在 real_weighting，则返回 None（保持旧训练路径）
        - 若存在 real_weighting，则默认 enable_real_weighting=True（由规范化函数处理）
    """
    if not isinstance(config, dict):
        return None
    raw = config.get("real_weighting")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("real_weighting 配置必须为JSON对象")
    normalized = normalize_real_flight_weighting_config(raw)
    allowed_timings = {"episode_end"}
    if normalized.update_timing not in allowed_timings:
        raise ValueError(
            f"real_weighting.update_timing 不支持: {normalized.update_timing!r}"
        )
    return normalized


def _should_use_weighted_training(real_weighting_config) -> bool:
    if real_weighting_config is None:
        return False
    return bool(getattr(real_weighting_config, "enable_real_weighting", True))


def _env_int(name: str) -> int | None:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return None
    try:
        return int(raw_value)
    except ValueError:
        print(f"[!] 忽略无效的环境变量 {name}={raw_value}")
        return None


def _compute_weighted_training_target(total_timesteps: int, current_steps: int) -> int:
    return int(total_timesteps) + max(0, int(current_steps))


class ProgressTimingState:
    def __init__(self):
        self.started = False
        self.start_time = 0.0
        self.last_print_time = 0.0
        self.last_print_step = 0


class TrainingProgressCallback(BaseCallback):
    """
    训练进度回调类

    功能：
        - 监控训练进度，定期打印进度信息（包含ETA）
        - 更新训练可视化统计信息
        - 支持按步数或时间间隔打印

    继承自：
        stable_baselines3.common.callbacks.BaseCallback
    """

    def __init__(
        self,
        total_timesteps: int,
        print_interval_steps: int = 50,
        print_interval_sec: int = 15,
        training_visualizer=None,
        data_logger=None,
        timing_state: ProgressTimingState | None = None,
        logger: logging.Logger | None = None,
    ):
        """
        初始化训练进度回调

        参数：
            total_timesteps: 总训练步数
            print_interval_steps: 按步数打印的间隔（每N步打印一次）
            print_interval_sec: 按时间打印的间隔（每N秒打印一次）
            training_visualizer: 训练可视化器实例（可选）
        """
        super().__init__()
        self.total_timesteps = max(int(total_timesteps), 0)  # 总训练步数
        self.print_interval_steps = max(int(print_interval_steps), 1)  # 步数打印间隔
        self.print_interval_sec = max(int(print_interval_sec), 1)  # 时间打印间隔
        self.training_visualizer = training_visualizer  # 可视化器引用
        self.last_episode_count = 0  # 上次记录的Episode数量
        self.data_logger = data_logger  # 数据记录器引用
        self.timing_state = timing_state or ProgressTimingState()
        self._progress_logger = logger or logging.getLogger("crazyflie_train_online")

    def _on_training_start(self) -> None:
        now = time.time()
        if not self.timing_state.started:
            self.timing_state.started = True
            self.timing_state.start_time = now
        self.timing_state.last_print_time = now
        self.timing_state.last_print_step = int(self.num_timesteps)
        self._print_progress(force=True)

    def _on_step(self) -> bool:
        """
        每个训练步骤调用一次

        功能：
            - 检查是否需要打印进度（按步数或时间）
            - 更新训练可视化统计信息
            - 检测新完成的Episode并更新可视化

        返回：
            bool: True继续训练，False停止训练
        """
        num_timesteps = int(self.num_timesteps)
        now = time.time()

        # 检查是否需要打印进度（满足步数间隔或时间间隔）
        need_by_steps = (
            num_timesteps - self.timing_state.last_print_step
        ) >= self.print_interval_steps
        need_by_time = (
            now - self.timing_state.last_print_time
        ) >= self.print_interval_sec
        if need_by_steps or need_by_time:
            self._print_progress()

        # ========== 更新可视化统计 ==========
        if self.training_visualizer:
            try:
                # 检查是否有新的episode完成
                if (
                    hasattr(self.model, "ep_info_buffer")
                    and len(self.model.ep_info_buffer) > 0
                ):
                    current_episode_count = len(self.model.ep_info_buffer)
                    if current_episode_count > self.last_episode_count:
                        # 新episode完成，更新统计
                        ep_info = self.model.ep_info_buffer[-1]
                        ep_reward = ep_info.get("r", 0.0)  # Episode总奖励
                        ep_length = ep_info.get("l", 0)  # Episode步数
                        self.training_visualizer.update_training_stats(
                            episode_reward=ep_reward,
                            episode_length=ep_length,
                            is_episode_done=True,
                        )
                        self.last_episode_count = current_episode_count

                        # 记录 Episode 统计信息
                        if self.data_logger:
                            self.data_logger.record_episode_stats(
                                episode=current_episode_count,
                                reward=ep_reward,
                                length=ep_length,
                            )

                # 更新当前步的奖励（从locals获取）
                if "rewards" in self.locals and len(self.locals["rewards"]) > 0:
                    step_reward = float(self.locals["rewards"][0])
                    self.training_visualizer.update_training_stats(
                        current_step_reward=step_reward
                    )

                # 更新权重历史（定期更新，不只在episode结束时）
                if hasattr(self.model, "env"):
                    env = self.model.env
                    # 处理VecEnv包装（stable-baselines3可能使用向量化环境）
                    if hasattr(env, "envs") and len(env.envs) > 0:
                        env = env.envs[0]  # 获取实际环境
                    if hasattr(env, "server") and env.server:
                        drone_name = getattr(env, "drone_name", None)
                        if drone_name and drone_name in env.server.algorithms:
                            # 获取当前权重并更新可视化
                            weights = env.server.algorithms[
                                drone_name
                            ].get_current_coefficients()
                            self.training_visualizer.update_weight_history(weights)

                            # 记录权重到数据记录器
                            if self.data_logger:
                                self.data_logger.record_weights(
                                    drone_name=drone_name,
                                    weights=weights,
                                    episode=current_episode_count,
                                    step=self.num_timesteps,
                                )
            except Exception:
                # 静默忽略可视化更新错误，避免影响训练
                pass

        # ========== 记录实体无人机飞行数据 ==========
        if self.data_logger:
            try:
                # 从环境中获取当前的 logging_data
                if hasattr(self.model, "env"):
                    env = self.model.env
                    # 处理VecEnv包装
                    if hasattr(env, "envs") and len(env.envs) > 0:
                        env = env.envs[0]
                    if hasattr(env, "server") and env.server:
                        drone_name = getattr(env, "drone_name", None)
                        if drone_name:
                            logging_data = (
                                env.server.crazyswarm.get_loggingData_by_droneName(
                                    drone_name
                                )
                            )
                            if logging_data:
                                self.data_logger.record_flight_data(
                                    drone_name, logging_data
                                )
            except Exception:
                # 静默忽略数据记录错误，避免影响训练
                pass
        # ===========================================
        # ====================================

        return True

    def _print_progress(self, force: bool = False) -> None:
        num_timesteps = int(self.num_timesteps)
        now = time.time()
        if (
            not force
            and num_timesteps == self.timing_state.last_print_step
            and (now - self.timing_state.last_print_time) < 1.0
        ):
            return
        self.timing_state.last_print_step = num_timesteps
        self.timing_state.last_print_time = now

        elapsed = now - self.timing_state.start_time
        if self.total_timesteps > 0:
            progress = min(num_timesteps / self.total_timesteps, 1.0)
            eta = (elapsed / progress - elapsed) if progress > 0 else 0.0
            percent = progress * 100.0
            self._progress_logger.info(
                "进度 %s/%s (%.1f%%) 已用%s 预计剩余%s",
                num_timesteps,
                self.total_timesteps,
                percent,
                _format_duration(elapsed),
                _format_duration(eta),
            )
        else:
            self._progress_logger.info(
                "进度 %s 步 已用%s", num_timesteps, _format_duration(elapsed)
            )


class _WeightedCallbackWrapper(BaseCallback):
    def __init__(self, progress_callback, episode_callback):
        super().__init__()
        self._progress_callback = progress_callback
        self._episode_callback = episode_callback

    @property
    def episode_finished(self):
        return bool(getattr(self._episode_callback, "episode_finished", False))

    @property
    def last_episode_index(self):
        return getattr(self._episode_callback, "last_episode_index", None)

    def _init_callback(self) -> None:
        self._delegate_init(self._progress_callback)
        self._delegate_init(self._episode_callback)

    def update_locals(self, locals_) -> None:
        super().update_locals(locals_)
        self._delegate_locals(self._progress_callback, locals_)
        self._delegate_locals(self._episode_callback, locals_)

    def _on_training_start(self) -> None:
        self._delegate_training_start(self._progress_callback)
        self._delegate_training_start(self._episode_callback)

    def _on_training_end(self) -> None:
        self._delegate_training_end(self._progress_callback)
        self._delegate_training_end(self._episode_callback)

    def _on_step(self) -> bool:
        self._sync_timesteps(self._progress_callback)
        self._sync_timesteps(self._episode_callback)
        progress_ok = self._delegate_step(self._progress_callback)
        episode_ok = self._delegate_step(self._episode_callback)
        return bool(progress_ok) and bool(episode_ok)

    def _delegate_init(self, callback) -> None:
        if hasattr(callback, "init_callback"):
            callback.init_callback(self.model)
        elif hasattr(callback, "_init_callback"):
            callback._init_callback()
        if hasattr(callback, "model"):
            callback.model = self.model

    @staticmethod
    def _delegate_locals(callback, locals_) -> None:
        if hasattr(callback, "update_locals"):
            callback.update_locals(locals_)
        else:
            setattr(callback, "locals", locals_)

    @staticmethod
    def _delegate_training_start(callback) -> None:
        if hasattr(callback, "_on_training_start"):
            callback._on_training_start()

    @staticmethod
    def _delegate_training_end(callback) -> None:
        if hasattr(callback, "_on_training_end"):
            callback._on_training_end()

    def _sync_timesteps(self, callback) -> None:
        if hasattr(callback, "num_timesteps"):
            callback.num_timesteps = getattr(self, "num_timesteps", 0)

    @staticmethod
    def _delegate_step(callback) -> bool:
        if hasattr(callback, "_on_step"):
            return bool(callback._on_step())
        if hasattr(callback, "on_step"):
            return bool(callback.on_step())
        return True


def _build_weighted_callback(progress_callback, episode_callback):
    return _WeightedCallbackWrapper(progress_callback, episode_callback)


def _get_config_value(cli_value, config: dict, key: str, default):
    """
    获取配置值（优先级：命令行 > 配置文件 > 默认值）

    参数：
        cli_value: 命令行参数值（优先级最高）
        config: 配置字典
        key: 配置键名
        default: 默认值（优先级最低）

    返回：
        配置值
    """
    if cli_value is not None:
        return cli_value
    if key in config:
        return config[key]
    return default


def _format_duration(seconds: float) -> str:
    """
    格式化时间持续时间为可读字符串

    功能：
        将秒数转换为 "HH:MM:SS" 或 "MM:SS" 格式

    参数：
        seconds: 秒数（浮点数）

    返回：
        str: 格式化后的时间字符串

    示例：
        _format_duration(3661) -> "01:01:01"
        _format_duration(125) -> "02:05"
    """
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _save_model(model, path: str, logger, note: str) -> bool:
    """
    保存训练模型到文件

    功能：
        将DDPG模型保存为.zip文件

    参数：
        model: DDPG模型实例
        path: 保存路径（不含.zip扩展名）
        logger: 日志记录器
        note: 保存说明（用于日志）

    返回：
        bool: 保存是否成功
    """
    if model is None:
        return False
    try:
        model.save(path)
        logger.info("%s: %s.zip", note, path)
        return True
    except Exception as exc:
        logger.error("保存模型失败: %s (%s)", path, exc)
        return False


def _log_weighted_update_result(logger, episode_index: int, result) -> None:
    if result is None:
        return
    status = getattr(result, "status", "unknown")
    real_samples = getattr(result, "episode_real_samples", 0)
    gradient_steps = getattr(result, "extra_gradient_steps", 0)
    rollback = getattr(result, "rollback_triggered", False)
    delta_norm = float(getattr(result, "policy_param_delta_norm", 0.0))
    logger.info(
        "[RealWeight] episode=%s status=%s real_samples=%s gradient_steps=%s "
        "rollback=%s delta_norm=%.6f",
        episode_index,
        status,
        real_samples,
        gradient_steps,
        rollback,
        delta_norm,
    )


def _save_final_weights(server, path: str, logger) -> None:
    """
    保存各无人机最后的权重系数到JSON文件

    功能：
        将训练完成后的权重系数保存到JSON文件，用于后续训练或部署

    参数：
        server: AlgorithmServer实例，包含所有无人机的算法对象
        path: 保存路径（JSON文件）
        logger: 日志记录器

    保存格式：
        {
            "UAV1": {
                "repulsionCoefficient": 1.0,
                "entropyCoefficient": 2.0,
                ...
            },
            "UAV2": {...}
        }
    """
    if not server or not path:
        return
    weights_by_drone = {}
    # 遍历所有无人机，收集权重系数
    for drone_name in server.drone_names:
        algo = server.algorithms.get(drone_name)
        if not algo:
            continue
        weights_by_drone[drone_name] = algo.get_current_coefficients()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(weights_by_drone, f, ensure_ascii=False, indent=2)
        logger.info("[OK] 权重已保存: %s", path)
    except Exception as exc:
        logger.error("[!]  保存权重失败: %s (%s)", path, exc)


def _load_initial_weights(path: str, drone_name: str, logger) -> dict:
    """
    加载初始权重（支持两种格式）

    功能：
        从JSON文件加载初始权重，支持两种格式：
        1. 单一权重字典：所有无人机使用相同权重
        2. 按无人机名映射：每个无人机有独立的权重

    参数：
        path: 权重文件路径（JSON格式）
        drone_name: 无人机名称（用于查找对应权重）
        logger: 日志记录器

    返回：
        dict: 权重字典，包含5个APF系数

    支持的格式：
        格式1（单一权重）:
        {
            "repulsionCoefficient": 1.0,
            "entropyCoefficient": 2.0,
            ...
        }

        格式2（按无人机）:
        {
            "UAV1": {"repulsionCoefficient": 1.0, ...},
            "UAV2": {"repulsionCoefficient": 1.5, ...}
        }
    """
    if not path:
        return {}
    if not os.path.exists(path):
        logger.error("初始权重文件不存在: %s", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.error("读取初始权重失败: %s (%s)", path, exc)
        return {}

    if not isinstance(data, dict):
        logger.error("初始权重格式无效: %s", path)
        return {}

    # 格式1: 检查是否为单一权重字典（包含所有必需的权重键）
    required_keys = [
        "repulsionCoefficient",  # α1: 排斥力系数
        "entropyCoefficient",  # α2: 熵值系数
        "distanceCoefficient",  # α3: 距离系数
        "leaderRangeCoefficient",  # α4: Leader范围系数
        "directionRetentionCoefficient",  # α5: 方向保持系数
    ]
    if all(k in data for k in required_keys):
        return data  # 单一权重格式，直接返回

    # 格式2: 按无人机名索引
    if drone_name in data and isinstance(data[drone_name], dict):
        return data[drone_name]

    logger.warning("未找到无人机%s的初始权重，跳过", drone_name)
    return {}


def _weights_to_action(weights: dict) -> np.ndarray:
    """
    将权重字典转换为动作向量（numpy数组）

    功能：
        将APF权重系数字典转换为DDPG算法所需的动作向量格式

    参数：
        weights: 权重字典，包含5个APF系数 + 2个避障参数（可选）

    返回：
        np.ndarray: 形状为(7,)的浮点数组，包含7个参数

    权重顺序：
        [repulsionCoefficient, entropyCoefficient, distanceCoefficient,
         leaderRangeCoefficient, directionRetentionCoefficient,
         obstacleRepulsionDistance, obstacleRepulsionCoefficient]
    """
    return np.array(
        [
            float(weights.get("repulsionCoefficient", 0.0)),
            float(weights.get("entropyCoefficient", 0.0)),
            float(weights.get("distanceCoefficient", 0.0)),
            float(weights.get("leaderRangeCoefficient", 0.0)),
            float(weights.get("directionRetentionCoefficient", 0.0)),
            float(weights.get("obstacleRepulsionDistance", 15.0)),
            float(weights.get("obstacleRepulsionCoefficient", 5.0)),
        ],
        dtype=np.float32,
    )


def _derive_weights_path(model_path: str) -> str:
    """
    根据模型路径推导权重文件路径

    功能：
        权重文件名与模型文件名一致（去掉.zip，加上.json）
        例如：model_20250123_120000.zip -> model_20250123_120000.json

    参数：
        model_path: 模型路径（不含.zip扩展名）

    返回：
        str: 权重文件路径（.json扩展名）
    """
    if not model_path:
        return ""
    # 如果路径以.zip结尾，去掉它
    if model_path.endswith(".zip"):
        model_path = model_path[:-4]
    # 返回与模型文件名一致的权重文件名
    return f"{model_path}.json"


def main():
    """
    主训练流程函数

    功能：
        1. 解析命令行参数和配置文件
        2. 初始化AlgorithmServer（连接实体无人机）
        3. 创建在线训练环境（CrazyflieOnlineWeightEnv）
        4. 启动训练可视化（可选）
        5. 创建并训练DDPG模型
        6. 保存训练结果和模型

    训练流程：
        1. 加载配置和参数
        2. 创建并启动AlgorithmServer
        3. 启动无人机任务
        4. 创建训练环境
        5. 启动训练可视化（可选）
        6. 创建DDPG模型并开始训练

    安全特性：
        - 训练前需要人工确认（避免误启动）
        - 权重变化安全限制（防止飞行不稳定）
        - 支持从已有模型继续训练

    异常处理：
        - KeyboardInterrupt: 用户中断，尝试保存当前模型
        - Exception: 其他错误，清理资源并退出
    """
    # ========== 初始化日志系统 ==========
    logging.basicConfig(
        level=logging.INFO,  # 日志级别：INFO
        format="%(asctime)s - %(levelname)s - %(message)s",  # 日志格式：时间 - 级别 - 消息
    )
    logger = logging.getLogger("crazyflie_train_online")
    # ====================================

    parser = argparse.ArgumentParser(description="Crazyflie在线训练（实体机）")
    parser.add_argument(
        "--config", type=str, default=None, help="训练配置文件路径（JSON）"
    )
    parser.add_argument("--drone-name", type=str, default=None, help="训练无人机名称")
    parser.add_argument("--total-timesteps", type=int, default=None, help="训练步数")
    parser.add_argument(
        "--step-duration", type=float, default=None, help="每步飞行时长（秒）"
    )
    parser.add_argument("--reward-config", type=str, default=None, help="奖励配置路径")
    parser.add_argument("--save-dir", type=str, default=None, help="模型保存目录")
    parser.add_argument(
        "--continue-model", type=str, default=None, help="继续训练模型路径（不含.zip）"
    )
    parser.add_argument(
        "--initial-model-path", type=str, default=None, help="初始模型路径（不含.zip）"
    )
    parser.add_argument(
        "--reset-unity",
        action="store_true",
        default=None,
        help="每个episode重置Unity环境",
    )
    parser.add_argument(
        "--safety-max-delta", type=float, default=None, help="权重变化最大幅度"
    )
    parser.add_argument(
        "--no-safety-limit", action="store_true", default=None, help="关闭权重变化限制"
    )
    parser.add_argument(
        "--progress-interval", type=int, default=None, help="进度打印间隔（步）"
    )
    parser.add_argument(
        "--enable-visualization",
        action="store_true",
        default=None,
        help="启用训练可视化",
    )
    parser.add_argument(
        "--no-visualization", action="store_true", default=None, help="禁用训练可视化"
    )
    args = parser.parse_args()

    # 读取配置文件（若未提供则用空配置，后续会回退到默认值）
    config = _load_train_config(args.config)
    real_weighting_config = _load_real_weighting_config(config)
    raw_real_weighting = config.get("real_weighting") if isinstance(config, dict) else None

    # 从命令行/配置中解析训练超参数
    # 规则：命令行优先，其次配置文件，最后默认值
    drone_name = _get_config_value(args.drone_name, config, "drone_name", "UAV1")
    total_timesteps = _get_config_value(
        args.total_timesteps, config, "total_timesteps", 500
    )
    quick_total_timesteps = _env_int("AIRSIM_QUICK_DDPG_TIMESTEPS")
    if quick_total_timesteps is not None:
        total_timesteps = quick_total_timesteps
    step_duration = _get_config_value(args.step_duration, config, "step_duration", 5.0)
    reward_config = _get_config_value(args.reward_config, config, "reward_config", None)
    save_dir = _get_config_value(args.save_dir, config, "save_dir", "models")
    continue_model = _get_config_value(
        args.continue_model, config, "continue_model", None
    )
    initial_model_path = _get_config_value(
        args.initial_model_path, config, "initial_model_path", None
    )
    reset_unity = _get_config_value(args.reset_unity, config, "reset_unity", False)
    safety_max_delta = _get_config_value(
        args.safety_max_delta, config, "safety_max_delta", 0.5
    )
    progress_interval = _get_config_value(
        args.progress_interval, config, "progress_interval", 50
    )

    # 可视化开关：命令行优先，其次配置文件，最后默认值（默认启用）
    if args.no_visualization:
        enable_visualization = False
    elif args.enable_visualization:
        enable_visualization = True
    else:
        enable_visualization = _get_config_value(
            None, config, "enable_visualization", True
        )

    if not initial_model_path and continue_model:
        initial_model_path = continue_model
    initial_weights_path = _derive_weights_path(initial_model_path)

    # 权重安全限制开关：可由命令行显式指定，或由配置推导
    # no_safety_limit=True 表示关闭安全限制
    no_safety_limit = args.no_safety_limit
    if no_safety_limit is None:
        if "no_safety_limit" in config:
            no_safety_limit = config["no_safety_limit"]
        elif "safety_limit" in config:
            no_safety_limit = not bool(config["safety_limit"])
        else:
            no_safety_limit = False

    # 训练依赖：SB3 的 DDPG 与回调机制
    try:
        from stable_baselines3 import DDPG
        from stable_baselines3.common.noise import NormalActionNoise
    except ImportError:
        logger.error("缺少stable-baselines3，请先安装")
        sys.exit(1)

    # 算法服务器：负责与实体机/仿真系统通信
    from AlgorithmServer import MultiDroneAlgorithmServer
    # 训练可视化模块（延迟导入以避免轻量环境依赖）
    from Visualization import DDPGTrainingVisualizer

    # ========== 训练进度回调类 ==========

    # 打印训练参数，便于复现实验
    logger.info(
        "训练参数: drone=%s total=%s step=%.2fs reset_unity=%s safety_limit=%s "
        "max_delta=%.3f progress_interval=%s save_dir=%s continue_model=%s "
        "initial_model_path=%s initial_weights_path=%s enable_visualization=%s",
        drone_name,
        total_timesteps,
        step_duration,
        reset_unity,
        not no_safety_limit,
        safety_max_delta,
        progress_interval,
        save_dir,
        continue_model,
        initial_model_path,
        initial_weights_path,
        enable_visualization,
    )

    # 可视化状态提示
    if enable_visualization:
        logger.info("=" * 60)
        logger.info("[Eye]  训练可视化: 已启用")
        logger.info("   可视化窗口将在训练开始后自动弹出")
        logger.info("   显示内容: 训练统计、奖励曲线、权重变化、环境状态")
        logger.info("   操作提示: 按ESC键可关闭可视化窗口（不影响训练）")
        logger.info("=" * 60)
    else:
        logger.info("[Eye]  训练可视化: 已禁用")

    # ========== 安全确认 ==========
    # 实体机训练需要人工确认，避免误启动导致安全事故
    logger.info("确认已连接实体无人机并确保安全？(Y/N)")
    confirm = input().strip().upper()
    if confirm != "Y":
        logger.warning("已取消")
        return
    # =============================

    # ========== 初始化运行时对象 ==========
    # 这些变量在finally块中用于资源清理
    server = None  # AlgorithmServer实例
    model = None  # DDPG模型实例
    model_saved = False  # 模型是否已保存标志
    training_visualizer = None  # 训练可视化器实例
    data_logger = None  # 实体无人机数据记录器实例
    # ====================================

    try:
        # ========== 创建并启动算法服务器 ==========
        # AlgorithmServer负责与实体无人机通信和控制
        server = MultiDroneAlgorithmServer(
            drone_names=[drone_name],  # 训练无人机名称列表
            use_learned_weights=False,  # 训练模式：不使用已学习的权重
            model_path=None,  # 训练模式：不加载模型
            enable_visualization=False,  # 使用训练专用可视化，禁用服务器自带可视化
        )

        # 启动通信与后台线程
        if not server.start():
            logger.error("AlgorithmServer启动失败")
            return

        # 启动任务（让系统进入可训练状态）
        if not server.start_mission():
            logger.error("任务启动失败")
            return

        # 等待系统稳定
        time.sleep(2.0)

        # ========== 创建实体无人机数据记录器 ==========
        # 用于记录训练过程中的实体无人机飞行数据
        logger.info("创建实体无人机数据记录器...")
        data_logger = CrazyflieDataLogger(
            drone_names=[drone_name],
            output_dir=os.path.join(os.path.dirname(__file__), "crazyflie_logs"),
        )
        data_logger.start_recording()
        logger.info("[OK] 数据记录器已启动")
        # =============================================

        # ========== 创建在线训练环境 ==========
        # CrazyflieOnlineWeightEnv: 实体无人机在线训练环境
        # 环境功能：
        #   - 每一步都与实体无人机实时交互
        #   - 执行飞行动作并收集状态数据
        #   - 计算奖励信号（基于扫描效果、电量等）
        #   - 支持权重安全限制（防止权重变化过大）
        env = CrazyflieOnlineWeightEnv(
            server=server,  # 算法服务器引用
            drone_name=drone_name,  # 训练无人机名称
            reward_config_path=reward_config,  # 奖励配置文件路径
            step_duration=step_duration,  # 每步飞行时长（秒）
            reset_unity=reset_unity,  # 是否在每个episode重置Unity环境
            safety_limit=not no_safety_limit,  # 是否启用权重变化安全限制
            max_weight_delta=safety_max_delta,  # 权重变化最大幅度（安全限制）
        )

        # 应用初始权重（若提供）
        if initial_model_path:
            # 自动查找同名权重文件
            initial_weights_path = _derive_weights_path(initial_model_path)
            if os.path.exists(initial_weights_path):
                logger.info("[Folder] 找到权重文件: %s", initial_weights_path)
                weights = _load_initial_weights(
                    initial_weights_path, drone_name, logger
                )
                if weights:
                    server.algorithms[drone_name].set_coefficients(weights)
                    env.set_initial_action(_weights_to_action(weights))
                    logger.info("[OK] 已加载初始权重: %s", initial_weights_path)
                else:
                    logger.warning("[!]  权重文件格式无效: %s", initial_weights_path)
            else:
                logger.warning("[!]  权重文件不存在: %s", initial_weights_path)
                logger.info("   模型路径: %s", initial_model_path)
                logger.info("   将使用默认配置权重")

        # 创建并启动训练专用可视化
        if enable_visualization:
            logger.info("启动训练专用可视化...")
            try:
                training_visualizer = DDPGTrainingVisualizer(server=server, env=env)
                if training_visualizer.start_visualization():
                    logger.info("[OK] 训练可视化已启动")
                    logger.info("[Light] 可视化窗口应该会弹出，显示训练统计和环境状态")
                    logger.info("[Light] 按ESC键可关闭可视化窗口（不影响训练）")
                    # 给可视化窗口一些初始化时间
                    time.sleep(1.0)
                else:
                    logger.warning("[!]  训练可视化启动失败，但训练将继续")
            except Exception as e:
                logger.warning("[!]  训练可视化初始化失败: %s", str(e))
                logger.info("[Light] 训练将继续，但不显示可视化")
                training_visualizer = None

        # 动作维度决定噪声向量长度（用于探索）
        n_actions = env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=0.15 * np.ones(n_actions)
        )

        # 确保模型输出目录存在
        os.makedirs(save_dir, exist_ok=True)
        # 使用时间戳作为模型文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        final_path = os.path.join(
            save_dir, f"weight_predictor_crazyflie_online_{timestamp}"
        )
        logger.info("模型保存路径: %s.zip", os.path.abspath(final_path))

        # 继续训练：加载已有模型并保持步数累计
        if continue_model:
            logger.info("继续训练: 加载模型 %s.zip", continue_model)
            model = DDPG.load(continue_model, env=env, print_system_info=True)
            reset_num_timesteps = False
        else:
            # 新训练：从头初始化 DDPG
            model = DDPG(
                "MlpPolicy",
                env,
                action_noise=action_noise,
                learning_rate=1e-4,
                buffer_size=5000,
                learning_starts=200,
                batch_size=64,
                tau=0.005,
                gamma=0.99,
                train_freq=(1, "episode"),
                gradient_steps=-1,
                verbose=1,
                device="cpu",
            )
            reset_num_timesteps = True

        # 进度回调：定期打印训练进度
        progress_timing_state = ProgressTimingState()
        progress_cb = TrainingProgressCallback(
            total_timesteps=total_timesteps,
            print_interval_steps=progress_interval,
            print_interval_sec=15,
            training_visualizer=training_visualizer,
            data_logger=data_logger,
            timing_state=progress_timing_state,
            logger=logger,
        )
        if not _should_use_weighted_training(real_weighting_config):
            if real_weighting_config is None:
                logger.info("[RealWeight] 未配置 real_weighting，使用默认在线训练路径")
            else:
                logger.info("[RealWeight] real_weighting 已禁用，使用默认在线训练路径")
            # 训练主循环：达到 total_timesteps 视为训练完成
            model.learn(
                total_timesteps=total_timesteps,
                reset_num_timesteps=reset_num_timesteps,
                callback=progress_cb,
            )
        else:
            logger.info(
                "[RealWeight] 使用 real_weighting 配置: "
                "timing=%s enable=%s multiplier=%s min_samples=%s max_updates=%s",
                real_weighting_config.update_timing,
                real_weighting_config.enable_real_weighting,
                real_weighting_config.real_update_multiplier,
                real_weighting_config.min_real_samples_before_update,
                real_weighting_config.max_real_updates_per_episode,
            )
            if isinstance(raw_real_weighting, dict) and "real_batch_ratio" in raw_real_weighting:
                logger.info(
                    "[RealWeight] real_batch_ratio 当前版本仅保留兼容性，不参与实际训练采样"
                )
            priority_trainer = RealFlightPriorityTrainer(real_weighting_config)

            def _callback_factory():
                episode_callback = EpisodeAwareTrainingCallback(
                    total_timesteps=total_timesteps,
                    print_interval_steps=progress_interval,
                    print_interval_sec=15,
                    training_visualizer=training_visualizer,
                    data_logger=data_logger,
                    priority_trainer=priority_trainer,
                )
                return _build_weighted_callback(progress_cb, episode_callback)

            def _on_episode_end(episode_index: int):
                if real_weighting_config.update_timing != "episode_end":
                    logger.info(
                        "[RealWeight] update_timing=%s，跳过加权更新",
                        real_weighting_config.update_timing,
                    )
                    return None
                result = priority_trainer.apply_post_episode_update(
                    model=model, episode_index=episode_index
                )
                _log_weighted_update_result(logger, episode_index, result)
                return result

            adjusted_total_timesteps = total_timesteps
            if not reset_num_timesteps:
                adjusted_total_timesteps = _compute_weighted_training_target(
                    total_timesteps=total_timesteps,
                    current_steps=getattr(model, "num_timesteps", 0),
                )

            run_weighted_online_training(
                model=model,
                total_timesteps=adjusted_total_timesteps,
                reset_num_timesteps=reset_num_timesteps,
                callback_factory=_callback_factory,
                on_episode_end=_on_episode_end,
            )

        # 正常结束后保存模型
        model_saved = _save_model(model, final_path, logger, "训练完成，模型已保存")

        # 保存权重文件（与模型文件名一致）
        if model_saved and server:
            weights_path = _derive_weights_path(final_path)
            _save_final_weights(server, weights_path, logger)

        # 停止数据记录并保存
        if data_logger:
            logger.info("停止并保存实体无人机数据...")
            data_logger.stop_recording()
            data_logger.save_all()

    except KeyboardInterrupt:
        # 人工中断时尝试保存当前模型
        logger.warning("训练停止，尝试保存当前模型")
        if not model_saved:
            model_saved = _save_model(model, final_path, logger, "中断保存，模型已保存")
            # 保存权重文件（与模型文件名一致）
            if model_saved and server:
                weights_path = _derive_weights_path(final_path)
                _save_final_weights(server, weights_path, logger)
            # 停止数据记录并保存
            if data_logger:
                logger.warning("保存中断时的实体无人机数据...")
                data_logger.stop_recording()
                data_logger.save_all()
    finally:
        # 停止数据记录（最优先，确保数据被保存）
        if data_logger:
            try:
                logger.info("保存实体无人机训练数据...")
                if data_logger.is_recording:
                    data_logger.stop_recording()
                data_logger.save_all()
            except Exception as e:
                logger.warning("保存数据时出错: %s", e)

        # 停止可视化
        if training_visualizer:
            logger.info("停止训练可视化...")
            try:
                training_visualizer.stop_visualization()
                logger.info("[OK] 训练可视化已停止")
            except Exception as e:
                logger.warning("停止可视化时出错: %s", e)

        # 无论成功与否都释放服务器资源
        if server:
            server.stop()


if __name__ == "__main__":
    main()
