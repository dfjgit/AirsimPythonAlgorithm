"""
改进版AirSim环境训练脚本

功能说明：
    - 在AirSim仿真环境中使用DDPG算法训练APF（人工势场）权重系数
    - 支持多无人机协同训练模式
    - 集成训练可视化模块，实时显示训练进度和统计信息
    - 支持从已有权重继续训练
    - 自动保存最佳模型和检查点

主要改进：
    - 解决Unity卡死问题：改进异常处理和资源清理
    - 支持Ctrl+C强制退出：优雅处理中断信号
    - 增强的训练回调：显示详细的Episode统计信息
    - 训练可视化：实时显示训练状态、奖励曲线、权重变化

使用方法：
    python train_with_airsim_improved.py --config config.json
    python train_with_airsim_improved.py --total-timesteps 1000 --enable-visualization

日期：2026-01-23
"""

import os
import sys
import time
import signal
import argparse
import json
import numpy as np
import subprocess

# 添加项目根目录到Python路径，以便导入项目模块
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# ==================== 全局变量 ====================
# 全局标志，用于Ctrl+C中断处理
# 当用户按下Ctrl+C时，设置此标志为True，训练循环会优雅地停止
training_interrupted = False
# ==================================================


def signal_handler(sig, frame):
    """
    处理Ctrl+C中断信号

    功能：
        - 第一次按下Ctrl+C：设置中断标志，训练会优雅停止
        - 第二次按下Ctrl+C：强制退出程序

    参数：
        sig: 信号编号（SIGINT）
        frame: 当前堆栈帧
    """
    global training_interrupted
    if not training_interrupted:
        # 第一次中断：设置标志，允许训练优雅停止
        print("\n\n" + "=" * 60)
        print("[中断] 检测到Ctrl+C，正在停止训练...")
        print("=" * 60)
        training_interrupted = True
    else:
        # 第二次中断：强制退出
        print("\n[强制退出] 再次按Ctrl+C将强制退出程序")
        sys.exit(1)


# 注册信号处理器：捕获Ctrl+C信号
signal.signal(signal.SIGINT, signal_handler)

print("=" * 60)
print("DQN训练 - 改进版（防止Unity卡死）")
print("=" * 60)

# ==================== 依赖检查 ====================
# 检查并导入必要的第三方库
print("\n检查依赖...")
try:
    import torch  # PyTorch深度学习框架
    from stable_baselines3 import DDPG  # DDPG强化学习算法
    from stable_baselines3.common.noise import NormalActionNoise  # 动作噪声（用于探索）
    from stable_baselines3.common.callbacks import BaseCallback  # 训练回调基类

    print("[OK] 依赖检查通过")
except ImportError as e:
    print(f"[错误] 缺少依赖: {e}")
    print("请运行: pip install stable-baselines3 torch")
    input("按Enter退出...")
    sys.exit(1)
# ==================================================

# ==================== 导入项目模块 ====================
# 导入训练环境：用于AirSim仿真的权重训练环境
from envs.simple_weight_env import SimpleWeightEnv

# 独立进程可视化 (pygame 不与训练进程共享)
try:
    from multirotor.Visualization.visualization_ipc import VisualizationIPCServer

    HAS_EXT_VIS = True
except Exception:
    HAS_EXT_VIS = False
    print("警告: 无法导入 VisualizationIPCServer，独立可视化将被禁用")

# 导入算法服务器：负责与Unity AirSim通信和算法执行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AlgorithmServer import MultiDroneAlgorithmServer
# ==================================================


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
        dict: 配置参数字典，如果文件不存在或读取失败则返回空字典

    示例：
        config = _load_train_config("config.json")
    """
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"[!] 配置文件不存在: {path}")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}

        # 检查是否为统一配置格式（包含 common 和 airsim_virtual 键）
        if "common" in data and "airsim_virtual" in data:
            # 统一配置格式：合并 common 和 airsim_virtual 配置
            merged_config = {}
            merged_config.update(data.get("common", {}))
            merged_config.update(data.get("airsim_virtual", {}))
            return merged_config
        else:
            # 传统配置格式：直接返回
            return data
    except Exception as exc:
        print(f"[!] 配置文件读取失败: {exc}")
        return {}


def _get_config_value(cli_value, config: dict, key: str, default):
    """
    获取配置值（优先级：命令行 > 配置文件 > 默认值）

    功能：
        按照优先级顺序获取配置参数值

    参数：
        cli_value: 命令行参数值（优先级最高）
        config: 配置字典
        key: 配置键名
        default: 默认值（优先级最低）

    返回：
        配置值

    示例：
        total_steps = _get_config_value(args.total_timesteps, config, "total_timesteps", 100)
    """
    if cli_value is not None:
        return cli_value
    if key in config:
        return config[key]
    return default


def _save_final_weights(server, path: str) -> None:
    """
    保存各无人机最后的权重系数到JSON文件

    功能：
        将训练完成后的权重系数保存到JSON文件，用于后续训练或部署

    参数：
        server: AlgorithmServer实例，包含所有无人机的算法对象
        path: 保存路径（JSON文件）

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
        print(f"[OK] 初始权重已保存: {path}")
    except Exception as exc:
        print(f"[!]  保存初始权重失败: {exc}")


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


def _load_initial_weights(path: str) -> dict:
    """
    加载初始权重（支持按无人机名映射或单一字典）

    功能：
        从JSON文件加载初始权重，支持两种格式：
        1. 单一字典格式：所有无人机使用相同权重
        2. 按无人机名映射：每个无人机有独立的权重

    参数：
        path: 权重文件路径（JSON格式）

    返回：
        dict: 权重字典，格式为 {drone_name: weights} 或 {"__all__": weights}

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
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[!]  读取初始权重失败: {exc}")
        return {}

    if not isinstance(data, dict):
        return {}

    # 检查是否为单一权重字典格式（包含所有必需的权重键）
    keys = [
        "repulsionCoefficient",  # α1: 排斥力系数
        "entropyCoefficient",  # α2: 熵值系数
        "distanceCoefficient",  # α3: 距离系数
        "leaderRangeCoefficient",  # α4: Leader范围系数
        "directionRetentionCoefficient",  # α5: 方向保持系数
    ]
    if all(k in data for k in keys):
        # 单一权重格式，返回为 "__all__" 键
        return {"__all__": data}

    # 按无人机名映射格式
    return {k: v for k, v in data.items() if isinstance(v, dict)}


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
            float(weights.get("obstacleRepulsionDistance", 15.0)),  # 默认避障距离
            float(weights.get("obstacleRepulsionCoefficient", 5.0)),  # 默认避障系数
        ],
        dtype=np.float32,
    )


class ImprovedTrainingCallback(BaseCallback):
    """
    改进的训练回调类

    功能：
        - 监控训练进度，定期打印详细的Episode统计信息
        - 自动保存最佳模型和检查点
        - 更新训练可视化模块的统计信息
        - 支持Ctrl+C优雅中断

    主要特性：
        - 美观的Episode完成信息显示（带边框）
        - 奖励趋势分析（上升/下降）
        - 自动保存最佳模型（基于平均奖励）
        - 定期保存检查点（防止训练中断丢失进度）
        - 实时更新可视化窗口

    继承自：
        stable_baselines3.common.callbacks.BaseCallback
    """

    def __init__(
        self,
        total_timesteps,
        check_freq=1000,
        save_path="./models/",
        training_visualizer=None,
        server=None,
        vis_process=None,
        vis_log_path=None,
        overwrite_model=False,
        model_name="weight_predictor_airsim",
        verbose=1,
    ):
        """
        初始化训练回调

        参数:
            total_timesteps: 总训练步数
            check_freq: 检查点保存频率（每N步保存一次）
            save_path: 模型保存目录路径
            training_visualizer: 训练可视化器实例（可选）
            server: AlgorithmServer 实例（用于访问 DataCollector）
            vis_process: 独立可视化进程（subprocess.Popen对象，可选）
            vis_log_path: 独立可视化日志路径（可选）
            overwrite_model: 是否覆盖现有模型（不生成新时间戳）
            model_name: 模型名称（不含.zip）
            verbose: 详细程度（0=静默，1=显示信息）
        """
        super(ImprovedTrainingCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps  # 总训练步数
        self.check_freq = check_freq  # 检查点保存频率
        self.save_path = save_path  # 模型保存路径
        self.training_visualizer = training_visualizer  # 训练可视化器引用
        self.server = server  # AlgorithmServer 实例
        self.vis_process = vis_process  # 独立可视化进程
        self.vis_log_path = vis_log_path  # 独立可视化日志路径
        self.overwrite_model = overwrite_model  # 是否覆盖模型
        self.model_name = model_name  # 模型名称
        self.best_mean_reward = -np.inf  # 最佳平均奖励（用于保存最佳模型）
        self.last_print_step = 0  # 上次打印的步数
        self.print_interval = max(
            total_timesteps // 10, 100
        )  # 打印间隔（总共显示10次）
        self.episode_count = 0  # 已完成的Episode数量
        self.episode_rewards = []  # 所有Episode的奖励列表
        self.last_vis_check_step = 0  # 上次检查可视化进程的步数

        # 确保保存目录存在
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        """
        每个训练步骤调用一次

        功能：
            - 检查训练是否被中断（Ctrl+C）
            - 检查独立可视化进程状态
            - 检测新完成的Episode并显示详细信息
            - 更新训练可视化统计
            - 自动保存最佳模型和检查点

        返回：
            bool: True继续训练，False停止训练
        """
        try:
            # ========== 检查中断标志 ==========
            global training_interrupted
            if training_interrupted:
                print("\n[中断] 停止训练...")
                return False  # 返回False停止训练
            # ===================================

            # ========== 定期检查独立可视化进程状态 ==========
            # 每1000步检查一次可视化进程是否崩溃
            if self.vis_process and (self.num_timesteps - self.last_vis_check_step >= 1000):
                rc = self.vis_process.poll()
                if rc is not None:
                    print(f"\n{'!' * 60}")
                    print(f"[警告] 独立可视化进程已退出 (exit code: {rc})")
                    print(f"[日志] 请检查: {self.vis_log_path}")
                    print(f"{'!' * 60}\n")
                    # 不停止训练，只记录警告
                self.last_vis_check_step = self.num_timesteps
            # ============================================

            # [*] 每步都更新可视化（用于采集权重）
            if self.training_visualizer:
                try:
                    current_reward = 0.0
                    self.training_visualizer.update_training_stats(
                        current_step_reward=current_reward
                    )
                except Exception as e:
                    print(f"[警告] 更新训练可视化失败: {e}")

            # ========== Episode完成检测 ==========
            # 检查是否有新的Episode完成（通过比较ep_info_buffer长度）
            if (
                len(self.model.ep_info_buffer) > 0
                and len(self.model.ep_info_buffer) > self.episode_count
            ):
                # 获取最新完成的Episode信息
                ep_reward = self.model.ep_info_buffer[-1]["r"]  # Episode总奖励
                ep_length = self.model.ep_info_buffer[-1]["l"]  # Episode步数
                self.episode_rewards.append(ep_reward)
                self.episode_count = len(self.model.ep_info_buffer)

                # 更新训练可视化（如果启用）
                if self.training_visualizer:
                    try:
                        self.training_visualizer.update_training_stats(
                            episode_reward=ep_reward,
                            episode_length=ep_length,
                            is_episode_done=True,
                        )
                    except Exception as e:
                        print(f"[警告] 更新Episode统计到可视化失败: {e}")

                # 通知 DataCollector 记录训练数据 (仅更新全局统计，Episode 切换由 Env 触发)
                if hasattr(self, "server") and hasattr(self.server, "data_collector"):
                    try:
                        self.server.data_collector.set_external_data(
                            "global_step", self.num_timesteps
                        )
                        self.server.data_collector.set_external_data(
                            "global_reward", sum(self.episode_rewards)
                        )
                    except Exception as e:
                        print(f"[警告] 更新DataCollector失败: {e}")

                # ========== 美观的Episode完成信息显示 ==========
                print(f"\n{'╔' + '═' * 58 + '╗'}")
                print(
                    f"║  [*] Episode #{self.episode_count} 完成！{' ' * (45 - len(str(self.episode_count)))}║"
                )
                print(f"{'╠' + '═' * 58 + '╣'}")
                print(f"║  [^] 本次奖励: {ep_reward:+8.2f}{' ' * 40}║")
                print(f"║  [|] Episode长度: {ep_length:4.0f} 步{' ' * 36}║")

                # 显示统计信息（需要至少2个Episode）
                if len(self.episode_rewards) > 1:
                    avg_reward = np.mean(self.episode_rewards)  # 平均奖励
                    best_reward = max(self.episode_rewards)  # 最佳奖励
                    worst_reward = min(self.episode_rewards)  # 最差奖励
                    print(f"║{' ' * 58}║")
                    print(f"║  [#] 统计信息:{' ' * 43}║")
                    print(f"║    • 平均奖励: {avg_reward:+8.2f}{' ' * 35}║")
                    print(f"║    • 最佳奖励: {best_reward:+8.2f}{' ' * 35}║")
                    print(f"║    • 最差奖励: {worst_reward:+8.2f}{' ' * 35}║")

                    # 奖励趋势分析（需要至少3个Episode）
                    if len(self.episode_rewards) >= 3:
                        recent_avg = np.mean(
                            self.episode_rewards[-3:]
                        )  # 最近3个Episode的平均
                        trend = "[^] 上升" if recent_avg > avg_reward else "[v] 下降"
                        print(f"║    • 最近趋势: {trend}{' ' * 35}║")

                # 显示训练进度
                print(f"║{' ' * 58}║")
                remaining_steps = self.total_timesteps - self.num_timesteps
                progress = self.num_timesteps / self.total_timesteps * 100
                print(
                    f"║  [@] 训练进度: {self.num_timesteps}/{self.total_timesteps} ({progress:.1f}%){' ' * (24 - len(str(self.total_timesteps)) * 2 - len(f'{progress:.1f}'))}║"
                )
                print(
                    f"║  [T] 剩余步数: {remaining_steps}{' ' * (43 - len(str(remaining_steps)))}║"
                )
                print(f"{'╚' + '═' * 58 + '╝'}\n")

                # 如果训练还没结束，提示即将开始下一个Episode
                if self.num_timesteps < self.total_timesteps:
                    print(f"{'─' * 60}")
                    print(f"[R] 准备下一个Episode（#{self.episode_count + 1}）...")
                    print(f"   环境将自动重置...")
                    print(f"{'─' * 60}\n")
            # ============================================

            # ========== 定期打印和保存最佳模型 ==========
            # 减少打印频率，避免阻塞训练（总共显示10次）
            if self.num_timesteps - self.last_print_step >= self.print_interval:
                # 计算当前平均奖励
                if len(self.model.ep_info_buffer) > 0:
                    mean_reward = np.mean(
                        [ep_info["r"] for ep_info in self.model.ep_info_buffer]
                    )
                else:
                    mean_reward = 0

                # 如果当前平均奖励超过历史最佳，保存最佳模型
                if mean_reward > self.best_mean_reward and mean_reward > 0:
                    self.best_mean_reward = mean_reward

                    # 根据 overwrite_model 决定文件名
                    if self.overwrite_model:
                        # 覆盖模式：使用固定名称
                        model_path = os.path.join(self.save_path, f"best_{self.model_name}")
                    else:
                        # 生成新模型：添加时间戳
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        model_path = os.path.join(self.save_path, f"best_model_{timestamp}")

                    try:
                        self.model.save(model_path)
                        print(f"\n[*] 新最佳模型！奖励: {mean_reward:.2f}")
                        print(f"[S] 已保存: {model_path}.zip\n")
                    except Exception as e:
                        print(f"[错误] 保存最佳模型失败: {e}")

                self.last_print_step = self.num_timesteps
            # ============================================

            # ========== 定期保存检查点 ==========
            # 每check_freq步保存一次检查点，防止训练中断丢失进度
            if self.num_timesteps % self.check_freq == 0 and self.num_timesteps > 0:
                # 根据 overwrite_model 决定文件名
                if self.overwrite_model:
                    # 覆盖模式：使用固定名称
                    checkpoint_path = os.path.join(
                        self.save_path, f"checkpoint_{self.model_name}"
                    )
                    print(f"[S] 检查点: checkpoint_{self.model_name}.zip (覆盖)")
                else:
                    # 生成新模型：添加时间戳
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    checkpoint_path = os.path.join(
                        self.save_path, f"checkpoint_{self.num_timesteps}_{timestamp}"
                    )
                    print(f"[S] 检查点: checkpoint_{self.num_timesteps}_{timestamp}.zip")

                try:
                    self.model.save(checkpoint_path)
                except Exception as e:
                    print(f"[错误] 保存检查点失败: {e}")
            # ====================================

            return True  # 继续训练

        except Exception as e:
            # 捕获_on_step中的所有未处理异常
            print(f"\n{'=' * 60}")
            print(f"[严重错误] 训练回调中发生异常: {str(e)}")
            print(f"[步数] {self.num_timesteps}/{self.total_timesteps}")
            print(f"{'=' * 60}")
            import traceback
            traceback.print_exc()
            # 返回False停止训练，避免继续运行在错误状态下
            return False


# ==================== 训练参数默认配置 ====================
# 这些是训练参数的默认值，可以通过命令行参数或配置文件覆盖
DEFAULT_DRONE_NAMES = ["UAV1", "UAV2", "UAV3"]  # 默认使用3台无人机协同训练
DEFAULT_TOTAL_TIMESTEPS = 10000  # 默认总训练步数（完整训练）
DEFAULT_STEP_DURATION = 5.0  # 默认每步飞行时长（秒），与实体训练对齐
DEFAULT_CHECKPOINT_FREQ = 1000  # 默认检查点保存频率（每N步保存一次）
DEFAULT_ENABLE_VISUALIZATION = True  # 默认启用训练可视化
DEFAULT_INITIAL_MODEL_PATH = None
DEFAULT_USE_INITIAL_WEIGHTS = True  # 默认使用初始权重继承
DEFAULT_OVERWRITE_MODEL = False  # 默认不覆盖模型，生成新模型（带时间戳）
# =====================================================


def main():
    """
    主训练流程函数

    功能：
        1. 解析命令行参数和配置文件
        2. 初始化AlgorithmServer（连接Unity AirSim）
        3. 创建训练环境（SimpleWeightEnv）
        4. 启动训练可视化（可选）
        5. 创建并训练DDPG模型
        6. 保存训练结果和模型

    训练流程：
        [1/5] 启动AlgorithmServer
        [2/5] 启动无人机任务
        [3/5] 等待系统稳定
        [4/5] 创建训练环境
        [4.5/5] 启动训练可视化（可选）
        [5/5] 创建DDPG模型并开始训练

    异常处理：
        - KeyboardInterrupt: 用户中断（Ctrl+C），优雅停止
        - Exception: 其他错误，显示错误信息并清理资源
    """
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(description="AirSim权重训练（改进版）")
    parser.add_argument(
        "--config", type=str, default=None, help="训练配置文件路径（JSON）"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="总训练步数（覆盖配置文件和默认值）",
    )
    parser.add_argument(
        "--initial-model-path",
        type=str,
        default=DEFAULT_INITIAL_MODEL_PATH,
        help="初始模型路径（不含.zip），用于自动匹配同名权重文件",
    )
    parser.add_argument(
        "--use-initial-weights",
        action="store_true",
        default=None,
        help="启用初始权重继承",
    )
    parser.add_argument(
        "--no-initial-weights",
        action="store_true",
        default=None,
        help="禁用初始权重继承",
    )
    parser.add_argument(
        "--overwrite-model",
        action="store_true",
        default=None,
        help="覆盖现有模型（不生成新时间戳），用于未改变算法时的调试训练",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="指定模型名称（不含.zip），配合--overwrite-model使用。默认为weight_predictor_airsim",
    )
    args = parser.parse_args()

    # ========== 加载配置并解析参数 ==========
    # 优先级：命令行参数 > 配置文件 > 默认值
    config = _load_train_config(args.config)  # 加载JSON配置文件

    # 解析训练参数
    drone_names = _get_config_value(None, config, "drone_names", DEFAULT_DRONE_NAMES)
    total_timesteps = int(
        _get_config_value(args.total_timesteps, config, "total_timesteps", DEFAULT_TOTAL_TIMESTEPS)
    )
    step_duration = float(
        _get_config_value(None, config, "step_duration", DEFAULT_STEP_DURATION)
    )
    checkpoint_freq = int(
        _get_config_value(None, config, "checkpoint_freq", DEFAULT_CHECKPOINT_FREQ)
    )
    enable_visualization = bool(
        _get_config_value(
            None, config, "enable_visualization", DEFAULT_ENABLE_VISUALIZATION
        )
    )
    safety_limit = bool(
        _get_config_value(None, config, "safety_limit", True)
    )  # 权重变化安全限制
    max_weight_delta = float(
        _get_config_value(None, config, "max_weight_delta", 0.5)
    )  # 权重变化最大幅度

    # 模型覆盖逻辑：默认开启覆盖模式以满足用户需求
    overwrite_model = True

    # 允许命令行覆盖此默认值
    if args.overwrite_model is not None:
        overwrite_model = args.overwrite_model
    elif "overwrite_model" in config:
        overwrite_model = config["overwrite_model"]

    # 模型名称
    model_name = _get_config_value(
        args.model_name,
        config,
        "model_name",
        "weight_predictor_airsim",  # 默认模型名
    )

    # 初始权重使用逻辑：命令行优先
    if args.use_initial_weights is None and args.no_initial_weights is None:
        use_initial_weights = bool(
            _get_config_value(
                None, config, "use_initial_weights", DEFAULT_USE_INITIAL_WEIGHTS
            )
        )
    else:
        use_initial_weights = bool(args.use_initial_weights) and not bool(
            args.no_initial_weights
        )

    initial_model_path = _get_config_value(
        args.initial_model_path,
        config,
        "initial_model_path",
        DEFAULT_INITIAL_MODEL_PATH,
    )
    # 注意：initial_weights_path 将在加载时根据 initial_model_path 自动推导
    # ==========================================

    # ========== 初始化全局变量（用于资源清理） ==========
    server = None  # AlgorithmServer实例
    training_visualizer = None  # 训练可视化器实例
    # ====================================================

    print("\n" + "=" * 60)
    print("[>] DQN权重训练 - 多无人机协同模式")
    print("=" * 60)
    print(f"[Drone] 无人机数量: {len(drone_names)} 台 ({', '.join(drone_names)})")
    print(f"[#] 训练步数: {total_timesteps} 步")
    print(f"[Clock]  每步时长: {step_duration} 秒")
    print(f"[S] 检查点: 每 {checkpoint_freq} 步保存一次")
    print(f"[Eye]  可视化: {'启用' if enable_visualization else '禁用'}")
    print(
        f"[S] 模型策略: {'覆盖模式 (' + model_name + ')' if overwrite_model else '生成新模型（带时间戳）'}"
    )
    print(f"[^] 预计episode数: ~{total_timesteps // 50}")
    print("=" * 60)
    print(f"\n[Light] 说明: 使用{len(drone_names)}台无人机协同训练")
    print(f"   - 主训练无人机: {drone_names[0]} (用于DQN学习)")
    print(
        f"   - 协同无人机: {', '.join(drone_names[1:]) if len(drone_names) > 1 else '无'} (提供环境交互)"
    )
    print(f"   - 学到的权重策略将适用于所有无人机")
    print("\n[重要] 请确保Unity AirSim仿真已经运行！")

    confirm = input("Unity已运行？(Y/N): ").strip().upper()
    if confirm != "Y":
        print("请先启动Unity")
        return

    try:
        # ========== [1/5] 启动AlgorithmServer ==========
        print("\n[1/5] 启动AlgorithmServer...")

        # 创建算法服务器（负责与Unity AirSim通信）
        # 训练模式配置：
        #   - use_learned_weights=False: 训练时不使用已学习的权重，让DDPG动态调整
        #   - model_path=None: 训练模式不需要加载预训练模型
        #   - enable_visualization=False: 禁用AlgorithmServer自带的可视化，使用训练专用可视化
        #   - enable_data_collection_print=True: 训练模式下启用数据采集DEBUG打印，便于监控训练过程
        server = MultiDroneAlgorithmServer(
            drone_names=drone_names,
            use_learned_weights=False,  # 训练模式：不使用学习的权重
            model_path=None,  # 训练模式：不加载模型
            enable_visualization=False,  # 使用训练专用可视化，禁用服务器自带可视化
            enable_data_collection_print=True,  # 训练模式：启用数据采集DEBUG打印
        )

        print(f"[OK] 服务器创建成功")
        print(f"  无人机配置: {', '.join(drone_names)}")
        print(f"  使用训练专用可视化: {'是' if enable_visualization else '否'}")

        # 启动服务器
        if not server.start():
            print("[错误] AlgorithmServer启动失败")
            return

        print("[OK] AlgorithmServer已连接")

        # 启动无人机和算法线程
        print("\n[2/5] 启动无人机任务...")
        print("[重要] 训练模式：启动算法线程，训练环境动态改变权重")

        # 调用start_mission()启动完整流程
        if not server.start_mission():
            print("[错误] 任务启动失败")
            server.stop()
            return

        print("[OK] 无人机已起飞，算法线程运行中")

        # 启动独立进程可视化（默认启用，可通过环境变量 NO_VIS=1 禁用）
        ipc_server = None
        vis_process = None
        vis_log_f = None
        vis_log_path = None
        _tmp_vis_log_dir = os.path.join(
            os.path.dirname(__file__), "logs", "ddpg_airsim"
        )
        os.makedirs(_tmp_vis_log_dir, exist_ok=True)

        if HAS_EXT_VIS and os.environ.get("NO_VIS", "0") != "1":
            try:
                ipc_server = VisualizationIPCServer(
                    snapshot_provider=server.get_visualization_snapshot,
                    host="127.0.0.1",
                    port=0,
                    hz=10.0,
                    compress_level=1,
                )
                ipc_server.start()
                port = ipc_server.bound_port

                vis_log_path = os.path.join(_tmp_vis_log_dir, "external_vis.log")
                vis_log_f = open(vis_log_path, "w", encoding="utf-8")

                python_exe = sys.executable
                vis_entry = os.path.join(
                    os.path.dirname(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    ),
                    "multirotor",
                    "Visualization",
                    "external_visualizer_client.py",
                )
                vis_cmd = [
                    python_exe,
                    vis_entry,
                    "--mode",
                    "ddpg",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ]

                vis_env = os.environ.copy()
                vis_env["PYTHONIOENCODING"] = "utf-8"
                vis_env["PYTHONUTF8"] = "1"

                creationflags = 0
                if os.name == "nt" and os.environ.get("VIS_NEW_CONSOLE", "0") == "1":
                    creationflags = subprocess.CREATE_NEW_CONSOLE

                vis_process = subprocess.Popen(
                    vis_cmd,
                    stdout=vis_log_f,
                    stderr=vis_log_f,
                    creationflags=creationflags,
                    env=vis_env,
                )

                time.sleep(0.5)
                rc = vis_process.poll()
                if rc is not None:
                    print(f"! 独立可视化进程启动后立即退出 (returncode={rc})")
                    print(f"  - 请查看: {vis_log_path}")
                else:
                    print(f"[OK] 已启动独立可视化进程 (port={port})")
                    print(f"  - 外部可视化日志: {vis_log_path}")
            except Exception as e:
                print(f"! 启动独立可视化失败: {e}")
                if vis_log_path:
                    print(f"  - 外部可视化日志(若已生成): {vis_log_path}")
                try:
                    if ipc_server:
                        ipc_server.stop()
                except Exception:
                    pass
                ipc_server = None
                vis_process = None

        # [2.5] 设置实验元数据 (用于跨方案数据对比)
        if hasattr(server, "set_experiment_meta"):
            server.set_experiment_meta(
                algorithm_type="ddpg_apf", env_type="weight", control_mode="apf"
            )

        # 等待系统稳定
        print("\n[3/5] 等待系统稳定...")
        time.sleep(5)

        # 加载初始权重（若存在）
        initial_weights = {}
        if use_initial_weights:
            if not initial_model_path:
                print("[!]  未指定初始模型路径，跳过初始权重加载")
            else:
                # 自动查找同名权重文件
                initial_weights_path = _derive_weights_path(initial_model_path)
                if os.path.exists(initial_weights_path):
                    print(f"[Folder] 找到权重文件: {initial_weights_path}")
                    initial_weights = _load_initial_weights(initial_weights_path)
                else:
                    print(f"[!]  权重文件不存在: {initial_weights_path}")
                    print(f"   模型路径: {initial_model_path}")
                    print(f"   将使用默认配置权重")

            if initial_weights:
                for drone_name in drone_names:
                    weights = initial_weights.get(drone_name) or initial_weights.get(
                        "__all__"
                    )
                    if weights:
                        server.algorithms[drone_name].set_coefficients(weights)
                print(f"[OK] 已加载初始权重: {initial_weights_path}")
            else:
                print("[!]  未找到可用初始权重，使用默认配置权重")

        # ========== [4/5] 创建训练环境 ==========
        print("\n[4/5] 创建训练环境...")

        # 创建SimpleWeightEnv训练环境
        # 环境功能：
        #   - 将DDPG的动作（权重系数）应用到APF算法
        #   - 执行一步飞行并收集状态和奖励
        #   - 支持episode重置（reset_unity=True）
        env = SimpleWeightEnv(
            server=server,  # 算法服务器引用
            drone_name=drone_names[0],  # 使用第一台无人机进行DDPG训练（主训练机）
            reset_unity=True,  # 每个episode结束时重置Unity环境
            step_duration=step_duration,  # 每步飞行时长（秒）
            safety_limit=safety_limit,  # 是否启用权重变化安全限制
            max_weight_delta=max_weight_delta,  # 权重变化最大幅度（安全限制）
        )
        if use_initial_weights and initial_weights:
            training_weights = initial_weights.get(
                drone_names[0]
            ) or initial_weights.get("__all__")
            if training_weights:
                env.set_initial_action(_weights_to_action(training_weights))
        print(f"[OK] 环境创建成功")
        print(f"  [Mode] 模式: 多无人机协同训练")
        print(f"  [Train] 训练无人机: {drone_names[0]}")
        print(
            f"  [Hand] 协同无人机: {', '.join(drone_names[1:]) if len(drone_names) > 1 else '无'}"
        )
        print(f"  [Clock]  每步时长: {step_duration}秒")
        print(
            f"  [@] 每个episode: {env.reward_config.max_steps}步 = {env.reward_config.max_steps * step_duration / 60:.1f}分钟"
        )
        print(f"  [Light] 预计总训练时长: {total_timesteps * step_duration / 60:.1f}分钟")

        # 训练可视化（已迁移到独立进程 external_visualizer_client.py，避免 pygame 阻塞训练）
        if enable_visualization:
            print("\n[4.5/5] 训练可视化: 已使用独立进程模式启动")
            if vis_log_path:
                print(f"  - 外部可视化日志: {vis_log_path}")
            else:
                print("  - 外部可视化日志: (未生成)")
            # [!] 重要：仍然需要在主进程中创建训练可视化器用于收集统计数据（权重历史等）
            # 外部进程负责显示，主进程中的visualizer负责数据收集
            from multirotor.Visualization.ddpg_training_visualizer import (
                DDPGTrainingVisualizer,
            )

            training_visualizer = DDPGTrainingVisualizer(server=server, env=None)
            print("  - 主进程训练数据收集器: 已创建 (用于权重历史统计)")

        # ========== [5/5] 创建或加载 DDPG 模型 ==========
        print("\n[5/5] 获取 DDPG 模型...")

        # 确定模型路径
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(model_dir, exist_ok=True)
        fixed_model_path = os.path.join(model_dir, f"{model_name}.zip")

        # 检查是否存在旧模型
        reset_num_timesteps = True
        if os.path.exists(fixed_model_path):
            print(f"[R] 发现现有模型: {fixed_model_path}")
            print(f"   正在从旧模型加载神经网络参数进行增量训练...")
            try:
                # 加载旧模型，去掉 .zip 后缀
                model = DDPG.load(fixed_model_path[:-4], env=env)
                reset_num_timesteps = False
                print("[OK] 旧模型加载成功，将继续训练")
            except Exception as e:
                print(f"[!]  加载旧模型失败: {e}")
                print("🆕 将创建新的随机初始化模型")
                model = None
        else:
            print(f"🆕 未发现现有模型 ({model_name}.zip)，将创建新模型")
            model = None

        if model is None:
            # 获取动作空间维度（5个APF权重系数）
            n_actions = env.action_space.shape[0]

            # 创建动作噪声（用于探索）
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions),  # 噪声均值为0
                sigma=0.15 * np.ones(n_actions),  # 适度噪声，平衡探索与利用
            )

            # 创建DDPG模型
            model = DDPG(
                "MlpPolicy",  # 使用多层感知机（MLP）策略网络
                env,  # 训练环境
                action_noise=action_noise,  # 动作噪声（探索）
                learning_rate=1e-4,  # 学习率（较小值，稳定训练）
                buffer_size=5000,  # 经验回放缓冲区大小（小缓冲区，快速训练）
                learning_starts=200,  # 开始学习前的步数（收集经验）
                batch_size=64,  # 批次大小（每次训练使用的样本数）
                tau=0.005,  # 软更新系数（目标网络更新速度）
                gamma=0.99,  # 折扣因子（未来奖励的重要性）
                train_freq=(1, "episode"),  # 训练频率（每个episode训练一次）
                gradient_steps=-1,  # 梯度步数（-1表示使用所有可用数据）
                verbose=0,  # 详细程度（0=静默）
                device="cpu",  # 使用CPU（可改为'cuda'使用GPU）
            )
            print("[OK] DDPG模型初始化成功")

        # 开始训练
        print("\n" + "=" * 60)
        print("[@] 开始训练")
        print("=" * 60)
        print(f"[#] 训练步数: {total_timesteps}")
        print(f"[R] 增量训练: {'是' if not reset_num_timesteps else '否'}")
        print(f"[Pause]  按 Ctrl+C 可随时停止")
        print("=" * 60 + "\n")

        training_callback = ImprovedTrainingCallback(
            total_timesteps=total_timesteps,
            check_freq=checkpoint_freq,
            save_path=model_dir,
            training_visualizer=training_visualizer,  # 传入可视化器
            server=server,  # 传入 server 实例，用于访问 DataCollector
            vis_process=vis_process,  # 传入独立可视化进程，用于监控状态
            vis_log_path=vis_log_path,  # 传入可视化日志路径，用于诊断
            overwrite_model=overwrite_model,  # 传入覆盖模式标志
            model_name=model_name,  # 传入模型名称
            verbose=1,
        )

        model.learn(
            total_timesteps=total_timesteps,
            log_interval=None,
            callback=training_callback,
            reset_num_timesteps=reset_num_timesteps,
        )

        print("\n" + "=" * 60)
        print("[OK] 训练完成！")
        print("=" * 60)

        # 保存最终模型
        print("\n[S] 保存最终模型...")

        # 根据 overwrite_model 参数决定模型文件名
        if overwrite_model:
            # 覆盖模式：使用固定名称，不添加时间戳
            final_model_path = os.path.join(model_dir, model_name)
            print(f"[!]  覆盖模式：将覆盖现有模型 {model_name}")
        else:
            # 生成新模型：添加时间戳
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            final_model_path = os.path.join(model_dir, f"{model_name}_{timestamp}")
            print(f"[New] 生成新模型：使用时间戳 {timestamp}")

        model.save(final_model_path)
        print(f"[OK] 模型已保存: {final_model_path}.zip")

        # 保存最后权重系数（与模型同名）
        weights_path = _derive_weights_path(final_model_path)
        _save_final_weights(server, weights_path)

        # 显示训练统计
        print("\n" + "=" * 60)
        print("[训练统计]")
        print("=" * 60)
        if (
            hasattr(training_callback, "episode_rewards")
            and training_callback.episode_rewards
        ):
            print(f"完成episode数: {len(training_callback.episode_rewards)}")
            print(f"总奖励: {sum(training_callback.episode_rewards):.2f}")
            print(f"平均奖励: {np.mean(training_callback.episode_rewards):.2f}")
            print(f"最佳奖励: {max(training_callback.episode_rewards):.2f}")
            print(f"最差奖励: {min(training_callback.episode_rewards):.2f}")
        print("=" * 60)

        print("\n[生成的模型文件]:")
        if overwrite_model:
            print(f"  [最佳] 最佳模型: models/best_{model_name}.zip (覆盖模式)")
            print(f"  [文件] 最终模型: models/{model_name}.zip (覆盖模式)")
            if checkpoint_freq > 0:
                print(f"  [检查点] 检查点: models/checkpoint_{model_name}.zip (覆盖模式)")
        else:
            print(f"  [最佳] 最佳模型: models/best_model_*.zip")
            print(f"  [文件] 最终模型: models/{model_name}_<timestamp>.zip")
            if checkpoint_freq > 0:
                print(f"  [检查点] 检查点: models/checkpoint_*.zip")

        print("\n[下一步操作]:")
        print("  [1] 测试模型: python test_trained_model.py")
        print("  [2] 使用模型: python ../AlgorithmServer.py --use-learned-weights")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("[中断] 正在停止训练...")
        print("=" * 60)
        print("\n请稍候，正在清理资源...")

    except Exception as e:
        print(f"\n\n[错误] 训练出错: {str(e)}")
        import traceback

        traceback.print_exc()

    finally:
        # 确保清理资源

        # 停止可视化
        if training_visualizer:
            print("\n停止训练可视化...")
            try:
                training_visualizer.stop_visualization()
                print("[OK] 训练可视化已停止")
            except Exception as e:
                print(f"[警告] 停止可视化时出错: {e}")

        if server:
            print("\n停止AlgorithmServer...")
            try:
                # 先停止所有线程和服务（包括数据采集线程、算法线程）
                print("  停止数据采集线程...")
                server.data_collector.stop()

                print("  停止算法线程...")
                server.running = False  # 设置运行标志为False，停止所有算法线程

                # 等待算法线程结束（使用已导入的time模块）
                import time as time_module  # 使用别名避免变量冲突

                time_module.sleep(1)  # 等待1秒让线程正常退出

                # 降落无人机
                for drone_name in drone_names:
                    try:
                        print(f"  降落 {drone_name}...")
                        server.drone_controller.land(drone_name)
                    except:
                        pass

                # 停止Unity通信
                print("  断开Unity连接...")
                server.unity_socket.stop()

                print("[OK] AlgorithmServer已完全停止")
            except Exception as e:
                print(f"[警告] 清理资源时出现错误: {e}")

        print("\n训练已结束")
        print("按Enter键退出...")
        try:
            input()
        except:
            pass


if __name__ == "__main__":
    main()
