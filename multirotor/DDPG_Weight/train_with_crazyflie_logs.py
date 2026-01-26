"""
基于Crazyflie日志的离线训练脚本（动作不影响状态转移）

功能说明：
    - 使用历史飞行日志进行离线训练，无需实体无人机
    - 从日志文件中读取状态序列，动作不影响状态转移（仅用于奖励计算）
    - 适合快速迭代和实验，无需担心实体设备安全
    - 支持从随机位置开始episode，增加训练数据多样性

主要特性：
    - 离线训练：不需要实体无人机，使用历史日志数据
    - 快速迭代：可以快速测试不同的超参数和算法配置
    - 数据重用：可以多次使用同一份日志数据进行训练
    - 灵活配置：支持随机起始位置、步进间隔等参数

训练环境：
    - 环境类型：CrazyflieLogEnv（离线日志环境）
    - 算法：DDPG（Deep Deterministic Policy Gradient）
    - 动作空间：5维连续空间（APF权重系数）
    - 状态来源：从日志文件中读取的历史状态序列

使用方法：
    python train_with_crazyflie_logs.py --log-path logs/flight.json --total-timesteps 2000
    python train_with_crazyflie_logs.py --config config.json

注意事项：
    - 日志文件格式：支持JSON和CSV格式
    - 动作不影响状态：环境状态完全由日志决定，动作仅用于计算奖励
    - 适合场景：算法开发、超参数调优、快速实验

作者：训练模块开发团队
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

# 导入离线日志训练环境
from crazyflie_weight_env import CrazyflieLogEnv  # 基于日志的离线训练环境


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
    
    # 检查是否为统一配置格式（包含 common 和 crazyflie_logs 键）
    if "common" in data and "crazyflie_logs" in data:
        # 统一配置格式：合并 common 和 crazyflie_logs 配置
        merged_config = {}
        merged_config.update(data.get("common", {}))
        merged_config.update(data.get("crazyflie_logs", {}))
        return merged_config
    else:
        # 传统配置格式：直接返回
        return data


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


def main():
    """
    主训练流程函数
    
    功能：
        1. 解析命令行参数和配置文件
        2. 加载历史飞行日志数据
        3. 创建离线训练环境（CrazyflieLogEnv）
        4. 创建并训练DDPG模型
        5. 保存训练结果和模型
        
    训练流程：
        1. 加载配置和参数
        2. 加载历史日志文件
        3. 创建离线训练环境
        4. 创建DDPG模型
        5. 开始训练
        
    离线训练特点：
        - 不需要实体无人机，使用历史日志数据
        - 动作不影响状态转移（状态完全由日志决定）
        - 适合快速迭代和实验
        - 可以多次使用同一份日志数据
        
    异常处理：
        - KeyboardInterrupt: 用户中断，尝试保存当前模型
        - Exception: 其他错误，显示错误信息并退出
    """
    # ========== 初始化日志系统 ==========
    logging.basicConfig(
        level=logging.INFO,  # 日志级别：INFO
        format="%(asctime)s - %(levelname)s - %(message)s"  # 日志格式：时间 - 级别 - 消息
    )
    logger = logging.getLogger("crazyflie_train_logs")
    # ====================================

    parser = argparse.ArgumentParser(description="Crazyflie离线日志训练")
    parser.add_argument("--config", type=str, default=None, help="训练配置文件路径（JSON）")
    parser.add_argument("--log-path", type=str, default=None, help="日志文件路径（.json/.csv）")
    parser.add_argument("--total-timesteps", type=int, default=None, help="训练步数")
    parser.add_argument("--reward-config", type=str, default=None, help="奖励配置路径")
    parser.add_argument("--save-dir", type=str, default=None, help="模型保存目录")
    parser.add_argument("--continue-model", type=str, default=None, help="继续训练模型路径（不含.zip）")
    parser.add_argument("--max-steps", type=int, default=None, help="每个episode最大步数")
    parser.add_argument("--random-start", action="store_true", default=None, help="从随机位置开始episode")
    parser.add_argument("--step-stride", type=int, default=None, help="日志步进间隔")
    parser.add_argument("--progress-interval", type=int, default=None, help="进度打印间隔（步）")
    args = parser.parse_args()

    # 读取配置文件（若未提供则用空配置，后续会回退到默认值）
    config = _load_train_config(args.config)

    # ========== 加载日志文件路径 ==========
    # 离线训练的数据来源：历史飞行日志文件
    # 支持格式：JSON (.json) 或 CSV (.csv)
    log_path = _get_config_value(args.log_path, config, "log_path", None)
    if not log_path:
        raise ValueError("必须提供日志路径：--log-path 或配置文件中的 log_path")
    # ======================================

    # 从命令行/配置中解析训练超参数
    # 规则：命令行优先，其次配置文件，最后默认值
    total_timesteps = _get_config_value(args.total_timesteps, config, "total_timesteps", 2000)
    reward_config = _get_config_value(args.reward_config, config, "reward_config", None)
    save_dir = _get_config_value(args.save_dir, config, "save_dir", "models")
    continue_model = _get_config_value(args.continue_model, config, "continue_model", None)
    max_steps = _get_config_value(args.max_steps, config, "max_steps", None)
    random_start = _get_config_value(args.random_start, config, "random_start", False)
    step_stride = _get_config_value(args.step_stride, config, "step_stride", 1)
    progress_interval = _get_config_value(args.progress_interval, config, "progress_interval", 50)

    # 训练依赖：SB3 的 DDPG 与回调机制
    try:
        from stable_baselines3 import DDPG
        from stable_baselines3.common.noise import NormalActionNoise
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError:
        logger.error("缺少stable-baselines3，请先安装")
        sys.exit(1)

    # ========== 训练进度回调类 ==========
    class TrainingProgressCallback(BaseCallback):
        """
        训练进度回调类
        
        功能：
            - 监控训练进度，定期打印进度信息（包含ETA）
            - 支持按步数或时间间隔打印
            
        继承自：
            stable_baselines3.common.callbacks.BaseCallback
        """
        def __init__(
            self,
            total_timesteps: int,
            print_interval_steps: int = 50,
            print_interval_sec: int = 10,
        ):
            """
            初始化训练进度回调
            
            参数：
                total_timesteps: 总训练步数
                print_interval_steps: 按步数打印的间隔（每N步打印一次）
                print_interval_sec: 按时间打印的间隔（每N秒打印一次）
            """
            super().__init__()
            self.total_timesteps = max(int(total_timesteps), 0)  # 总训练步数
            self.print_interval_steps = max(int(print_interval_steps), 1)  # 步数打印间隔
            self.print_interval_sec = max(int(print_interval_sec), 1)  # 时间打印间隔
            self.start_time = 0.0  # 训练开始时间
            self.last_print_time = 0.0  # 上次打印时间
            self.last_print_step = 0  # 上次打印的步数

        def _on_training_start(self) -> None:
            now = time.time()
            self.start_time = now
            self.last_print_time = now
            self.last_print_step = int(self.num_timesteps)
            self._print_progress(force=True)

        def _on_step(self) -> bool:
            num_timesteps = int(self.num_timesteps)
            now = time.time()
            need_by_steps = (num_timesteps - self.last_print_step) >= self.print_interval_steps
            need_by_time = (now - self.last_print_time) >= self.print_interval_sec
            if need_by_steps or need_by_time:
                self._print_progress()
            return True

        def _print_progress(self, force: bool = False) -> None:
            num_timesteps = int(self.num_timesteps)
            now = time.time()
            if not force and num_timesteps == self.last_print_step and (now - self.last_print_time) < 1.0:
                return
            self.last_print_step = num_timesteps
            self.last_print_time = now

            elapsed = now - self.start_time
            if self.total_timesteps > 0:
                progress = min(num_timesteps / self.total_timesteps, 1.0)
                eta = (elapsed / progress - elapsed) if progress > 0 else 0.0
                percent = progress * 100.0
                logger.info(
                    "进度 %s/%s (%.1f%%) 已用%s 预计剩余%s",
                    num_timesteps,
                    self.total_timesteps,
                    percent,
                    _format_duration(elapsed),
                    _format_duration(eta)
                )
            else:
                logger.info("进度 %s 步 已用%s", num_timesteps, _format_duration(elapsed))

    # 打印训练参数，便于复现实验
    logger.info(
        "训练参数: log=%s total=%s max_steps=%s random_start=%s stride=%s "
        "progress_interval=%s save_dir=%s continue_model=%s",
        log_path,
        total_timesteps,
        max_steps,
        random_start,
        step_stride,
        progress_interval,
        save_dir,
        continue_model
    )

    # ========== 创建离线训练环境 ==========
    # CrazyflieLogEnv: 基于历史日志的离线训练环境
    # 环境特点：
    #   - 状态序列完全由日志文件决定
    #   - 动作不影响状态转移（仅用于计算奖励）
    #   - 支持随机起始位置（增加训练数据多样性）
    #   - 支持步进间隔（可以跳过部分日志数据）
    env = CrazyflieLogEnv(
        log_path=log_path,  # 日志文件路径（JSON或CSV格式）
        reward_config_path=reward_config,  # 奖励配置文件路径
        max_steps=max_steps,  # 每个episode最大步数（None表示使用日志全部数据）
        random_start=random_start,  # 是否从随机位置开始episode
        step_stride=step_stride  # 步进间隔（1表示使用所有数据，2表示每隔一步）
    )
    # ======================================

    # ========== 创建动作噪声 ==========
    # 动作维度决定噪声向量长度（用于探索）
    n_actions = env.action_space.shape[0]  # 动作空间维度（5个APF权重系数）
    
    # NormalActionNoise: 高斯噪声，帮助算法探索动作空间
    # sigma=0.2: 噪声标准差，控制探索强度（离线训练可以使用稍大的噪声）
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),  # 噪声均值为0
        sigma=0.2 * np.ones(n_actions)  # 噪声标准差（离线训练可以更激进地探索）
    )
    # ==================================

    # 确保模型输出目录存在
    os.makedirs(save_dir, exist_ok=True)

    model = None
    model_saved = False
    final_path = os.path.join(save_dir, "weight_predictor_crazyflie_logs")
    logger.info("模型保存路径: %s.zip", os.path.abspath(final_path))

    # ========== 创建或加载DDPG模型 ==========
    if continue_model:
        # 继续训练：加载已有模型并保持步数累计
        logger.info("继续训练: 加载模型 %s.zip", continue_model)
        model = DDPG.load(continue_model, env=env, print_system_info=True)
        reset_num_timesteps = False  # 不重置步数，继续累计
    else:
        # 新训练：从头初始化 DDPG
        # DDPG (Deep Deterministic Policy Gradient): 适用于连续动作空间的强化学习算法
        model = DDPG(
            "MlpPolicy",  # 使用多层感知机（MLP）策略网络
            env,  # 训练环境
            action_noise=action_noise,  # 动作噪声（探索）
            learning_rate=1e-4,  # 学习率
            buffer_size=50000,  # 经验回放缓冲区大小（离线训练可以使用更大的缓冲区）
            learning_starts=100,  # 开始学习前的步数（收集经验）
            batch_size=64,  # 批次大小
            tau=0.005,  # 软更新系数（目标网络更新速度）
            gamma=0.99,  # 折扣因子（未来奖励的重要性）
            train_freq=(1, "episode"),  # 训练频率（每个episode训练一次）
            gradient_steps=-1,  # 梯度步数（-1表示使用所有可用数据）
            verbose=1,  # 详细程度（1=显示信息）
            device="cpu"  # 使用CPU（可改为'cuda'使用GPU）
        )
        reset_num_timesteps = True  # 重置步数，从头开始计数
    # ========================================

    # 进度回调：定期打印训练进度
    progress_cb = TrainingProgressCallback(
        total_timesteps=total_timesteps,
        print_interval_steps=progress_interval,
        print_interval_sec=10
    )
    try:
        # 训练主循环：达到 total_timesteps 视为训练完成
        model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=reset_num_timesteps,
            callback=progress_cb
        )
        # 正常结束后保存模型
        model_saved = _save_model(model, final_path, logger, "训练完成，模型已保存")
    except KeyboardInterrupt:
        # 人工中断时尝试保存当前模型
        logger.warning("训练停止，尝试保存当前模型")
        if not model_saved:
            model_saved = _save_model(model, final_path, logger, "中断保存，模型已保存")


if __name__ == "__main__":
    main()
