"""
基于Crazyflie日志的离线训练脚本（动作不影响状态转移）
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from crazyflie_weight_env import CrazyflieLogEnv


def _load_train_config(path: str) -> dict:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("配置文件必须为JSON对象")
    return data


def _get_config_value(cli_value, config: dict, key: str, default):
    if cli_value is not None:
        return cli_value
    if key in config:
        return config[key]
    return default


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _save_model(model, path: str, logger, note: str) -> bool:
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("crazyflie_train_logs")

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

    config = _load_train_config(args.config)

    log_path = _get_config_value(args.log_path, config, "log_path", None)
    if not log_path:
        raise ValueError("必须提供日志路径：--log-path 或配置文件中的 log_path")

    total_timesteps = _get_config_value(args.total_timesteps, config, "total_timesteps", 2000)
    reward_config = _get_config_value(args.reward_config, config, "reward_config", None)
    save_dir = _get_config_value(args.save_dir, config, "save_dir", "models")
    continue_model = _get_config_value(args.continue_model, config, "continue_model", None)
    max_steps = _get_config_value(args.max_steps, config, "max_steps", None)
    random_start = _get_config_value(args.random_start, config, "random_start", False)
    step_stride = _get_config_value(args.step_stride, config, "step_stride", 1)
    progress_interval = _get_config_value(args.progress_interval, config, "progress_interval", 50)

    try:
        from stable_baselines3 import DDPG
        from stable_baselines3.common.noise import NormalActionNoise
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError:
        logger.error("缺少stable-baselines3，请先安装")
        sys.exit(1)

    class TrainingProgressCallback(BaseCallback):
        def __init__(
            self,
            total_timesteps: int,
            print_interval_steps: int = 50,
            print_interval_sec: int = 10,
        ):
            super().__init__()
            self.total_timesteps = max(int(total_timesteps), 0)
            self.print_interval_steps = max(int(print_interval_steps), 1)
            self.print_interval_sec = max(int(print_interval_sec), 1)
            self.start_time = 0.0
            self.last_print_time = 0.0
            self.last_print_step = 0

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

    env = CrazyflieLogEnv(
        log_path=log_path,
        reward_config_path=reward_config,
        max_steps=max_steps,
        random_start=random_start,
        step_stride=step_stride
    )

    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

    os.makedirs(save_dir, exist_ok=True)

    model = None
    model_saved = False
    final_path = os.path.join(save_dir, "weight_predictor_crazyflie_logs")
    logger.info("模型保存路径: %s.zip", os.path.abspath(final_path))

    if continue_model:
        logger.info("继续训练: 加载模型 %s.zip", continue_model)
        model = DDPG.load(continue_model, env=env, print_system_info=True)
        reset_num_timesteps = False
    else:
        model = DDPG(
            "MlpPolicy",
            env,
            action_noise=action_noise,
            learning_rate=1e-4,
            buffer_size=50000,
            learning_starts=100,
            batch_size=64,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "episode"),
            gradient_steps=-1,
            verbose=1,
            device="cpu"
        )
        reset_num_timesteps = True

    progress_cb = TrainingProgressCallback(
        total_timesteps=total_timesteps,
        print_interval_steps=progress_interval,
        print_interval_sec=10
    )
    try:
        model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=reset_num_timesteps,
            callback=progress_cb
        )
        model_saved = _save_model(model, final_path, logger, "训练完成，模型已保存")
    except KeyboardInterrupt:
        logger.warning("训练停止，尝试保存当前模型")
        if not model_saved:
            model_saved = _save_model(model, final_path, logger, "中断保存，模型已保存")


if __name__ == "__main__":
    main()
