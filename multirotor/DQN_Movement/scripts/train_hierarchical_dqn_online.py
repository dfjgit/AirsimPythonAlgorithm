import argparse
import os
import sys
import time
import logging
import json
import numpy as np

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from multirotor.AlgorithmServer import MultiDroneAlgorithmServer
from multirotor.DQN_Movement.envs.crazyflie_hierarchical_env import CrazyflieHierarchicalEnv

# 尝试导入训练库，根据实际项目使用的库（假设是 stable_baselines3 或自定义）
try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import CheckpointCallback
except ImportError:
    logging.error("缺少 stable-baselines3，请先安装")
    sys.exit(1)

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("HierarchicalDQN_Online")

    parser = argparse.ArgumentParser(description="双层DQN实机在线训练")
    parser.add_argument("--drone-name", type=str, default="UAV1", help="无人机名称")
    parser.add_argument("--total-timesteps", type=int, default=1000, help="总训练步数")
    parser.add_argument("--save-dir", type=str, default="models/hrl_online", help="模型保存目录")
    parser.add_argument("--load-model", type=str, default=None, help="加载预训练模型路径")
    args = parser.parse_args()

    # 1. 启动 AlgorithmServer (控制模式必须为 apf)
    server = MultiDroneAlgorithmServer(
        drone_names=[args.drone_name],
        control_mode='apf',
        use_learned_weights=False,  # 训练模式下由 Env 动态设置
        enable_visualization=True
    )

    if not server.start():
        logger.error("AlgorithmServer 启动失败")
        return

    if not server.start_mission():
        logger.error("任务启动失败")
        server.stop()
        return

    try:
        # 2. 创建实机分层环境
        env = CrazyflieHierarchicalEnv(
            server=server,
            drone_name=args.drone_name,
            step_duration=2.0  # 与实机物理特性对齐的决策周期
        )

        # 3. 初始化或加载模型
        if args.load_model and os.path.exists(args.load_model):
            logger.info(f"加载模型: {args.load_model}")
            model = DQN.load(args.load_model, env=env)
        else:
            logger.info("创建新模型")
            model = DQN(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=1e-3,
                buffer_size=5000,
                learning_starts=100,
                batch_size=32,
                tau=1.0,
                gamma=0.99,
                train_freq=4,
                gradient_steps=1,
                target_update_interval=100
            )

        # 4. 开始训练
        logger.info(f"开始实机在线训练，目标步数: {args.total_timesteps}")
        checkpoint_callback = CheckpointCallback(save_freq=500, save_path=args.save_dir, name_prefix="hrl_dqn")
        
        model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback)

        # 5. 保存最终模型
        final_path = os.path.join(args.save_dir, "hrl_dqn_final")
        model.save(final_path)
        logger.info(f"训练完成，模型已保存至: {final_path}")

    except KeyboardInterrupt:
        logger.warning("训练被人为中断")
    except Exception as e:
        logger.error(f"训练过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        server.stop()
        logger.info("系统已安全停止")

if __name__ == "__main__":
    main()
