"""
分层强化学习 (HRL) 训练脚本 - 与 AirSim 集成
使用真实的 AirSim 环境训练高层协同规划器 (DQN) 和底层控制器
"""
import os
import sys
import numpy as np
import json
from datetime import datetime
import threading
import time
import argparse

# 添加项目路径
# scripts -> DQN_Movement -> multirotor -> 项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# 添加 multirotor 目录
multirotor_dir = os.path.join(project_root, 'multirotor')
sys.path.insert(0, multirotor_dir)

# 添加 DQN_Movement 目录
dqn_movement_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, dqn_movement_dir)

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# 导入环境和服务器
from envs.hierarchical_movement_env import HierarchicalMovementEnv, MultiDroneHierarchicalMovementEnv
from AlgorithmServer import MultiDroneAlgorithmServer
from Algorithm.drones_config import DronesConfig

# 导入可视化器
try:
    from visualizers.hierarchical_visualizer import HierarchicalVisualizer
    HAS_VISUALIZER = True
except ImportError:
    HAS_VISUALIZER = False
    print("警告: 无法导入HierarchicalVisualizer，可视化功能将被禁用")

class VisualizationCallback(BaseCallback):
    """训练回调，用于更新可视化数据"""
    
    def __init__(self, visualizer, verbose=0):
        super(VisualizationCallback, self).__init__(verbose)
        self.visualizer = visualizer
        self.episode_count = 0
        self.episode_reward = 0
    
    def _on_step(self) -> bool:
        # 获取当前信息
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            drone_name = info.get('drone_name', 'UAV1')
        else:
            drone_name = 'UAV1'
        
        action = self.locals.get('actions', [0])[0]
        reward = self.locals.get('rewards', [0])[0]
        
        # 更新可视化数据
        self.visualizer.update_training_data(
            step=self.num_timesteps,
            action=int(action),
            reward=float(reward),
            drone_name=drone_name
        )
        
        self.episode_reward += reward
        
        # 检查Episode是否结束
        dones = self.locals.get('dones', [False])
        if dones[0]:
            self.visualizer.on_episode_end(self.episode_count)
            self.episode_count += 1
            self.episode_reward = 0
        
        return True

def train_hrl_with_airsim(enable_visualization=True):
    print("=" * 80)
    print("分层强化学习 (HRL) 训练 - 与 AirSim 集成")
    print("=" * 80)

    # 1. 加载配置
    drones_config = DronesConfig()
    drone_names = drones_config.get_training_drones('hierarchical')
    if not drone_names:
        print(f"  ✗ 错误: 没有可用的训练无人机，请检查 drones_config.json 中的 hierarchical 配置")
        sys.exit(1)
    
    print(f"✓ 训练无人机: {drone_names}")

    hrl_config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "hierarchical_dqn_config.json")
    with open(hrl_config_path, 'r', encoding='utf-8') as f:
        hrl_config = json.load(f)

    config_file = os.path.join(os.path.dirname(__file__), "..", "scanner_config.json")

    # 2. 启动 AirSim 服务器（禁用SimpleVisualizer，使用HierarchicalVisualizer代替）
    print(f"正在启动服务器 (DQN控制模式)...")
    server = MultiDroneAlgorithmServer(
        config_file=config_file,
        drone_names=drone_names,
        use_learned_weights=False,
        control_mode='dqn',
        enable_visualization=False  # 禁用SimpleVisualizer，避免与HierarchicalVisualizer冲突
    )

    if not server.start():
        print(f"  ✗ 服务器启动失败")
        sys.exit(1)

    print(f"✓ 服务器启动成功")

    if not server.start_mission():
        print(f"  ✗ 无人机任务启动失败")
        sys.exit(1)

    print(f"✓ 无人机任务启动成功")

    # 2.5 设置实验元数据 (用于跨方案数据对比)
    if hasattr(server, 'set_experiment_meta'):
        server.set_experiment_meta(
            algorithm_type='hrl_dqn_apf',
            env_type='hierarchical',
            control_mode='dqn'
        )

    # 3. 创建训练环境
    if len(drone_names) == 1:
        training_drone = drone_names[0]
        env = HierarchicalMovementEnv(server=server, drone_name=training_drone, config_path=hrl_config_path)
    else:
        print(f"模式: 多机分层训练 (无人机: {drone_names})")
        env = MultiDroneHierarchicalMovementEnv(server=server, drone_names=drone_names, config_path=hrl_config_path)
    
    env = Monitor(env)

    print(f"✓ 环境创建成功")
    print(f"  - HL 观察空间: {env.observation_space.shape}")
    print(f"  - HL 动作空间: {env.action_space.n}")

    # 4. 加载底层 (LL) 策略
    ll_model_path = os.path.join(os.path.dirname(__file__), "models", "movement_dqn_final.zip")
    if os.path.exists(ll_model_path):
        print(f"✓ 加载预训练底层模型: {ll_model_path}")
        ll_policy = DQN.load(ll_model_path)
        if len(drone_names) == 1:
            env.unwrapped.ll_policy = ll_policy
        else:
            env.unwrapped.set_ll_policy(ll_policy)
    else:
        print("! 未发现预训练底层模型，将使用启发式逻辑作为底层控制器")

    # 5. 创建或加载高层 (HL) 模型
    model_dir = os.path.join(os.path.dirname(__file__), 'models', 'hrl_planner_airsim')
    os.makedirs(model_dir, exist_ok=True)
    log_dir = os.path.join(os.path.dirname(__file__), 'logs', 'hrl_dqn_airsim')
    os.makedirs(log_dir, exist_ok=True)

    pretrained_hl = os.path.join(model_dir, 'hrl_hl_airsim_final.zip')
    if os.path.exists(pretrained_hl):
        print(f"✓ 加载预训练高层模型继续训练: {pretrained_hl}")
        model = DQN.load(pretrained_hl, env=env)
        model.tensorboard_log = log_dir
    else:
        print(f"创建新高层模型...")
        model = DQN(
            hrl_config['model']['policy'],
            env,
            learning_rate=hrl_config['training']['learning_rate'],
            buffer_size=hrl_config['training']['buffer_size'],
            learning_starts=hrl_config['training']['learning_starts'],
            batch_size=hrl_config['training']['batch_size'],
            tau=hrl_config['training']['tau'],
            gamma=hrl_config['training']['gamma'],
            target_update_interval=hrl_config['training']['target_update_interval'],
            exploration_fraction=hrl_config['training']['exploration_fraction'],
            exploration_initial_eps=hrl_config['training']['exploration_initial_eps'],
            exploration_final_eps=hrl_config['training']['exploration_final_eps'],
            policy_kwargs=dict(net_arch=hrl_config['model']['net_arch']),
            verbose=1,
            tensorboard_log=log_dir
        )

    # 6. 初始化可视化（如果启用）
    visualizer = None
    if enable_visualization and HAS_VISUALIZER:
        try:
            print(f"\n正在初始化分层训练可视化...")
            visualizer = HierarchicalVisualizer(env.unwrapped, server)
            visualizer.start_visualization()
            print(f"✓ 可视化已启动")
            time.sleep(1.0)  # 等待窗口初始化
        except Exception as e:
            print(f"! 可视化初始化失败: {str(e)}")
            print(f"  训练将继续，但不显示可视化")
            visualizer = None
    elif enable_visualization and not HAS_VISUALIZER:
        print(f"! 可视化模块未安装，训练将继续但不显示可视化")
        print(f"  提示: 安装pygame以启用可视化功能")
    
    # 7. 设置回调
    callbacks = []
    
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=model_dir,
        name_prefix='hrl_hl_airsim_checkpoint'
    )
    callbacks.append(checkpoint_callback)
    
    if visualizer:
        vis_callback = VisualizationCallback(visualizer)
        callbacks.append(vis_callback)

    # 8. 开始训练
    print("\n" + "=" * 80)
    print(f"开始分层 AirSim 融合训练")
    if visualizer:
        print(f"可视化: 已启用")
    else:
        print(f"可视化: 已禁用")
    print("=" * 80)

    try:
        model.learn(
            total_timesteps=hrl_config['training']['total_timesteps'],
            callback=callbacks,
            log_interval=1
        )
        
        # 保存最终模型
        final_model_path = os.path.join(model_dir, 'hrl_hl_airsim_final')
        model.save(final_model_path)
        print(f"\n✓ 高层模型已保存: {final_model_path}.zip")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n✗ 训练出错: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 停止可视化
        if visualizer:
            print(f"正在停止可视化...")
            visualizer.stop_visualization()
        
        # 停止服务器
        print(f"正在停止服务器...")
        server.stop()
        print(f"✓ 服务器已停止")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分层强化学习训练脚本')
    parser.add_argument('--no-visualization', action='store_true',
                       help='禁用实时可视化')
    args = parser.parse_args()
    
    train_hrl_with_airsim(enable_visualization=not args.no_visualization)
