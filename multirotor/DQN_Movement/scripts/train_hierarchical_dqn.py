import os
import sys
import numpy as np
import json
import time
import argparse
from datetime import datetime
import torch

# 添加项目路径
# scripts -> DQN_Movement -> multirotor -> 项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# 添加 DQN_Movement 目录
dqn_movement_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, dqn_movement_dir)

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from envs.hierarchical_movement_env import HierarchicalMovementEnv

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
        action = self.locals.get('actions', [0])[0]
        reward = self.locals.get('rewards', [0])[0]
        
        # 更新可视化数据
        self.visualizer.update_training_data(
            step=self.num_timesteps,
            action=int(action),
            reward=float(reward),
            drone_name='UAV1'
        )
        
        self.episode_reward += reward
        
        # 检查Episode是否结束
        dones = self.locals.get('dones', [False])
        if dones[0]:
            self.visualizer.on_episode_end(self.episode_count)
            self.episode_count += 1
            self.episode_reward = 0
        
        return True

def train_hrl(enable_visualization=True):
    print("=" * 80)
    print("分层强化学习 (HRL) 训练 - 高层协同规划器")
    print("=" * 80)
    
    # 1. 加载配置
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "hierarchical_dqn_config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 2. 创建高层训练环境
    # 注意: 这里 server=None 仅用于演示或纯离线逻辑测试。
    # 实际训练应连接 AirSimServer (AlgorithmServer.py)
    base_env = HierarchicalMovementEnv(server=None, drone_name="UAV1", config_path=config_path)
    env = Monitor(base_env)
    
    print(f"✓ 环境创建成功")
    print(f"  - HL 观察空间: {env.observation_space.shape}")
    print(f"  - HL 动作空间: {base_env.hl_action_space.n} (5x5 选点)")
    
    # 3. 加载底层 (LL) 策略 (可选)
    # 如果已有训练好的移动 DQN，可以在这里加载
    ll_model_path = os.path.join(os.path.dirname(__file__), "models", "movement_dqn_final.zip")
    if os.path.exists(ll_model_path):
        print(f"✓ 发现预训练底层模型: {ll_model_path}")
        env.unwrapped.ll_policy = DQN.load(ll_model_path)
    else:
        print("! 未发现预训练底层模型，将使用启发式趋向逻辑作为底层控制器")
    
    # 4. 创建高层 (HL) 模型
    model = DQN(
        config['model']['policy'],
        env,
        learning_rate=config['training']['learning_rate'],
        buffer_size=config['training']['buffer_size'],
        learning_starts=config['training']['learning_starts'],
        batch_size=config['training']['batch_size'],
        tau=config['training']['tau'],
        gamma=config['training']['gamma'],
        target_update_interval=config['training']['target_update_interval'],
        exploration_fraction=config['training']['exploration_fraction'],
        exploration_initial_eps=config['training']['exploration_initial_eps'],
        exploration_final_eps=config['training']['exploration_final_eps'],
        policy_kwargs=dict(net_arch=config['model']['net_arch']),
        verbose=1,
        tensorboard_log=os.path.join(os.path.dirname(__file__), "logs", "hrl_tensorboard")
    )
    
    # 模型保存目录
    model_dir = os.path.join(os.path.dirname(__file__), 'models', 'hrl_planner')
    os.makedirs(model_dir, exist_ok=True)
    
    # 5. 初始化可视化（如果启用）
    visualizer = None
    if enable_visualization and HAS_VISUALIZER:
        try:
            print(f"\n正在初始化分层训练可视化...")
            visualizer = HierarchicalVisualizer(env.unwrapped, server=None)
            visualizer.start_visualization()
            print(f"✓ 可视化已启动 (离线模式)")
            time.sleep(1.0)  # 等待窗口初始化
        except Exception as e:
            print(f"! 可视化初始化失败: {str(e)}")
            print(f"  训练将继续，但不显示可视化")
            visualizer = None
    elif enable_visualization and not HAS_VISUALIZER:
        print(f"! 可视化模块未安装，训练将继续但不显示可视化")
        print(f"  提示: 安装pygame以启用可视化功能")
    
    # 6. 设置回调
    callbacks = []
    
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=model_dir,
        name_prefix='hrl_hl_checkpoint'
    )
    callbacks.append(checkpoint_callback)
    
    if visualizer:
        vis_callback = VisualizationCallback(visualizer)
        callbacks.append(vis_callback)
    
    # 7. 开始训练
    print("\n" + "=" * 80)
    print(f"开始 HL 训练 (总计 {config['training']['total_timesteps']} 步)")
    if visualizer:
        print(f"可视化: 已启用 (离线模式)")
    else:
        print(f"可视化: 已禁用")
    print("=" * 80)
    
    try:
        model.learn(
            total_timesteps=config['training']['total_timesteps'],
            callback=callbacks,
            log_interval=1
        )
        
        # 7. 保存结果
        final_model_path = os.path.join(model_dir, 'hrl_hl_final')
        model.save(final_model_path)
        print(f"\n✓ 高层模型训练完成并保存: {final_model_path}.zip")
        
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分层强化学习训练脚本（离线模式）')
    parser.add_argument('--no-visualization', action='store_true',
                       help='禁用实时可视化')
    args = parser.parse_args()
    
    train_hrl(enable_visualization=not args.no_visualization)
