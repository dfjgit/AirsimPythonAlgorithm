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
import subprocess

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

dqn_logs_root = os.path.join(dqn_movement_dir, 'logs')
dqn_model_dir = os.path.join(dqn_movement_dir, 'models')
dqn_legacy_model_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(dqn_logs_root, exist_ok=True)
os.makedirs(dqn_model_dir, exist_ok=True)

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# 导入环境和服务器
from envs.hierarchical_movement_env import HierarchicalMovementEnv, MultiDroneHierarchicalMovementEnv
from AlgorithmServer import MultiDroneAlgorithmServer
from Algorithm.drones_config import DronesConfig

# 独立进程可视化 (pygame 不与训练进程共享)
try:
    from multirotor.Visualization.visualization_ipc import VisualizationIPCServer
    HAS_EXT_VIS = True
except Exception:
    HAS_EXT_VIS = False
    print("警告: 无法导入 VisualizationIPCServer，独立可视化将被禁用")

# 导入可视化器
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from Visualization import HierarchicalTrainingVisualizer
    HAS_VISUALIZER = True
except ImportError:
    HAS_VISUALIZER = False
    print("警告: 无法导入HierarchicalTrainingVisualizer，可视化功能将被禁用")

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

    config_file = os.path.join(os.path.dirname(__file__), "..", "..", "apf_algorithm_config.json")

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

    # 启动独立进程可视化（默认启用，可通过 --no-visualization 或环境变量 NO_VIS=1 禁用）
    ipc_server = None
    vis_process = None
    vis_log_f = None
    vis_log_path = None

    _disable_vis = (not enable_visualization) or (os.environ.get('NO_VIS', '0') == '1')
    if HAS_EXT_VIS and (not _disable_vis):
        try:
            ipc_server = VisualizationIPCServer(
                snapshot_provider=server.get_visualization_snapshot,
                host='127.0.0.1',
                port=0,
                hz=10.0,
                compress_level=1
            )
            ipc_server.start()
            port = ipc_server.bound_port

            # 外部可视化日志
            log_dir = os.path.join(dqn_logs_root, 'hrl_dqn_airsim')
            os.makedirs(log_dir, exist_ok=True)
            vis_log_path = os.path.join(log_dir, 'external_vis.log')
            vis_log_f = open(vis_log_path, 'w', encoding='utf-8')

            python_exe = sys.executable
            vis_entry = os.path.join(project_root, 'multirotor', 'Visualization', 'external_visualizer_client.py')
            vis_cmd = [python_exe, vis_entry, '--mode', 'hrl', '--host', '127.0.0.1', '--port', str(port)]

            vis_env = os.environ.copy()
            vis_env['PYTHONIOENCODING'] = 'utf-8'
            vis_env['PYTHONUTF8'] = '1'

            creationflags = 0
            if os.name == 'nt' and os.environ.get('VIS_NEW_CONSOLE', '0') == '1':
                creationflags = subprocess.CREATE_NEW_CONSOLE

            vis_process = subprocess.Popen(
                vis_cmd,
                stdout=vis_log_f,
                stderr=vis_log_f,
                creationflags=creationflags,
                env=vis_env
            )

            time.sleep(0.5)
            rc = vis_process.poll()
            if rc is not None:
                print(f"! 独立可视化进程启动后立即退出 (returncode={rc})")
                print(f"  - 请查看: {vis_log_path}")
            else:
                print(f"✓ 已启动独立可视化进程 (port={port})")
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
    ll_model_candidates = [
        os.path.join(dqn_model_dir, "movement_dqn_airsim_final.zip"),
        os.path.join(dqn_model_dir, "movement_dqn_final.zip"),
        os.path.join(dqn_legacy_model_dir, "movement_dqn_airsim_final.zip"),
        os.path.join(dqn_legacy_model_dir, "movement_dqn_final.zip"),
    ]
    ll_model_path = next((path for path in ll_model_candidates if os.path.exists(path)), None)
    if ll_model_path is not None:
        print(f"✓ 加载预训练底层模型: {ll_model_path}")
        ll_policy = DQN.load(ll_model_path)
        if len(drone_names) == 1:
            env.unwrapped.ll_policy = ll_policy
        else:
            env.unwrapped.set_ll_policy(ll_policy)
    else:
        print("! 未发现预训练底层模型，将使用启发式逻辑作为底层控制器")

    # 5. 创建或加载高层 (HL) 模型
    model_dir = os.path.join(dqn_model_dir, 'hrl_planner_airsim')
    os.makedirs(model_dir, exist_ok=True)
    log_dir = os.path.join(dqn_logs_root, 'hrl_dqn_airsim')
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
            visualizer = HierarchicalTrainingVisualizer(env.unwrapped, server)
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
        # 保存中断时的模型（固定名称，避免一直生成多个文件）
        interrupted_model_path = os.path.join(model_dir, 'hrl_hl_airsim_interrupted')
        model.save(interrupted_model_path)
        print(f"✓ 中断时的模型已保存(已覆盖): {interrupted_model_path}.zip")
        print(f"💡 提示: 可以使用此模型继续训练或用于测试")
    except Exception as e:
        print(f"\n✗ 训练出错: {str(e)}")
        import traceback
        traceback.print_exc()
        # 出错时也尝试保存模型
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_model_path = os.path.join(model_dir, f'hrl_hl_airsim_error_{timestamp}')
        try:
            model.save(error_model_path)
            print(f"✓ 出错时的模型已保存: {error_model_path}.zip")
        except Exception as save_error:
            print(f"✗ 保存出错模型失败: {str(save_error)}")
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
