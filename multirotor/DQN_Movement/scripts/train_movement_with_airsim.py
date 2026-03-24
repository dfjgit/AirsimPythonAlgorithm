"""
DQN训练脚本 - 与AirSim集成
使用真实的AirSim环境训练无人机移动策略
"""
import os
import sys
import numpy as np
import json
from datetime import datetime
import threading
import time
import subprocess


# 添加项目路径
# scripts -> DQN_Movement -> multirotor -> 项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) )
sys.path.insert(0, project_root)

# 添加 multirotor 目录
multirotor_dir = os.path.join(project_root, 'multirotor')
sys.path.insert(0, multirotor_dir)

# 添加 DQN_Movement 目录
dqn_movement_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, dqn_movement_dir)

print("=" * 80)
print("DQN训练 - 无人机移动控制 (AirSim集成)")
print("=" * 80)

# 检查依赖
print("\n[步骤1] 检查依赖...")
try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import get_linear_fn
    import gymnasium
    print(f"  ✓ Stable-Baselines3已安装")
except ImportError:
    print("  ✗ Stable-Baselines3未安装")
    print("    安装命令: pip install stable-baselines3 gymnasium")
    sys.exit(1)

# 导入环境和服务器
from envs.movement_env import MovementEnv, MultiDroneMovementEnv
from AlgorithmServer import MultiDroneAlgorithmServer
from Algorithm.drones_config import DronesConfig

# 独立进程可视化 (pygame 不与训练进程共享)
try:
    from multirotor.Visualization.visualization_ipc import VisualizationIPCServer
    HAS_EXT_VIS = True
except Exception:
    HAS_EXT_VIS = False
    print("警告: 无法导入 VisualizationIPCServer，独立可视化将被禁用")

print("\n" + "=" * 80)
print("[步骤2] 加载配置并确定训练无人机")
print("=" * 80)

# 加载无人机配置
drones_config = DronesConfig()
print(f"  ✓ 加载 drones_config.json")
print(f"    - 所有无人机: {drones_config.get_all_drones()}")
print(f"    - 启用的无人机: {drones_config.get_enabled_drones()}")

# 获取 DQN 训练使用的无人机列表
drone_names = drones_config.get_training_drones('dqn')
print(f"  ✓ DQN训练使用的无人机: {drone_names}")

if not drone_names:
    print(f"  ✗ 错误: 没有可用的训练无人机，请检查 drones_config.json")
    sys.exit(1)

# 显示无人机类型（虚拟/实体）
print(f"  \n  训练无人机详情:")
for drone in drone_names:
    is_crazyflie = drones_config.is_crazyflie_mirror(drone)
    type_display = "实体无人机(Crazyflie)" if is_crazyflie else "虚拟无人机(AirSim)"
    print(f"    - {drone}: {type_display}")

# 加载 DQN 训练配置
dqn_config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "movement_dqn_config.json")
with open(dqn_config_path, 'r', encoding='utf-8') as f:
    dqn_config = json.load(f)
print(f"  ✓ 加载 movement_dqn_config.json")

# apf_algorithm_config.json 路径（AlgorithmServer需要）
config_file = os.path.join(os.path.dirname(__file__), "..", "..", "apf_algorithm_config.json")
if not os.path.exists(config_file):
    print(f"  ✗ apf_algorithm_config.json 不存在: {config_file}")
    sys.exit(1)

print("\n" + "=" * 80)
print("[步骤3] 启动AirSim服务器")
print("=" * 80)

# 创建算法服务器（使用DQN控制模式）
print(f"  正在启动服务器 (DQN控制模式)...")
server = MultiDroneAlgorithmServer(
    config_file=config_file,
    drone_names=drone_names,
    use_learned_weights=False,
    control_mode='dqn',
    enable_visualization=False
)

# 启动服务器（连接Unity和AirSim）
print(f"  连接Unity和AirSim...")
if not server.start():
    print(f"  ✗ 服务器启动失败")
    sys.exit(1)

print(f"  ✓ 服务器启动成功")
if getattr(server, 'reset_trace_path', None):
    print(f"  Reset诊断日志: {server.reset_trace_path}")

# 关键：启动任务（让无人机起飞并启动算法线程）
print(f"  启动无人机任务...")
if not server.start_mission():
    print(f"  ✗ 无人机任务启动失败")
    sys.exit(1)

print(f"  ✓ 无人机任务启动成功")

# 独立进程可视化（默认启用，可通过环境变量 NO_VIS=1 禁用）
ipc_server = None
vis_process = None
vis_log_f = None
vis_log_path = None
# 先用临时目录存日志，避免 log_dir 尚未创建导致失败
_tmp_vis_log_dir = os.path.join(os.path.dirname(__file__), 'logs', 'movement_dqn_airsim')
os.makedirs(_tmp_vis_log_dir, exist_ok=True)

if HAS_EXT_VIS and os.environ.get('NO_VIS', '0') != '1':
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

        python_exe = sys.executable
        vis_entry = os.path.join(project_root, 'multirotor', 'Visualization', 'external_visualizer_client.py')
        vis_cmd = [python_exe, vis_entry, '--mode', 'dqn', '--host', '127.0.0.1', '--port', str(port)]

        vis_log_path = os.path.join(_tmp_vis_log_dir, 'external_vis.log')
        vis_log_f = open(vis_log_path, 'w', encoding='utf-8')

        creationflags = 0
        if os.name == 'nt' and os.environ.get('VIS_NEW_CONSOLE', '0') == '1':
            creationflags = subprocess.CREATE_NEW_CONSOLE

        vis_env = os.environ.copy()
        vis_env['PYTHONIOENCODING'] = 'utf-8'
        vis_env['PYTHONUTF8'] = '1'

        vis_process = subprocess.Popen(
            vis_cmd,
            stdout=vis_log_f,
            stderr=vis_log_f,
            creationflags=creationflags,
            env=vis_env
        )

        # 若子进程秒退，给出更明确提示
        time.sleep(0.5)
        rc = vis_process.poll()
        if rc is not None:
            print(f"  ! 独立可视化进程启动后立即退出 (returncode={rc})")
            print(f"    - 请查看: {vis_log_path}")
        else:
            print(f"  ✓ 已启动独立可视化进程 (port={port})")
            print(f"    - 外部可视化日志: {vis_log_path}")
    except Exception as e:
        print(f"  ! 启动独立可视化失败: {e}")
        if vis_log_path:
            print(f"    - 外部可视化日志(若已生成): {vis_log_path}")
        try:
            if ipc_server:
                ipc_server.stop()
        except Exception:
            pass
        ipc_server = None
        vis_process = None

if hasattr(server, 'set_experiment_meta'):
    server.set_experiment_meta(
        algorithm_type='pure_dqn',
        env_type='movement',
        control_mode='dqn'
    )

print("\n" + "=" * 80)
print("[步骤4] 创建训练环境")
print("=" * 80)

# 根据无人机数量选择环境类型
if len(drone_names) == 1:
    training_drone = drone_names[0]
    print(f"  模式: 单机训练")
    print(f"  训练无人机: {training_drone}")

    env = MovementEnv(server=server, drone_name=training_drone, config_path=dqn_config_path)
    env = Monitor(env)
else:
    print(f"  模式: 多机训练（参数共享）")
    print(f"  训练无人机: {drone_names}")
    print(f"  无人机数量: {len(drone_names)}")

    env = MultiDroneMovementEnv(server=server, drone_names=drone_names, config_path=dqn_config_path)
    env = Monitor(env)

print(f"  ✓ 环境创建成功")
print(f"    - 观察空间: {env.observation_space.shape}")
print(f"    - 动作空间: {env.action_space.n} (6方向)")
print(f"    - 连接到服务器: {server.running}")

print("\n" + "=" * 80)
print("[步骤5] 创建或加载DQN模型")
print("=" * 80)

model_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(model_dir, exist_ok=True)

log_dir = os.path.join(os.path.dirname(__file__), 'logs', 'movement_dqn_airsim')
os.makedirs(log_dir, exist_ok=True)

pretrained_model = os.path.join(model_dir, 'movement_dqn_final.zip')
resume_default = bool(dqn_config.get('training', {}).get('resume_from_pretrained', False))
resume_env = os.environ.get('USE_PRETRAINED')
resume_requested = resume_default if resume_env is None else (resume_env == '1')
use_pretrained = resume_requested and os.path.exists(pretrained_model)

def apply_resume_training_settings(model, train_cfg):
    """Apply fine-tune exploration settings when resuming from an existing model."""
    resume_initial_eps = float(train_cfg.get('resume_exploration_initial_eps', train_cfg['exploration_initial_eps']))
    resume_final_eps = float(train_cfg.get('resume_exploration_final_eps', train_cfg['exploration_final_eps']))
    resume_fraction = float(train_cfg.get('resume_exploration_fraction', train_cfg['exploration_fraction']))
    model.exploration_initial_eps = resume_initial_eps
    model.exploration_final_eps = resume_final_eps
    model.exploration_fraction = resume_fraction
    model.exploration_schedule = get_linear_fn(
        resume_initial_eps,
        resume_final_eps,
        resume_fraction,
    )
    model.learning_rate = float(train_cfg['learning_rate'])
    model.lr_schedule = lambda _: float(train_cfg['learning_rate'])
    model.batch_size = int(train_cfg['batch_size'])
    model.gamma = float(train_cfg['gamma'])
    model.target_update_interval = int(train_cfg['target_update_interval'])
    print(f"  ✓ 续训探索率: {resume_initial_eps:.3f} -> {resume_final_eps:.3f} (fraction={resume_fraction:.3f})")
    print(f"  ✓ 续训总步数: {int(train_cfg.get('resume_total_timesteps', train_cfg['total_timesteps']))}")

if use_pretrained:
    print(f"  ✓ 找到预训练模型: {pretrained_model}")
    print(f"  加载预训练模型继续训练...")
    model = DQN.load(pretrained_model, env=env)
    model.tensorboard_log = log_dir
    apply_resume_training_settings(model, dqn_config['training'])
    print(f"  ✓ 预训练模型加载成功")
    print(f"  ✓ TensorBoard 日志: {log_dir}")
else:
    if os.path.exists(pretrained_model):
        resume_hint = "1" if not resume_default else "0"
        action_hint = "继续叠加训练" if not resume_default else "从头训练"
        print(f"  ! 检测到旧模型但本次未加载: {pretrained_model}")
        print(f"    - 如需显式{action_hint}，请设置环境变量 USE_PRETRAINED={resume_hint}")
    print(f"  创建新模型...")
    model = DQN(
        dqn_config['model']['policy'],
        env,
        learning_rate=dqn_config['training']['learning_rate'],
        buffer_size=dqn_config['training']['buffer_size'],
        learning_starts=dqn_config['training']['learning_starts'],
        batch_size=dqn_config['training']['batch_size'],
        tau=dqn_config['training']['tau'],
        gamma=dqn_config['training']['gamma'],
        target_update_interval=dqn_config['training']['target_update_interval'],
        exploration_fraction=dqn_config['training']['exploration_fraction'],
        exploration_initial_eps=dqn_config['training']['exploration_initial_eps'],
        exploration_final_eps=dqn_config['training']['exploration_final_eps'],
        policy_kwargs=dict(net_arch=dqn_config['model']['net_arch']),
        verbose=1,
        tensorboard_log=log_dir
    )
    print(f"  ✓ 新模型创建成功")
    print(f"  ✓ TensorBoard 日志: {log_dir}")

print("\n" + "=" * 80)
print("[步骤6] 设置训练回调")
print("=" * 80)

class AirSimProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, print_freq=500, log_dir=None, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.print_freq = print_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_scanned = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.episode_count = 0
        self.start_time = datetime.now()
        self.log_dir = log_dir
        if self.log_dir:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.csv_path = os.path.join(self.log_dir, f'dqn_training_{timestamp}.csv')
            with open(self.csv_path, 'w', encoding='utf-8') as f:
                f.write('episode,reward,length,scanned_cells,timestep,elapsed_time,timestamp,collision_count,out_of_range_count,scan_efficiency\n')
            print(f"  ✓ CSV 日志: {self.csv_path}")

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0:
            progress = (self.num_timesteps / self.total_timesteps) * 100
            print(f"\n{'=' * 60}")
            print(f"进度: {progress:.1f}% ({self.num_timesteps}/{self.total_timesteps})")
            print(f"{'=' * 60}")
        return True

checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path=model_dir,
    name_prefix='movement_dqn_airsim'
)

progress_callback = AirSimProgressCallback(
    total_timesteps=dqn_config['training']['total_timesteps'],
    print_freq=500,
    log_dir=log_dir
)

class DQNVisualizationCallback(BaseCallback):
    def __init__(self, server, verbose=0):
        super().__init__(verbose)
        self.server = server
        self.drone_names = list(getattr(server, "drone_names", [])) if server is not None else []
        self.episode_reward = 0.0
        self.action_counts = {i: 0 for i in range(6)}
        self.last_action = None
        self.episode_count = 0
        self.total_steps = 0
        self.current_episode_steps = 0
        self.reward_history = []  # 记录每个episode的总奖励
        self.start_time = time.time()  # 用于计算速率
        self.per_drone_actions = {
            drone_name: {
                'last_action': None,
                'leader_distance': None,
                'is_out_of_range': False,
                'out_of_range_steps': 0,
                'out_of_range_duration_sec': 0.0,
                'out_of_range_count': 0,
                'current_drone_reward': 0.0,
            }
            for drone_name in self.drone_names
        }

    def _on_step(self) -> bool:
        if self.server is None:
            return True
        try:
            action = None
            if 'actions' in self.locals and len(self.locals['actions']) > 0:
                action = int(self.locals['actions'][0])
            reward = float(self.locals.get('rewards', [0.0])[0])
            self.episode_reward += reward
            self.total_steps += 1
            self.current_episode_steps += 1
            
            # 计算速率
            elapsed = time.time() - self.start_time
            steps_per_sec = self.total_steps / max(elapsed, 0.001)
            
            dones = self.locals.get('dones', [False])
            is_done = bool(dones[0]) if len(dones) > 0 else False
            infos = self.locals.get('infos', [{}])
            info = infos[0] if infos and isinstance(infos[0], dict) else {}
            drone_name = info.get('drone_name')

            if action is not None and action in self.action_counts:
                self.action_counts[action] += 1
                self.last_action = action
            if drone_name:
                if drone_name not in self.per_drone_actions:
                    self.per_drone_actions[drone_name] = {
                        'last_action': None,
                        'leader_distance': None,
                        'is_out_of_range': False,
                        'out_of_range_steps': 0,
                        'out_of_range_duration_sec': 0.0,
                        'out_of_range_count': 0,
                        'current_drone_reward': 0.0,
                    }
                self.per_drone_actions[drone_name].update(
                    {
                        'last_action': action,
                        'leader_distance': info.get('leader_distance'),
                        'is_out_of_range': bool(info.get('is_out_of_range', False)),
                        'out_of_range_steps': int(info.get('out_of_range_steps', 0) or 0),
                        'out_of_range_duration_sec': float(info.get('out_of_range_duration_sec', 0.0) or 0.0),
                        'out_of_range_count': int(info.get('out_of_range_count', 0) or 0),
                        'current_drone_reward': float(info.get('current_drone_reward', 0.0) or 0.0),
                    }
                )

            if is_done:
                self.episode_count += 1
                self.reward_history.append(float(self.episode_reward))
                # 保持奖励历史在合理范围内
                if len(self.reward_history) > 200:
                    self.reward_history = self.reward_history[-200:]

            # 写入到server，供IPC外部可视化快照读取
            self.server.current_training_stats = {
                'timestep': int(getattr(self, 'num_timesteps', 0)),
                'total_steps': self.total_steps,
                'current_episode_steps': self.current_episode_steps,
                'steps_per_sec': float(steps_per_sec),
                'episode_count': self.episode_count,
                'current_episode_reward': float(self.episode_reward),
                'reward_history': list(self.reward_history),
                'is_done': bool(is_done),
                'last_action': self.last_action,
                'action_counts': dict(self.action_counts),
                'drone_name': drone_name,
                'current_step_reward': reward,
                'leader_distance': info.get('leader_distance'),
                'is_out_of_range': bool(info.get('is_out_of_range', False)),
                'out_of_range_steps': int(info.get('out_of_range_steps', 0) or 0),
                'out_of_range_duration_sec': float(info.get('out_of_range_duration_sec', 0.0) or 0.0),
                'out_of_range_count': int(info.get('out_of_range_count', 0) or 0),
                'current_drone_reward': float(info.get('current_drone_reward', 0.0) or 0.0),
                'last_done_reason': info.get('last_done_reason'),
                'per_drone_actions': {
                    name: dict(values) for name, values in self.per_drone_actions.items()
                },
            }
            # 强制快照缓存失效
            try:
                self.server._vis_snapshot_cache = None
                self.server._vis_snapshot_cache_time = 0.0
            except Exception:
                pass

            if is_done:
                self.episode_reward = 0.0
                self.current_episode_steps = 0
        except Exception:
            pass
        return True

visualizer = None
vis_callback = None
# 临时禁用可视化以排查卡死问题
"""
if HAS_VISUALIZER:
    try:
        visualizer = DQNMovementTrainingVisualizer(env.unwrapped, server)

        def start_vis_thread():
            try:
                visualizer.start_visualization()
            except Exception as e:
                print(f"  ! 可视化窗口运行异常: {e}")

        vis_thread = threading.Thread(target=start_vis_thread, daemon=True)
        vis_thread.start()

        vis_callback = DQNVisualizationCallback(visualizer)
        print("  ✓ DQN训练可视化已在独立线程启动", flush=True)
    except Exception as e:
        print(f"  ! DQN训练可视化启动失败: {e}", flush=True)
        visualizer = None
        vis_callback = None
"""

print(f"  ✓ 回调设置完成", flush=True)
sys.stdout.flush()

print("\n" + "=" * 80)
print("[步骤7] 开始训练")
print("=" * 80)

total_timesteps = int(
    dqn_config['training'].get(
        'resume_total_timesteps' if use_pretrained else 'total_timesteps',
        dqn_config['training']['total_timesteps']
    )
)
reset_num_timesteps = True
if use_pretrained:
    reset_num_timesteps = bool(dqn_config['training'].get('reset_num_timesteps_on_resume', False))
print(f"训练步数: {total_timesteps}")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\n⚠ 请确保Unity客户端已连接到服务器")
print(f"按 Ctrl+C 可以随时中断训练并保存模型\n")

try:
    print(f"\n[DEBUG] 即将调用 model.learn()...", flush=True)

    callbacks = [checkpoint_callback, progress_callback]
    
    # 启用 DQN 训练数据同步回调，用于外部可视化面板
    vis_callback = DQNVisualizationCallback(server)
    callbacks.append(vis_callback)
    print("  ✓ DQN 训练数据同步回调已启用 (用于外部可视化)")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=10,
        reset_num_timesteps=reset_num_timesteps
    )

    print("\n" + "=" * 80)
    print("✓ 训练完成！")
    print("=" * 80)

except KeyboardInterrupt:
    print("\n\n训练被用户中断")
    print("正在保存当前模型...")
except Exception as e:
    print(f"\n\n✗ 训练出错: {str(e)}")
    import traceback
    traceback.print_exc()
    print("正在保存当前模型...")

print("\n" + "=" * 80)
print("[步骤8] 保存最终模型")
print("=" * 80)

final_model_path = os.path.join(model_dir, 'movement_dqn_airsim_final')
model.save(final_model_path)
print(f"  ✓ 模型已保存: {final_model_path}.zip")

print("\n" + "=" * 80)
print("[步骤9] 清理")
print("=" * 80)

print(f"  正在停止服务器...")
server.stop()
time.sleep(1)
print(f"  ✓ 服务器已停止")

# 清理独立可视化资源
try:
    if ipc_server:
        ipc_server.stop()
except Exception:
    pass

try:
    if vis_process and vis_process.poll() is None:
        vis_process.terminate()
except Exception:
    pass

print("\n" + "=" * 80)
print("训练完成总结")
print("=" * 80)
print(f"✓ 最终模型: {final_model_path}.zip")
print(f"✓ 训练日志: {log_dir}")
print(f"✓ 检查点: {model_dir}")
print(f"✓ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\n下一步:")
print(f"  1. 查看Tensorboard: tensorboard --logdir={log_dir}")
print(f"  2. 测试模型: python test_movement_dqn.py")
print("=" * 80)
