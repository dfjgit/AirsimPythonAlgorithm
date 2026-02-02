"""
DQN训练脚本 - 无人机移动控制
训练DQN智能体学习如何通过6方向移动控制无人机进行区域扫描
"""
import os
import sys
import numpy as np
import json
from datetime import datetime

# 添加项目路径
# scripts -> DQN_Movement -> multirotor -> 项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# 添加 DQN_Movement 目录
dqn_movement_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, dqn_movement_dir)

print("=" * 80)
print("DQN训练 - 无人机移动控制")
print("=" * 80)

# 检查依赖
print("\n[步骤1] 检查依赖...")
try:
    import torch
    print(f"  ✓ PyTorch: {torch.__version__}")
except ImportError:
    print("  ✗ PyTorch未安装")
    print("    安装命令: pip install torch")
    sys.exit(1)

try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    import gymnasium
    print(f"  ✓ Stable-Baselines3已安装")
except ImportError:
    print("  ✗ Stable-Baselines3未安装")
    print("    安装命令: pip install stable-baselines3 gymnasium")
    sys.exit(1)

# 导入环境
from envs.movement_env import MovementEnv

print("\n" + "=" * 80)
print("[步骤2] 加载配置")
print("=" * 80)

# 加载配置文件
config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "movement_dqn_config.json")
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

print(f"  ✓ 配置文件加载成功: {config_path}")
print(f"    - 动作步长: {config['movement']['step_size']}米")
print(f"    - 最大步数: {config['movement']['max_steps']}")
print(f"    - 训练步数: {config['training']['total_timesteps']}")
print(f"    - 学习率: {config['training']['learning_rate']}")

print("\n" + "=" * 80)
print("[步骤3] 创建训练环境")
print("=" * 80)

# 创建环境（无server，使用模拟数据）
env = MovementEnv(server=None, drone_name="UAV1", config_path=config_path)
env = Monitor(env)  # 包装监控器

print(f"  ✓ 环境创建成功")
print(f"    - 观察空间: {env.observation_space.shape}")
print(f"    - 动作空间: {env.action_space.n} (6方向)")
print(f"    - 动作映射: 0=上, 1=下, 2=左, 3=右, 4=前, 5=后")

print("\n" + "=" * 80)
print("[步骤4] 创建DQN模型")
print("=" * 80)

# 创建DQN模型
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
    tensorboard_log=None
)

print(f"  ✓ DQN模型创建成功")
print(f"    - 策略: {config['model']['policy']}")
print(f"    - 网络结构: {config['model']['net_arch']}")
print(f"    - 学习率: {config['training']['learning_rate']}")
print(f"    - 缓冲区大小: {config['training']['buffer_size']}")
print(f"    - 批次大小: {config['training']['batch_size']}")

print("\n" + "=" * 80)
print("[步骤5] 设置训练回调")
print("=" * 80)

# 创建模型保存目录
model_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(model_dir, exist_ok=True)
log_dir = os.path.join(os.path.dirname(__file__), 'logs', 'movement_dqn')
os.makedirs(log_dir, exist_ok=True)

# 自定义进度回调
class ProgressCallback(BaseCallback):
    """显示训练进度和统计信息，并保存训练数据"""
    
    def __init__(self, total_timesteps, print_freq=1000, save_dir=None, verbose=0):
        super(ProgressCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.print_freq = print_freq
        self.save_dir = save_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.episode_count = 0
        self.start_time = datetime.now()
        
        # 创建保存目录
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
        
    def _on_step(self) -> bool:
        # 累计episode统计
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        # episode结束
        if self.locals['dones'][0]:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.episode_times.append((datetime.now() - self.start_time).total_seconds())
            
            # 保存 episode 统计
            if self.save_dir:
                self._save_episode_stats()
            
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        # 定期打印进度
        if self.num_timesteps % self.print_freq == 0:
            progress = (self.num_timesteps / self.total_timesteps) * 100
            
            if len(self.episode_rewards) > 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                print(f"\n进度: {progress:.1f}% ({self.num_timesteps}/{self.total_timesteps})")
                print(f"  最近10个episode - 平均奖励: {avg_reward:.2f}, 平均长度: {avg_length:.1f}")
            else:
                print(f"\n进度: {progress:.1f}% ({self.num_timesteps}/{self.total_timesteps})")
        
        return True
    
    def _save_episode_stats(self):
        """保存 episode 统计数据为 CSV 格式"""
        import csv
        csv_path = os.path.join(self.save_dir, 'dqn_training_stats.csv')
        
        # 准备数据
        stats = {
            'episode': self.episode_count,
            'reward': self.episode_rewards[-1],
            'length': self.episode_lengths[-1],
            'elapsed_time': self.episode_times[-1]
        }
        
        # 写入 CSV
        file_exists = os.path.exists(csv_path)
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['episode', 'reward', 'length', 'elapsed_time'])
            if not file_exists:
                writer.writeheader()
            writer.writerow(stats)

# 检查点回调
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=model_dir,
    name_prefix='movement_dqn_checkpoint'
)

# 进度回调
progress_callback = ProgressCallback(
    total_timesteps=config['training']['total_timesteps'],
    print_freq=1000,
    save_dir=log_dir  # 添加保存目录
)

print(f"  ✓ 回调设置完成")
print(f"    - 模型保存目录: {model_dir}")
print(f"    - 日志目录: {log_dir}")
print(f"    - 检查点保存频率: 每10000步")

print("\n" + "=" * 80)
print("[步骤6] 开始训练")
print("=" * 80)

total_timesteps = config['training']['total_timesteps']
print(f"训练步数: {total_timesteps}")
print(f"预计时间: 约{total_timesteps // 1000}分钟（取决于硬件）")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

try:
    # 开始训练
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, progress_callback],
        log_interval=10
    )
    
    print("\n" + "=" * 80)
    print("✓ 训练完成！")
    print("=" * 80)
    
except KeyboardInterrupt:
    print("\n训练被用户中断")
except Exception as e:
    print(f"\n✗ 训练出错: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("[步骤7] 保存最终模型和训练数据")
print("=" * 80)

# 保存最终模型
final_model_path = os.path.join(model_dir, 'movement_dqn_final')
model.save(final_model_path)
print(f"  ✓ 最终模型已保存: {final_model_path}.zip")

# 保存训练元数据为 JSON
metadata = {
    'algorithm': 'DQN',
    'task': 'movement_control',
    'start_time': progress_callback.start_time.strftime('%Y-%m-%d %H:%M:%S'),
    'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'duration_seconds': (datetime.now() - progress_callback.start_time).total_seconds(),
    'total_timesteps': config['training']['total_timesteps'],
    'total_episodes': progress_callback.episode_count,
    'config': config,
    'observation_space': {
        'shape': list(env.observation_space.shape),
        'dtype': str(env.observation_space.dtype)
    },
    'action_space': {
        'n': env.action_space.n,
        'actions': ['up', 'down', 'left', 'right', 'forward', 'backward']
    },
    'final_model_path': f"{final_model_path}.zip",
    'training_stats_path': os.path.join(log_dir, 'dqn_training_stats.csv')
}

metadata_path = os.path.join(log_dir, 'dqn_training_metadata.json')
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)
print(f"  ✓ 训练元数据已保存: {metadata_path}")

print("\n" + "=" * 80)
print("[步骤8] 测试模型")
print("=" * 80)

# 简单测试
obs, info = env.reset()
print(f"测试状态shape: {obs.shape}")

print("\n执行5步测试:")
for i in range(5):
    action, _states = model.predict(obs, deterministic=True)
    action_name = ['上', '下', '左', '右', '前', '后'][action]
    
    print(f"\n测试 {i+1}:")
    print(f"  预测动作: {action} ({action_name})")
    
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(f"  奖励: {reward:.2f}")
    print(f"  已扫描: {info['scanned_cells']}")
    
    if done:
        obs, info = env.reset()
        print(f"  Episode结束，重置环境")

print("\n" + "=" * 80)
print("训练完成总结")
print("=" * 80)
print(f"✓ 最终模型: {final_model_path}.zip")
print(f"✓ 训练日志: {log_dir}")
print(f"✓ 检查点: {model_dir}")
print(f"✓ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\n下一步:")
print(f"  1. 查看Tensorboard日志: tensorboard --logdir={log_dir}")
print(f"  2. 测试训练好的模型: python test_movement_dqn.py")
print(f"  3. 与AirSim集成: python train_movement_with_airsim.py")
print("=" * 80)

