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

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

print("=" * 80)
print("DQN训练 - 无人机移动控制 (AirSim集成)")
print("=" * 80)

# 检查依赖
print("\n[步骤1] 检查依赖...")
try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    import gymnasium
    print(f"  ✓ Stable-Baselines3已安装")
except ImportError:
    print("  ✗ Stable-Baselines3未安装")
    print("    安装命令: pip install stable-baselines3 gymnasium")
    sys.exit(1)

# 导入环境和服务器
from movement_env import MovementEnv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from AlgorithmServer import MultiDroneAlgorithmServer

print("\n" + "=" * 80)
print("[步骤2] 启动AirSim服务器")
print("=" * 80)

# 配置文件路径
config_file = os.path.join(os.path.dirname(__file__), "..", "scanner_config.json")
if not os.path.exists(config_file):
    print(f"  ✗ 配置文件不存在: {config_file}")
    sys.exit(1)

# 创建算法服务器
print(f"  正在启动服务器...")
server = MultiDroneAlgorithmServer(
    config_file=config_file,
    drone_names=["UAV1"],
    use_learned_weights=False
)

# 在后台线程中运行服务器
server_thread = threading.Thread(target=server.start, daemon=True)
server_thread.start()

# 等待服务器初始化
print(f"  等待服务器初始化...")
time.sleep(3)

if server.running:
    print(f"  ✓ AirSim服务器启动成功")
else:
    print(f"  ⚠ 服务器可能未完全启动，但将继续尝试")

print("\n" + "=" * 80)
print("[步骤3] 创建训练环境")
print("=" * 80)

# 加载配置
config_path = os.path.join(os.path.dirname(__file__), "movement_dqn_config.json")
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

# 创建环境（连接到server）
env = MovementEnv(server=server, drone_name="UAV1", config_path=config_path)
env = Monitor(env)

print(f"  ✓ 环境创建成功")
print(f"    - 观察空间: {env.observation_space.shape}")
print(f"    - 动作空间: {env.action_space.n} (6方向)")
print(f"    - 连接到服务器: {server.running}")

print("\n" + "=" * 80)
print("[步骤4] 创建或加载DQN模型")
print("=" * 80)

# 检查是否有预训练模型
model_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(model_dir, exist_ok=True)

pretrained_model = os.path.join(model_dir, 'movement_dqn_final.zip')
use_pretrained = os.path.exists(pretrained_model)

if use_pretrained:
    print(f"  ✓ 找到预训练模型: {pretrained_model}")
    print(f"  加载预训练模型继续训练...")
    model = DQN.load(pretrained_model, env=env)
    model.tensorboard_log = None  # 禁用tensorboard
    print(f"  ✓ 预训练模型加载成功")
else:
    print(f"  创建新模型...")
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
    print(f"  ✓ 新模型创建成功")

print("\n" + "=" * 80)
print("[步骤5] 设置训练回调")
print("=" * 80)

# 日志目录
log_dir = os.path.join(os.path.dirname(__file__), 'logs', 'movement_dqn_airsim')
os.makedirs(log_dir, exist_ok=True)

# 自定义回调
class AirSimProgressCallback(BaseCallback):
    """AirSim训练进度回调"""
    
    def __init__(self, total_timesteps, print_freq=500, verbose=0):
        super(AirSimProgressCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.print_freq = print_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_scanned = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        # 累计统计
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        # episode结束
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # 获取扫描信息
            if 'infos' in self.locals and len(self.locals['infos']) > 0:
                info = self.locals['infos'][0]
                self.episode_scanned.append(info.get('scanned_cells', 0))
            
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        # 定期打印
        if self.num_timesteps % self.print_freq == 0:
            progress = (self.num_timesteps / self.total_timesteps) * 100
            
            print(f"\n{'=' * 60}")
            print(f"进度: {progress:.1f}% ({self.num_timesteps}/{self.total_timesteps})")
            
            if len(self.episode_rewards) > 0:
                recent_n = min(10, len(self.episode_rewards))
                avg_reward = np.mean(self.episode_rewards[-recent_n:])
                avg_length = np.mean(self.episode_lengths[-recent_n:])
                
                print(f"最近{recent_n}个episodes:")
                print(f"  平均奖励: {avg_reward:.2f}")
                print(f"  平均步数: {avg_length:.1f}")
                
                if len(self.episode_scanned) > 0:
                    avg_scanned = np.mean(self.episode_scanned[-recent_n:])
                    print(f"  平均扫描: {avg_scanned:.1f}个单元格")
            
            print(f"{'=' * 60}")
        
        return True

# 检查点回调
checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path=model_dir,
    name_prefix='movement_dqn_airsim'
)

# 进度回调
progress_callback = AirSimProgressCallback(
    total_timesteps=config['training']['total_timesteps'],
    print_freq=500
)

print(f"  ✓ 回调设置完成")

print("\n" + "=" * 80)
print("[步骤6] 开始训练")
print("=" * 80)

total_timesteps = config['training']['total_timesteps']
print(f"训练步数: {total_timesteps}")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\n⚠ 请确保Unity客户端已连接到服务器")
print(f"按 Ctrl+C 可以随时中断训练并保存模型\n")

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
    print("\n\n训练被用户中断")
    print("正在保存当前模型...")
except Exception as e:
    print(f"\n\n✗ 训练出错: {str(e)}")
    import traceback
    traceback.print_exc()
    print("正在保存当前模型...")

print("\n" + "=" * 80)
print("[步骤7] 保存最终模型")
print("=" * 80)

# 保存模型
final_model_path = os.path.join(model_dir, 'movement_dqn_airsim_final')
model.save(final_model_path)
print(f"  ✓ 模型已保存: {final_model_path}.zip")

print("\n" + "=" * 80)
print("[步骤8] 清理")
print("=" * 80)

# 停止服务器
print(f"  正在停止服务器...")
server.stop()
time.sleep(1)
print(f"  ✓ 服务器已停止")

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

