"""
简单的DDPG训练脚本
训练APF权重预测模型
"""
import os
import sys
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

print("=" * 60)
print("简单DDPG训练 - APF权重学习")
print("=" * 60)

# 检查依赖
print("\n检查依赖...")
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
except ImportError:
    print("✗ PyTorch未安装")
    print("  安装: pip install torch")
    sys.exit(1)

try:
    from stable_baselines3 import DDPG
    from stable_baselines3.common.noise import NormalActionNoise
    print(f"✓ Stable-Baselines3已安装")
except ImportError:
    print("✗ Stable-Baselines3未安装")
    print("  安装: pip install stable-baselines3")
    sys.exit(1)

# 导入环境
from envs.simple_weight_env import SimpleWeightEnv

print("\n" + "=" * 60)
print("步骤1: 创建训练环境")
print("=" * 60)

# 创建环境（无server，使用模拟数据）
# reset_unity参数对离线训练无效（没有Unity）
env = SimpleWeightEnv(server=None, drone_name="UAV1", reset_unity=False)

print(f"✓ 环境创建成功")
print(f"  观察空间: {env.observation_space.shape}")
print(f"  动作空间: {env.action_space.shape}")

print("\n" + "=" * 60)
print("步骤2: 创建DDPG模型")
print("=" * 60)

# 添加动作噪声（用于探索）
n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions),
    sigma=0.5 * np.ones(n_actions)  # 权重噪声
)

# 创建DDPG模型
model = DDPG(
    "MlpPolicy",
    env,
    action_noise=action_noise,
    learning_rate=1e-3,
    buffer_size=10000,
    learning_starts=100,
    batch_size=64,
    tau=0.005,
    gamma=0.99,
    train_freq=(1, "episode"),
    gradient_steps=-1,
    verbose=1
)

print(f"✓ DDPG模型创建成功")
print(f"  学习率: {model.learning_rate}")
print(f"  批次大小: {model.batch_size}")
print(f"  缓冲区大小: {model.buffer_size}")

print("\n" + "=" * 60)
print("步骤3: 开始训练")
print("=" * 60)

# 训练参数
total_timesteps = 200000  # 总步数
print(f"训练步数: {total_timesteps}")
print(f"预计时间: 约{total_timesteps // 1000}分钟\n")

# 自定义回调：显示简单进度条
from stable_baselines3.common.callbacks import BaseCallback

class SimpleProgressCallback(BaseCallback):
    """简单的进度显示回调"""
    
    def __init__(self, total_timesteps, verbose=0):
        super(SimpleProgressCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.last_print_step = 0
        self.print_interval = total_timesteps // 20  # 显示20次
        
    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_print_step >= self.print_interval:
            progress = (self.num_timesteps / self.total_timesteps) * 100
            bar_length = 40
            filled = int(bar_length * self.num_timesteps / self.total_timesteps)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            print(f"\r进度: [{bar}] {progress:.1f}% ({self.num_timesteps}/{self.total_timesteps})", end='', flush=True)
            self.last_print_step = self.num_timesteps
        
        return True
    
    def _on_training_end(self) -> None:
        print()  # 换行

# 创建回调
progress_callback = SimpleProgressCallback(total_timesteps)

# 开始训练
print("开始训练...\n")

try:
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=10,
        callback=progress_callback
    )
    print("\n✓ 训练完成！")
    
except KeyboardInterrupt:
    print("\n训练被中断")
except Exception as e:
    print(f"\n✗ 训练出错: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("步骤4: 保存模型")
print("=" * 60)

# 创建模型目录
model_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(model_dir, exist_ok=True)

# 保存模型
model_path = os.path.join(model_dir, 'weight_predictor_simple')
model.save(model_path)
print(f"✓ 模型已保存: {model_path}.zip")

print("\n" + "=" * 60)
print("步骤5: 测试模型")
print("=" * 60)

# 测试模型
obs = env.reset()
print(f"测试状态: {obs[:5]}...")

for i in range(5):
    action, _states = model.predict(obs, deterministic=True)
    print(f"\n测试 {i+1}:")
    print(f"  预测权重: α1={action[0]:.2f}, α2={action[1]:.2f}, α3={action[2]:.2f}, α4={action[3]:.2f}, α5={action[4]:.2f}")
    
    obs, reward, done, info = env.step(action)
    print(f"  奖励: {reward:.2f}")
    
    if done:
        obs = env.reset()

print("\n" + "=" * 60)
print("训练完成总结")
print("=" * 60)
print(f"✓ 模型文件: {model_path}.zip")
print(f"✓ 下一步: 使用 test_trained_model.py 测试模型")
print("=" * 60)

