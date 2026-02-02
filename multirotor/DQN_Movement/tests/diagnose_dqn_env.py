"""
DQN 环境诊断脚本
用于排查 Episode 提前终止的问题
"""
import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入环境
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from envs.movement_env import MovementEnv
import numpy as np

print("="*80)
print("DQN 环境诊断")
print("="*80)

# 创建环境（不连接 server，纯测试模式）
print("\n[步骤1] 创建测试环境...")
env = MovementEnv(server=None, drone_name="UAV1")
print(f"✓ 环境创建成功")
print(f"  - 观察空间: {env.observation_space.shape}")
print(f"  - 动作空间: {env.action_space.n}")
print(f"  - max_steps: {env.config['movement']['max_steps']}")
print(f"  - success_scan_ratio: {env.config['thresholds']['success_scan_ratio']}")

# 重置环境
print("\n[步骤2] 重置环境...")
obs, info = env.reset()
print(f"✓ 环境重置完成")
print(f"  - 观察维度: {obs.shape}")
print(f"  - step_count: {env.step_count}")
print(f"  - collision_count: {env.collision_count}")
print(f"  - out_of_range_count: {env.out_of_range_count}")

# 测试多步运行
print("\n[步骤3] 测试运行 10 步...")
for i in range(10):
    action = env.action_space.sample()  # 随机动作
    print(f"\n--- Step {i+1} ---")
    print(f"  动作: {action}")
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"  奖励: {reward:.2f}")
    print(f"  terminated: {terminated}")
    print(f"  truncated: {truncated}")
    print(f"  info: {info}")
    
    if terminated or truncated:
        print(f"\n⚠️  Episode 在第 {i+1} 步结束!")
        print(f"  - 原因: {'terminated' if terminated else 'truncated'}")
        break
else:
    print(f"\n✓ 成功运行 10 步，Episode 未结束")

print("\n" + "="*80)
print("诊断完成")
print("="*80)
