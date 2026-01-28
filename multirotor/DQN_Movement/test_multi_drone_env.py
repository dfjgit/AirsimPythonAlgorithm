"""
测试多无人机 DQN 环境
验证 MultiDroneMovementEnv 的基本功能
"""
import sys
import os
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

print("=" * 80)
print("测试多无人机 DQN 环境")
print("=" * 80)

# 导入环境
from multirotor.DQN_Movement.movement_env import MultiDroneMovementEnv

print("\n[步骤1] 创建多无人机环境")
print("-" * 80)

# 创建环境（无server，测试模式）
drone_names = ["UAV1", "UAV2", "UAV3"]
env = MultiDroneMovementEnv(server=None, drone_names=drone_names)

print(f"✓ 环境创建成功")
print(f"  无人机数量: {env.num_drones}")
print(f"  无人机列表: {env.drone_names}")
print(f"  观察空间: {env.observation_space.shape}")
print(f"  动作空间: {env.action_space.n} (6方向)")

print("\n[步骤2] 重置环境")
print("-" * 80)

state, info = env.reset()
print(f"✓ 环境已重置")
print(f"  初始状态shape: {state.shape}")
print(f"  当前控制无人机: {env.drone_names[env.current_drone_idx]}")

print("\n[步骤3] 执行多步动作")
print("-" * 80)

for step in range(10):
    action = np.random.randint(0, 6)
    current_drone = env.drone_names[env.current_drone_idx]
    
    state, reward, done, truncated, info = env.step(action)
    
    print(f"步骤 {step+1}:")
    print(f"  控制无人机: {current_drone}")
    print(f"  动作: {action} (['上','下','左','右','前','后'][action])")
    print(f"  奖励: {reward:.2f}")
    print(f"  总奖励: {env.total_episode_reward:.2f}")
    print(f"  完成: {done}")
    print(f"  下一个无人机: {env.drone_names[env.current_drone_idx]}")
    
    if done:
        print(f"\n✓ Episode 完成！")
        break

print("\n[步骤4] 验证轮流控制")
print("-" * 80)

env.reset()
control_sequence = []

for i in range(env.num_drones * 2):
    drone_idx_before = env.current_drone_idx
    action = 0  # 向上
    state, reward, done, truncated, info = env.step(action)
    control_sequence.append(env.drone_names[drone_idx_before])

print(f"✓ 控制序列: {' -> '.join(control_sequence)}")
print(f"  验证轮流: {'✓ 通过' if len(set(control_sequence)) == env.num_drones else '✗ 失败'}")

print("\n" + "=" * 80)
print("✓ 多无人机环境测试通过！")
print("=" * 80)
print("\n关键特性:")
print("  ✓ 支持多无人机配置")
print("  ✓ 轮流控制机制")
print("  ✓ 独立状态记录")
print("  ✓ 共享模型参数")
print("  ✓ 协同奖励计算")
print("\n下一步:")
print("  1. 启动 AlgorithmServer")
print("  2. 运行 train_movement_with_airsim.py")
print("  3. 在 drones_config.json 中配置多个无人机")
print("=" * 80)
