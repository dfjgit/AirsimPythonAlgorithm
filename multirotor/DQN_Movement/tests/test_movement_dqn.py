"""
测试 DQN 移动模型。

优先从 DQN_Movement/models 读取新模型，同时兼容旧的 scripts/models 目录。
"""
import os
import sys
from datetime import datetime

import numpy as np


# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

print("=" * 80)
print("测试 DQN 移动模型")
print("=" * 80)

print("\n[步骤1] 检查依赖...")
try:
    from stable_baselines3 import DQN
    import gymnasium  # noqa: F401
    print("  ✓ Stable-Baselines3 已安装")
except ImportError:
    print("  ✗ Stable-Baselines3 未安装")
    print("    安装命令: pip install stable-baselines3 gymnasium")
    sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.movement_env import MovementEnv

print("\n" + "=" * 80)
print("[步骤2] 加载模型")
print("=" * 80)

dqn_movement_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dirs = [
    os.path.join(dqn_movement_dir, "models"),
    os.path.join(dqn_movement_dir, "scripts", "models"),
    os.path.join(os.path.dirname(__file__), "models"),
]
model_files = [
    "movement_dqn_airsim_final.zip",
    "movement_dqn_final.zip",
    "movement_dqn_airsim_checkpoint_100000_steps.zip",
    "movement_dqn_airsim_checkpoint_50000_steps.zip",
    "movement_dqn_checkpoint_100000_steps.zip",
    "movement_dqn_checkpoint_50000_steps.zip",
]

model_path = None
searched_paths = []
for model_dir in model_dirs:
    for model_file in model_files:
        test_path = os.path.join(model_dir, model_file)
        searched_paths.append(test_path)
        if os.path.exists(test_path):
            model_path = test_path
            break
    if model_path is not None:
        break

if model_path is None:
    print("  ✗ 未找到训练好的模型")
    print("  请先运行训练: python multirotor\\DQN_Movement\\scripts\\train_movement_dqn.py")
    print(f"  已搜索路径数: {len(searched_paths)}")
    for searched_path in searched_paths:
        print(f"    - {searched_path}")
    sys.exit(1)

print(f"  ✓ 找到模型: {model_path}")

try:
    model = DQN.load(model_path)
    print("  ✓ 模型加载成功")
except Exception as exc:
    print(f"  ✗ 模型加载失败: {exc}")
    sys.exit(1)

print("\n" + "=" * 80)
print("[步骤3] 创建测试环境")
print("=" * 80)

env = MovementEnv(server=None, drone_name="UAV1")
print("  ✓ 测试环境创建成功")

print("\n" + "=" * 80)
print("[步骤4] 运行测试 Episodes")
print("=" * 80)

n_test_episodes = 5
print(f"测试 episodes 数量: {n_test_episodes}")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

episode_rewards = []
episode_lengths = []
episode_scanned = []

action_names = ["上升", "下降", "左移", "右移", "前进", "后退"]

for episode in range(n_test_episodes):
    print(f"\n{'=' * 60}")
    print(f"Episode {episode + 1}/{n_test_episodes}")
    print(f"{'=' * 60}")

    obs, info = env.reset()
    done = False
    episode_reward = 0.0
    episode_length = 0
    actions_taken = {i: 0 for i in range(6)}

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_reward += reward
        episode_length += 1
        actions_taken[int(action)] += 1

        if episode_length % 10 == 0 or done:
            print(
                f"  步数 {episode_length}: 动作={action_names[int(action)]}, "
                f"奖励={reward:.2f}, 累计奖励={episode_reward:.2f}, "
                f"已扫描={info['scanned_cells']}"
            )

        if episode_length >= 1000:
            print("  ! 达到最大步数限制，强制结束")
            break

    episode_rewards.append(episode_reward)
    episode_lengths.append(episode_length)
    episode_scanned.append(info["scanned_cells"])

    print(f"\nEpisode {episode + 1} 结果:")
    print(f"  总奖励: {episode_reward:.2f}")
    print(f"  总步数: {episode_length}")
    print(f"  已扫描单元格: {info['scanned_cells']}")
    print(f"  碰撞次数: {info['collision_count']}")
    print(f"  越界次数: {info['out_of_range_count']}")
    print("  动作分布:")
    for action, count in actions_taken.items():
        percentage = (count / episode_length * 100) if episode_length > 0 else 0.0
        print(f"    {action_names[action]}: {count} 次 ({percentage:.1f}%)")

print("\n" + "=" * 80)
print("测试总结")
print("=" * 80)

print(f"\n平均统计 ({n_test_episodes} episodes):")
print(f"  平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
print(f"  平均步数: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
print(f"  平均扫描: {np.mean(episode_scanned):.1f} ± {np.std(episode_scanned):.1f}")

print("\n详细结果:")
for i in range(n_test_episodes):
    print(
        f"  Episode {i + 1}: 奖励={episode_rewards[i]:.2f}, "
        f"步数={episode_lengths[i]}, 扫描={episode_scanned[i]}"
    )

print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\n" + "=" * 80)
print("✓ 测试完成")
print("=" * 80)

avg_reward = np.mean(episode_rewards)
print("\n性能评估:")
if avg_reward > 100:
    print(f"  优秀，平均奖励 {avg_reward:.2f} > 100")
elif avg_reward > 0:
    print(f"  良好，平均奖励 {avg_reward:.2f} > 0")
elif avg_reward > -100:
    print(f"  一般，平均奖励 {avg_reward:.2f}，模型可能还需要更多训练")
else:
    print(f"  较差，平均奖励 {avg_reward:.2f}，建议重新训练或继续优化")

print("\n下一步:")
print("  1. 增加 n_test_episodes 查看更多测试结果")
print("  2. 对接 AirSim 环境测试真实控制效果")
print("=" * 80)
