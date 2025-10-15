"""
测试DQN移动模型
加载训练好的DQN模型并测试无人机移动策略
"""
import os
import sys
import numpy as np
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

print("=" * 80)
print("测试DQN移动模型")
print("=" * 80)

# 检查依赖
print("\n[步骤1] 检查依赖...")
try:
    from stable_baselines3 import DQN
    import gymnasium
    print(f"  ✓ Stable-Baselines3已安装")
except ImportError:
    print("  ✗ Stable-Baselines3未安装")
    print("    安装命令: pip install stable-baselines3 gymnasium")
    sys.exit(1)

# 导入环境
from movement_env import MovementEnv

print("\n" + "=" * 80)
print("[步骤2] 加载模型")
print("=" * 80)

# 模型路径
model_dir = os.path.join(os.path.dirname(__file__), 'models')
model_files = [
    'movement_dqn_final.zip',
    'movement_dqn_checkpoint_100000_steps.zip',
    'movement_dqn_checkpoint_50000_steps.zip'
]

# 查找可用模型
model_path = None
for model_file in model_files:
    test_path = os.path.join(model_dir, model_file)
    if os.path.exists(test_path):
        model_path = test_path
        break

if model_path is None:
    print("  ✗ 未找到训练好的模型")
    print(f"  请先运行训练: python train_movement_dqn.py")
    print(f"  预期模型位置: {model_dir}")
    sys.exit(1)

print(f"  ✓ 找到模型: {model_path}")

# 加载模型
try:
    model = DQN.load(model_path)
    print(f"  ✓ 模型加载成功")
except Exception as e:
    print(f"  ✗ 模型加载失败: {str(e)}")
    sys.exit(1)

print("\n" + "=" * 80)
print("[步骤3] 创建测试环境")
print("=" * 80)

# 创建环境（无server，测试模式）
env = MovementEnv(server=None, drone_name="UAV1")
print(f"  ✓ 测试环境创建成功")

print("\n" + "=" * 80)
print("[步骤4] 运行测试Episodes")
print("=" * 80)

n_test_episodes = 5
print(f"测试episodes数量: {n_test_episodes}")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# 统计信息
episode_rewards = []
episode_lengths = []
episode_scanned = []

action_names = ['上', '下', '左', '右', '前', '后']

for episode in range(n_test_episodes):
    print(f"\n{'=' * 60}")
    print(f"Episode {episode + 1}/{n_test_episodes}")
    print(f"{'=' * 60}")
    
    obs, info = env.reset()
    done = False
    episode_reward = 0
    episode_length = 0
    actions_taken = {i: 0 for i in range(6)}
    
    while not done:
        # 使用确定性策略
        action, _states = model.predict(obs, deterministic=True)
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 统计
        episode_reward += reward
        episode_length += 1
        actions_taken[action] += 1
        
        # 显示关键信息（每10步或最后一步）
        if episode_length % 10 == 0 or done:
            print(f"  步骤 {episode_length}: 动作={action_names[action]}, "
                  f"奖励={reward:.2f}, 累计奖励={episode_reward:.2f}, "
                  f"已扫描={info['scanned_cells']}")
        
        # 防止无限循环
        if episode_length >= 1000:
            print(f"  ⚠ 达到最大步数限制，强制结束")
            break
    
    # Episode统计
    episode_rewards.append(episode_reward)
    episode_lengths.append(episode_length)
    episode_scanned.append(info['scanned_cells'])
    
    print(f"\nEpisode {episode + 1} 结果:")
    print(f"  总奖励: {episode_reward:.2f}")
    print(f"  总步数: {episode_length}")
    print(f"  已扫描单元格: {info['scanned_cells']}")
    print(f"  碰撞次数: {info['collision_count']}")
    print(f"  越界次数: {info['out_of_range_count']}")
    print(f"  动作分布:")
    for action, count in actions_taken.items():
        percentage = (count / episode_length * 100) if episode_length > 0 else 0
        print(f"    {action_names[action]}: {count}次 ({percentage:.1f}%)")

print("\n" + "=" * 80)
print("测试总结")
print("=" * 80)

print(f"\n平均统计 ({n_test_episodes} episodes):")
print(f"  平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
print(f"  平均步数: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
print(f"  平均扫描: {np.mean(episode_scanned):.1f} ± {np.std(episode_scanned):.1f}")

print(f"\n详细结果:")
for i in range(n_test_episodes):
    print(f"  Episode {i+1}: 奖励={episode_rewards[i]:.2f}, "
          f"步数={episode_lengths[i]}, 扫描={episode_scanned[i]}")

print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\n" + "=" * 80)
print("✓ 测试完成")
print("=" * 80)

# 性能评估
avg_reward = np.mean(episode_rewards)
print(f"\n性能评估:")
if avg_reward > 100:
    print(f"  🌟 优秀！平均奖励 {avg_reward:.2f} > 100")
elif avg_reward > 0:
    print(f"  ✓ 良好。平均奖励 {avg_reward:.2f} > 0")
elif avg_reward > -100:
    print(f"  ⚠ 一般。平均奖励 {avg_reward:.2f}，模型可能需要更多训练")
else:
    print(f"  ✗ 较差。平均奖励 {avg_reward:.2f}，建议重新训练")

print("\n下一步:")
print(f"  1. 查看更多测试: 修改 n_test_episodes 变量")
print(f"  2. 可视化测试: 添加可视化代码")
print(f"  3. 与AirSim集成: 使用真实环境测试")
print("=" * 80)

