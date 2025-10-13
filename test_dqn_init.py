"""
快速测试DQN初始化是否会卡住
"""
import os
import sys
import time

# 设置环境变量（在导入PyTorch之前）
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

print("=" * 60)
print("DQN初始化测试")
print("=" * 60)

print("\n步骤1: 导入基础模块...")
start = time.time()
import numpy as np
print(f"✓ 导入numpy成功 ({time.time() - start:.2f}秒)")

print("\n步骤2: 导入PyTorch...")
start = time.time()
import torch
print(f"✓ 导入torch成功 ({time.time() - start:.2f}秒)")
print(f"  PyTorch版本: {torch.__version__}")
print(f"  CUDA可用: {torch.cuda.is_available()}")
print(f"  当前设备: cpu")

print("\n步骤3: 设置PyTorch配置...")
start = time.time()
torch.set_num_threads(2)
print(f"✓ 设置线程数为2 ({time.time() - start:.2f}秒)")

print("\n步骤4: 导入DQN模块...")
start = time.time()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from multirotor.DQN.DqnLearning import DQNAgent, ReplayBuffer
print(f"✓ 导入DQNAgent成功 ({time.time() - start:.2f}秒)")

print("\n步骤5: 创建DQN智能体...")
start = time.time()
agent = DQNAgent(
    state_dim=18,
    action_dim=25,
    lr=0.001,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    batch_size=64,
    target_update=10,
    memory_capacity=10000
)
print(f"✓ 创建DQN智能体成功 ({time.time() - start:.2f}秒)")

print("\n步骤6: 测试动作选择...")
start = time.time()
state = np.random.randn(18).astype(np.float32)
action = agent.select_action(state)
print(f"✓ 选择动作成功: {action} ({time.time() - start:.2f}秒)")

print("\n步骤7: 测试经验存储...")
start = time.time()
for i in range(10):
    state = np.random.randn(18).astype(np.float32)
    action = np.random.randint(0, 25)
    reward = np.random.randn()
    next_state = np.random.randn(18).astype(np.float32)
    done = False
    agent.memory.push(state, action, reward, next_state, done)
print(f"✓ 存储10条经验成功 ({time.time() - start:.2f}秒)")
print(f"  缓冲区大小: {len(agent.memory)}")

print("\n" + "=" * 60)
print("✓ 所有测试通过！DQN可以正常初始化和使用。")
print("=" * 60)

print("\n建议:")
print("1. 如果测试通过但主程序仍然卡住，请禁用DQN:")
print("   在 scanner_config.json 中设置: \"dqn\": { \"enabled\": false }")
print("\n2. 如果测试在某个步骤卡住，记录卡住的步骤并报告")
print("\n3. 对于AMD 6800H无独显笔记本，推荐禁用DQN以获得最佳性能")

