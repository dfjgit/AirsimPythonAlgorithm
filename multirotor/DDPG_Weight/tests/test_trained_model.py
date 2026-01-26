"""
测试训练好的权重预测模型
"""
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

print("=" * 60)
print("测试训练好的权重预测模型")
print("=" * 60)

# 导入
from stable_baselines3 import DDPG
from envs.simple_weight_env import SimpleWeightEnv

# 加载模型
model_path = os.path.join(os.path.dirname(__file__), 'models', 'weight_predictor_simple')

if not os.path.exists(model_path + '.zip'):
    print(f"✗ 模型文件不存在: {model_path}.zip")
    print("  请先运行 train_simple.py 训练模型")
    sys.exit(1)

print(f"\n加载模型: {model_path}.zip")
model = DDPG.load(model_path)
print("✓ 模型加载成功")

# 创建环境
env = SimpleWeightEnv(server=None, drone_name="UAV1")

print("\n" + "=" * 60)
print("测试场景1: 随机状态")
print("=" * 60)

for i in range(10):
    # 随机状态
    obs = env.observation_space.sample()
    
    # 预测权重
    action, _states = model.predict(obs, deterministic=True)
    
    print(f"\n测试 {i+1}:")
    print(f"  状态: pos=({obs[0]:.1f}, {obs[1]:.1f}, {obs[2]:.1f}), entropy={obs[9]:.1f}")
    print(f"  预测权重:")
    print(f"    α1 (排斥力)  = {action[0]:.2f}")
    print(f"    α2 (熵)      = {action[1]:.2f}")
    print(f"    α3 (距离)    = {action[2]:.2f}")
    print(f"    α4 (Leader)  = {action[3]:.2f}")
    print(f"    α5 (方向)    = {action[4]:.2f}")

print("\n" + "=" * 60)
print("测试场景2: 特定场景")
print("=" * 60)

# 场景1: 高熵值区域（未扫描）
print("\n场景: 高熵值区域（未扫描）")
obs = np.zeros(18, dtype=np.float32)
obs[9] = 80.0   # 平均熵值高
obs[10] = 95.0  # 最大熵值高
obs[15] = 0.1   # 扫描比例低

action, _ = model.predict(obs, deterministic=True)
print(f"预测权重: α1={action[0]:.2f}, α2={action[1]:.2f}, α3={action[2]:.2f}, α4={action[3]:.2f}, α5={action[4]:.2f}")
print(f"期望: α2(熵)应该较高，引导探索")

# 场景2: 低熵值区域（已扫描）
print("\n场景: 低熵值区域（已扫描）")
obs = np.zeros(18, dtype=np.float32)
obs[9] = 10.0   # 平均熵值低
obs[10] = 20.0  # 最大熵值低
obs[15] = 0.8   # 扫描比例高

action, _ = model.predict(obs, deterministic=True)
print(f"预测权重: α1={action[0]:.2f}, α2={action[1]:.2f}, α3={action[2]:.2f}, α4={action[3]:.2f}, α5={action[4]:.2f}")
print(f"期望: α3(距离)或α4(Leader)应该较高，引导移动")

# 场景3: 靠近其他无人机
print("\n场景: 靠近其他无人机")
obs = np.zeros(18, dtype=np.float32)
obs[0] = 5.0    # 位置
obs[11] = 2.0   # 相对Leader距离近

action, _ = model.predict(obs, deterministic=True)
print(f"预测权重: α1={action[0]:.2f}, α2={action[1]:.2f}, α3={action[2]:.2f}, α4={action[3]:.2f}, α5={action[4]:.2f}")
print(f"期望: α1(排斥力)应该较高，避免碰撞")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
print("\n✓ 模型可以正常预测权重")
print("✓ 下一步: 集成到AlgorithmServer进行实际测试")

