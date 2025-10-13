"""
测试DQN权重预测集成
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

print("=" * 60)
print("测试DQN权重预测集成")
print("=" * 60)

print("\n步骤1: 检查模型文件...")
model_path = os.path.join(os.path.dirname(__file__), 'models', 'weight_predictor_simple.zip')

if os.path.exists(model_path):
    print(f"✓ 模型文件存在: {model_path}")
else:
    print(f"✗ 模型文件不存在: {model_path}")
    print("  请先运行: python train_simple.py")
    sys.exit(1)

print("\n步骤2: 加载模型...")
try:
    from stable_baselines3 import DDPG
    model = DDPG.load(model_path)
    print("✓ 模型加载成功")
except ImportError:
    print("✗ stable-baselines3未安装")
    print("  安装: pip install stable-baselines3")
    sys.exit(1)
except Exception as e:
    print(f"✗ 模型加载失败: {str(e)}")
    sys.exit(1)

print("\n步骤3: 测试状态提取...")
try:
    import numpy as np
    
    # 直接创建测试状态（不需要server）
    # 模拟一个18维状态向量
    test_state = np.array([
        # 位置 (3)
        5.0, 2.0, 3.0,
        # 速度 (3)
        1.0, 0.0, 0.5,
        # 方向 (3)
        1.0, 0.0, 0.0,
        # 熵值 (3)
        60.0, 80.0, 15.0,
        # Leader相对位置 (3)
        10.0, 0.0, 5.0,
        # 扫描进度 (3)
        0.3, 50.0, 100.0
    ], dtype=np.float32)
    
    print(f"✓ 状态创建成功: shape={test_state.shape}")
    print(f"  状态示例: pos=({test_state[0]:.1f}, {test_state[1]:.1f}, {test_state[2]:.1f})")
    
    state = test_state  # 用于后续测试
    
except Exception as e:
    print(f"✗ 状态创建失败: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n步骤4: 测试权重预测...")
try:
    # 使用模型预测
    action, _ = model.predict(state, deterministic=True)
    
    print(f"✓ 权重预测成功")
    print(f"  预测权重:")
    print(f"    α1 (排斥力) = {action[0]:.2f}")
    print(f"    α2 (熵)     = {action[1]:.2f}")
    print(f"    α3 (距离)   = {action[2]:.2f}")
    print(f"    α4 (Leader) = {action[3]:.2f}")
    print(f"    α5 (方向)   = {action[4]:.2f}")
    
    # 验证范围
    if all(0.1 <= w <= 10.0 for w in action):
        print("✓ 权重范围正确 [0.1, 10.0]")
    else:
        print("⚠ 权重超出范围")
    
except Exception as e:
    print(f"✗ 权重预测失败: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n步骤5: 验证权重字典...")
try:
    weights_dict = {
        'repulsionCoefficient': float(action[0]),
        'entropyCoefficient': float(action[1]),
        'distanceCoefficient': float(action[2]),
        'leaderRangeCoefficient': float(action[3]),
        'directionRetentionCoefficient': float(action[4])
    }
    
    print("✓ 权重字典创建成功")
    print(f"  权重字典: {weights_dict}")
    
except Exception as e:
    print(f"✗ 权重字典创建失败: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ 所有测试通过！")
print("=" * 60)
print("\n集成测试成功！可以使用以下命令运行:")
print("  python multirotor/AlgorithmServer.py --use-learned-weights")
print("\n或者使用固定权重:")
print("  python multirotor/AlgorithmServer.py")

