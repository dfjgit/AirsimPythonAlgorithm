"""
测试所有训练脚本的导入是否正常
"""
import os
import sys

# 添加项目路径
# scripts -> DQN_Movement -> multirotor -> 项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# 添加 multirotor 目录
multirotor_dir = os.path.join(project_root, 'multirotor')
sys.path.insert(0, multirotor_dir)

# 添加 DQN_Movement 目录
dqn_movement_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, dqn_movement_dir)

print("=" * 80)
print("测试导入模块")
print("=" * 80)

try:
    print("\n1. 测试基础库...")
    from stable_baselines3 import DQN
    import gymnasium
    print("   ✓ stable_baselines3 和 gymnasium 导入成功")
except ImportError as e:
    print(f"   ✗ 错误: {e}")
    sys.exit(1)

try:
    print("\n2. 测试环境模块...")
    from envs.movement_env import MovementEnv, MultiDroneMovementEnv
    print("   ✓ movement_env 导入成功")
    
    from envs.hierarchical_movement_env import HierarchicalMovementEnv, MultiDroneHierarchicalMovementEnv
    print("   ✓ hierarchical_movement_env 导入成功")
except ImportError as e:
    print(f"   ✗ 错误: {e}")
    sys.exit(1)

try:
    print("\n3. 测试服务器模块...")
    from AlgorithmServer import MultiDroneAlgorithmServer
    print("   ✓ AlgorithmServer 导入成功")
    
    from Algorithm.drones_config import DronesConfig
    print("   ✓ drones_config 导入成功")
except ImportError as e:
    print(f"   ✗ 错误: {e}")
    sys.exit(1)

try:
    print("\n4. 测试可视化模块...")
    from visualizers.hierarchical_visualizer import HierarchicalVisualizer
    print("   ✓ hierarchical_visualizer 导入成功")
except ImportError as e:
    print(f"   ✗ 错误: {e}")
    sys.exit(1)

try:
    print("\n5. 测试 airsim 模块...")
    import airsim
    print("   ✓ airsim 导入成功")
except ImportError as e:
    print(f"   ✗ 错误: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ 所有模块导入成功！")
print("=" * 80)
print("\n项目路径设置：")
print(f"  - 项目根目录: {project_root}")
print(f"  - multirotor 目录: {multirotor_dir}")
print(f"  - DQN_Movement 目录: {dqn_movement_dir}")
