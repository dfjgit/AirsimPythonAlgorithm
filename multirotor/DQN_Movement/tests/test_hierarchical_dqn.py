import os
import sys
import numpy as np

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
multirotor_dir = os.path.join(root_dir, "multirotor")
sys.path.append(root_dir)
sys.path.append(multirotor_dir)
sys.path.insert(0, os.path.join(current_dir, '..'))

from envs.hierarchical_movement_env import HierarchicalMovementEnv

def test_hrl_env():
    print("=" * 60)
    print("测试 HierarchicalMovementEnv (Mock Mode)")
    print("=" * 60)
    
    # 1. 初始化环境 (server=None)
    env = HierarchicalMovementEnv(server=None, drone_name="UAV1")
    
    obs, info = env.reset()
    print(f"初始 HL 观察空间 shape: {obs.shape}")
    
    # 2. 执行几步高层动作
    for hl_step in range(3):
        # 随机选择一个区域 (0-24)
        action = env.action_space.sample()
        print(f"\n[HL Step {hl_step+1}] 选择区域: {action}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"  HL 奖励: {reward:.2f}")
        print(f"  HL 观察前5维: {obs[:5]}")
        
        if terminated:
            print("任务完成/结束")
            break
            
    print("\n" + "=" * 60)
    print("环境 Mock 测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    test_hrl_env()
