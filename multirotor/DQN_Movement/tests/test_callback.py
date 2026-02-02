"""
测试 DQN 训练回调函数是否能正确检测 episode 结束
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym

print("=" * 80)
print("测试 DQN 回调函数 - Episode 结束检测")
print("=" * 80)

# 创建一个简单的测试环境
env = gym.make('CartPole-v1')

# 创建测试回调
class TestCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_count = 0
        self.step_count = 0
    
    def _on_step(self) -> bool:
        self.step_count += 1
        
        # 打印首次调用时的键
        if self.step_count == 1:
            print(f"\n[首次调用] self.locals 的键: {list(self.locals.keys())}")
            if 'dones' in self.locals:
                print(f"  - dones: {self.locals['dones']}, 类型: {type(self.locals['dones'])}")
            if 'terminations' in self.locals:
                print(f"  - terminations: {self.locals['terminations']}, 类型: {type(self.locals['terminations'])}")
            if 'infos' in self.locals and len(self.locals['infos']) > 0:
                print(f"  - infos[0] 的键: {list(self.locals['infos'][0].keys())}")
        
        # 检测 episode 结束
        is_done = False
        if 'dones' in self.locals and len(self.locals['dones']) > 0:
            is_done = bool(self.locals['dones'][0])
        elif 'terminations' in self.locals and len(self.locals['terminations']) > 0:
            is_done = bool(self.locals['terminations'][0])
        
        if is_done:
            self.episode_count += 1
            print(f"\n[Episode {self.episode_count} 结束] 总步数: {self.step_count}")
        
        return True

# 创建 DQN 模型
print("\n创建 DQN 模型...")
model = DQN("MlpPolicy", env, verbose=0)

# 创建回调
callback = TestCallback()

print("\n开始训练 1000 步...")
model.learn(total_timesteps=1000, callback=callback)

print(f"\n训练完成！")
print(f"  总步数: {callback.step_count}")
print(f"  完成 episodes: {callback.episode_count}")

if callback.episode_count > 0:
    print(f"\n✅ 回调函数能够正确检测 episode 结束！")
else:
    print(f"\n❌ 回调函数无法检测 episode 结束，需要进一步调试。")

env.close()
print("=" * 80)
