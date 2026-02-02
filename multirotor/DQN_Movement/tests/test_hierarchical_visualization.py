"""
测试分层训练可视化功能
"""
import os
import sys
import time

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from envs.hierarchical_movement_env import HierarchicalMovementEnv
from visualizers.hierarchical_visualizer import HierarchicalVisualizer

def test_visualization():
    print("=" * 60)
    print("测试分层训练可视化功能")
    print("=" * 60)
    
    # 1. 创建环境（Mock模式）
    print("1. 创建环境...")
    env = HierarchicalMovementEnv(server=None, drone_name="UAV1")
    print("   ✓ 环境创建成功")
    
    # 2. 创建可视化器
    print("2. 创建可视化器...")
    visualizer = HierarchicalVisualizer(env, server=None)
    print("   ✓ 可视化器创建成功")
    
    # 3. 启动可视化
    print("3. 启动可视化...")
    visualizer.start_visualization()
    print("   ✓ 可视化已启动")
    print("   💡 可视化窗口应该已经显示")
    
    # 4. 模拟训练过程
    print("4. 模拟训练数据...")
    print("   (按 Ctrl+C 或关闭窗口退出)")
    
    try:
        step = 0
        episode = 0
        
        while visualizer.running:
            # 模拟一个训练步骤
            action = step % 25  # 循环遍历所有25个区域
            reward = (step % 10) - 5  # 模拟奖励波动
            
            # 更新可视化数据
            visualizer.update_training_data(
                step=step,
                action=action,
                reward=reward,
                drone_name="UAV1"
            )
            
            step += 1
            
            # 每100步结束一个Episode
            if step % 100 == 0:
                visualizer.on_episode_end(episode)
                episode += 1
                print(f"   Episode {episode} 完成")
            
            time.sleep(0.1)  # 控制更新速度
            
    except KeyboardInterrupt:
        print("\n   测试被用户中断")
    
    # 5. 停止可视化
    print("5. 停止可视化...")
    visualizer.stop_visualization()
    print("   ✓ 可视化已停止")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    test_visualization()
