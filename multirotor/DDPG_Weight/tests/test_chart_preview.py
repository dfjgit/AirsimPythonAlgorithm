"""
测试训练可视化器的图表生成和预览功能

演示如何使用 TrainingVisualizer.generate_training_charts() 方法
"""
import sys
import os
import numpy as np

# 添加项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from multirotor.DDPG_Weight.training_visualizer import TrainingVisualizer


def test_chart_generation():
    """测试图表生成功能"""
    print("=" * 60)
    print("训练可视化器 - 图表预览功能测试")
    print("=" * 60)
    
    # 创建可视化器实例（不需要 server 和 env）
    visualizer = TrainingVisualizer(server=None, env=None)
    
    # 模拟训练数据
    print("\n📝 正在生成模拟训练数据...")
    
    # 模拟 30 个 episode 的训练过程
    for episode in range(30):
        # 模拟每个 episode 的步数和奖励
        episode_steps = np.random.randint(20, 50)
        base_reward = -100 + episode * 3  # 奖励逐渐提升
        episode_reward = base_reward + np.random.randn() * 10
        
        # 模拟每步的奖励更新
        for step in range(episode_steps):
            step_reward = np.random.randn() * 2
            visualizer.update_training_stats(
                current_step_reward=step_reward,
                is_episode_done=False
            )
        
        # Episode 结束
        visualizer.update_training_stats(
            episode_reward=episode_reward,
            episode_length=episode_steps,
            is_episode_done=True
        )
        
        # 模拟权重更新
        if episode % 3 == 0:  # 每3个episode更新一次权重
            weights = {
                'repulsionCoefficient': 1.0 + np.random.randn() * 0.2,
                'entropyCoefficient': 2.0 + np.random.randn() * 0.3,
                'distanceCoefficient': 1.5 + np.random.randn() * 0.25,
                'leaderRangeCoefficient': 0.8 + np.random.randn() * 0.15,
                'directionRetentionCoefficient': 0.6 + np.random.randn() * 0.1
            }
            visualizer.update_weight_history(weights)
    
    print(f"✅ 已生成 {visualizer.episode_count} 个 episode 的模拟数据")
    print(f"   总步数: {visualizer.total_steps}")
    print(f"   奖励历史: {len(visualizer.reward_history)} 条")
    print(f"   权重更新: {len(visualizer.weight_history['repulsionCoefficient'])} 次")
    
    # 测试不同模式
    print("\n" + "=" * 60)
    print("【模式 1】预览后手动确认是否保存（推荐）")
    print("=" * 60)
    
    saved_files = visualizer.generate_training_charts(
        preview_before_save=True,   # 显示预览窗口
        auto_save=False              # 需要用户确认
    )
    
    if saved_files:
        print(f"\n✅ 成功保存 {len(saved_files)} 个文件")
        for f in saved_files:
            print(f"   📁 {f}")
    else:
        print("\nℹ️  未保存文件")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


def test_auto_save_mode():
    """测试自动保存模式（预览后自动保存）"""
    print("\n" + "=" * 60)
    print("【模式 2】预览后自动保存")
    print("=" * 60)
    
    visualizer = TrainingVisualizer(server=None, env=None)
    
    # 生成少量测试数据
    for episode in range(10):
        episode_steps = 25
        episode_reward = -50 + episode * 5 + np.random.randn() * 5
        
        for step in range(episode_steps):
            visualizer.update_training_stats(current_step_reward=np.random.randn())
        
        visualizer.update_training_stats(
            episode_reward=episode_reward,
            episode_length=episode_steps,
            is_episode_done=True
        )
    
    saved_files = visualizer.generate_training_charts(
        preview_before_save=True,   # 显示预览
        auto_save=True               # 自动保存
    )
    
    print(f"\n✅ 已自动保存文件")


def test_no_preview_mode():
    """测试无预览直接保存模式"""
    print("\n" + "=" * 60)
    print("【模式 3】不预览直接保存")
    print("=" * 60)
    
    visualizer = TrainingVisualizer(server=None, env=None)
    
    # 生成少量测试数据
    for episode in range(10):
        episode_steps = 25
        episode_reward = -50 + episode * 5
        
        for step in range(episode_steps):
            visualizer.update_training_stats(current_step_reward=np.random.randn())
        
        visualizer.update_training_stats(
            episode_reward=episode_reward,
            episode_length=episode_steps,
            is_episode_done=True
        )
    
    saved_files = visualizer.generate_training_charts(
        preview_before_save=False,  # 不预览
        auto_save=False              # 此参数无效，因为不预览就直接保存
    )
    
    print(f"\n✅ 已直接保存文件（无预览）")


if __name__ == "__main__":
    try:
        # 主测试：预览后手动确认
        test_chart_generation()
        
        # 如果用户想测试其他模式，可以取消下面的注释
        # test_auto_save_mode()
        # test_no_preview_mode()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
