"""
测试可视化窗口启动
"""
import sys
import os
import time

# 设置SDL视频驱动为Windows（必须在导入pygame之前）
os.environ['SDL_VIDEODRIVER'] = 'windows'

# 添加路径
sys.path.insert(0, 'multirotor')

from Visualization import HierarchicalTrainingVisualizer

print("=" * 60)
print("可视化窗口测试")
print("=" * 60)
print("\n正在启动可视化窗口...")
print("请观察是否有pygame窗口在屏幕左上角(100,100)弹出\n")

# 创建可视化器
visualizer = HierarchicalTrainingVisualizer(env=None, server=None)

# 启动可视化
visualizer.start_visualization()

print("\n可视化已启动！")
print("窗口标题: '🎯 Hierarchical DQN Training Visualization'")
print("窗口位置: 屏幕左上角(100, 100)")
print("\n如果看不到窗口:")
print("  1. 检查任务栏是否有pygame图标")
print("  2. 按Alt+Tab查看所有窗口")
print("  3. 检查是否被其他窗口遮挡")
print("\n按Ctrl+C停止测试...")

try:
    # 保持运行
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n\n正在关闭...")
    visualizer.stop_visualization()
    print("测试完成")
