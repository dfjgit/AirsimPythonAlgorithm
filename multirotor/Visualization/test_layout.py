"""
测试新的2列布局系统

这个脚本演示如何使用新的智能多列布局
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Visualization import RuntimeVisualizer
import time

def test_layout():
    """测试布局"""
    print("=" * 60)
    print("测试新的左右两侧面板布局系统")
    print("=" * 60)
    print()
    print("功能特性:")
    print("  ✓ 左右两侧面板布局")
    print("  ✓ 中间突出显示熵值热力图")
    print("  ✓ 自动均衡两侧高度")
    print("  ✓ 面板按高度优先排序")
    print()
    print("布局架构:")
    print("  左侧面板区: 360px")
    print("  中间热力图: 1200px (大幅扩展!)")
    print("  右侧面板区: 360px")
    print("  总宽度: 1920px (Full HD)")
    print("  总高度: 1080px")
    print("  热力图占比: 62.5%")
    print()
    print("启动可视化窗口...")
    print("(按ESC键关闭)")
    print("=" * 60)
    
    # 创建可视化器（会自动应用2列布局）
    visualizer = RuntimeVisualizer(server=None)
    visualizer.start_visualization()
    
    try:
        # 等待用户关闭
        while visualizer.running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        visualizer.stop_visualization()
        print("✓ 测试完成")

if __name__ == "__main__":
    test_layout()
