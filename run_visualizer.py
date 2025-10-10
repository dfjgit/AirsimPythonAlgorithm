#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
无人机可视化程序启动器
此脚本用于启动无人机环境可视化程序，确保正确设置Python路径
"""

import os
import sys

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 添加项目根目录到Python路径
sys.path.append(script_dir)

try:
    # 导入并运行可视化程序
    from multirotor.DQN.drone_visualizer import DroneVisualizer
    
    print("正在启动无人机环境可视化程序...")
    print("使用说明:")
    print("- 点击UI面板中的参数可以使用上下箭头调整数值")
    print("- 鼠标点击并拖拽可以移动无人机")
    print("- 按R键可以重置所有位置")
    print("- 按ESC键退出程序")
    
    # 启动可视化程序
    visualizer = DroneVisualizer()
    visualizer.run()
    
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保项目结构正确，且已安装所需依赖")
    print("可以使用以下命令安装依赖:")
    print("pip install pygame")
    sys.exit(1)
except Exception as e:
    print(f"程序运行错误: {e}")
    sys.exit(1)