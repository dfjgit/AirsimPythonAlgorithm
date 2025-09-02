#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试DroneServer功能的简单脚本"""

import os
import sys
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
try:
    from multirotor.setup_path import SetupPath
    SetupPath.addAirSimModulePath()
    import airsim
    print("✓ AirSim模块导入成功")
except ImportError as e:
    print(f"✗ 导入AirSim模块失败: {e}")
    sys.exit(1)

try:
    # 检查依赖项
    import msgpack
    import numpy
    import cv2
    import tornado
    print("✓ 所有必要的依赖项都已安装")
except ImportError as e:
    print(f"✗ 缺少依赖项: {e}")
    sys.exit(1)

def test_connection():
    """测试与AirSim模拟器的连接"""
    print("\n正在测试与AirSim模拟器的连接...")
    try:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("✓ 成功连接到AirSim模拟器")
        # 获取无人机状态信息
        state = client.getMultirotorState()
        print(f"  无人机状态: {state.landed_state}")
        print(f"  位置: {state.kinematics_estimated.position}")
        return True
    except Exception as e:
        print(f"✗ 连接AirSim模拟器失败: {e}")
        print("  注意: 如果您没有运行AirSim模拟器，这个错误是正常的")
        return False

def main():
    print("=== AirSim DroneServer 测试脚本 ===")
    
    # 检查项目结构
    print("\n正在检查项目结构...")
    required_files = [
        "multirotor/DroneServer.py",
        "multirotor/components/drone_controller.py",
        "multirotor/components/command_processor.py",
        "multirotor/components/data_storage.py",
        "multirotor/components/socket_server.py",
        "multirotor/components/unity_environment.py",
        "multirotor/server_api.md",
        "requirements.txt",
        "setup.py",
        "install_offline.bat"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("✗ 缺少以下必要文件:")
        for file in missing_files:
            print(f"  - {file}")
    else:
        print("✓ 所有必要文件都存在")
    
    # 测试连接
    test_connection()
    
    print("\n=== 测试完成 ===")
    print("如需启动DroneServer，请运行:")
    print("cd multirotor && python DroneServer.py")

if __name__ == "__main__":
    main()