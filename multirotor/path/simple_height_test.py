#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的高度测试脚本
用于手动排查高度问题
"""

import sys
import os
import json
import time
import math

# 添加项目路径
current_dir = os.path.dirname(__file__)
multirotor_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(multirotor_dir)
sys.path.append(project_dir)

import airsim

def test_height():
    """简单的高度测试"""
    
    # 1. 连接
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("✓ 已连接到AirSim\n")
    
    # 2. 重置
    client.reset()
    time.sleep(1)
    
    # 3. 启用控制
    client.enableApiControl(True, "UAV1")
    client.armDisarm(True, "UAV1")
    print("✓ 无人机已解锁\n")
    
    # 4. 记录地面位置
    state = client.getMultirotorState(vehicle_name="UAV1")
    ground_pos = state.kinematics_estimated.position
    ground_z = ground_pos.z_val
    print(f"【地面位置】")
    print(f"  NED坐标: X={ground_pos.x_val:.4f}, Y={ground_pos.y_val:.4f}, Z={ground_z:.4f}")
    print()
    
    # 5. 起飞
    print("【起飞】")
    client.takeoffAsync(vehicle_name="UAV1").join()
    time.sleep(2)
    
    state = client.getMultirotorState(vehicle_name="UAV1")
    takeoff_pos = state.kinematics_estimated.position
    takeoff_height = -(takeoff_pos.z_val - ground_z)
    print(f"  起飞后NED: X={takeoff_pos.x_val:.4f}, Y={takeoff_pos.y_val:.4f}, Z={takeoff_pos.z_val:.4f}")
    print(f"  离地高度: {takeoff_height:.4f}m")
    print()
    
    # 6. 读取path1的第一个点
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path1_file = os.path.join(script_dir, 'path1.json')
    with open(path1_file, 'r') as f:
        path1 = json.load(f)
    first_point = path1['1'][0]
    
    target_height = first_point['z']  # 期望的离地高度
    print(f"【Path1第一个点】")
    print(f"  X={first_point['x']:.4f}, Y={first_point['y']:.4f}, 高度={target_height:.4f}m")
    print()
    
    # 7. 测试不同的高度转换方法
    print("=" * 80)
    print("测试三种高度转换方法")
    print("=" * 80)
    
    # 方法1: 直接取负
    method1_z = -target_height
    print(f"\n【方法1: 直接取负】")
    print(f"  目标高度: {target_height:.4f}m")
    print(f"  转换后Z: {method1_z:.4f}")
    print(f"  发送指令: moveToPositionAsync(x={first_point['x']:.4f}, y={first_point['y']:.4f}, z={method1_z:.4f}, speed=1.0)")
    
    client.moveToPositionAsync(
        first_point['x'], first_point['y'], method1_z, 1.0, vehicle_name="UAV1"
    ).join()
    time.sleep(3)
    
    state = client.getMultirotorState(vehicle_name="UAV1")
    pos = state.kinematics_estimated.position
    actual_height = -(pos.z_val - ground_z)
    print(f"  到达后NED: X={pos.x_val:.4f}, Y={pos.y_val:.4f}, Z={pos.z_val:.4f}")
    print(f"  实际离地高度: {actual_height:.4f}m")
    print(f"  高度误差: {actual_height - target_height:.4f}m")
    print()
    
    # 方法2: 使用地面Z作为基准
    method2_z = ground_z - target_height
    print(f"【方法2: 使用地面Z作为基准】")
    print(f"  地面Z: {ground_z:.4f}m")
    print(f"  目标高度: {target_height:.4f}m")
    print(f"  转换后Z: ground_z - height = {ground_z:.4f} - {target_height:.4f} = {method2_z:.4f}")
    print(f"  发送指令: moveToPositionAsync(x={first_point['x']:.4f}, y={first_point['y']:.4f}, z={method2_z:.4f}, speed=1.0)")
    
    client.moveToPositionAsync(
        first_point['x'], first_point['y'], method2_z, 1.0, vehicle_name="UAV1"
    ).join()
    time.sleep(3)
    
    state = client.getMultirotorState(vehicle_name="UAV1")
    pos = state.kinematics_estimated.position
    actual_height = -(pos.z_val - ground_z)
    print(f"  到达后NED: X={pos.x_val:.4f}, Y={pos.y_val:.4f}, Z={pos.z_val:.4f}")
    print(f"  实际离地高度: {actual_height:.4f}m")
    print(f"  高度误差: {actual_height - target_height:.4f}m")
    print()
    
    # 方法3: 直接设置绝对Z（假设Z=0是某个参考点）
    method3_z = -target_height
    print(f"【方法3: 使用moveToZAsync】")
    print(f"  目标Z: {method3_z:.4f}")
    
    client.moveToZAsync(method3_z, 1.0, vehicle_name="UAV1").join()
    time.sleep(3)
    
    state = client.getMultirotorState(vehicle_name="UAV1")
    pos = state.kinematics_estimated.position
    actual_height = -(pos.z_val - ground_z)
    print(f"  到达后NED: X={pos.x_val:.4f}, Y={pos.y_val:.4f}, Z={pos.z_val:.4f}")
    print(f"  实际离地高度: {actual_height:.4f}m")
    print(f"  高度误差: {actual_height - target_height:.4f}m")
    print()
    
    print("=" * 80)
    print("测试完成")
    print("=" * 80)
    print("\n分析：")
    print(f"地面Z={ground_z:.4f}，目标高度={target_height:.4f}m")
    print(f"如果用 -target_height，则Z={-target_height:.4f}")
    print(f"如果用 ground_z - target_height，则Z={ground_z - target_height:.4f}")
    print(f"这两者差了: {ground_z:.4f}米")
    print()
    
    # 降落
    print("降落...")
    client.landAsync(vehicle_name="UAV1").join()
    client.armDisarm(False, "UAV1")
    client.enableApiControl(False, "UAV1")
    print("完成！")

if __name__ == "__main__":
    try:
        test_height()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

