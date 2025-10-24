#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断AirSim高度问题
"""

import sys
import os
import json

# 添加项目路径
current_dir = os.path.dirname(__file__)
multirotor_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(multirotor_dir)
sys.path.append(project_dir)

import airsim
import time

def diagnose_height_issue():
    """诊断高度问题"""
    print("=" * 80)
    print("AirSim 高度问题诊断")
    print("=" * 80)
    
    # 连接AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("✓ 已连接到AirSim")
    
    # 重置
    client.reset()
    time.sleep(1)
    
    # 启用API控制
    client.enableApiControl(True, "UAV1")
    client.armDisarm(True, "UAV1")
    print("✓ 无人机已解锁")
    
    # 1. 检查起飞前的位置
    print("\n【步骤1：起飞前位置】")
    state_before = client.getMultirotorState(vehicle_name="UAV1")
    pos_before = state_before.kinematics_estimated.position
    print(f"起飞前位置(NED): X={pos_before.x_val:.4f}, Y={pos_before.y_val:.4f}, Z={pos_before.z_val:.4f}")
    print(f"起飞前高度(转换): {-pos_before.z_val:.4f}m")
    
    # 2. 起飞
    print("\n【步骤2：执行起飞】")
    client.takeoffAsync(vehicle_name="UAV1").join()
    time.sleep(2)
    print("✓ 起飞完成")
    
    # 3. 检查起飞后的位置
    print("\n【步骤3：起飞后位置】")
    state_after = client.getMultirotorState(vehicle_name="UAV1")
    pos_after = state_after.kinematics_estimated.position
    print(f"起飞后位置(NED): X={pos_after.x_val:.4f}, Y={pos_after.y_val:.4f}, Z={pos_after.z_val:.4f}")
    print(f"起飞后高度(转换): {-pos_after.z_val:.4f}m")
    print(f"起飞高度变化: {pos_before.z_val - pos_after.z_val:.4f}m")
    
    # 4. 读取path1的目标高度
    print("\n【步骤4：Path1目标高度】")
    with open('path1.json', 'r') as f:
        path1_data = json.load(f)
    first_point = path1_data['1'][0]
    target_height = first_point['z']
    target_z_ned = -target_height
    print(f"Path1第一个点目标高度: {target_height:.4f}m")
    print(f"转换为NED坐标: Z={target_z_ned:.4f}m")
    
    # 5. 尝试移动到目标高度
    print("\n【步骤5：移动到Path1的起点高度】")
    print(f"发送指令: moveToPositionAsync(x={first_point['x']:.4f}, y={first_point['y']:.4f}, z={target_z_ned:.4f}, speed=1.0)")
    client.moveToPositionAsync(
        first_point['x'], first_point['y'], target_z_ned, 1.0, vehicle_name="UAV1"
    ).join()
    time.sleep(2)
    
    # 6. 检查到达后的位置
    print("\n【步骤6：到达后位置】")
    state_arrived = client.getMultirotorState(vehicle_name="UAV1")
    pos_arrived = state_arrived.kinematics_estimated.position
    print(f"到达后位置(NED): X={pos_arrived.x_val:.4f}, Y={pos_arrived.y_val:.4f}, Z={pos_arrived.z_val:.4f}")
    print(f"到达后高度(转换): {-pos_arrived.z_val:.4f}m")
    
    # 7. 分析偏差
    print("\n【步骤7：偏差分析】")
    print(f"目标高度: {target_height:.4f}m")
    print(f"实际高度: {-pos_arrived.z_val:.4f}m")
    print(f"高度偏差: {-pos_arrived.z_val - target_height:.4f}m")
    print(f"偏差比例: {(-pos_arrived.z_val / target_height):.2f}倍")
    
    # 8. 尝试不同高度
    print("\n【步骤8：尝试飞到绝对高度1.0米】")
    print("发送指令: moveToPositionAsync(x=current, y=current, z=-1.0, speed=1.0)")
    client.moveToPositionAsync(
        pos_arrived.x_val, pos_arrived.y_val, -1.0, 1.0, vehicle_name="UAV1"
    ).join()
    time.sleep(2)
    
    state_test = client.getMultirotorState(vehicle_name="UAV1")
    pos_test = state_test.kinematics_estimated.position
    print(f"到达位置(NED): X={pos_test.x_val:.4f}, Y={pos_test.y_val:.4f}, Z={pos_test.z_val:.4f}")
    print(f"实际高度: {-pos_test.z_val:.4f}m")
    print(f"目标是-1.0，实际是{pos_test.z_val:.4f}")
    
    # 9. 检查Home位置
    print("\n【步骤9：检查Home位置】")
    home_geo = client.getHomeGeoPoint("UAV1")
    print(f"Home地理位置: lat={home_geo.latitude}, lon={home_geo.longitude}, alt={home_geo.altitude}")
    
    # 10. 降落
    print("\n【步骤10：降落】")
    client.landAsync(vehicle_name="UAV1").join()
    client.armDisarm(False, "UAV1")
    client.enableApiControl(False, "UAV1")
    print("✓ 降落完成")
    
    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)

if __name__ == "__main__":
    try:
        diagnose_height_issue()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

