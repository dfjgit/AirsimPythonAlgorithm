#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析路径坐标，检查是否有坐标轴错误
"""

import json
import math

# 读取对比数据
with open('path_comparison_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

expected = data['expected_path_data']
actual = data['actual_path_data']

print("=" * 80)
print("坐标轴分析")
print("=" * 80)

# 起点和终点
exp_start = expected[0]
exp_end = expected[-1]
act_start = actual[0]
act_end = actual[-1]

print("\n【起点对比】")
print(f"预期起点: x={exp_start['x']:.4f}, y={exp_start['y']:.4f}, z={exp_start['z']:.4f}")
print(f"实际起点: x={act_start['x']:.4f}, y={act_start['y']:.4f}, z={act_start['z']:.4f}")
print(f"起点差异: Δx={act_start['x']-exp_start['x']:.4f}, Δy={act_start['y']-exp_start['y']:.4f}, Δz={act_start['z']-exp_start['z']:.4f}")

print("\n【终点对比】")
print(f"预期终点: x={exp_end['x']:.4f}, y={exp_end['y']:.4f}, z={exp_end['z']:.4f}")
print(f"实际终点: x={act_end['x']:.4f}, y={act_end['y']:.4f}, z={act_end['z']:.4f}")
print(f"终点差异: Δx={act_end['x']-exp_end['x']:.4f}, Δy={act_end['y']-exp_end['y']:.4f}, Δz={act_end['z']-exp_end['z']:.4f}")

print("\n【预期路径变化范围】")
exp_x = [p['x'] for p in expected]
exp_y = [p['y'] for p in expected]
exp_z = [p['z'] for p in expected]
print(f"X: {min(exp_x):.4f} ~ {max(exp_x):.4f}, 变化量: {max(exp_x)-min(exp_x):.4f}")
print(f"Y: {min(exp_y):.4f} ~ {max(exp_y):.4f}, 变化量: {max(exp_y)-min(exp_y):.4f}")
print(f"Z: {min(exp_z):.4f} ~ {max(exp_z):.4f}, 变化量: {max(exp_z)-min(exp_z):.4f}")

print("\n【实际飞行路径变化范围】")
act_x = [p['x'] for p in actual]
act_y = [p['y'] for p in actual]
act_z = [p['z'] for p in actual]
print(f"X: {min(act_x):.4f} ~ {max(act_x):.4f}, 变化量: {max(act_x)-min(act_x):.4f}")
print(f"Y: {min(act_y):.4f} ~ {max(act_y):.4f}, 变化量: {max(act_y)-min(act_y):.4f}")
print(f"Z: {min(act_z):.4f} ~ {max(act_z):.4f}, 变化量: {max(act_z)-min(act_z):.4f}")

# 计算起点到终点的理论直线距离
exp_distance = math.sqrt(
    (exp_end['x'] - exp_start['x'])**2 +
    (exp_end['y'] - exp_start['y'])**2 +
    (exp_end['z'] - exp_start['z'])**2
)

act_distance = math.sqrt(
    (act_end['x'] - act_start['x'])**2 +
    (act_end['y'] - act_start['y'])**2 +
    (act_end['z'] - act_start['z'])**2
)

print("\n【直线距离】")
print(f"预期起点到终点直线距离: {exp_distance:.4f} m")
print(f"实际起点到终点直线距离: {act_distance:.4f} m")

# 计算实际飞行路径的总距离
total_actual = 0
for i in range(1, len(actual)):
    d = math.sqrt(
        (actual[i]['x'] - actual[i-1]['x'])**2 +
        (actual[i]['y'] - actual[i-1]['y'])**2 +
        (actual[i]['z'] - actual[i-1]['z'])**2
    )
    total_actual += d

print(f"实际飞行总路径长度: {total_actual:.4f} m")
print(f"路径长度比: {total_actual/act_distance:.2f}倍（应接近1.0表示是直线）")

# 检查是否实际飞到了正确的起点和终点
print("\n【诊断】")

# 检查Z坐标偏差模式
z_offset = act_start['z'] - exp_start['z']
print(f"起点Z坐标偏差: {z_offset:.4f} m")
print(f"Z坐标比值: {act_start['z']/exp_start['z']:.4f}")

# 检查是否所有点的Z都偏高相同的量
z_offsets = [actual[i]['z'] - expected[i]['z'] for i in range(len(expected))]
avg_z_offset = sum(z_offsets) / len(z_offsets)
print(f"所有点的平均Z偏差: {avg_z_offset:.4f} m")

# 检查是否起点位置根本没到
start_3d_error = math.sqrt(
    (act_start['x'] - exp_start['x'])**2 +
    (act_start['y'] - exp_start['y'])**2 +
    (act_start['z'] - exp_start['z'])**2
)
print(f"\n起点的3D位置误差: {start_3d_error:.4f} m")

if start_3d_error > 0.5:
    print("⚠️ 警告：起点位置误差超过0.5米，可能是：")
    print("   1. 无人机没有飞到预期起点就开始记录")
    print("   2. 坐标系转换有误")
    print("   3. 位置容差设置过大")

# 检查实际路径是否是直线
if total_actual / act_distance > 1.5:
    print("\n⚠️ 警告：实际飞行路径不是直线！")
    print(f"   实际路径长度({total_actual:.2f}m) 远大于直线距离({act_distance:.2f}m)")
    print("   可能原因：无人机在飞行过程中有大幅度摆动或路径规划问题")

print("\n" + "=" * 80)

