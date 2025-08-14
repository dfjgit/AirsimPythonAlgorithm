# 导入必要的库
import setup_path
import airsim
import numpy as np
import os
import tempfile
import pprint
import cv2
import time  # 导入 time 模块
import json  # 导入 json 模块

import socket

# 无人机初始位置
origin_x = [0]
origin_y = [0]

# 从 JSON 文件读取飞行轨迹

try:
    with open('originPath.json', 'r') as file:
        data = json.load(file)
        points = data.get("points", [])
        flight_path = [[point["z"], point["x"], -point["y"]] for point in points]
except FileNotFoundError:
    print("未找到 flight_path.json 文件，使用默认轨迹。")
    flight_path = []

# 连接到AirSim模拟器
client = airsim.MultirotorClient()
client.confirmConnection()
client.reset()

airsim.wait_key('Press any key to takeoff')


name = "UAV1"  # 假设无人机名称为 UAV1
client.enableApiControl(True, name)  # 获取控制权
client.armDisarm(True, name)  # 解锁（螺旋桨开始转动）
client.takeoffAsync(vehicle_name=name).join()  # 起飞并等待完成

airsim.wait_key('Press any key to follow the flight path')


# 让无人机按照预设路线飞行
for point in flight_path:
    x, y, z = point
    print("TO：",x, y, z)    
    client.moveToPositionAsync(x, y, z, 3, vehicle_name=name).join()  # 移动并等待完成

airsim.wait_key('Press any key to end the flight')

# 降落
client.landAsync(vehicle_name=name).join()  # 降落并等待完成
client.armDisarm(False, vehicle_name=name)  # 上锁
client.enableApiControl(False, vehicle_name=name)  # 释放控制权
