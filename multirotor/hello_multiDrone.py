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

# 无人机初始位置
origin_x = [0,-6, -6, 6, 6]
origin_y = [0,-6, 6, 6, -6]

# 无人机数量
uavNum = 1

# 从 JSON 文件读取飞行轨迹
try:
    with open('flight_path.json', 'r') as file:
        data = json.load(file)
        points = data.get("points", [])
        flight_path = [[point["x"], point["y"], point["z"]] for point in points]
except FileNotFoundError:
    print("未找到 flight_path.json 文件，使用默认轨迹。")
    flight_path = [
        [0, 0, -10],
        [5, 0, -10],
        [5, 5, -10],
        [0, 5, -10],
        [0, 0, -10]
    ]

# def get_UAV_pos(client, vehicle_name="SimpleFlight"):
#     global origin_x
#     global origin_y
#     state = client.simGetGroundTruthKinematics(vehicle_name=vehicle_name)
#     x = state.position.x_val
#     y = state.position.y_val
#     i = int(vehicle_name[3])
#     x += origin_x[i - 1]
#     y += origin_y[i - 1]
#     pos = np.array([[x], [y]])
#     return pos

# 连接到AirSim模拟器
client = airsim.MultirotorClient()
client.confirmConnection()
client.reset()

airsim.wait_key('Press any key to takeoff')
for i in range(uavNum):
    name = "UAV" + str(i+1)
    print(name)
    client.enableApiControl(True, name)     # 获取控制权
    client.armDisarm(True, name)            # 解锁（螺旋桨开始转动）
    client.takeoffAsync(vehicle_name=name).join()  # 起飞并等待完成

airsim.wait_key('Press any key to follow the flight path')

# 让无人机按照预设路线飞行
for i in range(uavNum):
    name = "UAV" + str(i + 1)
    for point in flight_path:
        x, y, z = point
        if i == 0:
            client.moveToPositionAsync(x, y, z, 3, vehicle_name=name) # 移动并等待完成
        else:
            client.moveToPositionAsync(x, y, z, 3, vehicle_name=name).join()  # 移动并等待完成

airsim.wait_key('Press any key to end the flight')

# 降落
for i in range(uavNum):
    name = "UAV" + str(i + 1)
    client.landAsync(vehicle_name=name).join()  # 降落并等待完成
    client.armDisarm(False, vehicle_name=name)              # 上锁
    client.enableApiControl(False, vehicle_name=name)       # 释放控制权

# # 让无人机进行编队
for i in range(uavNum):
    name = "UAV" + str(i + 1)
    client.moveByVelocityAsync(0, 0, -3, 1, vehicle_name=name).join() # 移动并等待完成
    client.moveToPositionAsync(origin_x[i], origin_y[i], -10, 3, vehicle_name=name).join()


    # if i != uavNum-1:
    #     client.moveByVelocityAsync(0, 0, -3, 1, vehicle_name=name).join() # 移动并等待完成
    #     client.moveToPositionAsync(origin_x[i], origin_y[i], -10, 3, vehicle_name=name).join() # 移动并等待完成
    # else:
    #     client.moveByVelocityAsync(0, 0, -3, 1, vehicle_name=name).join() # 移动并等待完成
    #     client.moveToPositionAsync(origin_x[i], origin_y[i], -10, 3, vehicle_name=name).join()

for i in range(uavNum):
    name = "UAV" + str(i + 1)
    if i != uavNum-1:
        client.hoverAsync(vehicle_name=name) # 悬停并等待完成
    else:
        client.hoverAsync(vehicle_name=name).join() # 悬停并等待完成

for i in range(uavNum):
    name = "UAV" + str(i + 1)
    if i != uavNum-1:
        client.moveByVelocityAsync(5, 0, 0, 20, vehicle_name=name)  
    else:
        client.moveByVelocityAsync(5, 0, 0, 20, vehicle_name=name).join()  

state = client.getMultirotorState(vehicle_name='UAV1')
print("state: %s" % pprint.pformat(state))

# 等待用户按键后拍摄图像
airsim.wait_key('Press any key to take images')
# 从无人机获取相机图像
responses = client.simGetImages([
    airsim.ImageRequest("0", airsim.ImageType.DepthVis),  # 深度可视化图像
    airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True), # 透视投影深度图像
    airsim.ImageRequest("1", airsim.ImageType.Scene), # PNG格式的场景视觉图像
    airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])  # 未压缩的RGBA数组场景视觉图像
print('Retrieved images: %d' % len(responses))

# 创建临时目录用于保存图像
tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_drone")
print ("Saving images to %s" % tmp_dir)
try:
    os.makedirs(tmp_dir)
except OSError:
    if not os.path.isdir(tmp_dir):
        raise

# 保存图像到临时目录
for idx, response in enumerate(responses):

    filename = os.path.join(tmp_dir, str(idx))

    if response.pixels_as_float:
        print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
        airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
    elif response.compress: # png格式
        print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
        airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
    else: # 未压缩数组
        print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) # 获取numpy数组
        img_rgb = img1d.reshape(response.height, response.width, 3) # 将数组重塑为4通道图像数组H X W X 3
        cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # 写入PNG文件

airsim.wait_key('Press any key to end')
# # 降落
for i in range(uavNum):
    name = "UAV" + str(i + 1)
    if i != uavNum-1:
       client.landAsync(vehicle_name=name)
    else:
       client.landAsync(vehicle_name=name).join()  # 降落并等待完成
    name = "UAV" + str(i + 1)
    client.armDisarm(False, vehicle_name=name)              # 上锁
    client.enableApiControl(False, vehicle_name=name)       # 释放控制权