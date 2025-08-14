# 导入必要的库
import setup_path
import airsim

import numpy as np
import os
import tempfile
import pprint
import cv2

# 连接到AirSim模拟器
client = airsim.MultirotorClient()
client.confirmConnection()



for i in range(1):
    name = "UAV" + str(i+1)
    client.enableApiControl(True, name)     # 获取控制权
    client.armDisarm(True, name)            # 解锁（螺旋桨开始转动）
    client.takeoffAsync(vehicle_name=name).join()  # 起飞并等待完成
    
    
# # 获取多旋翼状态并打印
# state = client.getMultirotorState()
# s = pprint.pformat(state)
# print("state: %s" % s)

# # 获取IMU数据并打印
# imu_data = client.getImuData()
# s = pprint.pformat(imu_data)
# print("imu_data: %s" % s)

# # 获取气压计数据并打印
# barometer_data = client.getBarometerData()
# s = pprint.pformat(barometer_data)
# print("barometer_data: %s" % s)

# # 获取磁力计数据并打印
# magnetometer_data = client.getMagnetometerData()
# s = pprint.pformat(magnetometer_data)
# print("magnetometer_data: %s" % s)

# # 获取GPS数据并打印
# gps_data = client.getGpsData()
# s = pprint.pformat(gps_data)
# print("gps_data: %s" % s)

# # 等待用户按键后起飞
airsim.wait_key('Press any key to takeoff')
print("Taking off...")
client.armDisarm(True,vehicle_name="UAV1")
# client.takeoffAsync().join()


client.takeoffAsync(5,).join()
client.takeoffAsync(6,"UAV2").join()

# # 获取多旋翼状态并打印
# state = client.getMultirotorState()
# print("state: %s" % pprint.pformat(state))

# 等待用户按键后移动到指定位置
airsim.wait_key('Press any key to move vehicle to (-10, 10, -10) at 5 m/s')
client.moveToPositionAsync(-10, 10, -10, 5).join()

# 悬停
client.hoverAsync().join()

# 获取多旋翼状态并打印
state = client.getMultirotorState()
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
 
# 等待用户按键后重置到原始状态
airsim.wait_key('Press any key to reset to original state')

client.reset()
client.armDisarm(False)

# 关闭API控制
client.enableApiControl(False)
