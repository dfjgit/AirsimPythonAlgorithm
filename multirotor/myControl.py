# 导入必要的库
import tempfile
import cv2
import numpy as np
import setup_path
import airsim

import msvcrt  # 用于捕获键盘输入
import sys  # 导入 sys 模块用于标准输出
import os  # 导入 os 模块用于文件操作

# 无人机初始位置
origin_x = [0, -6, -6, 6, 6]
origin_y = [0, -6, 6, 6, -6]

# 无人机数量
uavNum = 5

# 连接到AirSim模拟器
client = airsim.MultirotorClient()
client.confirmConnection()


# 默认移动速度
speed = 5

# 创建保存图片的目录
if not os.path.exists('photos'):
    os.makedirs('photos')
    
    
def reset_and_form_formation(client, uavNum, origin_x, origin_y):
    """
    重置所有无人机的状态并让它们进行编队。

    :param client: AirSim客户端对象，用于与AirSim模拟器进行交互。
    :param uavNum: 无人机的数量。
    :param origin_x: 一个列表，包含每个无人机编队位置的x坐标。
    :param origin_y: 一个列表，包含每个无人机编队位置的y坐标。
    """
    client.reset()
    for i in range(uavNum):
        name = "UAV" + str(i + 1)
        client.enableApiControl(True, name)
        client.armDisarm(True, name)

    for i in range(uavNum):
        name = "UAV" + str(i + 1)
        client.enableApiControl(True, name)  # 获取控制权
        client.armDisarm(True, name)  # 解锁（螺旋桨开始转动）
        client.takeoffAsync(vehicle_name=name).join()  # 起飞并等待完成

    # 让无人机进行编队
    for i in range(uavNum):
        name = "UAV" + str(i + 1)
        client.moveByVelocityAsync(0, 0, -3, 1, vehicle_name=name).join()  # 移动并等待完成
        client.moveToPositionAsync(origin_x[i], origin_y[i], -10, 3, vehicle_name=name).join()

reset_and_form_formation(client, uavNum, origin_x, origin_y)

while True:
    key = msvcrt.getch().decode().lower()
    # 打印按键信息并覆盖之前的输出
    sys.stdout.write(f"\rPressed key: {key}")
    sys.stdout.flush()  # 强制刷新输出缓冲区

    # 按下 'w' 键，控制无人机向前移动
    if key == 'w':
        for i in range(uavNum):
            name = "UAV" + str(i + 1)
            client.moveByVelocityAsync(0, speed, 0, 1e6, vehicle_name=name)
    # 按下 's' 键，控制无人机向后移动
    elif key == 's':
        for i in range(uavNum):
            name = "UAV" + str(i + 1)
            client.moveByVelocityAsync(0, -speed, 0, 1e6, vehicle_name=name)
    # 按下 'a' 键，控制无人机向左移动
    elif key == 'a':
        for i in range(uavNum):
            name = "UAV" + str(i + 1)
            client.moveByVelocityAsync(-speed, 0, 0, 1e6, vehicle_name=name)
    # 按下 'd' 键，控制无人机向右移动
    elif key == 'd':
        for i in range(uavNum):
            name = "UAV" + str(i + 1)
            client.moveByVelocityAsync(speed, 0, 0, 1e6, vehicle_name=name)
    # 按下 'q' 键，控制无人机上升
    elif key == 'q':
        for i in range(uavNum):
            name = "UAV" + str(i + 1)
            client.moveByVelocityAsync(0, 0, -3, 1, vehicle_name=name)
    # 按下 'e' 键，控制无人机下降
    elif key == 'e':
        for i in range(uavNum):
            name = "UAV" + str(i + 1)
            client.moveByVelocityAsync(0, 0, 3, 1, vehicle_name=name)
    # 按下 'x' 键，控制无人机停止移动
    elif key == 'x':
        for i in range(uavNum):
            name = "UAV" + str(i + 1)
            client.moveByVelocityAsync(0, 0, 0, 0, vehicle_name=name)
    # 按下 'p' 键，控制无人机拍照
    elif key == 'p':
        for i in range(uavNum):
            name = "UAV" + str(i + 1)
            # 请求获取单张图像
            responses = client.simGetImages([airsim.ImageRequest("1", airsim.ImageType.Scene)], vehicle_name=name)
            # airsim.ImageRequest("0", airsim.ImageType.DepthVis),  # 深度可视化图像
            # airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True), # 透视投影深度图像
            # airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])  # 未压缩的RGBA数组场景视觉图像
            # airsim.ImageRequest("1", airsim.ImageType.Scene), # PNG格式的场景视觉图像
            response = responses[0]
            # 将图像数据转换为字节数组
            
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
                    airsim.write_pfm(os.path.normpath(filename + name + '.pfm'), airsim.get_pfm_array(response))
                elif response.compress: # png格式
                    print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
                    airsim.write_file(os.path.normpath(filename + name + '.png'), response.image_data_uint8)
                else: # 未压缩数组
                    print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
                    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) # 获取numpy数组
                    img_rgb = img1d.reshape(response.height, response.width, 3) # 将数组重塑为4通道图像数组H X W X 3
                    cv2.imwrite(os.path.normpath(filename + name + '.png'), img_rgb) # 写入PNG文件
    
            # # 保存图像
            # filename = os.path.join('photos', f'{name}_photo_{int(time.time())}.png')
            # airsim.write_png(filename, img_rgb)
            # print(f"Photo saved for {name} at {filename}")
    # 按下 'r' 键，重置无人机状态

    
        # 在原代码中调用函数
    elif key == 'r':
        reset_and_form_formation(client, uavNum, origin_x, origin_y)
    # 按下 'ESC' 键，退出程序并让无人机降落
    elif key == '\x1b':
        for i in range(uavNum):
            name = "UAV" + str(i + 1)
            print(name+"is landing...")
            client.landAsync(vehicle_name=name).join()  # 降落并等待完成
            client.armDisarm(False, vehicle_name=name)  # 上锁
            client.enableApiControl(False, vehicle_name=name)  # 释放控制权
        break

print("Done.")



