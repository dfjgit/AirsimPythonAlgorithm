import re
import logging
import numpy as np
from typing import Dict, List
from multirotor.Algorithm.Vector3 import Vector3

from AirsimServer.unity_socket_server import UnitySocketServer
from .crazyflie_operate import CrazyflieOperate
from .crazyflie_operate import EnumCrazyflieOperate
from .crazyflie_logging_data import CrazyflieLoggingData
from .crazyflie_wayPoint import WayPath, WayPoint

logger = logging.getLogger("CrazyswarmManager")

class CrazyswarmManager:
    
    def __init__(self, unity_socket : UnitySocketServer):
        self.unity_socket = unity_socket  # Unity通信Socket服务
        self.crazyfliePositionById : Dict[int, CrazyflieLoggingData] = {}

    def update_crazyflies(self, crazyflie_running_datas : List[CrazyflieLoggingData]):
        # 遍历每个无人机的运行数据对象
        for data in crazyflie_running_datas:
        # 以id为键，loggingData为值，更新/写入字典
            self.crazyfliePositionById[data.Id] = data

    def take_off(self, drone_name, height : float, duration : float):
        operateData = {
            'height' : height,
            'duration' : duration
        }
        self.operate(drone_name, EnumCrazyflieOperate.TakeOff, operateData)

    def go_to(self, drone_name : str, velocity : Vector3, duration : float):
        id = self.get_id_by_name(drone_name)

        if id not in self.crazyfliePositionById:
            return

        position = self.get_position_by_velocity(id, velocity, duration)
        wayPoint = WayPoint.CreateFromPosition(position, duration)
        wayPath = WayPath.CreateFromPoint(wayPoint)
        
        self.operate(drone_name, EnumCrazyflieOperate.GoTo, wayPath)

    def land(self, drone_name : str, duration : float):
        operateData = {
            'duration' : duration
        }
        self.operate(drone_name, EnumCrazyflieOperate.Land, operateData)


    def operate(self, drone_name : str, operateType : EnumCrazyflieOperate, operateData):
        id = self.get_id_by_name(drone_name)

        if id not in self.crazyfliePositionById:
            logger.warning(f"Crazyflie实体无人机: {id}号不存在，当前Crazyflie实体无人机集群为：{self.crazyfliePositionById.keys()}")
            return

        operate = CrazyflieOperate(id, operateType, operateData)
        self.send_crazyflie_operate_data(operate)


    def get_position_by_velocity(self, id : int, velocity : Vector3, duration : float):
        """
        根据速度和持续时间，计算经过该时间后的物体位置
        :param velocity: 物体的速度向量（Vector3类型，包含x/y/z三个轴的速度）
        :param duration: 速度持续的时间（秒）
        :return: 经过duration时间后的物体新位置（Vector3类型）
        """
        if id not in self.crazyfliePositionById:
            return
        
        displacement = velocity * duration  # 计算这段时间内的位移
        currentPosition = Vector3(self.crazyfliePositionById[id].X, self.crazyfliePositionById[id].Y, self.crazyfliePositionById[id].Z)
        new_position = Vector3(currentPosition.x + displacement.x, currentPosition.y + displacement.y, currentPosition.z + displacement.z)  # 当前位置叠加位移得到新位置
        return new_position
    
    def send_crazyflie_operate_data(self, crazyflieOperateData : CrazyflieOperate):
        try:
            self.unity_socket.send_crazyflie_operate(crazyflieOperateData)
        except Exception as e:
            logger.warning(f"发送实体无人机操作指令数据到Unity失败: {str(e)}")
        

    def get_id_by_name(self, drone_name):
        # 匹配字符串中第一个正整数
        match = re.search(r'\d+', drone_name)  # 找第一个连续数字序列
        if match:
            return int(match.group())  # 找到则转成整数返回
        else:
            return 0  # 无数字时返回0