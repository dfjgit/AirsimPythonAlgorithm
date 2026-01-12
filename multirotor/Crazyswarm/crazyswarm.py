import re
import logging
import traceback
import numpy as np
import threading
from typing import Dict, List
from multirotor.Algorithm.Vector3 import Vector3

from AirsimServer.unity_socket_server import UnitySocketServer
from Algorithm.scanner_config_data import ScannerConfigData
from .crazyflie_operate import CrazyflieOperate
from .crazyflie_operate import EnumCrazyflieOperate
from .crazyflie_logging_data import CrazyflieLoggingData
from .crazyflie_wayPoint import WayPath, WayPoint

logger = logging.getLogger("CrazyswarmManager")

class CrazyswarmManager:
    
    def __init__(self, unity_socket: UnitySocketServer, configData: ScannerConfigData):
        self.unity_socket = unity_socket  # Unity通信Socket服务
        self.crazyflieLoggingDataById : Dict[int, CrazyflieLoggingData] = {}
        self.operateDatas: list[CrazyflieOperate] = []

        for drone_name in configData.droneSettings.keys():
            isCrazyflieMirror = configData.droneSettings[drone_name]["isCrazyflieMirror"]

            if isCrazyflieMirror:
                id = self.get_id_by_name(drone_name)
                self.crazyflieLoggingDataById[id] = None

        self.updateTimer = threading.Timer(0.02, self.update)
        self.updateTimer.start()

    def update(self):
        try:
            if len(self.operateDatas) > 0:
                self.send_crazyflie_operate_data()
        except Exception as e:
            logger.warning(f"Crazyswarm update has a error: {e}")
        finally:
            self.updateTimer = threading.Timer(0.02, self.update)
            self.updateTimer.start()

    def update_crazyflies(self, crazyflie_running_datas: List[CrazyflieLoggingData]):
        # 遍历每个无人机的运行数据对象
        for data in crazyflie_running_datas:
        # 以id为键，loggingData为值，更新/写入字典
            self.crazyflieLoggingDataById[data.Id] = data
            # logger.debug(f"实体无人机Crazyflie：{data.Id}号机更新日志数据 日志数据为：{data}")


    def take_off(self, drone_name, height: float, duration: float):
        operateData = {
            'height' : height,
            'duration' : duration
        }
        self.operate(drone_name, EnumCrazyflieOperate.TakeOff, operateData)
        self.operate(drone_name, EnumCrazyflieOperate.StartLogging, 0.2)
        self.operate(drone_name, EnumCrazyflieOperate.GetLogging, 
                     {
                         "isLoop": True,
                     })


    def go_to(self, drone_name: str, direction: Vector3, duration: float):
        id = self.get_id_by_name(drone_name)
        if id not in self.crazyflieLoggingDataById:
            return
        logging_data = self.crazyflieLoggingDataById[id]
        if logging_data is None:
            logger.warning(f"Crazyflie实体无人机: {id}号没有日志数据")
            return
        # 校验速度数据是否在合理范围（比如无人机速度不会超过1m/s，可根据实际调整）
        max_speed = 1.0
        if (abs(logging_data.XSpeed) > max_speed or 
            abs(logging_data.YSpeed) > max_speed or 
            abs(logging_data.ZSpeed) > max_speed):
            logger.warning(f"Crazyflie id={id} 速度异常：X = {logging_data.XSpeed}, Y = {logging_data.YSpeed}, Z = {logging_data.ZSpeed}")
            return
        # 校验位置数据是否合理（避免超大值）
        max_position = 5.0  # 假设飞行范围不超过2米
        if (abs(logging_data.X / 20) > max_position or 
            abs(logging_data.Y / 20) > max_position or 
            abs(logging_data.Z / 20) > max_position):
            logger.warning(f"Crazyflie id={id} 位置异常：X = {logging_data.X}, Y = {logging_data.Y}, Z = {logging_data.Z}")
            return
        # 正常计算速度和位置
        velocity = logging_data.Speed * direction
        position = self.get_position_by_velocity(id, velocity, duration)
        if position is None:
            return
        wayPoint = WayPoint.CreateFromPosition(position, duration)
        wayPath = WayPath.CreateFromPoint(wayPoint)
        self.operate(drone_name, EnumCrazyflieOperate.GoTo, wayPath)


    def land(self, drone_name: str, duration: float):
        operateData = {
            'duration' : duration
        }
        self.operate(drone_name, EnumCrazyflieOperate.Land, operateData)


    def start_logging(self,  drone_name: str, timeSpan: float):
        operateData = timeSpan
        self.operate(drone_name, EnumCrazyflieOperate.TakeOff, operateData)


    def operate(self, drone_name : str, operateType: EnumCrazyflieOperate, operateData):
        id = self.get_id_by_name(drone_name)

        if id not in self.crazyflieLoggingDataById:
            logger.warning(f"Crazyflie实体无人机: {id}号不存在，当前Crazyflie实体无人机集群为：{self.crazyflieLoggingDataById.keys()}")
            return

        operateData = CrazyflieOperate(id, operateType, operateData)
        self.operateDatas.append(operateData)


    def get_position_by_velocity(self, id: int, velocity: Vector3, duration: float):
        """
        根据速度和持续时间，计算经过该时间后的物体位置
        :param velocity: 物体的速度向量（Vector3类型，包含x/y/z三个轴的速度）
        :param duration: 速度持续的时间（秒）
        :return: 经过duration时间后的物体新位置（Vector3类型）
        """
        if id not in self.crazyflieLoggingDataById:
            return
        
        displacement = velocity * duration  # 计算这段时间内的位移
        logger.debug(f"当前速度为：{velocity} 持续时间为：{duration} 位移为：{displacement}")
        currentPosition = Vector3(self.crazyflieLoggingDataById[id].X, self.crazyflieLoggingDataById[id].Y, self.crazyflieLoggingDataById[id].Z)
        x: float = currentPosition.x / 20
        y: float = currentPosition.y / 20
        new_position = Vector3(x + displacement.x, y + displacement.y, currentPosition.z )  # 当前位置叠加位移得到新位置
        return new_position
    

    def get_loggingData_by_droneName(self, drone_name: str) -> CrazyflieLoggingData:
        id = self.get_id_by_name(drone_name)
        if id not in self.crazyflieLoggingDataById:
            # 返回空的CrazyflieLoggingData实例（需确保该类支持无参初始化）
            return CrazyflieLoggingData()
        # 返回对应数据
        return self.crazyflieLoggingDataById[id]

    
    def send_crazyflie_operate_data(self):
        try:
            self.unity_socket.send_crazyflie_operate(self.operateDatas)
            self.operateDatas.clear()
        except Exception as e:
            logger.warning(f"发送实体无人机操作指令数据到Unity失败: {str(e)}")


    def get_id_by_name(self, drone_name):
        # 匹配字符串中第一个正整数
        match = re.search(r'\d+', drone_name)  # 找第一个连续数字序列
        if match:
            return int(match.group())  # 找到则转成整数返回
        else:
            return 0  # 无数字时返回0
        
    def clear(self):
        self.updateTimer.cancel()