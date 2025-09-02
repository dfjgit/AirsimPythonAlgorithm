import time
import logging
from typing import Optional, Any, Dict

# 配置日志
logger = logging.getLogger("UnityEnvironment")

class DroneControllerProxy:
    """
    无人机控制器代理类
    提供对原始DroneController的代理功能
    """
    def __init__(self, drone_controller):
        # 初始化DroneController实例
        self.drone_controller = drone_controller
        
    # 代理DroneController的方法
    def connect(self) -> bool:
        return self.drone_controller.connect()
        
    def reset(self) -> bool:
        """调用原始控制器的重置方法"""
        return self.drone_controller.reset()
        
    def enable_api_control(self, enable: bool = True, vehicle_name: Optional[str] = None) -> bool:
        return self.drone_controller.enable_api_control(enable, vehicle_name)
        
    def arm_disarm(self, arm: bool = True, vehicle_name: Optional[str] = None) -> bool:
        return self.drone_controller.arm_disarm(arm, vehicle_name)
        
    def takeoff(self, vehicle_name: Optional[str] = None, timeout_sec: int = 30) -> bool:
        return self.drone_controller.takeoff(vehicle_name, timeout_sec)
        
    def land(self, vehicle_name: Optional[str] = None, timeout_sec: int = 30) -> bool:
        return self.drone_controller.land(vehicle_name, timeout_sec)
        
    def move_to_position(self, x: float, y: float, z: float, speed: float = 3, 
                        vehicle_name: Optional[str] = None, timeout_sec: int = 30) -> bool:
        return self.drone_controller.move_to_position(x, y, z, speed, vehicle_name, timeout_sec)
        
    def get_image(self, vehicle_name: Optional[str] = None, camera_name: str = "0", 
                 image_type: Any = "Scene") -> Optional[str]:
        return self.drone_controller.get_image(vehicle_name, camera_name, image_type)
        
    def get_vehicle_state(self, vehicle_name: Optional[str] = None) -> Dict[str, Any]:
        return self.drone_controller.get_vehicle_state(vehicle_name)
        
    # 属性访问代理
    @property
    def connection_status(self):
        return self.drone_controller.connection_status
        
    @property
    def default_vehicle(self):
        return self.drone_controller.default_vehicle