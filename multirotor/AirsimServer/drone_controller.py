import setup_path
import airsim
import numpy as np
import math
import logging
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict

# 配置日志
logger = logging.getLogger("DroneController")

# 图像类型映射表，确保与AirSim的ImageType完全对应
IMAGE_TYPE_MAPPING = {
    "Scene": airsim.ImageType.Scene,
    "DepthPlanar": airsim.ImageType.DepthPlanar,
    "DepthPerspective": airsim.ImageType.DepthPerspective,
    "DepthVis": airsim.ImageType.DepthVis,
    "DisparityNormalized": airsim.ImageType.DisparityNormalized,
    "Segmentation": airsim.ImageType.Segmentation,
    "SurfaceNormals": airsim.ImageType.SurfaceNormals,
    "Infrared": airsim.ImageType.Infrared,
    "OpticalFlow": airsim.ImageType.OpticalFlow,
    "OpticalFlowVis": airsim.ImageType.OpticalFlowVis
}

class DroneController:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.default_vehicle = "UAV1"
        self.connection_status = False
        
        # API调用锁（保护多线程并发调用）
        import threading
        self.api_lock = threading.Lock()
        self.state_lock = threading.Lock()  # 专门保护vehicle_states的锁
        
        # 无人机状态跟踪（使用普通字典，避免defaultdict引发的numpy视图冲突）
        self.vehicle_states = {}
    
    def _get_or_create_state(self, vehicle_name: str) -> Dict[str, Any]:
        """安全地获取或创建无人机状态（线程安全）"""
        with self.state_lock:
            if vehicle_name not in self.vehicle_states:
                self.vehicle_states[vehicle_name] = {
                    "armed": False,
                    "flying": False,
                    "api_enabled": False,
                    "position": (0.0, 0.0, 0.0),
                    "orientation": (0.0, 0.0, 0.0)
                }
            # 返回状态的深拷贝，避免外部修改
            return dict(self.vehicle_states[vehicle_name])
    
    def _update_state_field(self, vehicle_name: str, field: str, value: Any) -> None:
        """安全地更新单个状态字段（线程安全，避免numpy视图冲突）"""
        with self.state_lock:
            if vehicle_name not in self.vehicle_states:
                self.vehicle_states[vehicle_name] = {
                    "armed": False,
                    "flying": False,
                    "api_enabled": False,
                    "position": (0.0, 0.0, 0.0),
                    "orientation": (0.0, 0.0, 0.0)
                }
            # 创建新字典替换旧字典，完全避免修改现有对象
            new_state = dict(self.vehicle_states[vehicle_name])
            new_state[field] = value
            self.vehicle_states[vehicle_name] = new_state

    def connect(self) -> bool:
        """连接到AirSim模拟器"""
        try:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.connection_status = True
            logger.info("成功连接到AirSim模拟器")
            return True
        except Exception as e:
            self.connection_status = False
            logger.error(f"连接到AirSim模拟器失败: {str(e)}")
            return False

    def reset(self) -> bool:
        """重置模拟器状态"""
        try:
            self.client.reset()
            logger.info("模拟器已重置")
            self.vehicle_states.clear()
            return True
        except Exception as e:
            logger.error(f"重置模拟器失败: {str(e)}")
            return False

    def enable_api_control(self, enable: bool = True, vehicle_name: Optional[str] = None) -> bool:
        """启用/禁用API控制"""
        vehicle_name = vehicle_name or self.default_vehicle
        try:
            with self.api_lock:
                self.client.enableApiControl(enable, vehicle_name)
            # 无法直接获取API状态，通过操作结果记录
            self._update_state_field(vehicle_name, "api_enabled", enable)
            logger.info(f"无人机{vehicle_name}API控制已{'启用' if enable else '禁用'}")
            return True
        except Exception as e:
            logger.error(f"无人机{vehicle_name}API控制操作失败: {str(e)}")
            return False

    def arm_disarm(self, arm: bool = True, vehicle_name: Optional[str] = None) -> bool:
        """无人机解锁/上锁"""
        vehicle_name = vehicle_name or self.default_vehicle
        try:
            state = self._get_or_create_state(vehicle_name)
            if not state["api_enabled"]:
                logger.error(f"无人机{vehicle_name}API控制未启用，无法执行解锁/上锁操作")
                return False
            
            with self.api_lock:
                self.client.armDisarm(arm, vehicle_name)
            # 无法直接获取解锁状态，通过操作结果记录
            self._update_state_field(vehicle_name, "armed", arm)
            logger.info(f"无人机{vehicle_name}已{'解锁' if arm else '上锁'}")
            return True
        except Exception as e:
            logger.error(f"无人机{vehicle_name}解锁/上锁操作失败: {str(e)}")
            return False

    def takeoff(self, vehicle_name: Optional[str] = None, timeout_sec: int = 30) -> bool:
        """无人机起飞（适配API）"""
        vehicle_name = vehicle_name or self.default_vehicle
        try:
            state = self._get_or_create_state(vehicle_name)
            # 仅检查API是否启用
            if not state["api_enabled"]:
                logger.error(f"无人机{vehicle_name}API控制未启用，无法起飞")
                return False
                
            if state["flying"]:
                logger.warning(f"无人机{vehicle_name}已处于飞行状态")
                return True

            # 执行起飞（示例中的调用方式，不传递超时到join()）
            with self.api_lock:
                self.client.takeoffAsync(vehicle_name=vehicle_name).join()
            # 假设起飞成功
            self._update_state_field(vehicle_name, "flying", True)
            self._update_vehicle_position(vehicle_name)
            logger.info(f"无人机{vehicle_name}起飞完成")
            return True
        except Exception as e:
            logger.error(f"无人机{vehicle_name}起飞操作失败: {str(e)}")
            return False

    def land(self, vehicle_name: Optional[str] = None, timeout_sec: int = 30) -> bool:
        """无人机降落"""
        vehicle_name = vehicle_name or self.default_vehicle
        try:
            state = self._get_or_create_state(vehicle_name)
            if not state["flying"]:
                logger.warning(f"无人机{vehicle_name}未处于飞行状态，无需降落")
                return True

            # 执行降落
            with self.api_lock:
                self.client.landAsync(vehicle_name=vehicle_name).join()
            self._update_state_field(vehicle_name, "flying", False)
            self._update_vehicle_position(vehicle_name)
            logger.info(f"无人机{vehicle_name}降落完成")
            return True
        except Exception as e:
            logger.error(f"无人机{vehicle_name}降落操作失败: {str(e)}")
            return False

    def move_to_position(self, x: float, y: float, z: float, speed: float = 3, 
                        vehicle_name: Optional[str] = None, timeout_sec: int = 30) -> bool:
        """移动到指定位置"""
        vehicle_name = vehicle_name or self.default_vehicle
        try:
            state = self._get_or_create_state(vehicle_name)
            if not state["flying"]:
                logger.error(f"无人机{vehicle_name}未处于飞行状态，无法移动")
                return False

            if speed <= 0:
                logger.error(f"无人机{vehicle_name}速度必须大于0")
                return False
            # 执行移动并等待完成（与其他异步操作保持一致）
            self.client.moveToPositionAsync(
                x, y, z, speed, vehicle_name=vehicle_name
            ).join()
            
            self._update_vehicle_position(vehicle_name)
            logger.info(f"无人机{vehicle_name}已移动到({x},{y},{z})")
            return True
        except Exception as e:
            logger.error(f"无人机{vehicle_name}移动操作失败: {str(e)}")
            return False

    def move_by_velocity(self, x: float, y: float, z: float, duration: float = 3,
                         vehicle_name: Optional[str] = None, timeout_sec: int = 30) -> bool:
        """根据速度移动"""
        vehicle_name = vehicle_name or self.default_vehicle
        try:
            state = self._get_or_create_state(vehicle_name)
            if not state["flying"]:
                logger.error(f"无人机{vehicle_name}未处于飞行状态，无法移动")
                return False

            if duration <= 0:
                logger.error(f"无人机{vehicle_name}持续时间必须大于0")
                return False

            # 检查速度是否为零或过小
            velocity_magnitude = math.sqrt(x*x + y*y + z*z)
            if velocity_magnitude < 0.01:
                logger.debug(f"无人机{vehicle_name}速度过小({velocity_magnitude:.3f})，跳过移动")
                return True

            # 使用锁保护API调用，避免多线程冲突
            with self.api_lock:
                # 使用moveByVelocityAsync，但不等待完成，避免阻塞
                self.client.moveByVelocityAsync(
                    x, y, z, duration, vehicle_name=vehicle_name
                )

            # logger.debug(f"无人机{vehicle_name}移动向量({x:.3f},{y:.3f},{z:.3f})，持续时间{duration}秒")
            return True
        except Exception as e:
            logger.error(f"无人机{vehicle_name}移动操作失败: {str(e)}")
            return False
        
    def get_image(self, vehicle_name: Optional[str] = None, camera_name: str = "0", 
                 image_type: Any = "Scene") -> Optional[str]:
        """获取指定相机图像并返回Base64编码"""
        vehicle_name = vehicle_name or self.default_vehicle
        try:            
            
            # 使用get方法获取值并提供默认值
            image_newType = IMAGE_TYPE_MAPPING.get(image_type, None)
            if image_newType is None:
                image_newType = airsim.ImageType.Scene
                logger.warning(f"未找到{image_type}对应的新类型，使用默认类型Scene")
                 
            logger.info(f"获取{vehicle_name}的{camera_name}相机{image_newType}类型图像中...")
            # 匹配示例中的图像获取方式
            image_data = self.client.simGetImage(camera_name, image_newType, vehicle_name=vehicle_name)
            if image_data:
                import io
                from PIL import Image
                
                with Image.open(io.BytesIO(image_data)) as image:
                    logger.info(f"图像信息 - 无人机: {vehicle_name}, 相机: {camera_name}, 类型: {image_newType}, 尺寸: {image.size}, 格式: {image.format}, 模式: {image.mode}")
                return image_data
            
            logger.warning(f"未获取到{vehicle_name}的图像数据")
            return None
        except Exception as e:
            logger.error(f"获取图像失败: {str(e)}")
            return None

    def get_vehicle_state(self, vehicle_name: Optional[str] = None) -> Dict[str, Any]:
        """获取无人机当前状态"""
        vehicle_name = vehicle_name or self.default_vehicle
        self._update_vehicle_state(vehicle_name)
        return self._get_or_create_state(vehicle_name)
    
    def _update_vehicle_state(self, vehicle_name: str) -> None:
        """更新无人机状态（完全适配示例API）"""
        try:
            # 更新飞行状态（使用正确的LandedState属性名）
            state = self.client.getMultirotorState(vehicle_name=vehicle_name)
            # 修正LandedState属性名（移除前缀）
            flying_status = (state.landed_state == airsim.LandedState.Flying)
            self._update_state_field(vehicle_name, "flying", flying_status)
            
            # 更新位置
            self._update_vehicle_position(vehicle_name)
            
            # 更新姿态
            self._update_vehicle_orientation(vehicle_name)
            
        except Exception as e:
            logger.warning(f"更新无人机{vehicle_name}状态失败: {str(e)}")
    
    def _update_vehicle_position(self, vehicle_name: str) -> None:
        """更新无人机位置信息"""
        try:
            position = self.client.getMultirotorState(vehicle_name=vehicle_name).kinematics_estimated.position
            # 使用安全更新方法，避免numpy视图冲突
            self._update_state_field(vehicle_name, "position", (position.x_val, position.y_val, position.z_val))
        except Exception as e:
            logger.warning(f"更新无人机{vehicle_name}位置失败: {str(e)}")

    def _update_vehicle_orientation(self, vehicle_name: str) -> None:
        """更新无人机姿态信息（欧拉角）"""
        try:
            orientation_q = self.client.getMultirotorState(vehicle_name=vehicle_name).kinematics_estimated.orientation
            from airsim.utils import to_eularian_angles
            pitch, roll, yaw = to_eularian_angles(orientation_q)
            # 使用安全更新方法，避免numpy视图冲突
            self._update_state_field(
                vehicle_name, 
                "orientation", 
                (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))
            )
        except Exception as e:
            logger.warning(f"更新无人机{vehicle_name}姿态失败: {str(e)}")