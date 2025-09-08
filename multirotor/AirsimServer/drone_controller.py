import setup_path
import airsim
import numpy as np
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
        
        # 无人机状态跟踪（仅保留可获取的状态）
        self.vehicle_states = defaultdict(lambda: {
            "armed": False,  # 无法直接获取，通过操作记录
            "flying": False,
            "api_enabled": False,
            "position": (0.0, 0.0, 0.0)
        })

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
            self.client.enableApiControl(enable, vehicle_name)
            # 无法直接获取API状态，通过操作结果记录
            self.vehicle_states[vehicle_name]["api_enabled"] = enable
            logger.info(f"无人机{vehicle_name}API控制已{'启用' if enable else '禁用'}")
            return True
        except Exception as e:
            logger.error(f"无人机{vehicle_name}API控制操作失败: {str(e)}")
            return False

    def arm_disarm(self, arm: bool = True, vehicle_name: Optional[str] = None) -> bool:
        """无人机解锁/上锁"""
        vehicle_name = vehicle_name or self.default_vehicle
        try:
            if not self.vehicle_states[vehicle_name]["api_enabled"]:
                logger.error(f"无人机{vehicle_name}API控制未启用，无法执行解锁/上锁操作")
                return False
                
            self.client.armDisarm(arm, vehicle_name)
            # 无法直接获取解锁状态，通过操作结果记录
            self.vehicle_states[vehicle_name]["armed"] = arm
            logger.info(f"无人机{vehicle_name}已{'解锁' if arm else '上锁'}")
            return True
        except Exception as e:
            logger.error(f"无人机{vehicle_name}解锁/上锁操作失败: {str(e)}")
            return False

    def takeoff(self, vehicle_name: Optional[str] = None, timeout_sec: int = 30) -> bool:
        """无人机起飞（适配API）"""
        vehicle_name = vehicle_name or self.default_vehicle
        try:
            # 仅检查API是否启用
            if not self.vehicle_states[vehicle_name]["api_enabled"]:
                logger.error(f"无人机{vehicle_name}API控制未启用，无法起飞")
                return False
                
            if self.vehicle_states[vehicle_name]["flying"]:
                logger.warning(f"无人机{vehicle_name}已处于飞行状态")
                return True

            # 执行起飞（示例中的调用方式，不传递超时到join()）
            self.client.takeoffAsync(vehicle_name=vehicle_name).join()
            # 假设起飞成功
            self.vehicle_states[vehicle_name]["flying"] = True
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
            if not self.vehicle_states[vehicle_name]["flying"]:
                logger.warning(f"无人机{vehicle_name}未处于飞行状态，无需降落")
                return True

            # 执行降落
            self.client.landAsync(vehicle_name=vehicle_name).join()
            self.vehicle_states[vehicle_name]["flying"] = False
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
            if not self.vehicle_states[vehicle_name]["flying"]:
                logger.error(f"无人机{vehicle_name}未处于飞行状态，无法移动")
                return False

            if speed <= 0:
                logger.error(f"无人机{vehicle_name}速度必须大于0")
                return False
            # 执行移动（匹配示例中的API调用）
            self.client.moveToPositionAsync(
                x, y, z, speed, vehicle_name=vehicle_name
            )
            
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
            if not self.vehicle_states[vehicle_name]["flying"]:
                logger.error(f"无人机{vehicle_name}未处于飞行状态，无法移动")
                return False

            if duration <= 0:
                logger.error(f"无人机{vehicle_name}持续时间必须大于0")
                return False
            # 执行移动（添加join等待异步操作完成）
            self.client.moveByVelocityAsync(
                x, y, z, duration, vehicle_name=vehicle_name
            ).join()
            
            self._update_vehicle_position(vehicle_name)
            logger.info(f"无人机{vehicle_name}移动向量({x},{y},{z})，持续时间{duration}秒")
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
        return self.vehicle_states[vehicle_name].copy()
    
    def _update_vehicle_state(self, vehicle_name: str) -> None:
        """更新无人机状态（完全适配示例API）"""
        try:
            # 更新飞行状态（使用正确的LandedState属性名）
            state = self.client.getMultirotorState(vehicle_name=vehicle_name)
            # 修正LandedState属性名（移除前缀）
            self.vehicle_states[vehicle_name]["flying"] = (
                state.landed_state == airsim.LandedState.Flying
            )
            
            # 更新位置
            self._update_vehicle_position(vehicle_name)
            
        except Exception as e:
            logger.warning(f"更新无人机{vehicle_name}状态失败: {str(e)}")
    
    def _update_vehicle_position(self, vehicle_name: str) -> None:
        """更新无人机位置信息"""
        try:
            position = self.client.getMultirotorState(vehicle_name=vehicle_name).kinematics_estimated.position
            self.vehicle_states[vehicle_name]["position"] = (position.x_val, position.y_val, position.z_val)
        except Exception as e:
            logger.warning(f"更新无人机{vehicle_name}位置失败: {str(e)}")