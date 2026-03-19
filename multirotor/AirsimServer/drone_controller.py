import setup_path
import airsim
import numpy as np
import math
import time
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
    "OpticalFlowVis": airsim.ImageType.OpticalFlowVis,
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
                    "orientation": (0.0, 0.0, 0.0),
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
                    "orientation": (0.0, 0.0, 0.0),
                }
            # 创建新字典替换旧字典，完全避免修改现有对象
            new_state = dict(self.vehicle_states[vehicle_name])
            new_state[field] = value
            self.vehicle_states[vehicle_name] = new_state

    def connect(self) -> bool:
        """连接到AirSim模拟器"""
        try:
            self.client = airsim.MultirotorClient()
            with self.api_lock:
                self.client.confirmConnection()
            self.connection_status = True
            logger.info("成功连接到AirSim模拟器")
            return True
        except Exception as e:
            self.connection_status = False
            logger.error(f"连接到AirSim模拟器失败: {str(e)}")
            return False

    def reset(self) -> bool:
        """重置模拟器状态(增强防穿地保护)"""
        try:
            with self.api_lock:
                # 1. 暂停仿真,避免物理引擎在重置过程中导致穿地
                self.client.simPause(True)
                time.sleep(0.2)

                # 2. 执行重置
                self.client.reset()
                time.sleep(0.5)

                # 3. 多次暂停/恢复循环，确保物理引擎稳定
                for i in range(3):
                    self.client.simPause(False)
                    time.sleep(0.1)
                    self.client.simPause(True)
                    time.sleep(0.1)

                # 4. 最终恢复仿真
                self.client.simPause(False)
                time.sleep(0.3)

            logger.info("模拟器已安全重置(增强防穿地保护已启用)")
            self.vehicle_states.clear()
            return True
        except Exception as e:
            logger.error(f"重置模拟器失败: {str(e)}")
            # 确保仿真恢复运行
            try:
                with self.api_lock:
                    self.client.simPause(False)
            except:
                pass
            return False

    def enable_api_control(
        self, enable: bool = True, vehicle_name: Optional[str] = None
    ) -> bool:
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
                logger.error(
                    f"无人机{vehicle_name}API控制未启用，无法执行解锁/上锁操作"
                )
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

    def takeoff(
        self, vehicle_name: Optional[str] = None, timeout_sec: int = 30
    ) -> bool:
        """????????API?"""
        vehicle_name = vehicle_name or self.default_vehicle
        try:
            state = self._get_or_create_state(vehicle_name)
            if not state["api_enabled"]:
                logger.error(f"???{vehicle_name}API??????????")
                return False

            if state["flying"]:
                logger.warning(f"???{vehicle_name}???????")
                return True

            with self.api_lock:
                self.client.takeoffAsync(vehicle_name=vehicle_name).join()

            deadline = time.time() + max(1, timeout_sec)
            while time.time() < deadline:
                self._update_vehicle_state(vehicle_name)
                state = self._get_or_create_state(vehicle_name)
                pos = state.get("position", (0.0, 0.0, 0.0))
                altitude = -float(pos[2]) if pos is not None else 0.0
                if state.get("flying", False) and float(pos[2]) < -0.8:
                    logger.info(f"???{vehicle_name}????")
                    return True
                if altitude > 1.2:
                    logger.warning(
                        f"???{vehicle_name}???????? flying={state.get('flying', False)}, pos={pos}"
                    )
                    self._update_state_field(vehicle_name, "flying", True)
                    return True
                time.sleep(0.1)

            self._update_vehicle_state(vehicle_name)
            final_state = self._get_or_create_state(vehicle_name)
            final_pos = final_state.get("position", (0.0, 0.0, 0.0))
            logger.warning(
                f"???{vehicle_name}??????timeout={timeout_sec}s, "
                f"flying={final_state.get('flying', False)}, pos={final_pos}"
            )
            return False
        except Exception as e:
            logger.error(f"???{vehicle_name}??????: {str(e)}")
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
                # 状态更新也放在锁内
                self._update_vehicle_state_internal(vehicle_name)

            self._update_state_field(vehicle_name, "flying", False)
            logger.info(f"无人机{vehicle_name}降落完成")
            return True
        except Exception as e:
            logger.error(f"无人机{vehicle_name}降落操作失败: {str(e)}")
            return False

    def move_to_position(
        self,
        x: float,
        y: float,
        z: float,
        speed: float = 3,
        vehicle_name: Optional[str] = None,
        timeout_sec: int = 30,
    ) -> bool:
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

            # 执行移动并等待完成
            with self.api_lock:
                self.client.moveToPositionAsync(
                    x, y, z, speed, vehicle_name=vehicle_name
                ).join()
                # 状态更新也放在锁内
                self._update_vehicle_state_internal(vehicle_name)

            logger.info(f"无人机{vehicle_name}已移动到({x},{y},{z})")
            return True
        except Exception as e:
            logger.error(f"无人机{vehicle_name}移动操作失败: {str(e)}")
            return False

    def move_by_velocity(
        self,
        x: float,
        y: float,
        z: float,
        duration: float = 3,
        vehicle_name: Optional[str] = None,
        timeout_sec: int = 30,
    ) -> bool:
        """通过速度移动无人机（异步非阻塞版本）"""
        vehicle_name = vehicle_name or self.default_vehicle
        try:
            x = float(x)
            y = float(y)
            z = float(z)
            duration = float(duration)
            state = self._get_or_create_state(vehicle_name)
            if not state["flying"]:
                # 尝试同步一次状态
                self._update_vehicle_state(vehicle_name)
                state = self._get_or_create_state(vehicle_name)

                if not state["flying"]:
                    # 如果仍然不在飞行状态，尝试强制设置并继续（容错）
                    logger.warning(f"无人机{vehicle_name}状态未确认，尝试强制移动...")
                    self._update_state_field(vehicle_name, "flying", True)

            with self.api_lock:
                # 异步发送移动指令，不阻塞等待
                # 新指令会覆盖旧指令，实现平滑移动
                self.client.moveByVelocityAsync(
                    x, y, z, duration, vehicle_name=vehicle_name
                )
                # 状态更新也放在锁内
                self._update_vehicle_state_internal(vehicle_name)

            return True
        except Exception as e:
            logger.error(f"无人机{vehicle_name}速度移动失败: {str(e)}")
            return False

    def get_image(
        self,
        vehicle_name: Optional[str] = None,
        camera_name: str = "0",
        image_type: Any = "Scene",
    ) -> Optional[str]:
        """获取指定相机图像并返回Base64编码"""
        vehicle_name = vehicle_name or self.default_vehicle
        try:
            # 使用get方法获取值并提供默认值
            image_newType = IMAGE_TYPE_MAPPING.get(image_type, None)
            if image_newType is None:
                image_newType = airsim.ImageType.Scene
                logger.warning(f"未找到{image_type}对应的新类型，使用默认类型Scene")

            logger.info(
                f"获取{vehicle_name}的{camera_name}相机{image_newType}类型图像中..."
            )
            # 匹配示例中的图像获取方式
            with self.api_lock:
                image_data = self.client.simGetImage(
                    camera_name, image_newType, vehicle_name=vehicle_name
                )

            if image_data:
                import io
                from PIL import Image

                with Image.open(io.BytesIO(image_data)) as image:
                    logger.info(
                        f"图像信息 - 无人机: {vehicle_name}, 相机: {camera_name}, 类型: {image_newType}, 尺寸: {image.size}, 格式: {image.format}, 模式: {image.mode}"
                    )
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
            # 线程安全地一次性获取并更新状态，减少 API 调用次数和冲突概率
            with self.api_lock:
                self._update_vehicle_state_internal(vehicle_name)
        except Exception as e:
            logger.warning(f"更新无人机{vehicle_name}状态失败: {str(e)}")

    def _update_vehicle_state_internal(self, vehicle_name: str) -> None:
        """内部状态更新逻辑（假设已持有 api_lock）"""
        # 获取综合状态
        state = self.client.getMultirotorState(vehicle_name=vehicle_name)

        # 1. 更新飞行状态
        # LandedState: 0=Landed, 1=Flying, 2=TakingOff, 3=Landing
        # AirSim 有时会在已经离地后短暂仍返回 Landed，这里用高度做一次兜底。

        # 2. 更新位置
        pos = state.kinematics_estimated.position
        self._update_state_field(
            vehicle_name, "position", (pos.x_val, pos.y_val, pos.z_val)
        )

        altitude = -float(pos.z_val)
        flying_status = (
            state.landed_state == airsim.LandedState.Flying or altitude > 1.2
        )
        self._update_state_field(vehicle_name, "flying", flying_status)

        # 3. 更新姿态
        orientation_q = state.kinematics_estimated.orientation
        from airsim.utils import to_eularian_angles

        pitch, roll, yaw = to_eularian_angles(orientation_q)
        self._update_state_field(
            vehicle_name,
            "orientation",
            (math.degrees(roll), math.degrees(pitch), math.degrees(yaw)),
        )

    def _update_vehicle_position(self, vehicle_name: str) -> None:
        """更新无人机位置信息"""
        try:
            with self.api_lock:
                position = self.client.getMultirotorState(
                    vehicle_name=vehicle_name
                ).kinematics_estimated.position
            # 使用安全更新方法，避免numpy视图冲突
            self._update_state_field(
                vehicle_name,
                "position",
                (position.x_val, position.y_val, position.z_val),
            )
        except Exception as e:
            logger.warning(f"更新无人机{vehicle_name}位置失败: {str(e)}")

    def _update_vehicle_orientation(self, vehicle_name: str) -> None:
        """更新无人机姿态信息（欧拉角）"""
        try:
            with self.api_lock:
                orientation_q = self.client.getMultirotorState(
                    vehicle_name=vehicle_name
                ).kinematics_estimated.orientation
            from airsim.utils import to_eularian_angles

            pitch, roll, yaw = to_eularian_angles(orientation_q)
            # 使用安全更新方法，避免numpy视图冲突
            self._update_state_field(
                vehicle_name,
                "orientation",
                (math.degrees(roll), math.degrees(pitch), math.degrees(yaw)),
            )
        except Exception as e:
            logger.warning(f"更新无人机{vehicle_name}姿态失败: {str(e)}")

    def check_collision(self, vehicle_name: Optional[str] = None) -> Dict[str, Any]:
        """
        检查无人机碰撞状态

        Returns:
            Dict: {
                'has_collided': bool,
                'object_name': str,
                'penetration_depth': float,
                'impact_point': tuple,
                'normal': tuple
            }
        """
        vehicle_name = vehicle_name or self.default_vehicle
        try:
            with self.api_lock:
                collision_info = self.client.simGetCollisionInfo(vehicle_name)

            result = {
                "has_collided": collision_info.has_collided,
                "object_name": collision_info.object_name,
                "penetration_depth": collision_info.penetration_depth,
                "impact_point": (
                    collision_info.impact_point.x_val,
                    collision_info.impact_point.y_val,
                    collision_info.impact_point.z_val,
                ),
                "normal": (
                    collision_info.normal.x_val,
                    collision_info.normal.y_val,
                    collision_info.normal.z_val,
                ),
                "time_stamp": collision_info.time_stamp,
            }

            if result["has_collided"]:
                logger.warning(
                    f"无人机{vehicle_name}发生碰撞: 对象={result['object_name']}, "
                    f"穿透深度={result['penetration_depth']:.3f}m"
                )

            return result
        except Exception as e:
            logger.error(f"获取无人机{vehicle_name}碰撞信息失败: {str(e)}")
            return {
                "has_collided": False,
                "object_name": "",
                "penetration_depth": 0.0,
                "impact_point": (0, 0, 0),
                "normal": (0, 0, 0),
                "time_stamp": 0.0,
            }

    def recover_from_collision(self, vehicle_name: Optional[str] = None) -> bool:
        """
        从碰撞/穿地中恢复

        Returns:
            bool: 是否成功恢复
        """
        vehicle_name = vehicle_name or self.default_vehicle
        try:
            # 检查碰撞状态
            collision = self.check_collision(vehicle_name)

            # 检查是否穿地（高度异常低）
            with self.api_lock:
                pose = self.client.simGetVehiclePose(vehicle_name)
                current_z = pose.position.z_val
                is_underground = current_z > 0  # NED坐标系，z>0表示在地面以下

            if collision["has_collided"] or is_underground:
                logger.warning(f"检测到无人机{vehicle_name}异常状态，执行恢复...")
                if collision["has_collided"]:
                    logger.info(f"  - 碰撞检测: {collision['object_name']}")
                if is_underground:
                    logger.warning(f"  - 穿地检测: z={current_z:.2f}m")

                # 使用 simSetVehiclePose 强制设置位置（忽略碰撞）
                from airsim import Pose, Vector3r, Quaternionr

                with self.api_lock:
                    # 获取当前水平位置
                    current_pose = self.client.simGetVehiclePose(vehicle_name)

                    # 创建新的位置：保持水平位置，重置高度到安全位置
                    safe_pose = Pose()
                    safe_pose.position = Vector3r(
                        current_pose.position.x_val,
                        current_pose.position.y_val,
                        -3.0,  # 设置到3米高度（NED坐标系为负）
                    )
                    safe_pose.orientation = current_pose.orientation

                    # 强制设置位置，忽略碰撞检测
                    self.client.simSetVehiclePose(safe_pose, True, vehicle_name)
                    time.sleep(0.3)

                    # 执行悬停稳定
                    self.client.hoverAsync(vehicle_name=vehicle_name).join()
                    time.sleep(0.5)

                logger.info(f"无人机{vehicle_name}已恢复并悬停稳定")
                return True

            return False
        except Exception as e:
            logger.error(f"无人机{vehicle_name}恢复失败: {str(e)}")
            return False

    def reset_vehicle_to_pose(
        self,
        vehicle_name: Optional[str] = None,
        position: tuple = (0, 0, -3),
        ignore_collision: bool = True,
    ) -> bool:
        """
        将无人机重置到指定位置(可忽略碰撞)

        Args:
            vehicle_name: 无人机名称
            position: 目标位置 (x, y, z), NED坐标系
            ignore_collision: 是否忽略碰撞(用于从穿地状态恢复)

        Returns:
            bool: 是否成功
        """
        vehicle_name = vehicle_name or self.default_vehicle
        try:
            from airsim import Pose, Vector3r, Quaternionr

            # 创建姿态
            pose = Pose()
            pose.position = Vector3r(position[0], position[1], position[2])
            pose.orientation = Quaternionr(0, 0, 0, 1)  # 默认姿态

            with self.api_lock:
                self.client.simSetVehiclePose(pose, ignore_collision, vehicle_name)
                try:
                    self.client.hoverAsync(vehicle_name=vehicle_name).join()
                except Exception:
                    pass
                self._update_vehicle_state_internal(vehicle_name)

            time.sleep(0.2)
            logger.info(f"无人机{vehicle_name}已重置到位置{position}")
            return True
        except Exception as e:
            logger.error(f"无人机{vehicle_name}位置重置失败: {str(e)}")
            return False
