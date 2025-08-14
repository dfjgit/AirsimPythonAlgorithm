import setup_path
import airsim
import numpy as np
import os
import json
import socket
import threading
import base64
import io
import logging
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict

from PIL import Image

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("drone_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DroneServer")

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

            # 执行移动（匹配示例中的API调用）
            self.client.moveToPositionAsync(
                x, y, z, speed, vehicle_name=vehicle_name
            ).join()
            
            self._update_vehicle_position(vehicle_name)
            logger.info(f"无人机{vehicle_name}已移动到({x},{y},{z})")
            return True
        except Exception as e:
            logger.error(f"无人机{vehicle_name}移动操作失败: {str(e)}")
            return False

    def get_image(self, vehicle_name: Optional[str] = None, camera_name: str = "0", 
                 image_type: Any = airsim.ImageType.Scene) -> Optional[str]:
        """获取指定相机图像并返回Base64编码"""
        vehicle_name = vehicle_name or self.default_vehicle
        try:
            logger.info(f"获取{vehicle_name}的{camera_name}相机图像中...")
            
            if isinstance(image_type, str):
                image_type = getattr(airsim.ImageType, image_type, airsim.ImageType.DepthVis)
                
            # 匹配示例中的图像获取方式
            image_data = self.client.simGetImage(camera_name, image_type, vehicle_name=vehicle_name)
            if image_data:
                with Image.open(io.BytesIO(image_data)) as image:
                    logger.info(f"图像信息 - 无人机: {vehicle_name}, 相机: {camera_name}, 类型: {image_type}, "
                                f"尺寸: {image.size}, 格式: {image.format}, 模式: {image.mode}")
                return base64.b64encode(image_data).decode('utf-8')
            
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


class DroneSocketServer:
    def __init__(self, host: str = '0.0.0.0', port: int = 65432):
        self.host = host
        self.port = port
        self.drone = DroneController()
        self.server_socket: Optional[socket.socket] = None
        self.running = False
        self.client_handlers: List[threading.Thread] = []
        self.lock = threading.Lock()

    def start(self) -> None:
        """启动Socket服务器"""
        self.running = True
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            logger.info(f"无人机控制服务器已启动，监听 {self.host}:{self.port}")

            while self.running:
                try:
                    self.server_socket.settimeout(1.0)
                    client_socket, addr = self.server_socket.accept()
                    logger.info(f"新连接: {addr}")
                    
                    handler = threading.Thread(target=self.handle_client, args=(client_socket, addr))
                    handler.daemon = True
                    handler.start()
                    
                    with self.lock:
                        self.client_handlers.append(handler)
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        logger.error(f"接受连接时出错: {e}")

        except Exception as e:
            logger.error(f"服务器错误: {e}")
        finally:
            self.stop()

    def handle_client(self, client_socket: socket.socket, addr: Tuple[str, int]) -> None:
        """处理客户端连接"""
        handler_thread = threading.current_thread()
        try:
            with client_socket:
                client_socket.settimeout(60.0)
                logger.info(f"开始处理客户端: {addr}")
                
                while True:
                    try:
                        data = client_socket.recv(1024).decode('utf-8')
                        if not data:
                            logger.info(f"客户端断开连接: {addr}")
                            break

                        try:
                            command = json.loads(data)
                            logger.debug(f"收到命令 from {addr}: {command}")
                        except json.JSONDecodeError as e:
                            response = {"status": "error", "message": f"无效的JSON格式: {str(e)}"}
                            client_socket.sendall(json.dumps(response).encode('utf-8'))
                            continue

                        response = self.process_command(command)
                        client_socket.sendall(json.dumps(response).encode('utf-8'))

                    except UnicodeDecodeError:
                        response = {"status": "error", "message": "无法解码UTF-8数据"}
                        client_socket.sendall(json.dumps(response).encode('utf-8'))
                    except socket.timeout:
                        logger.warning(f"客户端{addr}超时未发送数据")
                        response = {"status": "error", "message": "连接超时"}
                        client_socket.sendall(json.dumps(response).encode('utf-8'))
                        break
                    except ConnectionResetError:
                        logger.info(f"客户端{addr}强制断开连接")
                        break
                    except Exception as e:
                        logger.error(f"处理客户端{addr}请求时出错: {str(e)}")
                        response = {"status": "error", "message": f"服务器内部错误: {str(e)}"}
                        try:
                            client_socket.sendall(json.dumps(response).encode('utf-8'))
                        except:
                            pass
                        break
        except Exception as e:
            logger.error(f"客户端{addr}处理线程出错: {str(e)}")
        finally:
            logger.info(f"客户端{addr}连接已关闭")
            with self.lock:
                if handler_thread in self.client_handlers:
                    self.client_handlers.remove(handler_thread)
            
    def process_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """处理命令并返回结果"""
        try:
            cmd = command.get("command", "").lower()
            params = command.get("params", {})
            
            if not cmd:
                return {"status": "error", "message": "命令不能为空"}

            vehicle_name = params.get("vehicle_name", self.drone.default_vehicle)

            if cmd == "connect":
                result = self.drone.connect()
                return {"status": "success", "message": "已连接到AirSim模拟器"} if result else \
                       {"status": "error", "message": "连接失败"}

            if not self.drone.connection_status:
                return {"status": "error", "message": "未连接到AirSim模拟器，请先执行connect命令"}

            if cmd == "reset":
                result = self.drone.reset()
                return {"status": "success", "message": "模拟器已重置"} if result else \
                       {"status": "error", "message": "模拟器重置失败"}

            elif cmd == "enable_api":
                enable = params.get("enable", True)
                if not isinstance(enable, bool):
                    return {"status": "error", "message": "enable参数必须是布尔值"}
                    
                result = self.drone.enable_api_control(enable, vehicle_name)
                return {"status": "success", "message": f"无人机{vehicle_name}API控制已{'启用' if enable else '禁用'}"} if result else \
                       {"status": "error", "message": f"无人机{vehicle_name}API控制{'启用' if enable else '禁用'}失败"}

            elif cmd == "arm":
                arm = params.get("arm", True)
                if not isinstance(arm, bool):
                    return {"status": "error", "message": "arm参数必须是布尔值"}
                    
                result = self.drone.arm_disarm(arm, vehicle_name)
                return {"status": "success", "message": f"无人机{vehicle_name}已{'解锁' if arm else '上锁'}"} if result else \
                       {"status": "error", "message": f"无人机{vehicle_name}{'解锁' if arm else '上锁'}失败"}

            elif cmd == "takeoff":
                result = self.drone.takeoff(vehicle_name)
                return {"status": "success", "message": f"无人机{vehicle_name}起飞完成"} if result else \
                       {"status": "error", "message": f"无人机{vehicle_name}起飞失败"}

            elif cmd == "land":
                result = self.drone.land(vehicle_name)
                return {"status": "success", "message": f"无人机{vehicle_name}降落完成"} if result else \
                       {"status": "error", "message": f"无人机{vehicle_name}降落失败"}

            elif cmd == "move_to_position":
                required_params = ["x", "y", "z"]
                for param in required_params:
                    if param not in params:
                        return {"status": "error", "message": f"缺少位置参数: {param}"}
                
                try:
                    x, y, z = float(params["x"]), float(params["y"]), float(params["z"])
                    speed = float(params.get("speed", 3))
                except ValueError:
                    return {"status": "error", "message": "位置或速度参数不是有效的数字"}
                    
                result = self.drone.move_to_position(x, y, z, speed, vehicle_name)
                return {"status": "success", "message": f"无人机{vehicle_name}已移动到({x},{y},{z})"} if result else \
                       {"status": "error", "message": f"无人机{vehicle_name}移动失败"}

            elif cmd == "get_image":
                camera_name = params.get("camera_name", "0")
                image_type = params.get("image_type", "Scene")
                
                image_b64 = self.drone.get_image(vehicle_name, camera_name, image_type)
                if image_b64:
                    return {"status": "success", "image_data": image_b64, "message": "图像获取成功"}
                else:
                    return {"status": "error", "message": f"无人机{vehicle_name}图像获取失败"}

            elif cmd == "get_state":
                state = self.drone.get_vehicle_state(vehicle_name)
                return {"status": "success", "state": state, "message": f"已获取无人机{vehicle_name}状态"}

            elif cmd == "disconnect":
                return {"status": "success", "message": "连接已关闭"}

            else:
                return {"status": "error", "message": f"未知命令: {cmd}"}

        except Exception as e:
            logger.error(f"处理命令时出错: {str(e)}")
            return {"status": "error", "message": str(e)}

    def stop(self) -> None:
        """停止服务器"""
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
                logger.info("服务器套接字已关闭")
            except Exception as e:
                logger.error(f"关闭服务器套接字时出错: {e}")
        
        with self.lock:
            for handler in self.client_handlers:
                try:
                    handler.join(timeout=5.0)
                    if handler.is_alive():
                        logger.warning("客户端处理线程未能及时终止")
                except Exception as e:
                    logger.error(f"等待客户端处理线程结束时出错: {e}")
        
        logger.info("服务器已完全停止")


if __name__ == "__main__":
    server = DroneSocketServer()
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止服务器...")
        server.stop()
    except Exception as e:
        logger.critical(f"服务器运行时发生致命错误: {e}", exc_info=True)
        server.stop()
