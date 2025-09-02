import logging
from typing import Dict, Any, Optional
import base64

# 配置日志
logger = logging.getLogger("CommandProcessor")

class CommandProcessor:
    """
    处理从客户端接收到的命令
    解析命令并调用相应的功能模块
    """
    def __init__(self, drone_controller, data_manager=None):
        self.drone_controller = drone_controller
        self.data_manager = data_manager

    def process_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理命令并返回结果
        
        :param command: 包含命令和参数的字典
        :return: 包含处理结果的字典
        """
        try:
            cmd = command.get("command", "").lower()
            params = command.get("params", {})
            
            if not cmd:
                return {"status": "error", "message": "命令不能为空"}

            vehicle_name = params.get("vehicle_name", self.drone_controller.default_vehicle)

            # 处理数据存储相关命令
            if self.data_manager and cmd in ["store_data", "retrieve_data", "delete_data", "list_data_ids"]:
                return self._process_data_commands(cmd, params)

            # 处理无人机控制相关命令
            if cmd == "connect":
                result = self.drone_controller.connect()
                return {"status": "success", "message": "已连接到AirSim模拟器"} if result else \
                       {"status": "error", "message": "连接失败"}

            if not self.drone_controller.connection_status:
                return {"status": "error", "message": "未连接到AirSim模拟器，请先执行connect命令"}

            if cmd == "reset":
                result = self.drone_controller.reset()
                return {"status": "success", "message": "模拟器已重置"} if result else \
                       {"status": "error", "message": "模拟器重置失败"}

            elif cmd == "enable_api":
                enable = params.get("enable", True)
                if not isinstance(enable, bool):
                    return {"status": "error", "message": "enable参数必须是布尔值"}
                    
                result = self.drone_controller.enable_api_control(enable, vehicle_name)
                return {"status": "success", "message": f"无人机{vehicle_name}API控制已{'启用' if enable else '禁用'}"} if result else \
                       {"status": "error", "message": f"无人机{vehicle_name}API控制{'启用' if enable else '禁用'}失败"}

            elif cmd == "arm":
                arm = params.get("arm", True)
                if not isinstance(arm, bool):
                    return {"status": "error", "message": "arm参数必须是布尔值"}
                    
                result = self.drone_controller.arm_disarm(arm, vehicle_name)
                return {"status": "success", "message": f"无人机{vehicle_name}已{'解锁' if arm else '上锁'}"} if result else \
                       {"status": "error", "message": f"无人机{vehicle_name}{'解锁' if arm else '上锁'}失败"}

            elif cmd == "takeoff":
                result = self.drone_controller.takeoff(vehicle_name)
                return {"status": "success", "message": f"无人机{vehicle_name}起飞完成"} if result else \
                       {"status": "error", "message": f"无人机{vehicle_name}起飞失败"}

            elif cmd == "land":
                result = self.drone_controller.land(vehicle_name)
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
                    
                result = self.drone_controller.move_to_position(x, y, z, speed, vehicle_name)
                return {"status": "success", "message": f"无人机{vehicle_name}已移动到({x},{y},{z})"} if result else \
                       {"status": "error", "message": f"无人机{vehicle_name}移动失败"}

            elif cmd == "get_image":
                camera_name = params.get("camera_name", "0")
                image_type = params.get("image_type", "Scene")
                
                image_data = self.drone_controller.get_image(vehicle_name, camera_name, image_type)
                if image_data:
                    return {"status": "success", "image_data": base64.b64encode(image_data).decode('utf-8'), "message": "图像获取成功"}
                else:
                    return {"status": "error", "message": f"无人机{vehicle_name}图像获取失败"}

            elif cmd == "get_state":
                state = self.drone_controller.get_vehicle_state(vehicle_name)
                return {"status": "success", "state": state, "message": f"已获取无人机{vehicle_name}状态"}

            elif cmd == "disconnect":
                return {"status": "success", "message": "连接已关闭"}

            else:
                return {"status": "error", "message": f"未知命令: {cmd}"}

        except Exception as e:
            logger.error(f"处理命令时出错: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _process_data_commands(self, cmd: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理数据存储相关命令
        
        :param cmd: 命令名称
        :param params: 命令参数
        :return: 包含处理结果的字典
        """
        if cmd == "store_data":
            data_id = params.get("data_id")
            content = params.get("content")
            return self.data_manager.store_data(data_id, content)
            
        elif cmd == "retrieve_data":
            data_id = params.get("data_id")
            return self.data_manager.retrieve_data(data_id)
            
        elif cmd == "delete_data":
            data_id = params.get("data_id")
            return self.data_manager.delete_data(data_id)
            
        elif cmd == "list_data_ids":
            return self.data_manager.list_data_ids()
            
        return {"status": "error", "message": f"未知的数据命令: {cmd}"}