import sys
import time
import logging
import json
import threading
import os
from typing import Dict, Any, Optional
import traceback


# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AlgorithmServer")

# 导入必要的模块
from AirsimServer.drone_controller import DroneController
from AirsimServer.unity_socket_server import UnitySocketServer
from Algorithm.scanner_algorithm import ScannerAlgorithm
from Algorithm.scanner_config_data import ScannerConfigData
from Algorithm.scanner_runtime_data import ScannerRuntimeData
from Algorithm.HexGridDataModel import HexGridDataModel
from Algorithm.Vector3 import Vector3

class AlgorithmServer:
    """
    融合算法和AirsimServer的主类
    通过Socket从Unity获取网格数据和运行时数据
    直接调用Airsim进行操作，将计算结果通过Socket发送给Unity
    """
    def __init__(self, config_file=None):
        # 如果没有提供配置文件路径，使用默认路径
        if config_file is None:
            config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scanner_config.json")
        # 初始化控制器
        self.drone_controller = DroneController()
        # 初始化配置数据
        self.config_data = ScannerConfigData(config_file)
        # 初始化运行时数据（将从Unity接收）
        self.unity_runtime_data = ScannerRuntimeData()
        # 初始化网格数据（将从Unity接收）
        self.unity_grid_data = HexGridDataModel()
        # 算法计算后的运行时数据
        self.processed_runtime_data = ScannerRuntimeData()
        # 初始化算法
        self.algorithm = ScannerAlgorithm(
            config_data=self.config_data
        )
        # 线程控制标志
        self.running = False
        # 无人机处理线程
        self.drone_thread = None
        # 保存最后一次无人机状态
        self.last_state = None
        # 初始化Unity Socket服务器
        self.unity_socket_server = UnitySocketServer()
        # 注册数据接收回调函数（关键：从Unity获取数据）
        self.unity_socket_server.set_data_received_callback(self._on_unity_data_received)
        # 数据同步锁
        self.data_lock = threading.Lock()

    def connect(self) -> bool:
        """
        启动Socket服务器并等待Unity连接，然后再连接到AirSim模拟器
        """
        logger.info("正在启动Unity Socket服务器...")
        # 先启动Unity Socket服务器（关键：用于接收Unity数据）
        if not self.unity_socket_server.start():
            logger.error("Unity Socket服务器启动失败")
            return False
        
        # 等待Unity连接，最多等待120秒
        max_wait_time = 120.0
        start_time = time.time()
        logger.info("等待Unity客户端连接...")
        
        while time.time() - start_time < max_wait_time:
            # 检查Unity连接状态（通过connection是否存在来判断）
            if hasattr(self.unity_socket_server, 'connection') and self.unity_socket_server.connection is not None:
                logger.info("Unity客户端已成功连接")
                # Unity连接成功后，立即发送初始配置数据
                self.unity_socket_server.send_config_data(self.config_data)
                logger.info("已向Unity发送初始配置数据")
                
                # 再连接到AirSim模拟器
                logger.info("正在连接到AirSim模拟器...")
                result = self.drone_controller.connect()
                if result:
                    logger.info("成功连接到AirSim模拟器")
                    # 启用API控制
                    self.drone_controller.enable_api_control(True)
                    # 解锁无人机
                    self.drone_controller.arm_disarm(True)
                    return True
                else:
                    logger.error("连接到AirSim模拟器失败")
                    return False
            
            time.sleep(0.5)  # 每0.5秒检查一次
        
        logger.error(f"等待Unity连接超时（{max_wait_time}秒）")
        self.unity_socket_server.stop()  # 停止Socket服务器
        return False

    def disconnect(self) -> None:
        """
        断开与AirSim模拟器的连接并停止Socket服务器
        """
        logger.info("正在断开与AirSim模拟器的连接...")
        # 停止运行
        self.running = False
        # 等待线程结束
        if self.drone_thread and self.drone_thread.is_alive():
            self.drone_thread.join(5.0)  # 等待最多5秒
        # 上锁无人机
        try:
            self.drone_controller.arm_disarm(False)
            # 禁用API控制
            self.drone_controller.enable_api_control(False)
        except Exception as e:
            logger.error(f"断开连接时出错: {str(e)}")
        # 停止Unity Socket服务器
        self.unity_socket_server.stop()
        logger.info("已断开与AirSim模拟器的连接")

    def takeoff(self) -> bool:
        """
        控制无人机起飞
        """
        logger.info("无人机准备起飞...")
        result = self.drone_controller.takeoff()
        if result:
            logger.info("无人机起飞成功")
            # 起飞后上升到指定高度
            self.drone_controller.move_to_position(
                0, 0, -self.config_data.altitude, 2.0
            )
        else:
            logger.error("无人机起飞失败")
        return result

    def land(self) -> bool:
        """
        控制无人机降落
        """
        logger.info("无人机准备降落...")
        result = self.drone_controller.land()
        if result:
            logger.info("无人机降落成功")
        else:
            logger.error("无人机降落失败")
        return result

    def _on_unity_data_received(self, received_data: Dict[str, Any]) -> None:
        """处理从Unity接收到的数据（关键：获取网格和运行时数据，响应配置请求）
        
        Args:
            received_data: 从Unity接收到的数据，包含grid_data、runtime_data或配置请求
        """
        try:
            # 打印接收到的Unity数据类型
            data_types = []
            if 'request' in received_data:
                data_types.append(f"request:{received_data['request']}")
            if 'grid_data' in received_data:
                data_types.append("grid_data")
            if 'runtime_data' in received_data:
                data_types.append("runtime_data")
            
            # 记录数据类型信息
            if data_types:
                logger.info(f"接收到Unity数据，类型: {', '.join(data_types)}")
            else:
                logger.warning("接收到Unity数据，但未检测到已知的数据类型")
                logger.debug(f"接收到的原始数据结构: {list(received_data.keys())}")
            
            with self.data_lock:  # 确保线程安全
                # 处理配置数据请求
                if 'request' in received_data and received_data['request'] == 'config_data':
                    logger.debug("收到Unity的配置数据请求，准备发送当前配置")
                    self.unity_socket_server.send_config_data(self.config_data)
                    return
                
                # 解析并更新网格数据
                if 'grid_data' in received_data and received_data['grid_data'] is not None:
                    logger.debug("更新Unity发送的grid_data")
                    self.unity_grid_data.from_dict(received_data['grid_data'])
                
                # 解析并更新运行时数据
                if 'runtime_data' in received_data and received_data['runtime_data'] is not None:
                    logger.debug("更新Unity发送的runtime_data")
                    self.unity_runtime_data.from_dict(received_data['runtime_data'])
                    
                    # 同步无人机位置到本地运行时数据
                    self.last_state = {
                        "position_x": self.unity_runtime_data.position.x,
                        "position_y": self.unity_runtime_data.position.y,
                        "position_z": self.unity_runtime_data.position.z
                    }
                    
        except Exception as e:
            logger.error(f"处理Unity数据时出错: {str(e)}")
            logger.debug(traceback.format_exc())

    def process_drone_and_algorithm(self) -> None:
        """
        处理无人机移动和算法计算（使用从Unity获取的数据）
        在单独线程中运行
        """
        while self.running:
            try:
                with self.data_lock:  # 确保数据读取线程安全
                    # 检查是否已从Unity获取到必要数据
                    if not self.unity_grid_data.cells or not self.unity_runtime_data.position:
                        time.sleep(0.1)
                        continue

                    # 1. 使用从Unity获取的网格数据和运行时数据执行算法计算
                    # 计算权重
                    weights = self.algorithm.calculate_weights(
                        self.unity_grid_data, 
                        self.unity_runtime_data
                    )
                    
                    # 计算最佳移动方向
                    best_direction = self.algorithm.calculate_score_direction(
                        weights, 
                        self.unity_runtime_data.position, 
                        self.unity_grid_data, 
                        self.unity_runtime_data
                    )
                    
                    # 更新运行时数据（算法计算结果）
                    self.processed_runtime_data = self.algorithm.update_runtime_data(
                        self.unity_grid_data, 
                        self.unity_runtime_data
                    )
                    self.processed_runtime_data.finalMoveDir = best_direction

                    # 2. 控制无人机移动（使用算法计算结果）
                    current_position = self.unity_runtime_data.position
                    # 计算新位置（基于最佳方向）
                    move_distance = self.config_data.moveSpeed * self.config_data.updateInterval
                    new_position = Vector3(
                        current_position.x + best_direction.x * move_distance,
                        current_position.y + best_direction.y * move_distance,
                        current_position.z + best_direction.z * move_distance
                    )
                    
                    # 移动到新位置（Z轴取负值因为AirSim中Z轴向下为正）
                    self.drone_controller.move_to_position(
                        new_position.x, 
                        new_position.y, 
                        -self.config_data.altitude,  # 保持指定高度
                        self.config_data.moveSpeed
                    )
                    
                    logger.info(f"无人机移动指令: 从({current_position.x}, {current_position.y})到({new_position.x}, {new_position.y})")

                    # 3. 发送算法处理后的运行时数据到Unity
                    self.unity_socket_server.send_runtime_data(self.processed_runtime_data)

            except Exception as e:
                logger.error(f"处理无人机和算法时出错: {str(e)}")
                logger.debug(traceback.format_exc())
            
            # 按照配置的更新频率休眠
            time.sleep(self.config_data.updateInterval)

    def run(self) -> None:
        """
        启动算法服务器的主方法
        """
        try:
            # 连接到AirSim和启动Socket
            if not self.connect():
                return
            
            # 起飞
            if not self.takeoff():
                self.disconnect()
                return
            
            # 设置运行标志
            self.running = True
            
            # 创建并启动主处理线程（融合数据处理和无人机控制）
            self.drone_thread = threading.Thread(target=self.process_drone_and_algorithm)
            self.drone_thread.daemon = True
            self.drone_thread.start()
            
            logger.info("AlgorithmServer已成功启动，正在等待Unity数据...")
            
            # 主循环，保持程序运行
            try:
                while self.running:
                    time.sleep(1)  # 避免CPU占用过高
            except KeyboardInterrupt:
                logger.info("接收到中断信号，正在停止...")
                self.running = False
            
        finally:
            # 确保断开连接
            self.disconnect()
            logger.info("AlgorithmServer已停止")

    def load_config(self, config_file: str) -> Dict[str, Any]:
        """加载配置文件并更新算法，同时通知Unity配置已更新"""
        try:
            self.config_data = ScannerConfigData(config_file)
            self.algorithm = ScannerAlgorithm(config_data=self.config_data)
            logger.info(f"已加载配置文件: {config_file}")
            # 配置更新后主动通知Unity
            self.unity_socket_server.send_config_data(self.config_data)
            return {
                "status": "success",
                "message": f"已加载配置文件: {config_file}",
                "config": self.config_data.to_dict()
            }
        except Exception as e:
            logger.error(f"加载配置文件时出错: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

if __name__ == "__main__":
    server = AlgorithmServer()
    server.run()
    