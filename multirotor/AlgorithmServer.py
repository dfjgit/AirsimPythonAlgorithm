import sys
import time
import logging
import json
import threading
import os
from typing import Dict, Any, Optional, List, Tuple
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

class MultiDroneAlgorithmServer:
    """
    支持多无人机的融合算法和AirsimServer的主类
    通过Socket从Unity获取网格数据和运行时数据
    直接调用Airsim进行多无人机操作，将计算结果通过Socket发送给Unity
    """
    def __init__(self, config_file=None, drone_names: List[str] = None):
        # 如果没有提供配置文件路径，使用默认路径
        if config_file is None:
            config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scanner_config.json")
        # 初始化无人机名称列表，默认使用UAV1, UAV2, UAV3
        self.drone_names = drone_names if drone_names else ["UAV1", "UAV2", "UAV3"]
        
        # 初始化控制器
        self.drone_controller = DroneController()
        # 初始化配置数据
        self.config_data = ScannerConfigData(config_file)
        
        # 为每个无人机维护独立的数据结构
        self.unity_runtime_data = {name: ScannerRuntimeData() for name in self.drone_names}
        self.processed_runtime_data = {name: ScannerRuntimeData() for name in self.drone_names}
        self.algorithms = {name: ScannerAlgorithm(config_data=self.config_data) for name in self.drone_names}
        self.last_states = {name: None for name in self.drone_names}
        self.drone_threads = {name: None for name in self.drone_names}
        
        # 共享的网格数据（从Unity接收）
        self.unity_grid_data = HexGridDataModel()
        
        # 线程控制标志
        self.running = False
        # 初始化Unity Socket服务器
        self.unity_socket_server = UnitySocketServer()
        # 注册数据接收回调函数
        self.unity_socket_server.set_data_received_callback(self._on_unity_data_received)
        # 数据同步锁
        self.data_lock = threading.Lock()
        self.grid_lock = threading.Lock()

    def connect(self) -> bool:
        """
        启动Socket服务器并等待Unity连接，然后再连接到AirSim模拟器
        """
        logger.info("正在启动Unity Socket服务器...")
        # 先启动Unity Socket服务器
        if not self.unity_socket_server.start():
            logger.error("Unity Socket服务器启动失败")
            return False
        
        # 等待Unity连接，最多等待120秒
        max_wait_time = 120.0
        start_time = time.time()
        logger.info("等待Unity客户端连接...")
        
        while time.time() - start_time < max_wait_time:
            # 检查Unity连接状态
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
                    # 为所有无人机启用API控制并解锁
                    for drone_name in self.drone_names:
                        self.drone_controller.enable_api_control(True, drone_name)
                        self.drone_controller.arm_disarm(True, drone_name)
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
        # 等待所有无人机线程结束
        for drone_name in self.drone_names:
            if self.drone_threads[drone_name] and self.drone_threads[drone_name].is_alive():
                self.drone_threads[drone_name].join(5.0)  # 等待最多5秒
        
        # 为所有无人机上锁并禁用API控制
        try:
            for drone_name in self.drone_names:
                self.drone_controller.arm_disarm(False, drone_name)
                self.drone_controller.enable_api_control(False, drone_name)
        except Exception as e:
            logger.error(f"断开连接时出错: {str(e)}")
        
        # 停止Unity Socket服务器
        self.unity_socket_server.stop()
        logger.info("已断开与AirSim模拟器的连接")

    def takeoff_all(self) -> bool:
        """
        控制所有无人机起飞
        """
        logger.info("所有无人机准备起飞...")
        all_success = True
        for drone_name in self.drone_names:
            logger.info(f"无人机{drone_name}准备起飞...")
            result = self.drone_controller.takeoff(drone_name)
            if result:
                logger.info(f"无人机{drone_name}起飞成功")
            else:
                logger.error(f"无人机{drone_name}起飞失败")
                all_success = False
            time.sleep(1)  # 错开起飞时间
        return all_success

    def land_all(self) -> bool:
        """
        控制所有无人机降落
        """
        logger.info("所有无人机准备降落...")
        all_success = True
        for drone_name in self.drone_names:
            logger.info(f"无人机{drone_name}准备降落...")
            result = self.drone_controller.land(drone_name)
            if result:
                logger.info(f"无人机{drone_name}降落成功")
            else:
                logger.error(f"无人机{drone_name}降落失败")
                all_success = False
            time.sleep(1)  # 错降落时间
        return all_success

    def _on_unity_data_received(self, received_data: Dict[str, Any]) -> None:
        """处理从Unity接收到的数据
        
        Args:
            received_data: 从Unity接收到的数据，包含grid_data、runtime_data或配置请求
            注意：数据类型已由UnitySocketServer处理并确定
        """
        try:
            # 直接使用UnitySocketServer处理后的数据类型，不再自行判断
            data_type_info = []
            if 'request' in received_data:
                data_type_info.append(f"request:{received_data['request']}")
            if 'grid_data' in received_data:
                data_type_info.append("grid_data")
            if 'runtime_data' in received_data:
                data_type_info.append("runtime_data")
            
            # 记录数据类型信息
            if data_type_info:
                logger.info(f"接收到Unity数据，类型: {', '.join(data_type_info)}")
            else:
                logger.warning("接收到Unity数据，但未检测到已知的数据类型")
                logger.debug(f"接收到的原始数据结构: {list(received_data.keys())}")
            
            with self.data_lock:  # 确保线程安全
                # 处理配置数据请求
                if 'request' in received_data and received_data['request'] == 'config_data':
                    logger.debug("收到Unity的配置数据请求，准备发送当前配置")
                    self.unity_socket_server.send_config_data(self.config_data)
                    return
                
                # 解析并更新网格数据（共享数据）
                if 'grid_data' in received_data and received_data['grid_data'] is not None:
                    with self.grid_lock:
                        logger.debug("更新Unity发送的grid_data")
                        self.unity_grid_data.from_dict(received_data['grid_data'])
                
                # 解析并更新运行时数据（区分不同无人机）
                if 'runtime_data' in received_data and received_data['runtime_data'] is not None:
                    # 使用顶级的uav_name字段来标识无人机
                    if 'uav_name' in received_data:
                        drone_name = received_data['uav_name']
                        if drone_name in self.unity_runtime_data:
                            logger.debug(f"更新Unity发送的{drone_name}的runtime_data")
                            # 创建临时runtime_data对象进行解析
                            temp_runtime = ScannerRuntimeData.from_dict(received_data['runtime_data'])
                            # 保存无人机名称
                            self.unity_runtime_data[drone_name] = temp_runtime
                            
                            # 同步无人机位置到本地运行时数据
                            self.last_states[drone_name] = {
                                "position_x": temp_runtime.position.x,
                                "position_y": temp_runtime.position.y,
                                "position_z": temp_runtime.position.z
                            }
                        else:
                            logger.warning(f"收到未知无人机{drone_name}的运行时数据")
                    else:
                        logger.warning("运行时数据中未包含无人机标识(drone_name)")
                        
        except Exception as e:
            logger.error(f"处理Unity数据时出错: {str(e)}")
            logger.debug(traceback.format_exc())

    def _coordinate_drones(self) -> Dict[str, Vector3]:
        """
        多无人机协同控制逻辑，为每个无人机分配不同的目标区域
        避免无人机之间的碰撞
        """
        with self.grid_lock:
            # 获取所有无人机当前位置
            positions = {name: data.position for name, data in self.unity_runtime_data.items()}
            
            # 简单的区域分配策略：基于网格单元的熵值和无人机位置进行分配
            target_directions = {}
            
            # 为每个无人机计算避开其他无人机的方向修正
            for drone_name in self.drone_names:
                current_pos = positions[drone_name]
                avoidance_dir = Vector3()
                
                # 检查与其他无人机的距离，添加避碰方向
                for other_name, other_pos in positions.items():
                    if drone_name != other_name:
                        diff = current_pos - other_pos
                        distance = diff.magnitude()
                        # 如果距离小于安全距离，计算避碰方向
                        if distance < self.config_data.collisionAvoidanceRadius:
                            avoidance_dir = avoidance_dir + diff.normalized() * (self.config_data.collisionAvoidanceRadius - distance)
                
                target_directions[drone_name] = avoidance_dir
        
        return target_directions

    def process_single_drone(self, drone_name: str) -> None:
        """
        处理单个无人机的移动和算法计算
        在单独线程中运行
        """
        while self.running:
            try:
                with self.data_lock:  # 确保数据读取线程安全
                    # 检查是否已从Unity获取到必要数据
                    with self.grid_lock:
                        has_grid_data = bool(self.unity_grid_data.cells)
                    
                    if not has_grid_data or not self.unity_runtime_data[drone_name].position:
                        time.sleep(0.1)
                        continue

                    # 1. 获取协同控制方向（避碰等）
                    avoidance_directions = self._coordinate_drones()
                    avoidance_dir = avoidance_directions[drone_name]

                    # 2. 使用从Unity获取的网格数据和运行时数据执行算法计算
                    # 计算权重
                    weights = self.algorithms[drone_name].calculate_weights(
                        self.unity_grid_data, 
                        self.unity_runtime_data[drone_name]
                    )
                    
                    # 计算最佳移动方向
                    best_direction = self.algorithms[drone_name].calculate_score_direction(
                        weights, 
                        self.unity_runtime_data[drone_name].position, 
                        self.unity_grid_data, 
                        self.unity_runtime_data[drone_name]
                    )
                    
                    # 记录最佳方向的详细信息
                    logger.debug(f"无人机{drone_name}的最佳方向: ({best_direction.x:.4f}, {best_direction.y:.4f}, {best_direction.z:.4f}), 模长: {best_direction.magnitude():.4f}")
                    
                    # 应用避碰修正
                    if avoidance_dir.magnitude() > 0.01:
                        logger.debug(f"无人机{drone_name}的避碰方向: ({avoidance_dir.x:.4f}, {avoidance_dir.y:.4f}, {avoidance_dir.z:.4f})，权重: {self.config_data.avoidanceWeight}")
                        combined_dir = best_direction * (1 - self.config_data.avoidanceWeight) + \
                                      avoidance_dir.normalized() * self.config_data.avoidanceWeight
                        best_direction = combined_dir.normalized()
                        logger.debug(f"无人机{drone_name}的最终方向: ({best_direction.x:.4f}, {best_direction.y:.4f}, {best_direction.z:.4f})")
                    
                    # 更新运行时数据（算法计算结果）
                    self.processed_runtime_data[drone_name] = self.algorithms[drone_name].update_runtime_data(
                        self.unity_grid_data, 
                        self.unity_runtime_data[drone_name]
                    )
                    self.processed_runtime_data[drone_name].finalMoveDir = best_direction
                    # 添加无人机名称标识
                    self.processed_runtime_data[drone_name].drone_name = drone_name

                    # 3. 控制无人机移动（使用算法计算结果）
                    current_position = self.unity_runtime_data[drone_name].position
                    # 计算新位置（基于最佳方向）
                    move_distance = self.config_data.moveSpeed * self.config_data.updateInterval
                    new_position = Vector3(
                        current_position.x + best_direction.x * move_distance,
                        current_position.y + best_direction.y * move_distance,
                        current_position.z + best_direction.z * move_distance
                    )
                    
                    # 使用速度控制无人机移动
                    velocity_x = best_direction.x * self.config_data.moveSpeed
                    velocity_y = best_direction.y * self.config_data.moveSpeed
                    velocity_z = 0  # 保持当前高度
                    
                    # 记录移动参数
                    logger.debug(f"无人机{drone_name}移动参数 - 速度: ({velocity_x:.4f}, {velocity_y:.4f}, {velocity_z:.4f}), 持续时间: {duration:.4f}秒, 移动速度配置: {self.config_data.moveSpeed}, 更新间隔: {self.config_data.updateInterval}")
                    
                    # 计算持续时间（基于配置的更新间隔）
                    duration = self.config_data.updateInterval
                    
                    # 调用移动方法并记录结果
                    move_result = self.drone_controller.move_by_velocity(
                        velocity_x, 
                        velocity_y, 
                        velocity_z, 
                        duration, 
                        drone_name
                    )
                    
                    if move_result:
                        logger.info(f"无人机{drone_name}移动指令执行成功: 从({current_position.x:.2f}, {current_position.y:.2f})到({new_position.x:.2f}, {new_position.y:.2f})")
                    else:
                        logger.error(f"无人机{drone_name}移动指令执行失败")

                    # 4. 发送算法处理后的运行时数据到Unity
                    logger.debug(f"准备发送算法处理后的运行时数据到Unity，无人机: {drone_name}")
                    self.unity_socket_server.send_runtime_data(self.processed_runtime_data[drone_name])

            except Exception as e:
                logger.error(f"处理无人机{drone_name}和算法时出错: {str(e)}")
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
            
            # 所有无人机起飞
            if not self.takeoff_all():
                self.disconnect()
                return
            
            # 设置运行标志
            self.running = True
            
            # 为每个无人机创建并启动独立的处理线程
            for drone_name in self.drone_names:
                self.drone_threads[drone_name] = threading.Thread(
                    target=self.process_single_drone, 
                    args=(drone_name,)
                )
                self.drone_threads[drone_name].daemon = True
                self.drone_threads[drone_name].start()
            
            logger.info(f"MultiDroneAlgorithmServer已成功启动，正在控制{len(self.drone_names)}架无人机...")
            
            # 主循环，保持程序运行并检查Unity连接状态
            try:
                while self.running:
                    time.sleep(1)  # 避免CPU占用过高
                    
                    # 检查Unity连接状态
                    if hasattr(self.unity_socket_server, 'connection') and self.unity_socket_server.connection is None:
                        logger.warning("检测到Unity已断开Socket连接，正在断开与Airsim的连接...")
                        self.running = False
            except KeyboardInterrupt:
                logger.info("接收到中断信号，正在停止...")
                self.running = False
            
        finally:
            # 确保断开连接
            self.disconnect()
            logger.info("MultiDroneAlgorithmServer已停止")

    def load_config(self, config_file: str) -> Dict[str, Any]:
        """加载配置文件并更新算法，同时通知Unity配置已更新"""
        try:
            self.config_data = ScannerConfigData(config_file)
            # 更新所有无人机的算法配置
            for drone_name in self.drone_names:
                self.algorithms[drone_name] = ScannerAlgorithm(config_data=self.config_data)
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
    # 可以通过命令行参数指定无人机数量或名称列表
    drone_names = ["UAV1"]  # 默认使用1架无人机
    if len(sys.argv) > 1:
        try:
            num_drones = int(sys.argv[1])
            drone_names = [f"UAV{i+1}" for i in range(num_drones)]
        except ValueError:
            drone_names = sys.argv[1].split(',')
    
    server = MultiDroneAlgorithmServer(drone_names=drone_names)
    server.run()