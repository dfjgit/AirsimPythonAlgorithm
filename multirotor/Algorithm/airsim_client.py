import json
import time
import logging
import sys
import os
import json
from typing import Dict, Any, Optional
import threading

# 导入项目中的必要模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AirsimServer.drone_controller import DroneController
from AirsimServer.data_storage import DataStorageManager
from Algorithm.scanner_algorithm import ScannerAlgorithm
from Algorithm.scannerData import ScannerData
from Algorithm.HexGridDataModelData import HexGridDataModel
from Algorithm.Vector3 import Vector3

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AirsimClient")

# 数据存储Key常量定义
# 配置文件Key（Python一次性配置）
KEY_SCANNER_CONFIG = "scanner_config"
KEY_DRONE_CONFIG = "drone_config"
KEY_ENVIRONMENT_CONFIG = "environment_config"
KEY_RESULT_CONFIG = "result_config"

# 数据交换Key
KEY_SCANNER_DATA = "scannerData"
KEY_GRID_DATA = "girdData"

class AirsimClient:
    """
    Airsim客户端算法类
    集成scanner_algorithm计算方向向量，控制无人机飞行，并存储数据供Unity可视化
    使用专门设计的Key实现与Unity的数据交换
    """
    def __init__(self, grid_data_file: str, scanner_data_file: str):
        # 初始化数据存储管理器
        self.data_manager = DataStorageManager()
        
        # 初始化无人机控制器
        self.drone_controller = DroneController()
        
        # 初始化扫描器算法
        self.scanner_algorithm = None
        
        # 初始化网格数据和扫描器数据
        self.grid_data_file = grid_data_file
        self.scanner_data_file = scanner_data_file
        self.hex_grid_model = None
        self.scanner_data = None
        
        # 控制标志
        self.running = False
        self.control_thread = None
        
        # 初始化
        self._initialize()
        
        # 设置配置信息
        self._setup_configurations()
    
    def _initialize(self):
        """初始化算法和数据"""
        try:
            # 加载网格数据
            logger.info(f"正在加载网格数据: {self.grid_data_file}")
            with open(self.grid_data_file, 'r') as f:
                grid_data_json = json.load(f)
            self.hex_grid_model = HexGridDataModel.from_dict(grid_data_json)
            logger.info("网格数据加载成功")
            
            # 加载扫描器数据
            logger.info(f"正在加载扫描器数据: {self.scanner_data_file}")
            with open(self.scanner_data_file, 'r') as f:
                scanner_data_json = json.load(f)
            self.scanner_data = ScannerData(scanner_data_json)
            logger.info("扫描器数据加载成功")
            
            # 初始化扫描器算法
            self.scanner_algorithm = ScannerAlgorithm(self.hex_grid_model, self.scanner_data)
            logger.info("扫描器算法初始化成功")
            
        except Exception as e:
            logger.error(f"初始化失败: {str(e)}")
            raise
            
    def _setup_configurations(self):
        """设置一次性配置信息，供Unity显示"""
        try:
            # 设置扫描器配置信息
            scanner_config = {
                "entropy_coefficient": self.scanner_data.entropyCoefficient,
                "repulsion_coefficient": self.scanner_data.repulsionCoefficient,
                "leader_range_coefficient": self.scanner_data.leaderRangeCoefficient,
                "direction_retention_coefficient": self.scanner_data.directionRetentionCoefficient,
                "distance_coefficient": self.scanner_data.distanceCoefficient,
                "update_interval": self.scanner_data.updateInterval
            }
            self.data_manager.store_data(KEY_SCANNER_CONFIG, scanner_config)
            logger.info(f"已设置{KEY_SCANNER_CONFIG}配置")
            
            # 设置无人机配置信息
            drone_config = {
                "move_speed": self.scanner_data.moveSpeed,
                "enable_collision_avoidance": False,  # 可根据实际需求设置
                "takeoff_height": 10.0,  # 默认起飞高度
                "mission_timeout": 3600  # 默认任务超时时间（秒）
            }
            self.data_manager.store_data(KEY_DRONE_CONFIG, drone_config)
            logger.info(f"已设置{KEY_DRONE_CONFIG}配置")
            
            # 设置环境配置信息
            environment_config = {
                "grid_size": len(self.hex_grid_model.cells),
                "grid_radius": 0,  # HexGridDataModel可能没有radius属性
                "world_boundary": {
                    "min_x": -100,  # 默认边界，可根据实际环境调整
                    "max_x": 100,
                    "min_y": -100,
                    "max_y": 100,
                    "min_z": -100,
                    "max_z": 100
                }
            }
            self.data_manager.store_data(KEY_ENVIRONMENT_CONFIG, environment_config)
            logger.info(f"已设置{KEY_ENVIRONMENT_CONFIG}配置")
            
            # 设置结果配置信息
            result_config = {
                "success_threshold": 0.95,  # 完成度阈值
                "save_interval": 10.0,  # 数据保存间隔（秒）
                "visualization_frequency": 10  # 可视化频率（Hz）
            }
            self.data_manager.store_data(KEY_RESULT_CONFIG, result_config)
            logger.info(f"已设置{KEY_RESULT_CONFIG}配置")
            
        except Exception as e:
            logger.error(f"设置配置信息时出错: {str(e)}")
            raise
            
    def retrieve_grid_data_from_unity(self):
        """从Unity获取网格数据"""
        try:
            result = self.data_manager.retrieve_data(KEY_GRID_DATA)
            if result.get("status") == "success":
                grid_data = result.get("content")
                logger.info(f"成功获取来自Unity的{KEY_GRID_DATA}")
                # 更新本地网格数据
                if grid_data:
                    self.hex_grid_model = HexGridDataModel.from_dict(grid_data)
                    # 如果已经初始化了算法，也更新算法中的网格数据
                    if self.scanner_algorithm:
                        self.scanner_algorithm.grid_model = self.hex_grid_model
                return True
            else:
                logger.warning(f"无法获取来自Unity的{KEY_GRID_DATA}: {result.get('message')}")
                return False
        except Exception as e:
            logger.error(f"获取网格数据时出错: {str(e)}")
            return False
            
    def retrieve_scanner_data_from_unity(self):
        """从Unity获取扫描数据（包括无人机坐标）"""
        try:
            result = self.data_manager.retrieve_data(KEY_SCANNER_DATA)
            if result.get("status") == "success":
                scanner_data = result.get("content")
                logger.info(f"成功获取来自Unity的{KEY_SCANNER_DATA}")
                # 更新本地扫描器数据
                if scanner_data:
                    # 解析无人机位置数据
                    if "position" in scanner_data:
                        pos = scanner_data["position"]
                        self.scanner_data.position = Vector3(pos["x"], pos["y"], pos["z"])
                    # 其他可能的更新...
                return True
            else:
                logger.warning(f"无法获取来自Unity的{KEY_SCANNER_DATA}: {result.get('message')}")
                return False
        except Exception as e:
            logger.error(f"获取扫描数据时出错: {str(e)}")
            return False
    
    def connect(self) -> bool:
        """连接到Airsim模拟器"""
        logger.info("正在连接到Airsim模拟器...")
        result = self.drone_controller.connect()
        if result:
            logger.info("连接成功")
            # 启用API控制
            self.drone_controller.enable_api_control(True)
            # 解锁无人机
            self.drone_controller.arm_disarm(True)
        else:
            logger.error("连接失败")
        return result
    
    def start_mission(self):
        """开始任务"""
        if self.running:
            logger.warning("任务已经在运行中")
            return
        
        if not self.drone_controller.connection_status:
            logger.error("未连接到Airsim模拟器，请先连接")
            return
        
        self.running = True
        self.control_thread = threading.Thread(target=self._mission_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        logger.info("任务已开始")
    
    def stop_mission(self):
        """停止任务"""
        if not self.running:
            logger.warning("任务没有在运行")
            return
        
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=5.0)
        logger.info("任务已停止")
    
    def _mission_loop(self):
        """任务主循环"""
        try:
            # 起飞
            logger.info("无人机起飞中...")
            self.drone_controller.takeoff()
            logger.info("无人机起飞完成")
            
            # 等待起飞稳定
            time.sleep(5)
            
            # 主循环
            while self.running:
                try:
                    # 从Unity获取最新的网格数据（每5个循环获取一次）
                    if int(time.time() * 10) % 5 == 0:
                        self.retrieve_grid_data_from_unity()
                        
                    # 从Unity获取最新的扫描数据（包括无人机坐标）
                    self.retrieve_scanner_data_from_unity()
                    
                    # 获取当前无人机状态作为备选
                    vehicle_state = self.drone_controller.get_vehicle_state()
                    
                    # 更新扫描器数据中的当前位置（优先使用Unity提供的数据，其次使用AirSim数据）
                    if not self.scanner_data.position and vehicle_state and "position" in vehicle_state:
                        pos = vehicle_state["position"]
                        self.scanner_data.position = Vector3(pos["x"], pos["y"], pos["z"])
                    
                    # 使用扫描器算法计算方向向量
                    direction_vector = self.scanner_algorithm.process(self.hex_grid_model, self.scanner_data)
                    
                    if direction_vector:
                        # 将方向向量应用于无人机控制
                        logger.info(f"应用方向向量: ({direction_vector.x}, {direction_vector.y}, {direction_vector.z})")
                        
                        # 使用move_by_velocity方法控制无人机
                        # 可以根据需要调整速度和持续时间
                        speed = self.scanner_data.moveSpeed
                        duration = 1.0  # 持续1秒
                        self.drone_controller.move_by_velocity(
                            direction_vector.x * speed,
                            direction_vector.y * speed,
                            direction_vector.z * speed,
                            duration
                        )
                        
                        # 存储处理后的扫描数据供Unity绘制
                        self._store_data_for_unity(
                            direction_vector,
                            self.scanner_data,
                            self.scanner_algorithm.get_visited_cells()
                        )
                    
                    # 控制循环频率
                    time.sleep(0.1)  # 10Hz控制频率
                    
                except Exception as e:
                    logger.error(f"任务循环出错: {str(e)}")
                    time.sleep(1)  # 出错后暂停1秒
            
            # 任务结束，降落无人机
            logger.info("任务结束，无人机降落中...")
            self.drone_controller.land()
            logger.info("无人机降落完成")
            
        except Exception as e:
            logger.error(f"任务执行出错: {str(e)}")
    
    def _store_data_for_unity(self, direction_vector: Vector3, scanner_data: ScannerData, visited_cells: list):
        """存储处理后的扫描数据供Unity绘制"""
        try:
            # 准备要存储的数据（包含方向向量数据）
            processed_scanner_data = {
                "timestamp": time.time(),
                "direction_vector": direction_vector.to_dict(),
                "current_position": scanner_data.current_position.to_dict() if scanner_data.current_position else None,
                "target_position": scanner_data.target_position.to_dict() if scanner_data.target_position else None,
                "move_speed": scanner_data.move_speed,
                "visited_cells": visited_cells,
                "max_visited_count": scanner_data.max_visited_count,
                "min_entropy": scanner_data.min_entropy,
                "max_entropy": scanner_data.max_entropy,
                "entropy_weight": scanner_data.entropy_weight,
                "repulsion_weight": scanner_data.repulsion_weight,
                "leader_weight": scanner_data.leader_weight,
                "direction_persistence_weight": scanner_data.direction_persistence_weight
            }
            
            # 存储处理后的扫描数据，使用KEY_SCANNER_DATA键
            self.data_manager.store_data(KEY_SCANNER_DATA, processed_scanner_data)
            logger.debug(f"已更新{KEY_SCANNER_DATA}数据")
            
            # 同时存储当前的网格数据，使用KEY_GRID_DATA键
            if self.hex_grid_model:
                grid_data = self.hex_grid_model.to_dict()
                self.data_manager.store_data(KEY_GRID_DATA, grid_data)
                logger.debug(f"已更新{KEY_GRID_DATA}数据")
                
        except Exception as e:
            logger.error(f"存储数据供Unity绘制时出错: {str(e)}")
    
    def disconnect(self):
        """断开与Airsim模拟器的连接"""
        if self.running:
            self.stop_mission()
            
        if self.drone_controller.connection_status:
            # 上锁无人机
            self.drone_controller.arm_disarm(False)
            # 禁用API控制
            self.drone_controller.enable_api_control(False)
            logger.info("已断开与Airsim模拟器的连接")

# 主函数，用于测试客户端算法
if __name__ == "__main__":
    import sys
    import os
    
    # 设置默认文件路径
    DEFAULT_GRID_DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hex_grid_data.json")
    DEFAULT_SCANNER_DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scanner_data.json")
    
    # 获取命令行参数或使用默认值
    grid_data_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GRID_DATA_FILE
    scanner_data_file = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_SCANNER_DATA_FILE
    
    try:
        # 创建并初始化客户端
        client = AirsimClient(grid_data_file, scanner_data_file)
        
        # 连接到Airsim
        if client.connect():
            # 启动任务
            client.start_mission()
            
            # 运行一段时间后停止（这里设置为30秒，可以根据需要调整）
            mission_duration = 30  # 30秒
            logger.info(f"任务将运行{mission_duration}秒...")
            time.sleep(mission_duration)
            
            # 停止任务
            client.stop_mission()
            
        else:
            logger.error("无法连接到Airsim，任务无法启动")
            
    except KeyboardInterrupt:
        logger.info("用户中断程序")
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
    finally:
        # 确保断开连接
        if 'client' in locals():
            client.disconnect()
        
        logger.info("程序已退出")