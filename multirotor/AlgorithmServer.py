import sys
import time
import logging
import json
import threading
import os
import sys
from typing import Dict, Any, Optional, List, Tuple
import traceback
from pathlib import Path
import numpy as np

# 配置日志系统
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("AlgorithmServer")

# 导入核心模块
from AirsimServer.drone_controller import DroneController
from AirsimServer.unity_socket_server import UnitySocketServer
from Algorithm.scanner_algorithm import ScannerAlgorithm
from Algorithm.scanner_config_data import ScannerConfigData
from Algorithm.scanner_runtime_data import ScannerRuntimeData
from Algorithm.HexGridDataModel import HexGridDataModel
from Algorithm.Vector3 import Vector3
from AirsimServer.data_pack import PackType

# 尝试导入可视化模块
try:
    from Algorithm.simple_visualizer import SimpleVisualizer
    HAS_VISUALIZATION = True
except ImportError as e:
    logging.warning(f"无法导入可视化模块: {str(e)}")
    HAS_VISUALIZATION = False

class MultiDroneAlgorithmServer:
    """
    多无人机算法服务核心类
    功能：连接AirSim模拟器与Unity客户端，处理数据交互，执行扫描算法，控制多无人机协同作业
    """

    def __init__(self, config_file: Optional[str] = None, drone_names: Optional[List[str]] = None, enable_learning: Optional[bool] = None):
        """
        初始化服务器实例
        :param config_file: 算法配置文件路径（默认使用scanner_config.json）
        :param drone_names: 无人机名称列表（默认使用["UAV1", "UAV2", "UAV3"]）
        :param enable_learning: 是否启用DQN学习（None表示从配置文件读取）
        """
        # 配置文件路径处理
        self.config_path = self._resolve_config_path(config_file)
        # 无人机名称初始化
        self.drone_names = drone_names if drone_names else ["UAV1"]
        logger.info(f"初始化多无人机算法服务，控制无人机: {self.drone_names}")

        # 核心组件初始化
        self.drone_controller = DroneController()  # 无人机控制器
        self.unity_socket = UnitySocketServer()  # Unity通信Socket服务
        self.config_data = self._load_config()  # 算法配置数据

        # 数据存储结构（按无人机名称区分）
        self.unity_runtime_data: Dict[str, ScannerRuntimeData] = {
            name: ScannerRuntimeData() for name in self.drone_names
        }
        self.processed_runtime_data: Dict[str, ScannerRuntimeData] = {
            name: ScannerRuntimeData() for name in self.drone_names
        }
        self.algorithms: Dict[str, ScannerAlgorithm] = {
            name: ScannerAlgorithm(self.config_data) for name in self.drone_names
        }
        self.last_positions: Dict[str, Dict[str, float]] = {
            name: {} for name in self.drone_names
        }

        # 共享网格数据
        self.grid_data = HexGridDataModel()

        # 线程与状态管理
        self.running = False
        self.drone_threads: Dict[str, Optional[threading.Thread]] = {
            name: None for name in self.drone_names
        }
        self.data_lock = threading.Lock()  # 运行时数据锁
        self.grid_lock = threading.Lock()  # 网格数据锁
        
        # 可视化组件
        self.visualizer = None

        # 注册Unity数据接收回调
        self.unity_socket.set_callback(self._handle_unity_data)
        
        # DQN学习相关初始化
        # 如果enable_learning为None，则从配置文件读取
        if enable_learning is None:
            self.enable_learning = self.config_data.dqn_enabled
        else:
            self.enable_learning = enable_learning
            
        self.learning_envs = {}
        self.dqn_agents = {}
        
        if self.enable_learning:
            self._init_dqn_learning()
            
        # 初始化可视化组件
        self._init_visualization()

    def _resolve_config_path(self, config_file: Optional[str]) -> str:
        """解析配置文件路径，默认使用项目根目录下的scanner_config.json"""
        if config_file:
            if os.path.exists(config_file):
                return config_file
            logger.warning(f"指定的配置文件不存在: {config_file}，将使用默认配置")

        default_path = Path(__file__).parent / "scanner_config.json"
        if not default_path.exists():
            raise FileNotFoundError(f"默认配置文件不存在: {default_path}")
        return str(default_path)

    def _load_config(self) -> ScannerConfigData:
        """加载并解析配置文件"""
        try:
            logger.info(f"加载配置文件: {self.config_path}")
            return ScannerConfigData(self.config_path)
        except Exception as e:
            logger.error(f"配置文件加载失败: {str(e)}")
            raise

    def _init_visualization(self):
        """初始化可视化组件"""
        if HAS_VISUALIZATION:
            try:
                self.visualizer = SimpleVisualizer(self)
                logger.info("可视化组件初始化成功")
            except Exception as e:
                logger.warning(f"可视化组件初始化失败: {str(e)}")
                self.visualizer = None

    def start(self) -> bool:
        """启动服务主流程：连接Unity与AirSim，初始化无人机"""
        try:
            # 1. 启动Unity Socket服务并等待连接
            if not self._start_unity_socket():
                return False

            # 2. 连接AirSim模拟器
            if not self._connect_airsim():
                self.unity_socket.stop()
                return False

            # 3. 初始化无人机（启用API控制、解锁）
            if not self._init_drones():
                self._disconnect_airsim()
                self.unity_socket.stop()
                return False

            # 4. 启动可视化（如果已初始化）
            if self.visualizer:
                self.visualizer.start_visualization()
                logger.info("可视化功能已启动")

            logger.info("服务初始化成功")
            return True
        except Exception as e:
            logger.error(f"服务启动失败: {str(e)}")
            self.stop()
            return False

    def _start_unity_socket(self) -> bool:
        """启动Unity Socket服务并等待连接"""
        logger.info("启动Unity Socket服务...")
        if not self.unity_socket.start():
            logger.error("Unity Socket服务启动失败")
            return False

        # 等待Unity连接（超时120秒）
        timeout = 120
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.unity_socket.is_connected():
                logger.info("Unity客户端已连接")
                self.unity_socket.send_config(self.config_data)
                # logger.info("已发送初始配置数据到Unity")
                return True
            time.sleep(0.5)

        logger.error(f"等待Unity连接超时（{timeout}秒）")
        return False

    def _connect_airsim(self) -> bool:
        """连接到AirSim模拟器"""
        logger.info("连接到AirSim模拟器...")
        if self.drone_controller.connect():
            logger.info("AirSim连接成功")
            # 起飞前先重置airsim
            self.drone_controller.reset()
            return True
        logger.error("AirSim连接失败")
        return False

    def _init_drones(self) -> bool:
        """初始化无人机：启用API控制并解锁"""
        all_success = True
        for drone_name in self.drone_names:
            if not self.drone_controller.enable_api_control(True, drone_name):
                logger.error(f"无人机{drone_name}启用API控制失败")
                all_success = False
            if not self.drone_controller.arm_disarm(True, drone_name):
                logger.error(f"无人机{drone_name}解锁失败")
                all_success = False
        return all_success

    def start_mission(self) -> bool:
        """开始任务：控制所有无人机起飞并启动算法线程"""
        if not self.running:

            # 1. 所有无人机起飞
            if not self._takeoff_all():
                return False
            time.sleep(1)
            # 2. 启动算法处理线程
            self.running = True
            for drone_name in self.drone_names:
                self.drone_threads[drone_name] = threading.Thread(
                    target=self._process_drone,
                    args=(drone_name,),
                    daemon=True
                )
                self.drone_threads[drone_name].start()
                logger.info(f"无人机{drone_name}算法线程启动")

            logger.info("所有无人机任务启动完成")
            return True
        logger.warning("任务已在运行中")
        return False

    def _takeoff_all(self) -> bool:
        """控制所有无人机起飞"""
        logger.info("开始所有无人机起飞流程")
        all_success = True
        for drone_name in self.drone_names:
            if not self.drone_controller.takeoff(drone_name):
                logger.error(f"无人机{drone_name}起飞失败")
                all_success = False
            time.sleep(1)  # 错开起飞时间，避免碰撞
        return all_success

    # 修改MultiDroneAlgorithmServer类中的_handle_unity_data方法
    def _handle_unity_data(self, received_data: Dict[str, Any]) -> None:
        """处理从Unity接收的新格式数据
        注意：unity_socket_server.py会将原始DataPacks格式转换为包含特定数据类型的字典
        例如：{runtime_data: [...], time_span: "..."} 或 {grid_data: {...}, time_span: "..."}
        """
        try:
            with self.data_lock:
                # logger.debug(f"收到Unity数据: {received_data}")

                # 检查是否包含runtime_data字段
                if 'runtime_data' in received_data:
                    runtime_data_list = received_data['runtime_data']
                    if isinstance(runtime_data_list, list):
                        # logger.info(f"收到运行时数据，包含{len(runtime_data_list)}个无人机数据")
                        # 处理每个无人机的运行时数据
                        for runtime_data in runtime_data_list:
                            drone_name = runtime_data.get('uavname')
                            if drone_name in self.unity_runtime_data and isinstance(runtime_data, dict):
                                try:
                                    self.unity_runtime_data[drone_name] = ScannerRuntimeData.from_dict(runtime_data)
                                    # 更新位置信息
                                    pos = self.unity_runtime_data[drone_name].position
                                    self.last_positions[drone_name] = {
                                        'x': pos.x,
                                        'y': pos.y,
                                        'z': pos.z,
                                        'timestamp': time.time()
                                    }
                                    # logger.debug(f"更新无人机{drone_name}运行时数据: {pos}NAME：{self.unity_runtime_data[drone_name].uavname}NAME2{runtime_data['uavname']}")
                                except Exception as e:
                                    logger.error(f"解析无人机{drone_name}运行时数据失败: {str(e)}")
                            else:
                                logger.warning(f"无效的运行时数据或无人机名称: {drone_name}")

                # 检查是否包含grid_data字段
                elif 'grid_data' in received_data:
                    grid_data = received_data['grid_data']
                    if isinstance(grid_data, dict) and 'cells' in grid_data:
                        # logger.info(f"收到网格数据，包含{len(grid_data['cells'])}个单元")
                        with self.grid_lock:
                            self.grid_data.update_from_dict(grid_data)
                    else:
                        logger.warning(f"网格数据格式错误: {grid_data}")

                # 检查是否包含配置数据
                elif 'config_data' in received_data:
                    config_data = received_data['config_data']
                    logger.info("收到配置数据更新，准备重新加载配置")
                    try:
                        # 重新加载配置
                        temp_config = ScannerConfigData.from_dict(config_data)
                        self.config_data = temp_config
                        # 更新所有无人机的算法配置
                        for algo in self.algorithms.values():
                            algo.config = self.config_data
                        logger.info("配置数据更新成功")
                    except Exception as e:
                        logger.error(f"更新配置数据失败: {str(e)}")

                # 未知数据类型处理
                else:
                    logger.warning(f"收到未知格式数据: {received_data}")

        except Exception as e:
            logger.error(f"处理Unity数据时发生错误: {str(e)}，堆栈信息: {traceback.format_exc()}")


    def _init_dqn_learning(self):
        """初始化DQN学习组件"""
        try:
            # 使用正确的相对导入方式
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from multirotor.DQN.DqnLearning import DQNAgent
            from multirotor.DQN.DroneLearningEnv import DroneLearningEnv
            
            for drone_name in self.drone_names:
                # 创建学习环境
                env = DroneLearningEnv(self, drone_name)
                self.learning_envs[drone_name] = env
                
                # 从配置中读取DQN参数
                config = self.config_data
                
                # 创建DQN智能体
                agent = DQNAgent(
                    state_dim=env.state_dim,
                    action_dim=env.action_dim,
                    lr=config.dqn_learning_rate,
                    gamma=config.dqn_gamma,
                    epsilon=config.dqn_epsilon,
                    epsilon_min=config.dqn_epsilon_min,
                    epsilon_decay=config.dqn_epsilon_decay,
                    batch_size=config.dqn_batch_size,
                    target_update=config.dqn_target_update,
                    memory_capacity=config.dqn_memory_capacity
                )
                self.dqn_agents[drone_name] = agent
                
                # 尝试加载已训练的模型
                try:
                    model_path = f"DQN/dqn_{drone_name}_model.pth"
                    agent.load_model(model_path)
                    logger.info(f"已加载无人机{drone_name}的DQN模型")
                except Exception as e:
                    logger.info(f"未找到无人机{drone_name}的DQN模型，将从头开始训练: {str(e)}")
        except Exception as e:
            logger.error(f"初始化DQN学习组件失败: {str(e)}")
            self.enable_learning = False
    
    def _process_drone(self, drone_name: str) -> None:
        """无人机算法处理线程：计算移动方向并控制无人机"""
        logger.info(f"无人机{drone_name}算法线程启动")
        while self.running:
            try:
                # 检查数据就绪状态
                has_grid = bool(self.grid_data.cells)
                has_runtime = bool(self.unity_runtime_data[drone_name].position)

                if not (has_grid and has_runtime):
                    time.sleep(1)
                    continue

                if self.enable_learning and drone_name in self.learning_envs and drone_name in self.dqn_agents:
                    # DQN学习模式
                    try:
                        # 获取当前状态
                        state = self.learning_envs[drone_name].get_state()
                        
                        # DQN智能体选择动作
                        action = self.dqn_agents[drone_name].select_action(state)
                        
                        # 执行动作并获取反馈
                        next_state, reward, done, info = self.learning_envs[drone_name].step(action)
                        
                        # 存储经验
                        self.dqn_agents[drone_name].memory.push(state, action, reward, next_state, done)
                        
                        # DQN智能体学习
                        self.dqn_agents[drone_name].learn()
                        
                        # 定期保存模型
                        save_interval = self.config_data.dqn_model_save_interval
                        if self.dqn_agents[drone_name].steps_done % save_interval == 0:
                            model_path = f"DQN/dqn_{drone_name}_model_{self.dqn_agents[drone_name].steps_done}.pth"
                            self.dqn_agents[drone_name].save_model(model_path)
                            logger.info(f"已保存无人机{drone_name}的DQN模型: {model_path}")
                        
                        # 执行算法计算最终方向（使用调整后的权重）
                        final_dir = self.algorithms[drone_name].update_runtime_data(
                            self.grid_data, self.unity_runtime_data[drone_name]
                        )
                        
                        # 控制无人机移动
                        self._control_drone_movement(drone_name, final_dir.finalMoveDir)
                        
                        # 发送处理后的数据到Unity
                        self._send_processed_data(drone_name, final_dir)
                        
                    except Exception as e:
                        logger.error(f"无人机{drone_name}DQN学习处理出错: {str(e)}")
                        # 出错时回退到传统模式
                        final_dir = self.algorithms[drone_name].update_runtime_data(
                            self.grid_data, self.unity_runtime_data[drone_name]
                        )
                        self._control_drone_movement(drone_name, final_dir.finalMoveDir)
                        self._send_processed_data(drone_name, final_dir)
                else:
                    # 传统人工势场算法模式
                    # 执行算法计算最终方向
                    final_dir = self.algorithms[drone_name].update_runtime_data(
                        self.grid_data, self.unity_runtime_data[drone_name]
                    )
                    
                    # 控制无人机移动
                    self._control_drone_movement(drone_name, final_dir.finalMoveDir)
                    
                    # 发送处理后的数据到Unity
                    self._send_processed_data(drone_name, final_dir)

                # 按配置间隔休眠
                time.sleep(self.config_data.updateInterval)

            except Exception as e:
                logger.error(f"无人机{drone_name}处理出错: {str(e)}")
                logger.debug(traceback.format_exc())
                time.sleep(self.config_data.updateInterval)  # 出错后延迟重试


    def _control_drone_movement(self, drone_name: str, direction: Vector3) -> None:
        """控制无人机按指定方向移动"""
        with self.data_lock:
            current_pos = self.unity_runtime_data[drone_name].position

        # 计算移动参数
        move_speed = self.config_data.moveSpeed
        velocity = direction * move_speed
        velocity = velocity.unity_to_air_sim()
        # 发送速度控制指令
        success = self.drone_controller.move_by_velocity(
            velocity.x, velocity.y, velocity.z,  # z方向速度为0（保持高度）
            self.config_data.updateInterval, drone_name
        )

        # if success :
        #
        # else:
        #     logger.error(f"无人机{drone_name}移动指令发送失败")


    def _send_processed_data(self, drone_name: str, scannerRuntimeData: ScannerRuntimeData) -> None:
        """发送处理后的运行时数据到Unity"""
        with self.data_lock:
            # 直接使用传入的scannerRuntimeData数据
            self.processed_runtime_data[drone_name] = scannerRuntimeData
            self.processed_runtime_data[drone_name].drone_name = drone_name
            # 发送到Unity - 注意：send_runtime需要一个可迭代对象（列表）
            self.unity_socket.send_runtime([self.processed_runtime_data[drone_name]])
            # logger.debug(f"已发送无人机{drone_name}的处理后数据到Unity")


    def stop(self) -> None:
        """停止服务：降落无人机，断开连接，清理资源"""
        self.running = False
        logger.info("开始停止服务...")

        # 停止可视化
        if self.visualizer:
            self.visualizer.stop_visualization()
            logger.info("可视化功能已停止")

        # 等待无人机线程结束
        # for drone_name, thread in self.drone_threads.items():
        #     if thread and thread.is_alive():
        #         thread.join(5)
        #         logger.info(f"无人机{drone_name}线程已停止")

        # 控制所有无人机降落
        # self._land_all()

        # 断开无人机连接
        # self._disconnect_airsim()

        # 停止Unity Socket服务
        self.unity_socket.stop()
        logger.info("服务已完全停止")


    def _land_all(self) -> None:
        """控制所有无人机降落"""
        logger.info("开始所有无人机降落流程")
        for drone_name in self.drone_names:
            if self.drone_controller.land(drone_name):
                logger.info(f"无人机{drone_name}降落成功")
            else:
                logger.error(f"无人机{drone_name}降落失败")
            time.sleep(1)


    def _disconnect_airsim(self) -> None:
        """断开与AirSim的连接"""
        try:
            for drone_name in self.drone_names:
                self.drone_controller.arm_disarm(False, drone_name)
                self.drone_controller.enable_api_control(False, drone_name)
            logger.info("已断开与AirSim的连接")
        except Exception as e:
            logger.error(f"断开AirSim连接出错: {str(e)}")


if __name__ == "__main__":
    try:
        # 创建服务器实例，启用DQN学习功能
        server = MultiDroneAlgorithmServer(enable_learning=True)
        if server.start():
            server.start_mission()
            # 主循环保持运行
            while server.running:
                time.sleep(1)
    except KeyboardInterrupt:
        logger.info("用户中断，停止服务")
    except Exception as e:
        logger.critical(f"服务运行出错: {str(e)}", exc_info=True)
    finally:
        if 'server' in locals():
            server.stop()
        sys.exit(0)
