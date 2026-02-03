import sys
import time
import logging
import json
import math
import threading
import os
import sys
from typing import Dict, Any, Optional, List, Tuple
import traceback
from pathlib import Path
import numpy as np

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
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
from Algorithm.battery_data import BatteryManager, BatteryInfo, BatteryStatus  # 新增导入
from Algorithm.drones_config import DronesConfig
from Crazyswarm.crazyswarm import CrazyswarmManager
from Crazyswarm.crazyflie_operate import CrazyflieOperate
from Crazyswarm.crazyflie_logging_data import CrazyflieLoggingData
from Algorithm.Vector3 import Vector3
from Algorithm.data_collector import DataCollector
from AirsimServer.data_pack import PackType

# 尝试导入可视化模块
try:
    from Visualization import RuntimeVisualizer
    HAS_VISUALIZATION = True
except ImportError as e:
    logging.warning(f"无法导入可视化模块: {str(e)}")
    HAS_VISUALIZATION = False

class MultiDroneAlgorithmServer:
    """
    多无人机算法服务核心类
    功能：连接AirSim模拟器与Unity客户端，处理数据交互，执行扫描算法，控制多无人机协同作业
    """

    def __init__(self, config_file: Optional[str] = None, drone_names: Optional[List[str]] = None, use_learned_weights: bool = False, model_path: Optional[str] = None, enable_visualization: bool = True, enable_data_collection_print: bool = False, control_mode: str = 'apf'):
        """
        初始化服务器实例
        :param config_file: 算法配置文件路径（默认使用scanner_config.json）
        :param drone_names: 无人机名称列表（默认使用["UAV1", "UAV2", "UAV3"]）
        :param use_learned_weights: 是否使用学习的权重（DDPG模型预测，仅在control_mode='apf'时有效）
        :param model_path: DDPG模型路径（不含.zip后缀），如果为None则使用默认模型
        :param enable_visualization: 是否启用可视化（默认True）
        :param enable_data_collection_print: 是否启用数据采集DEBUG打印（默认False，训练模式下应设为True）
        :param control_mode: 控制模式，'apf'=APF算法控制（默认），'dqn'=DQN外部控制
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
        logger.info(f"配置文件加载完成 {self.drone_names}")
        
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

        # 加载无人机配置
        self.drones_config = DronesConfig()
        
        # 电量数据管理
        self.battery_manager = BatteryManager(self.config_data, self.drones_config)  
        self.battery_lock = threading.Lock()  # 电量数据锁

        self.crazyswarm = CrazyswarmManager(self.unity_socket, self.battery_manager, self.config_data, self.drones_config)

        # 共享网格数据
        self.grid_data = HexGridDataModel()

        # 线程与状态管理
        self.running = False
        self.drone_threads: Dict[str, Optional[threading.Thread]] = {
            name: None for name in self.drone_names
        }
        self.data_lock = threading.Lock()  # 运行时数据锁
        self.grid_lock = threading.Lock()  # 网格数据锁

        # 熵值记录
        self.entropy_history: List[Tuple[float, float]] = []
        self.entropy_history_lock = threading.Lock()
        self._start_time = time.time()
        self._last_entropy_record_time = 0.0
        self.entropy_dist_history: List[Tuple[float, List[int], List[float]]] = []
        self.entropy_bins: List[int] = []
        self.entropy_dist_history_lock = threading.Lock()
        
        # 可视化组件
        self.visualizer = None
        self.enable_visualization = enable_visualization

        # 数据采集系统（根据控制模式选择不同的数据目录）
        if control_mode.lower() == 'dqn':
            # DQN 模式：保存到 DQN_Movement/logs/dqn_scan_data
            dqn_data_dir = os.path.join(os.path.dirname(__file__), 'DQN_Movement', 'logs', 'dqn_scan_data')
            self.data_collector = DataCollector(
                data_dir=dqn_data_dir,
                collection_interval=1.0,
                enable_debug_print=enable_data_collection_print,
                training_prefix="dqn"
            )
        else:
            # APF/DDPG 模式：保存到 DDPG_Weight/airsim_training_logs（默认）
            self.data_collector = DataCollector(
                collection_interval=1.0,
                enable_debug_print=enable_data_collection_print
            )

        # 注册Unity数据接收回调
        self.unity_socket.set_callback(self._handle_unity_data)
        
        # 控制模式：'apf' 或 'dqn'
        self.control_mode = control_mode.lower()
        if self.control_mode not in ['apf', 'dqn']:
            logger.warning(f"未知的控制模式: {control_mode}，使用默认APF模式")
            self.control_mode = 'apf'
        logger.info(f"控制模式: {self.control_mode.upper()}")
        
        # DQN控制模式相关
        self.dqn_commands: Dict[str, Vector3] = {name: Vector3(0, 0, 0) for name in self.drone_names}  # 存储DQN移动指令
        self.dqn_command_lock = threading.Lock()  # DQN指令锁
        
        # DDPG权重预测（仅在APF模式下使用）
        self.use_learned_weights = use_learned_weights and (self.control_mode == 'apf')
        self.model_path = model_path  # 保存模型路径参数
        self.weight_model = None
        if self.use_learned_weights:
            self._init_weight_predictor()
        elif use_learned_weights and self.control_mode == 'dqn':
            logger.info("DQN控制模式下，use_learned_weights参数被忽略")
        
        # 初始化可视化组件（如果启用）
        if self.enable_visualization:
            self._init_visualization()
        else:
            logger.info("可视化已禁用")

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

    def _init_weight_predictor(self):
        """初始化权重预测器（DDPG模型）"""
        try:
            logger.info("=" * 60)
            logger.info("🔧 初始化DDPG权重预测器...")
            from stable_baselines3 import DDPG
            
            # 确定模型路径
            if self.model_path:
                # 使用用户指定的模型路径
                if os.path.isabs(self.model_path):
                    model_path = self.model_path
                else:
                    # 相对路径，相对于当前文件所在目录
                    model_path = os.path.join(os.path.dirname(__file__), self.model_path)
                logger.info(f"📂 使用指定模型: {model_path}")
            else:
                # 使用默认模型路径（优先级：best_model > weight_predictor_airsim > weight_predictor_simple）
                models_dir = os.path.join(os.path.dirname(__file__), 'DDPG_Weight', 'models')
                
                # 尝试多个默认模型
                default_models = [
                    os.path.join(models_dir, 'best_model'),
                    os.path.join(models_dir, 'weight_predictor_airsim'),
                    os.path.join(models_dir, 'weight_predictor_simple')
                ]
                
                model_path = None
                for candidate in default_models:
                    if os.path.exists(candidate + '.zip'):
                        model_path = candidate
                        logger.info(f"📂 使用默认模型: {os.path.basename(model_path)}")
                        break
                
                if not model_path:
                    logger.warning("❌ 未找到任何可用的模型文件")
                    logger.info("💡 可用模型列表：")
                    if os.path.exists(models_dir):
                        for f in os.listdir(models_dir):
                            if f.endswith('.zip'):
                                logger.info(f"   - {f}")
                    logger.warning("将使用配置文件中的固定权重")
                    self.use_learned_weights = False
                    logger.info("=" * 60)
                    return
            
            # 加载模型
            if os.path.exists(model_path + '.zip'):
                self.weight_model = DDPG.load(model_path)
                logger.info("=" * 60)
                logger.info("✅ DDPG权重预测模型加载成功！")
                logger.info(f"📦 模型文件: {model_path}.zip")
                logger.info("=" * 60)
            else:
                logger.warning(f"❌ 模型文件不存在: {model_path}.zip")
                logger.warning("将使用配置文件中的固定权重")
                self.use_learned_weights = False
                logger.info("=" * 60)
                
        except ImportError:
            logger.error("=" * 60)
            logger.error("❌ stable-baselines3未安装，无法使用DDPG权重预测")
            logger.info("💡 安装方法: pip install stable-baselines3")
            self.use_learned_weights = False
            logger.info("=" * 60)
        except Exception as e:
            logger.error("=" * 60)
            logger.error(f"❌ 权重预测器初始化失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self.use_learned_weights = False
            logger.info("=" * 60)
    
    def _init_visualization(self):
        """初始化可视化组件"""
        logger.info("=" * 60)
        logger.info("🎨 初始化可视化组件...")
        
        if not HAS_VISUALIZATION:
            logger.warning("❌ 可视化模块未导入（SimpleVisualizer导入失败）")
            logger.info("💡 请检查是否安装了pygame: pip install pygame")
            logger.info("=" * 60)
            self.visualizer = None
            return
        
        try:
            self.visualizer = RuntimeVisualizer(self)
            logger.info("✅ 可视化组件初始化成功")
            logger.info("💡 可视化将在start()后启动")
            logger.info("=" * 60)
        except Exception as e:
            logger.warning("=" * 60)
            logger.warning(f"❌ 可视化组件初始化失败: {str(e)}")
            import traceback
            logger.warning(traceback.format_exc())
            logger.info("💡 系统将继续运行，但不显示可视化界面")
            logger.info("=" * 60)
            self.visualizer = None

    def get_battery_voltage(self, drone_name: str) -> float:
        """获取指定无人机的当前电压"""
        return self.battery_manager.get_voltage(drone_name)

    def update_battery_voltage(self, drone_name: str, action_intensity: float = 0.0) -> float:
        """更新指定无人机的电量消耗
        :param drone_name: 无人机名称
        :param action_intensity: 动作强度（0-1），影响额外消耗
        :return: 更新后的电压值
        """
        return self.battery_manager.update_voltage(drone_name, action_intensity)

    def reset_battery_voltage(self, drone_name: str) -> float:
        """重置指定无人机的电量为初始值"""
        return self.battery_manager.reset_voltage(drone_name)

    def set_training_stats(self, episode: int, step: int, reward: float, total_reward: float):
        """设置训练统计信息，用于数据采集记录"""
        if self.data_collector:
            self.data_collector.set_external_data('episode', episode)
            self.data_collector.set_external_data('step', step)
            self.data_collector.set_external_data('reward', reward)
            self.data_collector.set_external_data('total_reward', total_reward)

    def set_experiment_meta(self, algorithm_type: str, env_type: str, control_mode: str):
        """
        设置实验元数据，用于数据采集的统一标签
        :param algorithm_type: 算法类型 (如 'hrl_dqn_apf', 'pure_dqn', 'ddpg_apf')
        :param env_type: 环境类型 (如 'hierarchical', 'movement', 'weight')
        :param control_mode: 控制模式 (如 'dqn', 'apf')
        """
        if self.data_collector:
            self.data_collector.set_external_data('algorithm_type', algorithm_type)
            self.data_collector.set_external_data('env_type', env_type)
            self.data_collector.set_external_data('control_mode', control_mode)
            logger.info(f"实验元数据已设置: {algorithm_type}/{env_type}/{control_mode}")

    def get_all_battery_data(self) -> Dict[str, Dict[str, float]]:
        """获取所有无人机的电量数据"""
        return self.battery_manager.get_all_battery_data()

    def set_battery_consumption_rate(self, drone_name: str, rate: float) -> None:
        """设置指定无人机的电量消耗率"""
        self.battery_manager.set_consumption_rate(drone_name, rate)

    # 新增方法：获取完整的电池信息
    def get_battery_info(self, drone_name: str) -> Optional[BatteryInfo]:
        """获取指定无人机的完整电池信息"""
        return self.battery_manager.get_battery_info(drone_name)

    # 新增方法：保存电池数据到文件
    def save_battery_data(self, file_path: str) -> None:
        """保存电池数据到JSON文件"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.battery_manager.to_json())
            logger.info(f"电池数据已保存到: {file_path}")
        except Exception as e:
            logger.error(f"保存电池数据失败: {str(e)}")

    # 新增方法：从文件加载电池数据
    def load_battery_data(self, file_path: str) -> None:
        """从JSON文件加载电池数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = f.read()
            self.battery_manager = BatteryManager.from_json(json_data)
            logger.info(f"电池数据已从文件加载: {file_path}")
        except Exception as e:
            logger.error(f"加载电池数据失败: {str(e)}")

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
                logger.info("=" * 60)
                logger.info("🎨 启动可视化线程...")
                if self.visualizer.start_visualization():
                    logger.info("✅ 可视化线程已启动")
                    logger.info("💡 可视化窗口应该会弹出")
                else:
                    logger.warning("❌ 可视化线程启动失败")
                logger.info("=" * 60)

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
            logger.info("重置AirSim模拟器...")
            self.drone_controller.reset()
            # 重置后等待几秒，让系统稳定
            logger.info("等待AirSim系统稳定...")
            time.sleep(3)
            return True
        logger.error("AirSim连接失败")
        return False

    def _init_drones(self) -> bool:
        """初始化无人机：启用API控制并解锁"""
        all_success = True
        for drone_name in self.drone_names:
            # 添加是否为实体无人机镜像判断
            if not self.drones_config.is_crazyflie_mirror(drone_name):
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
            logger.info("准备开始任务，等待系统完全稳定...")
            time.sleep(2)  # 额外等待2秒确保系统稳定

            # 1. 所有无人机起飞
            if not self._takeoff_all():
                return False
            
            # 起飞后等待更长时间，确保无人机稳定
            logger.info("无人机起飞完成，等待稳定...")
            time.sleep(3)
            
            # 2. 发送开始仿真指令到Unity（让领导者开始移动）
            if self.unity_socket and self.unity_socket.is_connected():
                logger.info("发送开始仿真指令到Unity...")
                self.unity_socket.send_start_simulation_command()
                time.sleep(0.5)  # 等待Unity处理指令
            
            # 3. 启动算法处理线程
            logger.info("启动算法处理线程...")
            self.running = True
            for drone_name in self.drone_names:
                self.drone_threads[drone_name] = threading.Thread(
                    target=self._process_drone,
                    args=(drone_name,),
                    daemon=True
                )
                self.drone_threads[drone_name].start()
                logger.info(f"无人机{drone_name}算法线程启动")

            # 4. 启动数据采集线程
            logger.info("启动数据采集线程...")
            self.data_collector.start(
                get_grid_data_func=lambda: self.grid_data,
                get_runtime_data_func=lambda: self.unity_runtime_data,
                get_algorithms_func=lambda: self.algorithms,
                get_drone_names_func=lambda: self.drone_names,
                get_battery_data_func=lambda: self.get_all_battery_data(),
                data_lock=self.data_lock,
                grid_lock=self.grid_lock
            )

            logger.info("所有无人机任务启动完成")
            return True
        logger.warning("任务已在运行中")
        return False

    def _takeoff_all(self) -> bool:
        """控制所有无人机起飞"""
        logger.info("开始所有无人机起飞流程")
        all_success = True
        for drone_name in self.drone_names:
            # 添加是否为实体无人机镜像判断
            if self.drones_config.is_crazyflie_mirror(drone_name):
                self.crazyswarm.take_off(drone_name, 0.5, 2)
            else:
                logger.info(f"无人机{drone_name}准备起飞...")
                if not self.drone_controller.takeoff(drone_name):
                    logger.error(f"无人机{drone_name}起飞失败")
                    all_success = False
                else:
                    logger.info(f"无人机{drone_name}起飞成功")
            time.sleep(2)  # 增加延迟时间，确保每个无人机起飞后稳定
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
                                    
                                except Exception as e:
                                    logger.error(f"解析无人机{drone_name}运行时数据失败: {str(e)}")
                                    logger.error(f"原始数据: {runtime_data}")
                            else:
                                logger.warning(f"无效的运行时数据或无人机名称: {drone_name}")

                # 检查是否包含grid_data字段
                elif 'grid_data' in received_data:
                    grid_data = received_data['grid_data']
                    if isinstance(grid_data, dict) and 'cells' in grid_data:
                        cells_count = len(grid_data['cells'])
                        # logger.debug(f"收到网格数据，包含{cells_count}个单元（Delta更新）")
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
                elif 'crazyflie_logging' in received_data:
                    try:
                        crazyflie_logging_json = CrazyflieLoggingData.from_json_to_dicts(received_data['crazyflie_logging'])
                        crazyflie_logging_list = CrazyflieLoggingData.from_dict_list(crazyflie_logging_json)
                        # logger.info("收到Crazyflies实体无人机日志数据更新")
                        self.crazyswarm.update_crazyflies_logging(crazyflie_logging_list)
                    except Exception as e:
                        logger.error(f"更新Crazyflies实体无人机日志数据失败: {str(e)}")
                # 未知数据类型处理
                else:
                    logger.warning(f"收到未知格式数据: {received_data}")

        except Exception as e:
            logger.error(f"处理Unity数据时发生错误: {str(e)}，堆栈信息: {traceback.format_exc()}")
        finally:
            self._record_entropy_snapshot()


    def _crazyflie_get_state_for_prediction(self, drone_name: str) -> np.ndarray:
        """提取Crazyflie实体无人机状态用于权重预测（18维）"""
        try:
            with self.data_lock:
                runtime_data = self.unity_runtime_data[drone_name]

                # 获取实体无人机当前日志数据
                logging_data = self.crazyswarm.get_loggingData_by_droneName(drone_name)
                grid_data = self.grid_data 
                
                # 位置 (3)
                pos = Vector3(logging_data.X, logging_data.Y, logging_data.Z)
                position = [pos.x, pos.y, pos.z]
                
                # 速度 (3)
                velocity = [logging_data.XSpeed, logging_data.YSpeed, logging_data.ZSpeed]

                direction = []
                if logging_data.Speed < 0.05:
                    direction = [1, 0, 0]
                else:
                    # 方向 (3) 通过速度计算当前移动方向
                    direction = self._calculate_move_direction(logging_data.XSpeed, logging_data.YSpeed, logging_data.ZSpeed)
                
                # 附近熵值 (3)
                nearby_cells = [c for c in grid_data.cells[:50] if (c.center - pos).magnitude() < 10.0]
                if nearby_cells:
                    entropies = [c.entropy for c in nearby_cells]
                    entropy_info = [float(np.mean(entropies)), float(np.max(entropies)), float(np.std(entropies))]
                else:
                    entropy_info = [50.0, 50.0, 0.0]
                
                # Leader相对位置 (3)
                if runtime_data.leader_position:
                    leader_rel = [
                        runtime_data.leader_position.x - pos.x,
                        runtime_data.leader_position.y - pos.y,
                        runtime_data.leader_position.z - pos.z
                    ]
                else:
                    leader_rel = [0.0, 0.0, 0.0]
                
                # 扫描进度 (3)
                total = len(grid_data.cells)
                scanned = sum(1 for c in grid_data.cells if c.entropy < 30)
                scan_info = [scanned / max(total, 1), float(scanned), float(total - scanned)]
                
                state = position + velocity + direction + entropy_info + leader_rel + scan_info
                return np.array(state, dtype = np.float32)
                
        except Exception as e:
            logger.debug(f"状态提取失败: {str(e)}")
            return np.zeros(18, dtype = np.float32)
        
    def _calculate_move_direction(self, vx: float, vy: float, vz: float) -> tuple[float, float, float]:
        """
        通过三维速度计算移动方向（返回单位方向向量）
        :param vx: 速度x分量
        :param vy: 速度y分量
        :param vz: 速度z分量
        :return: 归一化后的方向向量 (dx, dy, dz)，模长=1；速度为0时返回(0,0,0)
        """
        # 1. 计算速度向量的模长（速率）
        speed = math.sqrt(vx**2 + vy**2 + vz**2)
        
        # 2. 避免除以0（速度为0时，无移动方向）
        if speed < 1e-6:  # 浮点精度容错，避免极小值
            return (0.0, 0.0, 0.0)
        
        # 3. 归一化得到方向向量
        dx = vx / speed
        dy = vy / speed
        dz = vz / speed
        
        return (dx, dy, dz)

    def _get_state_for_prediction(self, drone_name: str) -> np.ndarray:
        """提取状态用于权重预测（18维）"""
        try:
            with self.data_lock:
                runtime_data = self.unity_runtime_data[drone_name]
                grid_data = self.grid_data
                
                # 位置 (3)
                pos = runtime_data.position
                position = [pos.x, pos.y, pos.z]
                
                # 速度 (3)
                vel = runtime_data.finalMoveDir
                velocity = [vel.x * self.config_data.moveSpeed, vel.y * self.config_data.moveSpeed, vel.z * self.config_data.moveSpeed]
                
                # 方向 (3)
                fwd = runtime_data.forward
                direction = [fwd.x, fwd.y, fwd.z]
                
                # 附近熵值 (3)
                nearby_cells = [c for c in grid_data.cells[:50] if (c.center - pos).magnitude() < 10.0]
                if nearby_cells:
                    entropies = [c.entropy for c in nearby_cells]
                    entropy_info = [float(np.mean(entropies)), float(np.max(entropies)), float(np.std(entropies))]
                else:
                    entropy_info = [50.0, 50.0, 0.0]
                
                # Leader相对位置 (3)
                if runtime_data.leader_position:
                    leader_rel = [
                        runtime_data.leader_position.x - pos.x,
                        runtime_data.leader_position.y - pos.y,
                        runtime_data.leader_position.z - pos.z
                    ]
                else:
                    leader_rel = [0.0, 0.0, 0.0]
                
                # 扫描进度 (3)
                total = len(grid_data.cells)
                scanned = sum(1 for c in grid_data.cells if c.entropy < 30)
                scan_info = [scanned / max(total, 1), float(scanned), float(total - scanned)]
                
                state = position + velocity + direction + entropy_info + leader_rel + scan_info
                return np.array(state, dtype=np.float32)
                
        except Exception as e:
            logger.debug(f"状态提取失败: {str(e)}")
            return np.zeros(18, dtype=np.float32)

    def get_entropy_history(self, limit: int = 600) -> List[Tuple[float, float]]:
        """获取最近的熵值历史记录"""
        with self.entropy_history_lock:
            return list(self.entropy_history[-limit:])

    def get_entropy_distribution(self, limit: int = 1) -> List[Tuple[float, List[int], List[float]]]:
        """获取最近的熵值分布（直方图和CDF）"""
        with self.entropy_dist_history_lock:
            return list(self.entropy_dist_history[-limit:])

    def _calc_entropy_distribution(self, entropies: List[float], bin_size: int = 5, max_entropy: int = 100) -> Tuple[List[int], List[int], List[float]]:
        """计算熵值直方图与累积分布（CDF）"""
        if bin_size <= 0:
            bin_size = 5
        if max_entropy <= 0:
            max_entropy = 100

        bins = list(range(0, max_entropy + bin_size, bin_size))
        hist = [0] * (len(bins) - 1)

        for e in entropies:
            idx = int(e // bin_size)
            if idx < 0:
                idx = 0
            if idx >= len(hist):
                idx = len(hist) - 1
            hist[idx] += 1

        total = max(sum(hist), 1)
        cdf: List[float] = []
        running = 0
        for count in hist:
            running += count
            cdf.append(running / total)

        return bins, hist, cdf

    def _record_entropy_snapshot(self) -> None:
        """定期记录网格平均熵值，用于可视化"""
        current_time = time.time()
        if current_time - self._last_entropy_record_time < 1.0:
            return

        with self.grid_lock:
            if not self.grid_data or not hasattr(self.grid_data, 'cells'):
                return

            total = len(self.grid_data.cells)
            if total == 0:
                return

            entropies = [cell.entropy for cell in self.grid_data.cells]
            total_entropy = sum(entropies)

        avg_entropy = total_entropy / total
        elapsed = current_time - self._start_time

        with self.entropy_history_lock:
            self.entropy_history.append((elapsed, avg_entropy))
            if len(self.entropy_history) > 1800:
                self.entropy_history = self.entropy_history[-1800:]

        bins, hist, cdf = self._calc_entropy_distribution(entropies)
        with self.entropy_dist_history_lock:
            self.entropy_dist_history.append((elapsed, hist, cdf))
            if len(self.entropy_dist_history) > 1800:
                self.entropy_dist_history = self.entropy_dist_history[-1800:]
        self.entropy_bins = bins

        self._last_entropy_record_time = current_time

    def _predict_weights(self, drone_name: str) -> Dict[str, float]:
        """使用模型预测权重并进行平衡处理"""
        if not self.weight_model:
            return None
        
        try:
            # 是否为实体无人机镜像
            isCrazyflieMirror = self.config_data.get_uav_crazyflie_mirror(drone_name)
            state = self._get_state_for_prediction(drone_name) if not isCrazyflieMirror else self._crazyflie_get_state_for_prediction(drone_name)

            action, _ = self.weight_model.predict(state, deterministic=True)
            
            # 权重范围限制 [0.5, 5.0]
            action = np.clip(action, 0.5, 5.0)
            
            # 优化权重平衡处理：减少平滑程度，增加探索性
            action_mean = np.mean(action)
            action_std = np.std(action)
            
            # 只有当标准差过大时才进行平滑（提高阈值）
            if action_std > 2.0:  # 从1.5提高到2.0
                action = action_mean + (action - action_mean) * 0.8  # 减少平滑程度
                action = np.clip(action, 0.5, 5.0)
            
            # 确保最大权重不超过最小权重的5倍（但允许更大的差异）
            min_weight = np.min(action)
            max_weight = np.max(action)
            if max_weight > min_weight * 8:  # 从5倍提高到8倍
                scale = (min_weight * 8) / max_weight
                action = action * scale
                action = np.clip(action, 0.5, 5.0)
            
            weights = {
                'repulsionCoefficient': float(action[0]),
                'entropyCoefficient': float(action[1]),
                'distanceCoefficient': float(action[2]),
                'leaderRangeCoefficient': float(action[3]),
                'directionRetentionCoefficient': float(action[4])
            }
            
            logger.debug(f"预测权重(平衡后): {weights}")
            return weights
            
        except Exception as e:
            logger.error(f"权重预测失败: {str(e)}")
            return None
    
    def _process_drone(self, drone_name: str) -> None:
        """无人机算法处理线程：计算移动方向并控制无人机"""
        logger.info(f"无人机{drone_name}算法线程启动 (控制模式: {self.control_mode.upper()})")
        while self.running:
            try:
                # 检查数据就绪状态
                has_grid = bool(self.grid_data.cells)
                has_runtime = bool(self.unity_runtime_data[drone_name].position)

                if not (has_grid and has_runtime):
                    time.sleep(1)
                    continue

                # 根据控制模式选择不同的控制逻辑
                if self.control_mode == 'apf':
                    # APF模式：使用算法计算移动方向
                    # 如果启用权重预测，更新APF权重
                    if self.use_learned_weights:
                        predicted_weights = self._predict_weights(drone_name)
                        if predicted_weights:
                            self.algorithms[drone_name].set_coefficients(predicted_weights)
                            logger.debug(f"无人机{drone_name}使用DDPG预测权重: {predicted_weights}")
                        else:
                            logger.warning(f"无人机{drone_name}权重预测失败，使用默认权重")
                    
                    # 同步 AirSim 中的姿态数据到运行时数据（用于数据采集分析）
                    if not self.config_data.get_uav_crazyflie_mirror(drone_name):
                        try:
                            state = self.drone_controller.get_vehicle_state(drone_name)
                            if "orientation" in state:
                                roll, pitch, yaw = state["orientation"]
                                with self.data_lock:
                                    self.unity_runtime_data[drone_name].orientation = Vector3(roll, pitch, yaw)
                        except Exception as e:
                            logger.debug(f"同步无人机{drone_name}姿态失败: {e}")

                    # 执行算法计算最终方向
                    final_dir = self.algorithms[drone_name].update_runtime_data(
                        self.grid_data, self.unity_runtime_data[drone_name]
                    )
                    
                    if not self.config_data.get_uav_crazyflie_mirror(drone_name):
                        # 控制无人机移动
                         self._control_drone_movement(drone_name, final_dir.finalMoveDir)
                    else:
                        # 获取实体无人机前往指令
                        self.crazyswarm.go_to(drone_name, final_dir.finalMoveDir, self.config_data.updateInterval)
                    
                    # 发送处理后的数据到Unity
                    self._send_processed_data(drone_name, final_dir)
                    
                elif self.control_mode == 'dqn':
                    # DQN模式：使用外部DQN提供的移动指令
                    with self.dqn_command_lock:
                        move_direction = self.dqn_commands.get(drone_name, Vector3(0, 0, 0))
                    
                    # 如果有有效的DQN指令，执行移动
                    if move_direction.magnitude() > 0.001:
                        if not self.config_data.get_uav_crazyflie_mirror(drone_name):
                            self._control_drone_movement(drone_name, move_direction)
                        else:
                            self.crazyswarm.go_to(drone_name, move_direction, self.config_data.updateInterval)
                        
                        logger.debug(f"无人机{drone_name} DQN控制: {move_direction}")
                    
                    # DQN模式下不需要运行APF算法，因为DQN已经直接控制无人机
                    # 只需要保持runtime_data的基本更新即可（由Unity发送）
                    logger.debug(f"[DQN模式] DQN直接控制，跳过APF算法计算")

                # 按配置间隔休眠
                time.sleep(self.config_data.updateInterval)

            except Exception as e:
                logger.error(f"无人机{drone_name}处理出错: {str(e)}")
                logger.debug(traceback.format_exc())
                time.sleep(self.config_data.updateInterval)  # 出错后延迟重试


    def _control_drone_movement(self, drone_name: str, direction: Vector3) -> None:
        """控制无人机按指定方向移动，水平和垂直分离计算"""
        with self.data_lock:
            current_pos = self.unity_runtime_data[drone_name].position

        # 检查方向向量是否有效
        if direction.magnitude() < 0.001:
            logger.debug(f"无人机{drone_name}方向向量过小，跳过移动")
            return

        # ===== 第一步：分离水平和垂直方向 =====
        # Unity坐标系：X前后，Y高度，Z左右
        horizontal_direction = Vector3(direction.x, 0.0, direction.z)  # 只保留X和Z（水平）
        vertical_direction = Vector3(0.0, direction.y, 0.0)  # 只保留Y（高度）
        
        # ===== 第二步：分别计算水平和垂直速度 =====
        move_speed = self.config_data.moveSpeed
        
        # 水平速度：使用完整的移动速度
        if horizontal_direction.magnitude() > 0.001:
            horizontal_velocity = horizontal_direction.normalized() * move_speed
        else:
            horizontal_velocity = Vector3(0.0, 0.0, 0.0)
        
        # 垂直速度：使用较慢的速度进行高度调整
        vertical_speed = move_speed * 0.5  # 高度调整速度为水平速度的50%
        if abs(direction.y) > 0.001:
            vertical_velocity = Vector3(0.0, direction.y * vertical_speed, 0.0)
        else:
            vertical_velocity = Vector3(0.0, 0.0, 0.0)
        
        # ===== 第三步：合成最终速度向量（Unity坐标系） =====
        final_velocity = horizontal_velocity + vertical_velocity
        
        # ===== 第四步：坐标转换：Unity到AirSim =====
        velocity_airsim = final_velocity.unity_to_air_sim()
        
        # ===== 第五步：限制速度范围 =====
        # 分别限制水平和垂直速度
        horizontal_speed_airsim = (velocity_airsim.x**2 + velocity_airsim.y**2)**0.5
        max_horizontal_velocity = 3.0  # 最大水平速度
        max_vertical_velocity = 4.5    # 最大垂直速度
        
        if horizontal_speed_airsim > max_horizontal_velocity:
            scale = max_horizontal_velocity / horizontal_speed_airsim
            velocity_airsim.x *= scale
            velocity_airsim.y *= scale
        
        if abs(velocity_airsim.z) > max_vertical_velocity:
            velocity_airsim.z = max_vertical_velocity if velocity_airsim.z > 0 else -max_vertical_velocity
        
        # ===== 第六步：检查无人机是否卡住 =====
        self._check_drone_stuck(drone_name, current_pos)
        
        # ===== 第七步：发送速度控制指令 =====
        success = self.drone_controller.move_by_velocity(
            velocity_airsim.x, velocity_airsim.y, velocity_airsim.z,
            self.config_data.updateInterval, drone_name
        )



        # if not success:
        #     logger.error(f"无人机{drone_name}移动指令发送失败")
        # else:
        #     logger.debug(
        #         f"无人机{drone_name}移动: Unity方向{direction} -> "
        #         f"水平{horizontal_direction} + 垂直{vertical_direction} -> "
        #         f"AirSim速度{velocity_airsim} (水平:{horizontal_speed_airsim:.2f}, 垂直:{abs(velocity_airsim.z):.2f})"
        #     )


    def _check_drone_stuck(self, drone_name: str, current_pos: Vector3) -> None:
        """检查无人机是否卡住（位置长时间不变）"""
        # 如果服务已停止，不再进行卡住检测（避免训练结束后继续打印警告）
        if not self.running:
            return
        
        current_time = time.time()
        
        # 检查位置是否发生变化
        if drone_name in self.last_positions and self.last_positions[drone_name]:
            last_pos = self.last_positions[drone_name]
            
            # 检查last_pos是否包含必要的键
            if not all(key in last_pos for key in ['x', 'y', 'z', 'timestamp']):
                # 如果数据不完整，更新为当前位置
                self.last_positions[drone_name] = {
                    'x': current_pos.x,
                    'y': current_pos.y,
                    'z': current_pos.z,
                    'timestamp': current_time
                }
                return
            
            distance = (current_pos - Vector3(last_pos['x'], last_pos['y'], last_pos['z'])).magnitude()
            time_diff = current_time - last_pos['timestamp']
            
            # 如果位置变化很小且时间超过阈值，认为卡住了
            if distance < 0.1 and time_diff > 5.0:  # 5秒内移动距离小于0.1米
                logger.warning(f"无人机{drone_name}可能卡住了！位置变化: {distance:.3f}m，时间: {time_diff:.1f}s")
                
                # 尝试发送一个小的随机移动来解除卡住状态（保持高度）
                import random
                random_dir = Vector3(
                    random.uniform(-0.5, 0.5),
                    random.uniform(-0.5, 0.5),
                    0.0  # Z轴方向为0，保持高度
                )
                
                # 计算随机移动速度
                random_velocity = random_dir * 1.0  # 小速度
                # 坐标转换：Unity -> AirSim
                random_velocity_airsim = random_velocity.unity_to_air_sim()
                random_velocity_airsim.z = 0.0  # 确保Z轴速度为0，保持高度
                
                logger.info(f"尝试解除无人机{drone_name}卡住状态，发送随机移动指令（保持高度）")
                self.drone_controller.move_by_velocity(
                    random_velocity_airsim.x, random_velocity_airsim.y, random_velocity_airsim.z,
                    1.0, drone_name  # 1秒的短时间移动
                )
                
                # 更新位置记录
                self.last_positions[drone_name] = {
                    'x': current_pos.x,
                    'y': current_pos.y,
                    'z': current_pos.z,
                    'timestamp': current_time
                }
        else:
            # 首次记录位置
            self.last_positions[drone_name] = {
                'x': current_pos.x,
                'y': current_pos.y,
                'z': current_pos.z,
                'timestamp': current_time
            }

    def _send_processed_data(self, drone_name: str, scannerRuntimeData: ScannerRuntimeData) -> None:
        """发送处理后的运行时数据到Unity"""
        # 检查是否正在重置（通过checking运行状态）
        if not self.running:
            return  # 重置期间不发送数据
            
        with self.data_lock:
            try:
                # 直接使用传入的scannerRuntimeData数据
                self.processed_runtime_data[drone_name] = scannerRuntimeData
                self.processed_runtime_data[drone_name].drone_name = drone_name
                # 发送到Unity - 注意：send_runtime需要一个可迭代对象（列表）
                self.unity_socket.send_runtime([self.processed_runtime_data[drone_name]])
                # logger.debug(f"已发送无人机{drone_name}的处理后数据到Unity")
            except Exception as e:
                # 捕获发送异常，避免影响主流程
                logger.warning(f"发送运行时数据到Unity失败: {str(e)}")


    def set_dqn_movement(self, drone_name: str, direction: Vector3) -> None:
        """
        为DQN控制模式设置移动指令
        :param drone_name: 无人机名称
        :param direction: 移动方向向量（Unity坐标系）
        """
        if self.control_mode != 'dqn':
            logger.warning(f"当前控制模式为{self.control_mode}，无法设置DQN移动指令")
            return
        
        with self.dqn_command_lock:
            self.dqn_commands[drone_name] = direction
        logger.debug(f"DQN设置移动指令: {drone_name} -> {direction}")


    def reset_environment(self) -> None:
        """重置运行环境（网格熵值、无人机位置、Leader等）"""
        logger.info("[重置] 正在重置运行环境...")
        
        # 智能重置判断：检查无人机是否已经在起点且处于飞行状态
        need_physical_reset = False
        
        # 如果是实体无人机镜像，我们通常不执行物理重置 (交给外部控制或手动重置)
        # 仅针对虚拟无人机 (AirSim) 进行智能判断
        for drone_name in self.drone_names:
            if not self.drones_config.is_crazyflie_mirror(drone_name):
                try:
                    state = self.drone_controller.get_vehicle_state(drone_name)
                    pos = state.get("position", (0, 0, 0)) # AirSim 坐标: (x, y, z)
                    api_enabled = state.get("api_enabled", False)
                    flying = state.get("flying", False)
                    
                    # 计算水平距离 (x, y) 距离起点的偏移
                    horizontal_dist = math.sqrt(pos[0]**2 + pos[1]**2)
                    
                    # 记录详细日志以便排查
                    logger.info(f"[重置检测] 无人机:{drone_name}, 位置:({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), "
                                f"水平距离:{horizontal_dist:.2f}m, API启用:{api_enabled}, 飞行状态:{flying}")
                    
                    # 只要无人机在飞（只要不在地面，flying即为True）且距离起点不远，就跳过物理重置
                    if horizontal_dist > 3.0 or not flying:
                        need_physical_reset = True
                        reason = "距离过远" if horizontal_dist > 3.0 else "未处于飞行状态"
                        logger.info(f"[重置检测] 判定需要物理重置，原因: {reason}")
                        break
                except Exception as e:
                    logger.debug(f"[重置] 获取无人机{drone_name}状态失败，默认执行物理重置: {e}")
                    need_physical_reset = True
                    break
        
        # 1. 重置AirSim模拟器和无人机位置 (仅针对虚拟无人机)
        if need_physical_reset and hasattr(self, 'drone_controller') and self.drone_controller.connection_status:
            try:
                logger.info("[重置] 检测到无人机偏移或未就绪，执行 AirSim 物理重置...")
                
                # 检查所有无人机的碰撞状态
                for drone_name in self.drone_names:
                    if not self.drones_config.is_crazyflie_mirror(drone_name):
                        collision = self.drone_controller.check_collision(drone_name)
                        if collision['has_collided']:
                            logger.warning(f"[重置] 无人机{drone_name}处于碰撞状态，将执行安全恢复")
                            # 先尝试悬停恢复
                            self.drone_controller.recover_from_collision(drone_name)
                
                # 执行安全重置(带防穿地保护)
                self.drone_controller.reset()
                
                # 重置后需要重新初始化无人机并起飞
                time.sleep(1)
                logger.info("[重置] 重新初始化无人机并重置电量...")
                self._init_drones()
                
                # 重置所有无人机电量数据
                for drone_name in self.drone_names:
                    self.reset_battery_voltage(drone_name)
                
                logger.info("[重置] 所有无人机重新起飞...")
                self._takeoff_all()
            except Exception as e:
                logger.error(f"[重置] AirSim重置失败: {str(e)}")
        else:
            logger.info("[重置] 🚀 无人机已在起点并处于飞行状态，跳过 AirSim 物理重置(降落/起飞)")
            # 即使跳过物理重置，也总是重置电量数据以保证统计准确
            for drone_name in self.drone_names:
                self.reset_battery_voltage(drone_name)

        # 2. 发送重置命令到Unity（重置网格数据、Leader状态等）
        if self.unity_socket and self.unity_socket.is_connected():
            logger.info("[重置] 发送重置命令到Unity...")
            self.unity_socket.send_reset_command()
            
            # 等待较长时间，确保Unity完成重置并发送新数据
            time.sleep(3.0) 
            logger.info("[重置] Unity重置指令已发送，等待接收新数据")
            
            # 关键修复：重置后必须重新发送start_simulation指令，让Leader恢复移动
            logger.info("[重置] 发送start_simulation指令，让Leader恢复移动...")
            self.unity_socket.send_start_simulation_command()
            time.sleep(0.5)  # 等待Unity处理指令
            logger.info("[重置] Leader已恢复移动")
        else:
            logger.warning("[重置] Unity未连接，仅清空本地网格数据")
            # 清空Python端的网格数据
            with self.grid_lock:
                self.grid_data.cells.clear()
        
        logger.info("[重置] 环境重置流程执行完毕")
    
    def stop(self) -> None:
        """停止服务：降落无人机，断开连接，清理资源"""
        self.running = False
        logger.info("开始停止服务...")

        # 停止数据采集线程
        if self.data_collector:
            self.data_collector.stop()

        # 停止可视化
        if self.visualizer:
            self.visualizer.stop_visualization()
            logger.info("可视化功能已停止")

        self._crazyflie_all_land()
        self.crazyswarm.clear()

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

    def _crazyflie_all_land(self):
        """控制所有实体无人机降落"""
        logger.info("开始所有实体无人机降落流程")
        for drone_name in self.drone_names:
            if self.config_data.get_uav_crazyflie_mirror(drone_name):
                self.crazyswarm.land(drone_name, 2)
                time.sleep(2)


    def _disconnect_airsim(self) -> None:
        """断开与AirSim的连接"""
        try:
            for drone_name in self.drone_names:
                self.drone_controller.arm_disarm(False, drone_name)
                self.drone_controller.enable_api_control(False, drone_name)
            logger.info("已断开与AirSim的连接")
        except Exception as e:
            logger.error(f"断开AirSim连接出错: {str(e)}")

    def reset_simulation(self) -> bool:
        """重置仿真环境（AirSim和Unity）"""
        try:
            logger.info("=" * 60)
            logger.info("🔄 开始重置仿真环境...")
            logger.info("=" * 60)
            
            # 保存当前运行状态
            was_running = self.running
            
            # 重要：检查Unity连接状态
            if not self.unity_socket.is_connected():
                logger.warning("[重置] Unity未连接，无法执行重置")
                return False
            
            # 1. 停止算法处理线程（但不影响Unity socket）
            if was_running:
                logger.info("[步骤1/8] 停止算法处理线程...")
                self.running = False
                
                # 等待所有线程结束
                logger.info("等待算法线程结束...")
                for drone_name, thread in self.drone_threads.items():
                    if thread and thread.is_alive():
                        thread.join(timeout=5.0)  # 最多等待5秒
                        if thread.is_alive():
                            logger.warning(f"无人机{drone_name}算法线程未能正常结束")
                        else:
                            logger.info(f"无人机{drone_name}算法线程已停止")
                time.sleep(0.5)  # 减少等待时间
            else:
                logger.info("[步骤1/8] 跳过（算法未运行）")
            
            # 2. 所有无人机降落
            logger.info("[步骤2/8] 所有无人机降落...")
            self._land_all()
            time.sleep(1)  # 减少等待时间
            
            # 3. 发送Unity重置命令
            logger.info("[步骤3/8] 发送重置命令到Unity...")
            self.unity_socket.send_reset_command()
            time.sleep(2)  # 等待Unity处理重置命令并完成
            
            # 4. 重置AirSim模拟器
            logger.info("[步骤4/8] 重置AirSim模拟器...")
            if not self.drone_controller.reset():
                logger.error("AirSim模拟器重置失败")
                return False
            time.sleep(1.5)  # 等待AirSim重置完成
            
            # 5. 清理本地数据
            logger.info("[步骤5/8] 清理本地数据...")
            self._clear_local_data()
            
            # 6. 重新初始化无人机
            logger.info("[步骤6/8] 重新初始化无人机...")
            if not self._init_drones():
                logger.error("无人机重新初始化失败")
                return False
            time.sleep(1)
            
            # 7. 发送配置数据到Unity（包含Leader位置等初始配置）
            logger.info("[步骤7/8] 发送配置数据到Unity...")
            self.unity_socket.send_config(self.config_data)
            time.sleep(0.5)
            
            # 8. 如果之前在运行，重新启动任务
            if was_running:
                logger.info("[步骤8/8] 重新启动任务...")
                if not self.start_mission():
                    logger.error("任务重新启动失败")
                    return False
            else:
                logger.info("[步骤8/8] 跳过（之前未运行任务）")
            
            logger.info("=" * 60)
            logger.info("✅ 仿真环境重置成功！")
            logger.info("=" * 60)
            return True
            
        except Exception as e:
            logger.error(f"❌ 重置仿真环境失败: {str(e)}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            # 尝试恢复运行状态
            if was_running and not self.running:
                logger.info("尝试恢复系统运行...")
                self.start_mission()
            return False

    def _clear_local_data(self) -> None:
        """清理本地数据状态"""
        try:
            # 重置运行时数据
            for drone_name in self.drone_names:
                self.unity_runtime_data[drone_name] = ScannerRuntimeData()
                self.processed_runtime_data[drone_name] = ScannerRuntimeData()
                self.last_positions[drone_name] = {}
            
            # 重置网格数据
            self.grid_data = HexGridDataModel()
            
            # 重新创建算法实例
            self.algorithms = {
                name: ScannerAlgorithm(self.config_data) for name in self.drone_names
            }
            
            logger.info("本地数据清理完成")
        except Exception as e:
            logger.error(f"清理本地数据失败: {str(e)}")


if __name__ == "__main__":
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='多无人机算法服务器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  1. 使用固定权重（默认）:
     python AlgorithmServer.py
     
  2. 使用DQN权重预测（自动选择最佳模型）:
     python AlgorithmServer.py --use-learned-weights
     
  3. 使用指定的DDPG模型:
     python AlgorithmServer.py --use-learned-weights --model-path DDPG_Weight/models/best_model
     python AlgorithmServer.py --use-learned-weights --model-path DDPG_Weight/models/checkpoint_5000
     
  4. 多无人机 + DDPG:
     python AlgorithmServer.py --use-learned-weights --drones 3
     
  5. 禁用可视化:
     python AlgorithmServer.py --no-visualization
        """
    )
    parser.add_argument('--use-learned-weights', action='store_true', 
                        help='使用DDPG学习的权重（需要先训练模型）')
    parser.add_argument('--model-path', type=str, default=None,
                        help='DDPG模型路径（相对或绝对路径，不含.zip后缀）。如果不指定，将自动选择：best_model > weight_predictor_airsim > weight_predictor_simple')
    parser.add_argument('--drones', type=int, default=1,
                        help='无人机数量（默认1）')
    parser.add_argument('--no-visualization', action='store_true',
                        help='禁用可视化（默认启用）')
    args = parser.parse_args()
    
    try:
        # 生成无人机名称列表
        drone_names = [f"UAV{i}" for i in range(1, args.drones + 1)]
        
        logger.info("=" * 60)
        logger.info(f"启动多无人机系统 - {args.drones}台无人机")
        logger.info(f"无人机列表: {drone_names}")
        if args.use_learned_weights:
            logger.info("模式: DDPG权重预测")
            if args.model_path:
                logger.info(f"模型: {args.model_path}")
            else:
                logger.info("模型: 自动选择（best_model > weight_predictor_airsim > weight_predictor_simple）")
        else:
            logger.info("模式: 固定权重")
        logger.info(f"可视化: {'禁用' if args.no_visualization else '启用'}")
        logger.info("=" * 60)
        
        # 创建服务器实例
        server = MultiDroneAlgorithmServer(
            drone_names=drone_names,
            use_learned_weights=args.use_learned_weights,
            model_path=args.model_path,
            enable_visualization=not args.no_visualization
        )
        
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