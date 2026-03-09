import sys
import time as _time
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
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("AlgorithmServer")

# 导入核心模块
from AirsimServer.drone_controller import DroneController
from AirsimServer.unity_socket_server import UnitySocketServer
from Algorithm.scanner_algorithm import ScannerAlgorithm
from Algorithm.scanner_config_data import ScannerConfigData
from Algorithm.scanner_runtime_data import ScannerRuntimeData
from Algorithm.HexGridDataModel import HexGridDataModel
from Algorithm.battery_data import (
    BatteryManager,
    BatteryInfo,
    BatteryStatus,
)  # 新增导入
from Algorithm.drones_config import DronesConfig
from Crazyswarm.crazyswarm import CrazyswarmManager
from Crazyswarm.crazyflie_operate import CrazyflieOperate
from Crazyswarm.crazyflie_logging_data import CrazyflieLoggingData
from Algorithm.Vector3 import Vector3
from Algorithm.data_collector import DataCollector
from AirsimServer.data_pack import PackType
from diagnostic_logger import get_diagnostic_logger, DroneDiagnosticLogger

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

    def __init__(
        self,
        config_file: Optional[str] = None,
        drone_names: Optional[List[str]] = None,
        use_learned_weights: bool = False,
        model_path: Optional[str] = None,
        enable_visualization: bool = True,
        enable_data_collection_print: bool = False,
        control_mode: str = "apf",
        max_episode_seconds: int = 300,
    ):
        """
        初始化服务器实例
        :param config_file: 算法配置文件路径（默认使用apf_algorithm_config.json）
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
        # 为每架无人机创建算法实例（所有无人机使用相同的权重）
        self.algorithms: Dict[str, ScannerAlgorithm] = {}
        for name in self.drone_names:
            algo = ScannerAlgorithm(self.config_data)
            self.algorithms[name] = algo
        self.last_positions: Dict[str, Dict[str, float]] = {
            name: {} for name in self.drone_names
        }

        # 加载无人机配置
        self.drones_config = DronesConfig()

        # 电量数据管理
        self.battery_manager = BatteryManager(self.config_data, self.drones_config)
        self.battery_lock = threading.Lock()  # 电量数据锁

        self.crazyswarm = CrazyswarmManager(
            self.unity_socket,
            self.battery_manager,
            self.config_data,
            self.drones_config,
        )

        # 共享网格数据
        self.grid_data = HexGridDataModel()

        # 线程与状态管理
        self.max_episode_seconds = max_episode_seconds  # 单轮训练最大时长（秒）
        self._episode_start_time = _time.time()
        self.ready_event = threading.Event()  # 同步所有无人机首帧runtime到位
        self.reset_ack_event = threading.Event()  # 用于重置闭环同步
        self.resetting = False  # 正在重置标志
        self.running = False
        self.drone_threads: Dict[str, Optional[threading.Thread]] = {
            name: None for name in self.drone_names
        }
        self.data_lock = threading.Lock()  # 运行时数据锁
        self.grid_lock = threading.Lock()  # 网格数据锁
        self.timeout_lock = threading.Lock()  # 超时重置锁

        # 记录每台无人机起飞后的初始位置（用于 reset_environment 水平偏移判定）
        self.home_positions: Dict[str, Tuple[float, float]] = {}

        # 熵值记录
        self.entropy_history: List[Tuple[float, float]] = []
        self.entropy_history_lock = threading.Lock()
        self._start_time = _time.time()
        self._last_entropy_record_time = 0.0
        self.entropy_dist_history: List[Tuple[float, List[int], List[float]]] = []
        self.entropy_bins: List[int] = []
        self.entropy_dist_history_lock = threading.Lock()

        # 可视化组件
        self.visualizer = None
        self.enable_visualization = enable_visualization

        # 可视化快照缓存（供独立可视化进程读取）
        self._vis_snapshot_cache = None
        self._vis_snapshot_cache_time = 0.0
        self._last_reset_time = 0.0  # 记录最后一次重置时间，用于客户端清除缓存
        self._last_reset_reason = ""  # 记录最后一次重置原因
        self._reset_history = []  # 重置历史记录 (时间, 原因)
        # 重置握手状态（避免旧包误触发 ACK，导致重置后不扫描）
        self._reset_command_sent_time = 0.0
        self._reset_runtime_fresh = False
        self._reset_grid_fresh = False
        self._reset_ack_delay = 0.5

        # 数据采集系统（根据控制模式选择不同的数据目录）
        if control_mode.lower() == "dqn":
            # DQN 模式：保存到 DQN_Movement/logs/dqn_scan_data
            dqn_data_dir = os.path.join(
                os.path.dirname(__file__), "DQN_Movement", "logs", "dqn_scan_data"
            )
            self.data_collector = DataCollector(
                data_dir=dqn_data_dir,
                collection_interval=1.0,
                enable_debug_print=enable_data_collection_print,
                training_prefix="dqn",
            )
        else:
            # APF/DDPG 模式：保存到 DDPG_Weight/airsim_training_logs（默认）
            self.data_collector = DataCollector(
                collection_interval=1.0, enable_debug_print=enable_data_collection_print
            )

        # 注册Unity数据接收回调
        self.unity_socket.set_callback(self._handle_unity_data)

        # 调试：初始化后检查received_obstacles状态
        if self.unity_socket:
            logger.info(
                f"[初始化] unity_socket已连接，received_obstacles初始长度: {len(self.unity_socket.received_obstacles) if hasattr(self.unity_socket, 'received_obstacles') else '属性不存在'}"
            )
        else:
            logger.warning("[初始化] ⚠️ unity_socket为None")

        # 控制模式：'apf' 或 'dqn'
        self.control_mode = control_mode.lower()
        if self.control_mode not in ["apf", "dqn"]:
            logger.warning(f"未知的控制模式: {control_mode}，使用默认APF模式")
            self.control_mode = "apf"
        logger.info(f"控制模式: {self.control_mode.upper()}")

        # DQN控制模式相关
        self.dqn_commands: Dict[str, Vector3] = {
            name: Vector3(0, 0, 0) for name in self.drone_names
        }  # 存储DQN移动指令
        self.dqn_command_lock = threading.Lock()  # DQN指令锁

        # DDPG权重预测（仅在APF模式下使用）
        self.use_learned_weights = use_learned_weights and (self.control_mode == "apf")
        self.model_path = model_path  # 保存模型路径参数
        self.weight_model = None
        if self.use_learned_weights:
            self._init_weight_predictor()
        elif use_learned_weights and self.control_mode == "dqn":
            logger.info("DQN控制模式下，use_learned_weights参数被忽略")

        # 训练统计（用于IPC可视化进程）
        self.current_training_stats: Dict[str, Any] = {
            "episode_count": 0,
            "total_steps": 0,
            "current_episode_steps": 0,
            "current_step_reward": 0.0,
            "current_episode_reward": 0.0,
            "episode_elapsed_time": 0.0,
            "avg_reward": 0.0,
            "max_reward": 0.0,
            "min_reward": 0.0,
            "reward_history": [],
            "steps_per_sec": 0.0,
        }
        self._training_stats_lock = threading.Lock()  # 训练统计锁

        # 初始化可视化组件（如果启用）
        if self.enable_visualization:
            self._init_visualization()
        else:
            logger.info("可视化已禁用")

        # 初始化诊断日志记录器
        try:
            self.diagnostic_logger = get_diagnostic_logger()
            logger.info(
                f"✅ 诊断日志已启用: {self.diagnostic_logger.get_log_file_path()}"
            )
        except Exception as e:
            logger.warning(f"⚠️ 诊断日志初始化失败: {e}")
            self.diagnostic_logger = None

    def _resolve_config_path(self, config_file: Optional[str]) -> str:
        """解析配置文件路径，默认使用项目根目录下的apf_algorithm_config.json"""
        if config_file:
            if os.path.exists(config_file):
                return config_file
            logger.warning(f"指定的配置文件不存在: {config_file}，将使用默认配置")

        default_path = Path(__file__).parent / "apf_algorithm_config.json"
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
                    model_path = os.path.join(
                        os.path.dirname(__file__), self.model_path
                    )
                logger.info(f"📂 使用指定模型: {model_path}")
            else:
                # 使用默认模型路径（优先级：best_model > weight_predictor_airsim > weight_predictor_simple）
                models_dir = os.path.join(
                    os.path.dirname(__file__), "DDPG_Weight", "models"
                )

                # 尝试多个默认模型
                default_models = [
                    os.path.join(models_dir, "best_model"),
                    os.path.join(models_dir, "weight_predictor_airsim"),
                    os.path.join(models_dir, "weight_predictor_simple"),
                ]

                model_path = None
                for candidate in default_models:
                    if os.path.exists(candidate + ".zip"):
                        model_path = candidate
                        logger.info(f"📂 使用默认模型: {os.path.basename(model_path)}")
                        break

                if not model_path:
                    logger.warning("❌ 未找到任何可用的模型文件")
                    logger.info("💡 可用模型列表：")
                    if os.path.exists(models_dir):
                        for f in os.listdir(models_dir):
                            if f.endswith(".zip"):
                                logger.info(f"   - {f}")
                    logger.warning("将使用配置文件中的固定权重")
                    self.use_learned_weights = False
                    logger.info("=" * 60)
                    return

            # 加载模型
            if os.path.exists(model_path + ".zip"):
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

    def update_battery_voltage(
        self, drone_name: str, action_intensity: float = 0.0
    ) -> float:
        """更新指定无人机的电量消耗
        :param drone_name: 无人机名称
        :param action_intensity: 动作强度（0-1），影响额外消耗
        :return: 更新后的电压值
        """
        return self.battery_manager.update_voltage(drone_name, action_intensity)

    def reset_battery_voltage(self, drone_name: str) -> float:
        """重置指定无人机的电量为初始值"""
        return self.battery_manager.reset_voltage(drone_name)

    def set_training_stats(
        self, episode: int, step: int, reward: float, total_reward: float
    ):
        """?????????????????? IPC ????"""
        now = _time.time()

        with self._training_stats_lock:
            previous_episode = self.current_training_stats.get("episode_count", -1)
            if episode != previous_episode or step <= 0:
                self._episode_start_time = now

            episode_elapsed_time = max(0.0, now - self._episode_start_time)

            self.current_training_stats["episode_count"] = episode
            self.current_training_stats["total_steps"] = step
            self.current_training_stats["current_step_reward"] = reward
            self.current_training_stats["current_episode_reward"] = total_reward
            self.current_training_stats["current_episode_steps"] = step
            self.current_training_stats["episode_elapsed_time"] = episode_elapsed_time

            if "reward_history" not in self.current_training_stats:
                self.current_training_stats["reward_history"] = []
            reward_history = self.current_training_stats["reward_history"]
            if step > 0:
                reward_history.append(total_reward)
            if len(reward_history) > 500:
                reward_history.pop(0)

            if reward_history:
                self.current_training_stats["avg_reward"] = sum(reward_history) / len(
                    reward_history
                )
                self.current_training_stats["max_reward"] = max(reward_history)
                self.current_training_stats["min_reward"] = min(reward_history)
            else:
                self.current_training_stats["avg_reward"] = 0.0
                self.current_training_stats["max_reward"] = 0.0
                self.current_training_stats["min_reward"] = 0.0

        if self.data_collector:
            self.data_collector.set_external_data("episode", episode)
            self.data_collector.set_external_data("step", step)
            self.data_collector.set_external_data("reward", reward)
            self.data_collector.set_external_data("step_reward", reward)
            self.data_collector.set_external_data("total_reward", total_reward)
            self.data_collector.set_external_data("episode_reward", total_reward)
            self.data_collector.set_external_data(
                "episode_elapsed_time", episode_elapsed_time
            )

    def set_experiment_meta(
        self, algorithm_type: str, env_type: str, control_mode: str
    ):
        """
        设置实验元数据，用于数据采集的统一标签
        :param algorithm_type: 算法类型 (如 'hrl_dqn_apf', 'pure_dqn', 'ddpg_apf')
        :param env_type: 环境类型 (如 'hierarchical', 'movement', 'weight')
        :param control_mode: 控制模式 (如 'dqn', 'apf')
        """
        if self.data_collector:
            self.data_collector.set_external_data("algorithm_type", algorithm_type)
            self.data_collector.set_external_data("env_type", env_type)
            self.data_collector.set_external_data("control_mode", control_mode)
            logger.info(f"实验元数据已设置: {algorithm_type}/{env_type}/{control_mode}")

    def get_all_battery_data(self) -> Dict[str, Dict[str, float]]:
        """获取所有无人机的电量数据"""
        return self.battery_manager.get_all_battery_data()

    def _get_training_data(self) -> Dict[str, Any]:
        """?????????????????"""
        if hasattr(self, "current_training_stats"):
            with self._training_stats_lock:
                stats = dict(self.current_training_stats)
            step_reward = float(stats.get("current_step_reward", 0.0))
            episode_reward = float(stats.get("current_episode_reward", 0.0))
            return {
                "episode": stats.get("episode_count", -1),
                "step": stats.get("total_steps", -1),
                "reward": step_reward,
                "step_reward": step_reward,
                "episode_reward": episode_reward,
                "total_reward": episode_reward,
                "episode_elapsed_time": float(stats.get("episode_elapsed_time", 0.0)),
            }
        return {
            "episode": -1,
            "step": -1,
            "reward": 0.0,
            "step_reward": 0.0,
            "episode_reward": 0.0,
            "total_reward": 0.0,
            "episode_elapsed_time": 0.0,
        }

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
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.battery_manager.to_json())
            logger.info(f"电池数据已保存到: {file_path}")
        except Exception as e:
            logger.error(f"保存电池数据失败: {str(e)}")

    # 新增方法：从文件加载电池数据
    def load_battery_data(self, file_path: str) -> None:
        """从JSON文件加载电池数据"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
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
        start_time = _time.time()
        while _time.time() - start_time < timeout:
            if self.unity_socket.is_connected():
                logger.info("Unity客户端已连接")
                self.unity_socket.send_config(self.config_data)
                self.unity_socket.send_drone_config(self.drones_config)
                # logger.info("已发送初始配置数据到Unity")
                return True
            _time.sleep(0.5)

        logger.error(f"等待Unity连接超时（{timeout}秒）")
        return False

    def _connect_airsim(self) -> bool:
        """连接到AirSim模拟器"""
        logger.info("连接到AirSim模拟器...")
        logger.info("[DEBUG-TEMP] 调用 drone_controller.connect()...")
        result = self.drone_controller.connect()
        logger.info(f"[DEBUG-TEMP] drone_controller.connect() 返回: {result}")
        if result:
            logger.info("AirSim连接成功")
            # 起飞前先重置airsim
            logger.info("重置AirSim模拟器...")
            self.drone_controller.reset()
            # 重置后等待几秒，让系统稳定
            logger.info("等待AirSim系统稳定...")
            _time.sleep(3)
            logger.info("[DEBUG-TEMP] _connect_airsim() 返回 True")
            return True
        logger.error("AirSim连接失败")
        logger.info("[DEBUG-TEMP] _connect_airsim() 返回 False")
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

    def _wait_for_takeoff(self, timeout: float = 5.0) -> bool:
        """等待所有虚拟无人机 flying=True（短暂等待即可）"""
        start = _time.time()
        while _time.time() - start < timeout:
            all_ready = True
            for name in self.drone_names:
                if self.drones_config.is_crazyflie_mirror(name):
                    continue
                state = self.drone_controller.get_vehicle_state(name)
                if not state.get("flying", False):
                    all_ready = False
                    break
            if all_ready:
                logger.info("所有虚拟无人机已稳定在空中")
                return True
            _time.sleep(0.1)
        logger.info("[起飞检测] 完成（所有无人机已起飞）")
        return True  # 即使超时也返回成功，因为takeoff已经设置了flying标志

    def start_mission(self) -> bool:
        """开始任务：控制所有无人机起飞并启动算法线程"""
        logger.info(f"[DEBUG-TEMP] start_mission() 被调用，running={self.running}")
        if not self.running:
            logger.info("准备开始任务，等待系统完全稳定...")
            _time.sleep(2)  # 额外等待2秒确保系统稳定

            # 1. 所有无人机起飞
            if not self._takeoff_all():
                return False

            # 等待所有虚拟无人机确认 flying=True 再开始仿真
            logger.info("无人机起飞完成，等待所有无人机稳定...")
            self._wait_for_takeoff()

            # 记录每台无人机起飞后的初始位置（AirSim NED: x,y 为水平）
            for name in self.drone_names:
                if self.drones_config.is_crazyflie_mirror(name):
                    continue
                try:
                    state = self.drone_controller.get_vehicle_state(name)
                    pos = state.get("position", (0.0, 0.0, 0.0))
                    self.home_positions[name] = (float(pos[0]), float(pos[1]))
                    logger.info(
                        f"[Home] {name} home_xy=({self.home_positions[name][0]:.2f},{self.home_positions[name][1]:.2f})"
                    )
                except Exception as e:
                    logger.warning(f"[Home] 记录{name} home位置失败: {e}")

            # 2. 发送开始仿真指令到Unity（让领导者开始移动）
            if self.unity_socket and self.unity_socket.is_connected():
                logger.info("发送开始仿真指令到Unity...")
                self.unity_socket.send_start_simulation_command()
                _time.sleep(0.5)  # 等待Unity处理指令

            # 在启动算法线程前清空同步事件
            self.ready_event.clear()
            self.resetting = False  # 确保首次启动时resetting为False
            logger.info(
                f"首帧同步事件已清空，等待Unity推送首帧runtime数据 (resetting={self.resetting})"
            )
            # 3. 启动算法处理线程
            logger.info("启动算法处理线程...")
            self.running = True
            # 记录Episode开始时间
            self._episode_start_time = _time.time()
            for drone_name in self.drone_names:
                self.drone_threads[drone_name] = threading.Thread(
                    target=self._process_drone, args=(drone_name,), daemon=True
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
                get_training_data_func=lambda: self._get_training_data(),  # 新增：提供训练数据获取函数
                data_lock=self.data_lock,
                grid_lock=self.grid_lock,
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
            _time.sleep(0.5)  # 减少延迟时间，加快起飞流程
        return all_success

    def _try_set_reset_ack(self) -> None:
        """当重置后的 runtime 与 grid 都到达后，触发 ACK。"""
        if (
            self.resetting
            and self._reset_runtime_fresh
            and self._reset_grid_fresh
            and not self.reset_ack_event.is_set()
        ):
            logger.info("[重置] 收到重置后的 runtime + grid 新数据，触发 ACK")
            self.reset_ack_event.set()

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
                if "runtime_data" in received_data:
                    # 收集首帧同步

                    runtime_data_list = received_data["runtime_data"]
                    if isinstance(runtime_data_list, list):
                        # logger.info(f"收到运行时数据，包含{len(runtime_data_list)}个无人机数据")
                        # 处理每个无人机的运行时数据
                        for runtime_data in runtime_data_list:
                            drone_name = runtime_data.get("uavname")
                            if drone_name in self.unity_runtime_data and isinstance(
                                runtime_data, dict
                            ):
                                try:
                                    self.unity_runtime_data[drone_name] = (
                                        ScannerRuntimeData.from_dict(runtime_data)
                                    )
                                    # 更新位置信息
                                    pos = self.unity_runtime_data[drone_name].position
                                    self.last_positions[drone_name] = {
                                        "x": pos.x,
                                        "y": pos.y,
                                        "z": pos.z,
                                        "timestamp": _time.time(),
                                    }

                                except Exception as e:
                                    logger.error(
                                        f"解析无人机{drone_name}运行时数据失败: {str(e)}"
                                    )
                                    logger.error(f"原始数据: {runtime_data}")
                            else:
                                logger.warning(
                                    f"无效的运行时数据或无人机名称: {drone_name}"
                                )

                        # ---- 首帧同步检查 ----
                        if not self.ready_event.is_set():
                            # 确保本次包里包含全部无人机，且每机都有有效位置
                            received_names = {
                                runtime_data.get("uavname")
                                for runtime_data in runtime_data_list
                                if runtime_data.get("uavname")
                                in self.unity_runtime_data
                            }

                            # 严格检查：必须在 unity_runtime_data 中有值，且 position 不能为 None (None 表示重置后尚未收到新包)
                            all_valid = True
                            invalid_drones = []
                            for n in self.drone_names:
                                if n not in self.unity_runtime_data:
                                    all_valid = False
                                    invalid_drones.append(f"{n}:无数据")
                                elif self.unity_runtime_data[n].position is None:
                                    all_valid = False
                                    invalid_drones.append(f"{n}:position为None")

                            # 每5秒打印一次调试信息
                            if not hasattr(self, "_last_sync_debug_time"):
                                self._last_sync_debug_time = 0
                            if _time.time() - self._last_sync_debug_time > 5.0:
                                self._last_sync_debug_time = _time.time()
                                logger.info(
                                    f"[首帧同步] 收到{len(received_names)}台, 需要{len(self.drone_names)}台, 有效={all_valid}"
                                )
                                if invalid_drones:
                                    logger.info(
                                        f"[首帧同步] 无效无人机: {', '.join(invalid_drones)}"
                                    )

                            if received_names == set(self.drone_names) and all_valid:
                                # 关键逻辑：重置期间禁止自动解锁 ready_event
                                if not self.resetting:
                                    logger.info(
                                        f"首帧 runtime_data 收齐（{len(received_names)}台），解除同步锁"
                                    )
                                    self.ready_event.set()
                                else:
                                    # 仅接受 reset 指令发送后的“新数据”，避免旧runtime包误触发 ACK
                                    now = _time.time()
                                    if (
                                        self._reset_command_sent_time > 0
                                        and now - self._reset_command_sent_time
                                        >= self._reset_ack_delay
                                    ):
                                        self._reset_runtime_fresh = True
                                        self._try_set_reset_ack()
                        # ---------------------

                # 检查是否包含grid_data字段
                elif "grid_data" in received_data:
                    grid_data = received_data["grid_data"]
                    if isinstance(grid_data, dict) and "cells" in grid_data:
                        cells_count = len(grid_data["cells"])
                        
                        # 调试：计算熵值统计
                        entropies = [c.get('entropy', 100) for c in grid_data['cells']]
                        avg_entropy = sum(entropies) / len(entropies) if entropies else 100
                        low_entropy_count = sum(1 for e in entropies if e < 30)
                        
                        # 使用warning级别让日志更显眼
                        # 频率控制：避免日志输出过快
                        if not hasattr(self, "_last_grid_log_time"):
                            self._last_grid_log_time = 0
                            self._last_low_entropy_count = 0
                        if not hasattr(self, "_last_avg_entropy"):
                            self._last_avg_entropy = 100

                        current_time = _time.time()
                        time_diff = current_time - self._last_grid_log_time
                        entropy_diff = low_entropy_count - self._last_low_entropy_count
                        
                        # 只在关键变化时输出日志
                        should_log = (
                            time_diff > 3.0 or  # 超过3秒
                            entropy_diff > 3 or  # 低熵格子显著增加
                            (self._last_avg_entropy - avg_entropy) > 2  # 平均熵值显著降低
                        )
                        
                        if should_log:
                            logger.warning(f"🔴 [网格更新] 收到{cells_count}个格子，平均熵值={avg_entropy:.1f}, 低熵格子={low_entropy_count}")
                            self._last_grid_log_time = current_time
                            self._last_low_entropy_count = low_entropy_count
                            self._last_avg_entropy = avg_entropy
                        

                        with self.grid_lock:
                            self.grid_data.update_from_dict(grid_data)

                        # 重置期间：标记 reset 指令后的新 grid 数据
                        if self.resetting:
                            now = _time.time()
                            if (
                                self._reset_command_sent_time > 0
                                and now - self._reset_command_sent_time
                                >= self._reset_ack_delay
                                and cells_count > 0
                            ):
                                self._reset_grid_fresh = True
                                self._try_set_reset_ack()
                    else:
                        logger.warning(f"网格数据格式错误: {grid_data}")

                # 检查是否包含配置数据
                elif "config_data" in received_data:
                    config_data = received_data["config_data"]
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
                # 检查是否包含统一障碍物数据（支持Static/Dynamic，Normal/RestrictedZone，Polygon/Circle）
                elif "obstacles" in received_data:
                    obstacles = received_data.get("obstacles", [])
                    if isinstance(obstacles, list):
                        # 分类处理障碍物（兼容数字枚举和字符串）
                        normal_obstacles = [
                            obs
                            for obs in obstacles
                            if obs.get("category") in [0, "Normal"]
                        ]
                        restricted_zones = [
                            obs
                            for obs in obstacles
                            if obs.get("category") in [1, "RestrictedZone"]
                        ]

                        logger.info(
                            f"收到障碍物数据 - 普通: {len(normal_obstacles)}, 禁飞区: {len(restricted_zones)}, 总计: {len(obstacles)}"
                        )

                        try:
                            # 处理普通障碍物（运行时动态数据）
                            for algo in self.algorithms.values():
                                if hasattr(algo, "set_normal_obstacles"):
                                    algo.set_normal_obstacles(normal_obstacles)

                            # 处理禁飞区（静态配置数据）
                            if restricted_zones:
                                for algo in self.algorithms.values():
                                    if hasattr(algo, "set_restricted_zones"):
                                        algo.set_restricted_zones(restricted_zones)

                            logger.info(
                                f"障碍物数据已更新到各算法实例 - 普通障碍物: {len(normal_obstacles)}, 禁飞区: {len(restricted_zones)}"
                            )
                        except Exception as e:
                            logger.error(f"更新障碍物数据失败: {str(e)}")
                    else:
                        logger.warning(f"障碍物数据格式错误: {type(obstacles)}")
                elif "crazyflie_logging" in received_data:
                    try:
                        crazyflie_logging_json = (
                            CrazyflieLoggingData.from_json_to_dicts(
                                received_data["crazyflie_logging"]
                            )
                        )
                        crazyflie_logging_list = CrazyflieLoggingData.from_dict_list(
                            crazyflie_logging_json
                        )
                        # logger.info("收到Crazyflies实体无人机日志数据更新")
                        self.crazyswarm.update_crazyflies_logging(
                            crazyflie_logging_list
                        )
                    except Exception as e:
                        logger.error(f"更新Crazyflies实体无人机日志数据失败: {str(e)}")
                # 未知数据类型处理
                else:
                    logger.warning(f"收到未知格式数据: {received_data}")

        except Exception as e:
            logger.error(
                f"处理Unity数据时发生错误: {str(e)}，堆栈信息: {traceback.format_exc()}"
            )
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
                velocity = [
                    logging_data.XSpeed,
                    logging_data.YSpeed,
                    logging_data.ZSpeed,
                ]

                direction = []
                if logging_data.Speed < 0.05:
                    direction = [1, 0, 0]
                else:
                    # 方向 (3) 通过速度计算当前移动方向
                    direction = self._calculate_move_direction(
                        logging_data.XSpeed, logging_data.YSpeed, logging_data.ZSpeed
                    )

                # 附近熵值 (3)
                nearby_cells = [
                    c
                    for c in grid_data.cells[:50]
                    if (c.center - pos).magnitude() < 10.0
                ]
                if nearby_cells:
                    entropies = [c.entropy for c in nearby_cells]
                    entropy_info = [
                        float(np.mean(entropies)),
                        float(np.max(entropies)),
                        float(np.std(entropies)),
                    ]
                else:
                    entropy_info = [50.0, 50.0, 0.0]

                # Leader相对位置 (3)
                if runtime_data.leader_position:
                    leader_rel = [
                        runtime_data.leader_position.x - pos.x,
                        runtime_data.leader_position.y - pos.y,
                        runtime_data.leader_position.z - pos.z,
                    ]
                else:
                    leader_rel = [0.0, 0.0, 0.0]

                # 扫描进度 (3)
                total = len(grid_data.cells)
                scanned = sum(1 for c in grid_data.cells if c.entropy < 30)
                scan_info = [
                    scanned / max(total, 1),
                    float(scanned),
                    float(total - scanned),
                ]

                state = (
                    position
                    + velocity
                    + direction
                    + entropy_info
                    + leader_rel
                    + scan_info
                )
                return np.array(state, dtype=np.float32)

        except Exception as e:
            logger.debug(f"状态提取失败: {str(e)}")
            return np.zeros(18, dtype=np.float32)

    def _calculate_move_direction(
        self, vx: float, vy: float, vz: float
    ) -> tuple[float, float, float]:
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
                velocity = [
                    vel.x * self.config_data.moveSpeed,
                    vel.y * self.config_data.moveSpeed,
                    vel.z * self.config_data.moveSpeed,
                ]

                # 方向 (3)
                fwd = runtime_data.forward
                direction = [fwd.x, fwd.y, fwd.z]

                # 附近熵值 (3)
                nearby_cells = [
                    c
                    for c in grid_data.cells[:50]
                    if (c.center - pos).magnitude() < 10.0
                ]
                if nearby_cells:
                    entropies = [c.entropy for c in nearby_cells]
                    entropy_info = [
                        float(np.mean(entropies)),
                        float(np.max(entropies)),
                        float(np.std(entropies)),
                    ]
                else:
                    entropy_info = [50.0, 50.0, 0.0]

                # Leader相对位置 (3)
                if runtime_data.leader_position:
                    leader_rel = [
                        runtime_data.leader_position.x - pos.x,
                        runtime_data.leader_position.y - pos.y,
                        runtime_data.leader_position.z - pos.z,
                    ]
                else:
                    leader_rel = [0.0, 0.0, 0.0]

                # 扫描进度 (3)
                total = len(grid_data.cells)
                scanned = sum(1 for c in grid_data.cells if c.entropy < 30)
                scan_info = [
                    scanned / max(total, 1),
                    float(scanned),
                    float(total - scanned),
                ]

                state = (
                    position
                    + velocity
                    + direction
                    + entropy_info
                    + leader_rel
                    + scan_info
                )
                return np.array(state, dtype=np.float32)

        except Exception as e:
            logger.debug(f"状态提取失败: {str(e)}")
            return np.zeros(18, dtype=np.float32)

    def get_entropy_history(self, limit: int = 600) -> List[Tuple[float, float]]:
        """获取最近的熵值历史记录"""
        with self.entropy_history_lock:
            return list(self.entropy_history[-limit:])

    def get_entropy_distribution(
        self, limit: int = 1
    ) -> List[Tuple[float, List[int], List[float]]]:
        """获取最近的熵值分布（直方图和CDF）"""
        with self.entropy_dist_history_lock:
            return list(self.entropy_dist_history[-limit:])

    def _calc_entropy_distribution(
        self, entropies: List[float], bin_size: int = 5, max_entropy: int = 100
    ) -> Tuple[List[int], List[int], List[float]]:
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
        current_time = _time.time()
        if current_time - self._last_entropy_record_time < 1.0:
            return

        with self.grid_lock:
            if not self.grid_data or not hasattr(self.grid_data, "cells"):
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
            isCrazyflieMirror = self.drones_config.is_crazyflie_mirror(drone_name)
            state = (
                self._get_state_for_prediction(drone_name)
                if not isCrazyflieMirror
                else self._crazyflie_get_state_for_prediction(drone_name)
            )

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
                "repulsionCoefficient": float(action[0]),
                "entropyCoefficient": float(action[1]),
                "distanceCoefficient": float(action[2]),
                "leaderRangeCoefficient": float(action[3]),
                "directionRetentionCoefficient": float(action[4]),
            }

            logger.debug(f"预测权重(平衡后): {weights}")
            return weights

        except Exception as e:
            logger.error(f"权重预测失败: {str(e)}")
            return None

    def _process_drone(self, drone_name: str) -> None:
        """无人机算法处理线程：计算移动方向并控制无人机"""
        logger.info(
            f"无人机{drone_name}算法线程启动 (控制模式: {self.control_mode.upper()})"
        )
        # 等待所有无人机接收首帧runtime数据后同步开始决策
        logger.info(
            f"[{drone_name}] 等待首帧同步... (ready_event={self.ready_event.is_set()}, resetting={self.resetting})"
        )

        # 添加超时等待，避免永久阻塞
        wait_timeout = 30.0  # 30秒超时
        start_wait = _time.time()
        while (
            not self.ready_event.is_set() and _time.time() - start_wait < wait_timeout
        ):
            _time.sleep(0.1)

        if not self.ready_event.is_set():
            logger.warning(
                f"[{drone_name}] ⚠️ 首帧同步超时({wait_timeout}s)，强制继续... (resetting={self.resetting})"
            )
            # 强制设置ready_event，避免永久阻塞
            self.ready_event.set()

        logger.info(f"[{drone_name}] 首帧同步完成，开始决策循环")
        while self.running:
            # 重置/同步期间需要再次阻塞，避免起飞未完成就开始执行动作
            if not self.ready_event.is_set():
                logger.info(f"[{drone_name}] 等待重置后同步... (ready_event=False)")
                # 添加超时等待
                reset_wait_timeout = 20.0  # 20秒超时
                reset_wait_start = _time.time()
                while (
                    not self.ready_event.is_set()
                    and _time.time() - reset_wait_start < reset_wait_timeout
                ):
                    _time.sleep(0.1)

                if not self.ready_event.is_set():
                    logger.warning(
                        f"[{drone_name}] ⚠️ 重置后同步超时({reset_wait_timeout}s)，强制继续..."
                    )
                    self.ready_event.set()
                else:
                    logger.info(f"[{drone_name}] 重置后同步完成，继续决策")
            try:
                # 训练模式下 episode/step 由训练端环境统一管理：
                # AlgorithmServer 不再在循环内基于超时自动 reset。
                # 如果需要超时终止，请在 Gym Env 中判断并调用 server.reset_environment()。
                # 检查数据就绪状态
                has_grid = bool(self.grid_data.cells)
                has_runtime = bool(self.unity_runtime_data[drone_name].position)

                if not (has_grid and has_runtime):
                    if not has_grid:
                        logger.info(f"[{drone_name}] ⏳ 等待网格数据...")
                    if not has_runtime:
                        logger.info(f"[{drone_name}] ⏳ 等待位置数据...")
                    _time.sleep(0.5)
                    continue

                # 诊断：检查位置是否在更新（每5秒打印一次）
                if not hasattr(self, "_last_pos_debug_time"):
                    self._last_pos_debug_time = {}
                if not hasattr(self, "_last_positions_debug"):
                    self._last_positions_debug = {}

                current_pos = self.unity_runtime_data[drone_name].position
                current_time = _time.time()
                last_debug_time = self._last_pos_debug_time.get(drone_name, 0)

                if current_time - last_debug_time > 5.0:
                    self._last_pos_debug_time[drone_name] = current_time
                    last_pos = self._last_positions_debug.get(drone_name)

                    if last_pos and current_pos:
                        dist_moved = (current_pos - last_pos).magnitude()
                        if dist_moved < 0.5:
                            logger.warning(
                                f"[{drone_name}] 🚨 位置几乎未变！5秒内移动了 {dist_moved:.2f}m"
                            )
                            logger.warning(
                                f"[{drone_name}]    当前位置: ({current_pos.x:.2f}, {current_pos.y:.2f}, {current_pos.z:.2f})"
                            )
                        else:
                            logger.info(
                                f"[{drone_name}] ✅ 位置正常: 5秒移动 {dist_moved:.2f}m"
                            )
                    else:
                        logger.info(
                            f"[{drone_name}] 📍 初始位置: ({current_pos.x:.2f}, {current_pos.y:.2f}, {current_pos.z:.2f})"
                        )

                    self._last_positions_debug[drone_name] = current_pos

                # 根据控制模式选择不同的控制逻辑
                if self.control_mode == "apf":
                    # APF模式：使用算法计算移动方向
                    # 如果启用权重预测，更新APF权重
                    if self.use_learned_weights:
                        predicted_weights = self._predict_weights(drone_name)
                        if predicted_weights:
                            self.algorithms[drone_name].set_coefficients(
                                predicted_weights
                            )
                            logger.debug(
                                f"无人机{drone_name}使用DDPG预测权重: {predicted_weights}"
                            )
                        else:
                            logger.warning(
                                f"无人机{drone_name}权重预测失败，使用默认权重"
                            )

                    # 同步 AirSim 中的姿态数据到运行时数据（用于数据采集分析）
                    if not self.drones_config.is_crazyflie_mirror(drone_name):
                        try:
                            state = self.drone_controller.get_vehicle_state(drone_name)
                            if "orientation" in state:
                                roll, pitch, yaw = state["orientation"]
                                with self.data_lock:
                                    self.unity_runtime_data[
                                        drone_name
                                    ].orientation = Vector3(roll, pitch, yaw)
                        except Exception as e:
                            logger.debug(f"同步无人机{drone_name}姿态失败: {e}")

                    # 执行算法计算最终方向
                    final_dir = self.algorithms[drone_name].update_runtime_data(
                        self.grid_data, self.unity_runtime_data[drone_name]
                    )

                    # 记录诊断日志（详细记录每台无人机的状态）
                    if self.diagnostic_logger and hasattr(
                        self.diagnostic_logger, "log_drone_status"
                    ):
                        try:
                            self.diagnostic_logger.log_drone_status(
                                drone_name,
                                self,
                                self.algorithms[drone_name],
                                self.unity_runtime_data[drone_name],
                            )
                        except Exception as e:
                            logger.debug(f"[{drone_name}] 诊断日志记录失败: {e}")

                    # 检查计算出的方向是否有效
                    move_dir = final_dir.finalMoveDir if final_dir else None
                    has_valid_dir = move_dir and move_dir.magnitude() > 0.001

                    # 定期打印诊断信息（每5秒一次，避免日志刷屏）
                    if not hasattr(self, "_last_dir_debug_time"):
                        self._last_dir_debug_time = {}
                    current_time = _time.time()
                    if (
                        drone_name not in self._last_dir_debug_time
                        or current_time - self._last_dir_debug_time.get(drone_name, 0)
                        > 5.0
                    ):
                        self._last_dir_debug_time[drone_name] = current_time
                        if has_valid_dir:
                            logger.info(
                                f"[{drone_name}] ✅ 方向正常: ({move_dir.x:.2f}, {move_dir.y:.2f}, {move_dir.z:.2f})"
                            )
                        else:
                            logger.warning(
                                f"[{drone_name}] ⚠️ 方向无效! final_dir={'有' if final_dir else '无'}, mag={move_dir.magnitude() if move_dir else 0:.3f}"
                            )

                    if not self.drones_config.is_crazyflie_mirror(drone_name):
                        # 控制无人机移动
                        if move_dir and move_dir.magnitude() > 0.001:
                            self._control_drone_movement(drone_name, move_dir)
                        else:
                            logger.warning(f"[{drone_name}] 跳过移动指令（方向无效）")
                    else:
                        # 获取实体无人机前往指令
                        self.crazyswarm.go_to(
                            drone_name,
                            final_dir.finalMoveDir,
                            self.config_data.updateInterval,
                        )

                    # 发送处理后的数据到Unity
                    self._send_processed_data(drone_name, final_dir)

                elif self.control_mode == "dqn":
                    # DQN模式：使用外部DQN提供的移动指令
                    with self.dqn_command_lock:
                        move_direction = self.dqn_commands.get(
                            drone_name, Vector3(0, 0, 0)
                        )

                    # 如果有有效的DQN指令，执行移动
                    if move_direction.magnitude() > 0.001:
                        if not self.drones_config.is_crazyflie_mirror(drone_name):
                            self._control_drone_movement(drone_name, move_direction)
                        else:
                            self.crazyswarm.go_to(
                                drone_name,
                                move_direction,
                                self.config_data.updateInterval,
                            )

                        logger.debug(f"无人机{drone_name} DQN控制: {move_direction}")

                    # DQN模式下不需要运行APF算法，因为DQN已经直接控制无人机
                    # 只需要保持runtime_data的基本更新即可（由Unity发送）
                    logger.debug(f"[DQN模式] DQN直接控制，跳过APF算法计算")

                # 按配置间隔休眠，保持训练速度与之前一致
                _time.sleep(self.config_data.updateInterval)

            except Exception as e:
                logger.error(f"无人机{drone_name}处理出错: {str(e)}")
                logger.debug(traceback.format_exc())
                _time.sleep(self.config_data.updateInterval)  # 出错后延迟重试

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
        horizontal_direction = Vector3(
            direction.x, 0.0, direction.z
        )  # 只保留X和Z（水平）
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
        horizontal_speed_airsim = (velocity_airsim.x**2 + velocity_airsim.y**2) ** 0.5
        max_horizontal_velocity = 3.0  # 最大水平速度
        max_vertical_velocity = 4.5  # 最大垂直速度

        if horizontal_speed_airsim > max_horizontal_velocity:
            scale = max_horizontal_velocity / horizontal_speed_airsim
            velocity_airsim.x *= scale
            velocity_airsim.y *= scale

        if abs(velocity_airsim.z) > max_vertical_velocity:
            velocity_airsim.z = (
                max_vertical_velocity
                if velocity_airsim.z > 0
                else -max_vertical_velocity
            )

        # ===== 第六步：检查无人机是否卡住 =====
        self._check_drone_stuck(drone_name, current_pos)

        # ===== 第七步：发送速度控制指令 =====
        # 诊断日志：记录详细的速度指令（每5秒一次）
        if not hasattr(self, "_last_move_debug_time"):
            self._last_move_debug_time = {}
        current_debug_time = _time.time()
        last_debug = self._last_move_debug_time.get(drone_name, 0)

        if current_debug_time - last_debug > 5.0:
            self._last_move_debug_time[drone_name] = current_debug_time
            logger.info(
                f"[{drone_name}] 📤 发送移动指令:\n"
                f"    Unity方向: ({direction.x:.2f}, {direction.y:.2f}, {direction.z:.2f})\n"
                f"    Unity速度: ({final_velocity.x:.2f}, {final_velocity.y:.2f}, {final_velocity.z:.2f})\n"
                f"    AirSim速度: ({velocity_airsim.x:.2f}, {velocity_airsim.y:.2f}, {velocity_airsim.z:.2f})\n"
                f"    水平速度: {horizontal_speed_airsim:.2f} m/s, 垂直: {velocity_airsim.z:.2f} m/s\n"
                f"    当前高度: {current_pos.y:.2f}m"
            )

            # 记录到诊断日志文件
            if self.diagnostic_logger and hasattr(
                self.diagnostic_logger, "log_move_command"
            ):
                try:
                    self.diagnostic_logger.log_move_command(
                        drone_name,
                        direction,
                        final_velocity,
                        velocity_airsim,
                        horizontal_speed_airsim,
                        current_pos.y if current_pos else 0,
                    )
                except Exception as e:
                    logger.debug(f"[{drone_name}] 移动指令诊断日志记录失败: {e}")

        success = self.drone_controller.move_by_velocity(
            velocity_airsim.x,
            velocity_airsim.y,
            velocity_airsim.z,
            self.config_data.updateInterval,  # 使用配置的更新间隔
            drone_name,
        )

        if not success:
            logger.error(f"[{drone_name}] ❌ 移动指令发送失败！")

    def _check_drone_stuck(self, drone_name: str, current_pos: Vector3) -> None:
        """检查无人机是否卡住（位置长时间不变）"""
        # 如果服务已停止，不再进行卡住检测（避免训练结束后继续打印警告）
        if not self.running:
            return

        current_time = _time.time()

        # 检查位置是否发生变化
        if drone_name in self.last_positions and self.last_positions[drone_name]:
            last_pos = self.last_positions[drone_name]

            # 检查last_pos是否包含必要的键
            if not all(key in last_pos for key in ["x", "y", "z", "timestamp"]):
                # 如果数据不完整，更新为当前位置
                self.last_positions[drone_name] = {
                    "x": current_pos.x,
                    "y": current_pos.y,
                    "z": current_pos.z,
                    "timestamp": current_time,
                }
                return

            distance = (
                current_pos - Vector3(last_pos["x"], last_pos["y"], last_pos["z"])
            ).magnitude()
            time_diff = current_time - last_pos["timestamp"]

            # 如果位置变化很小且时间超过阈值，认为卡住了
            if distance < 0.1 and time_diff > 5.0:  # 5秒内移动距离小于0.1米
                logger.warning(
                    f"无人机{drone_name}可能卡住了！位置变化: {distance:.3f}m，时间: {time_diff:.1f}s"
                )

                # 尝试发送一个小的随机移动来解除卡住状态（保持高度）
                import random

                random_dir = Vector3(
                    random.uniform(-0.5, 0.5),
                    random.uniform(-0.5, 0.5),
                    0.0,  # Z轴方向为0，保持高度
                )

                # 计算随机移动速度
                random_velocity = random_dir * 1.0  # 小速度
                # 坐标转换：Unity -> AirSim
                random_velocity_airsim = random_velocity.unity_to_air_sim()
                random_velocity_airsim.z = 0.0  # 确保Z轴速度为0，保持高度

                logger.info(
                    f"尝试解除无人机{drone_name}卡住状态，发送随机移动指令（保持高度）"
                )
                self.drone_controller.move_by_velocity(
                    random_velocity_airsim.x,
                    random_velocity_airsim.y,
                    random_velocity_airsim.z,
                    1.0,
                    drone_name,  # 1秒的短时间移动
                )

                # 更新位置记录
                self.last_positions[drone_name] = {
                    "x": current_pos.x,
                    "y": current_pos.y,
                    "z": current_pos.z,
                    "timestamp": current_time,
                }
        else:
            # 首次记录位置
            self.last_positions[drone_name] = {
                "x": current_pos.x,
                "y": current_pos.y,
                "z": current_pos.z,
                "timestamp": current_time,
            }

    def _send_processed_data(
        self, drone_name: str, scannerRuntimeData: ScannerRuntimeData
    ) -> None:
        """发送处理后的运行时数据到Unity"""
        # 检查是否正在重置（通过checking运行状态）
        if not self.running or self.resetting:
            return  # 重置期间或正在重置时不发送数据，避免发送脏数据

        with self.data_lock:
            try:
                # 直接使用传入的scannerRuntimeData数据
                self.processed_runtime_data[drone_name] = scannerRuntimeData
                self.processed_runtime_data[drone_name].drone_name = drone_name
                # 发送到Unity - 注意：send_runtime需要一个可迭代对象（列表）
                self.unity_socket.send_runtime(
                    [self.processed_runtime_data[drone_name]]
                )
                # # logger.debug(f"已发送无人机{drone_name}的处理后数据到Unity")
            except Exception as e:
                # 捕获发送异常，避免影响主流程
                logger.warning(f"发送运行时数据到Unity失败: {str(e)}")

    def set_dqn_movement(self, drone_name: str, direction: Vector3) -> None:
        """
        为DQN控制模式设置移动指令
        :param drone_name: 无人机名称
        :param direction: 移动方向向量（Unity坐标系）
        """
        if self.control_mode != "dqn":
            logger.warning(f"当前控制模式为{self.control_mode}，无法设置DQN移动指令")
            return

        with self.dqn_command_lock:
            self.dqn_commands[drone_name] = direction
        logger.debug(f"DQN设置移动指令: {drone_name} -> {direction}")

    def get_visualization_snapshot(self) -> Dict[str, Any]:
        """为独立可视化进程提取数据快照（非阻塞）"""
        # 显式声明使用全局time模块（避免Python误认为time是局部变量）
        global _time
        now = _time.time()
        
        # 修复：重置后立即清除缓存，确保返回最新数据
        # 如果上次重置时间晚于缓存时间，强制刷新
        if self._last_reset_time and self._vis_snapshot_cache_time < self._last_reset_time:
            self._vis_snapshot_cache = None
            logger.info("[可视化] 检测到重置，清除快照缓存")
        
        # 频率限制，避免过度消耗 CPU 序列化大数据
        # 只有缓存存在且时间差小于阈值时才返回缓存
        if self._vis_snapshot_cache is not None and (
            now - self._vis_snapshot_cache_time < 0.1
        ):
            return self._vis_snapshot_cache

        snapshot = {
            "timestamp": now,
            "drone_names": self.drone_names,
            "control_mode": self.control_mode,
            "last_reset_time": self._last_reset_time,
            "config_data": {
                "scanRadius": self.config_data.scanRadius,
                "moveSpeed": self.config_data.moveSpeed,
                "updateInterval": self.config_data.updateInterval,
            },
        }

        # 0. 提取电量数据（供外部可视化进程显示）
        try:
            snapshot["battery_data"] = self.get_all_battery_data()
        except Exception:
            pass

        # 1. 尝试提取网格数据 (带极短超时)
        # 修复：重置期间不要返回空cells，否则客户端proxy会被清空且无法恢复
        if self.grid_lock.acquire(timeout=0.01):
            try:
                if (
                    self.grid_data
                    and hasattr(self.grid_data, "cells")
                    and len(self.grid_data.cells) > 0
                ):
                    snapshot["grid_data"] = {
                        "cells": [
                            {
                                "x": c.center.x,
                                "y": c.center.y,
                                "z": c.center.z,
                                "entropy": c.entropy,
                            }
                            for c in self.grid_data.cells
                        ]
                    }
                else:
                    # 网格数据为空时才返回空列表
                    snapshot["grid_data"] = {"cells": []}
            finally:
                self.grid_lock.release()
        # 获取锁失败时：
        # - 如果正在重置，不添加grid_data字段，让客户端延用旧数据
        # - 如果不在重置，可能是短暂的锁竞争，跳过本次更新
        # 这样可以避免重置期间客户端收到空cells导致热力图消失
        # else:
        #     # 不添加grid_data字段，客户端会延用上一帧的数据

        # 2. 尝试提取运行时数据
        if self.data_lock.acquire(timeout=0.01):
            try:
                runtimes = {}
                for name, rd in self.unity_runtime_data.items():
                    runtimes[name] = {
                        "position": {
                            "x": rd.position.x,
                            "y": rd.position.y,
                            "z": rd.position.z,
                        }
                        if rd.position
                        else None,
                        "forward": {
                            "x": rd.forward.x,
                            "y": rd.forward.y,
                            "z": rd.forward.z,
                        }
                        if rd.forward
                        else None,
                        "finalMoveDir": {
                            "x": rd.finalMoveDir.x,
                            "y": rd.finalMoveDir.y,
                            "z": rd.finalMoveDir.z,
                        }
                        if rd.finalMoveDir
                        else None,
                        "leader_position": {
                            "x": rd.leader_position.x,
                            "y": rd.leader_position.y,
                            "z": rd.leader_position.z,
                        }
                        if rd.leader_position
                        else None,
                        "leader_scan_radius": rd.leader_scan_radius,
                    }
                snapshot["unity_runtime_data"] = runtimes
            finally:
                self.data_lock.release()

        # 3. 提取训练统计 (如果有)
        if hasattr(self, "data_collector") and self.data_collector:
            snapshot["training_stats"] = self.data_collector.external_data

        # 增加额外的训练实时统计（用于 DQN 面板）
        if hasattr(self, "current_training_stats"):
            snapshot["current_training_stats"] = self.current_training_stats

        # 5. 添加重置原因和历史记录
        snapshot["last_reset_reason"] = self._last_reset_reason
        snapshot["last_reset_time"] = self._last_reset_time
        snapshot["reset_history"] = list(self._reset_history)

        # 4. 提取障碍物数据（用于可视化绘制）
        # 添加详细调试日志
        if not self.unity_socket:
            logger.warning("[快照-障碍物] ⚠️ unity_socket为None，无法提取障碍物数据")
            snapshot["obstacles"] = []
        elif not hasattr(self.unity_socket, "received_obstacles"):
            logger.warning("[快照-障碍物] ⚠️ unity_socket没有received_obstacles属性")
            snapshot["obstacles"] = []
        else:
            # 正常情况：直接提取obstacles数据
            snapshot["obstacles"] = self.unity_socket.received_obstacles
            # 调试日志：每秒输出一次（避免日志刷屏）
            current_time = _time.time()
            if not hasattr(self, "_last_obstacle_log_time"):
                self._last_obstacle_log_time = 0
            if current_time - self._last_obstacle_log_time > 1.0:  # 每秒最多输出一次
                self._last_obstacle_log_time = current_time
                obs_count = len(snapshot["obstacles"]) if snapshot["obstacles"] else 0
                logger.debug(f"[快照-障碍物] 障碍物: {obs_count} 个已添加到snapshot")

        # 5. 提取当前权重数据（用于DDPG训练可视化）
        if self.drone_names and len(self.drone_names) > 0:
            first_drone = self.drone_names[0]
            # ⚠️ 优化：放宽algorithms访问检查，即使访问失败也尝试提取数据
            if first_drone in self.algorithms:
                try:
                    algo = self.algorithms[first_drone]
                    if hasattr(algo, "get_current_coefficients"):
                        weights = algo.get_current_coefficients()
                        snapshot["current_weights"] = weights
                        logger.debug(
                            f"[快照-权重] first_drone={first_drone}, 权重数: {len(weights)}"
                        )
                except Exception as e:
                    logger.error(f"[快照-权重] first_drone={first_drone} 提取失败: {e}")
                    # 即使失败也要添加空字典，避免可视化界面完全没有权重数据
                    snapshot["current_weights"] = {}

        # 调试：输出快照的所有字段（每5秒一次）
        if not hasattr(self, "_last_snapshot_debug_time"):
            self._last_snapshot_debug_time = 0
        current_time_debug = _time.time()
        if current_time_debug - self._last_snapshot_debug_time > 5.0:
            self._last_snapshot_debug_time = current_time_debug
            snap_keys = list(snapshot.keys())
            # logger.info(
            #     f"[快照-调试] 📤 准备发送snapshot，字段数: {len(snap_keys)}, 字段列表: {snap_keys}"
            # )
            # # 特别检查关键字段
            # logger.info(
            #     f"[快照-调试]   grid_data: {'有' if 'grid_data' in snapshot else '无'}, "
            #     + f"unity_runtime_data: {len(snapshot.get('unity_runtime_data', {}))} 个, "
            #     + f"obstacles: {len(snapshot.get('obstacles', []))} 个, "
            #     + f"current_weights: {len(snapshot.get('current_weights', {}))} 个"
            # )

        self._vis_snapshot_cache = snapshot
        self._vis_snapshot_cache_time = now
        return snapshot

    def reset_environment(self, reason: str = "Unknown", reset_grid: bool = True) -> None:
        """重置运行环境（严格闭合流程：停止-物理重置-起飞-清空数据-Unity重置-等待反馈-启动）

        Args:
            reason: 重置原因
            reset_grid: 是否重置网格熵值（默认True，完全重新扫描）
                       设为True时会将所有格子的熵值重置为80（完全重新扫描）
                       设为False时保持已扫描区域的低熵值（累积扫描进度）
        """
        logger.info(f"[重置] 🔄 开始严格重置流程... (原因: {reason}, 重置网格熵值: {reset_grid})")
        self.resetting = True
        self.reset_ack_event.clear()
        self.ready_event.clear()  # 确保算法线程在重置期间阻塞
        self._reset_command_sent_time = 0.0
        self._reset_runtime_fresh = False
        self._reset_grid_fresh = False

        # 记录重置时间和原因，用于客户端清除缓存
        self._last_reset_time = _time.time()
        self._last_reset_reason = reason
        # 添加到重置历史（保留最近20条）
        self._reset_history.append((self._last_reset_time, reason))
        if len(self._reset_history) > 20:
            self._reset_history = self._reset_history[-20:]

        # 0. 立即停止仿真并刹车
        for d_name in self.drone_names:
            try:
                if not self.drones_config.is_crazyflie_mirror(d_name):
                    self.drone_controller.move_by_velocity(0, 0, 0, 0.1, d_name)
                else:
                    self.crazyswarm.hover(d_name)
            except Exception:
                pass
        _time.sleep(0.2)

        # 1. 强制执行 AirSim 物理重置 (回到地面未起飞状态)
        # 不再进行水平距离和飞行状态判断，为了实验严谨性，每轮都重新开始
        if (
            hasattr(self, "drone_controller")
            and self.drone_controller.connection_status
        ):
            try:
                logger.info("[重置] 1/5 执行 AirSim 物理重置 (强制回到地面)...")

                # 检查碰撞状态并安全恢复（避免重置后卡在地面）
                for drone_name in self.drone_names:
                    if not self.drones_config.is_crazyflie_mirror(drone_name):
                        collision = self.drone_controller.check_collision(drone_name)
                        if collision["has_collided"]:
                            self.drone_controller.recover_from_collision(drone_name)

                # 执行模拟器 reset
                self.drone_controller.reset()
                _time.sleep(1.0)

                # 重新初始化 API 控制和解锁
                logger.info("[重置] 重新初始化无人机控制权...")
                if not self._init_drones():
                    logger.error("[重置] 无人机重新初始化失败")

                # 重新执行起飞流程
                logger.info("[重置] 无人机重新起飞...")
                if not self._takeoff_all():
                    logger.error("[重置] 无人机重新起飞失败")

                # 等待起飞稳定
                self._wait_for_takeoff()

                # 防穿地保护：额外等待物理引擎稳定
                logger.info("[重置] 等待物理引擎稳定...")
                _time.sleep(0.5)

                # 检查所有无人机高度，确保没有穿地
                for drone_name in self.drone_names:
                    if not self.drones_config.is_crazyflie_mirror(drone_name):
                        try:
                            state = self.drone_controller.get_vehicle_state(drone_name)
                            pos = state.get("position", (0.0, 0.0, 0.0))
                            height = -pos[2]  # NED坐标系，转换为正高度

                            if height < 0.5:  # 高度低于0.5米，可能存在穿地风险
                                logger.warning(
                                    f"[重置] {drone_name} 高度异常({height:.2f}m)，执行恢复..."
                                )
                                self.drone_controller.recover_from_collision(drone_name)
                                _time.sleep(0.3)
                            else:
                                logger.info(
                                    f"[重置] {drone_name} 高度正常({height:.2f}m)"
                                )
                        except Exception as e:
                            logger.warning(f"[重置] 检查{drone_name}高度失败: {e}")

            except Exception as e:
                logger.error(f"[重置] AirSim 物理重置流程异常: {e}")

        # 2. 清理本地数据状态与电量
        logger.info("[重置] 2/5 清理本地算法与统计数据...")
        self._clear_local_data(reset_grid_entropy=reset_grid)
        if reset_grid:
            logger.info("[重置] 网格熵值已重置为80（完全重新扫描）")
        else:
            logger.info("[重置] 保持网格熵值（扫描进度累积）")
        with self.dqn_command_lock:
            for k in self.dqn_commands:
                self.dqn_commands[k] = Vector3(0, 0, 0)

        # 强制立即刷新可视化快照，清除旧网格缓存
        self._vis_snapshot_cache = None
        self._vis_snapshot_cache_time = 0.0

        # 3. 发送重置命令到 Unity (重置网格和 Leader)
        if self.unity_socket and self.unity_socket.is_connected():
            logger.info("[重置] 3/5 发送重置命令到 Unity，等待反馈 (ACK)...")
            cleared = self.unity_socket.clear_pending_packs()
            if cleared > 0:
                logger.info(f"[重置] 已清空发送队列历史包: {cleared} 个")
            self._reset_runtime_fresh = False
            self._reset_grid_fresh = False
            self._reset_command_sent_time = _time.time()
            self.unity_socket.send_reset_command()

            # 4. 等待 Unity 返回重置完毕的数据 (ACK)
            wait_start = _time.time()
            success = self.reset_ack_event.wait(timeout=10.0)

            if success:
                logger.info(
                    f"[重置] ✅ 收到 Unity 反馈，耗时: {_time.time() - wait_start:.2f}s"
                )
            else:
                logger.warning("[重置] ⚠️ 等待 Unity 重置反馈超时")
                logger.info("[重置] 尝试重发一次 reset 指令...")
                self.reset_ack_event.clear()
                self._reset_runtime_fresh = False
                self._reset_grid_fresh = False
                self._reset_command_sent_time = _time.time()
                retry_start = _time.time()
                self.unity_socket.send_reset_command()
                retry_success = self.reset_ack_event.wait(timeout=6.0)
                if retry_success:
                    logger.info(
                        f"[重置] ✅ 重试后收到 Unity 反馈，耗时: {_time.time() - retry_start:.2f}s"
                    )
                else:
                    logger.warning("[重置] ⚠️ 重试后仍未收到 ACK，继续执行启动流程")
            # 5. 重发配置并启动仿真，确保Unity扫描状态机恢复
            logger.info("[重置] 4/5 重发无人机与算法配置到Unity...")
            self.unity_socket.send_drone_config(self.drones_config)
            self.unity_socket.send_config(self.config_data)
            _time.sleep(0.5)

            logger.info("[重置] 5/5 发送 start_simulation 指令，Leader 开始移动")
            self.unity_socket.send_start_simulation_command()
            _time.sleep(1.0)
            # 二次触发，提升多次重置后的可靠性（幂等）
            self.unity_socket.send_start_simulation_command()

            # 等待Unity启动熵值收集功能并稳定
            logger.info("[重置] 熵值收集启动中...")
            _time.sleep(3.0)

            # 确保Unity准备好接收runtime数据
            logger.info("[重置] Unity应该已启动熵值收集，算法线程即将发送数据")
        else:
            logger.warning("[重置] Unity 未连接，仅清空本地数据")
            with self.grid_lock:
                self.grid_data.cells.clear()

        # 所有重置流程结束后，显式放行算法线程
        self._reset_command_sent_time = 0.0
        self._reset_runtime_fresh = False
        self._reset_grid_fresh = False
        self.resetting = False
        self.ready_event.set()
        logger.info(
            "[重置] ✨ 严格重置流程执行完毕，系统已回到初始状态，算法线程已放行"
        )

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
            _time.sleep(1)

    def _crazyflie_all_land(self):
        """控制所有实体无人机降落"""
        logger.info("开始所有实体无人机降落流程")
        for drone_name in self.drone_names:
            if self.config_data.get_uav_crazyflie_mirror(drone_name):
                self.crazyswarm.land(drone_name, 2)
                _time.sleep(2)

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
                # 立即发送刹车指令，防止上一帧速度延续导致无控飘移
                for d_name in self.drone_names:
                    try:
                        if not self.drones_config.is_crazyflie_mirror(d_name):
                            # AirSim 虚拟无人机：发 0 速度短指令，相当于 hover
                            self.drone_controller.move_by_velocity(0, 0, 0, 0.1, d_name)
                        else:
                            # Crazyflie 镜像：直接发送悬停
                            self.crazyswarm.hover(d_name)
                    except Exception:
                        pass

                # 等待所有线程结束
                logger.info("等待算法线程结束...")
                for drone_name, thread in self.drone_threads.items():
                    if thread and thread.is_alive():
                        thread.join(timeout=5.0)  # 最多等待5秒
                        if thread.is_alive():
                            logger.warning(f"无人机{drone_name}算法线程未能正常结束")
                        else:
                            logger.info(f"无人机{drone_name}算法线程已停止")
                _time.sleep(0.5)  # 减少等待时间
            else:
                logger.info("[步骤1/8] 跳过（算法未运行）")

            # 2. 所有无人机降落
            logger.info("[步骤2/8] 所有无人机降落...")
            self._land_all()
            _time.sleep(1)  # 减少等待时间

            # 3. 发送Unity重置命令
            logger.info("[步骤3/8] 发送重置命令到Unity...")
            self.unity_socket.send_reset_command()
            _time.sleep(2)  # 等待Unity处理重置命令并完成

            # 4. 重置AirSim模拟器
            logger.info("[步骤4/8] 重置AirSim模拟器...")
            if not self.drone_controller.reset():
                logger.error("AirSim模拟器重置失败")
                return False
            _time.sleep(1.5)  # 等待AirSim重置完成

            # 5. 清理本地数据
            logger.info("[步骤5/8] 清理本地数据...")
            self._clear_local_data(reset_grid_entropy=True)

            # 6. 重新初始化无人机
            logger.info("[步骤6/8] 重新初始化无人机...")
            if not self._init_drones():
                logger.error("无人机重新初始化失败")
                return False
            _time.sleep(1)

            # 7. 发送配置数据到Unity（包含Leader位置等初始配置）
            logger.info("[步骤7/8] 发送配置数据到Unity...")
            self.unity_socket.send_config(self.config_data)
            _time.sleep(0.5)

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

    def _clear_local_data(self, reset_grid_entropy: bool = True) -> None:
        """清理本地数据状态"""
        try:
            # 重置运行时数据
            for drone_name in self.drone_names:
                # 显式将 position 设为 None，避免默认 Vector3() 触发同步误判
                self.unity_runtime_data[drone_name] = ScannerRuntimeData()
                self.unity_runtime_data[drone_name].position = None

                self.processed_runtime_data[drone_name] = ScannerRuntimeData()
                self.last_positions[drone_name] = {}

            # 重置网格数据 (按参数决定是否重置熵值，保持格子对象引用和列表结构稳定)
            with self.grid_lock:
                if reset_grid_entropy:
                    self.grid_data.set_preserve_entropy(False)
                    self.grid_data.reset_entropy()
                else:
                    self.grid_data.set_preserve_entropy(True)
            # 重新创建算法实例（所有无人机使用相同的权重）
            self.algorithms = {}
            for name in self.drone_names:
                algo = ScannerAlgorithm(self.config_data)
                self.algorithms[name] = algo

            logger.info("本地数据清理完成 (算法实例已重新创建)")
        except Exception as e:
            logger.error(f"清理本地数据失败: {str(e)}")


if __name__ == "__main__":
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="多无人机算法服务器",
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
        """,
    )
    parser.add_argument(
        "--use-learned-weights",
        action="store_true",
        help="使用DDPG学习的权重（需要先训练模型）",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="DDPG模型路径（相对或绝对路径，不含.zip后缀）。如果不指定，将自动选择：best_model > weight_predictor_airsim > weight_predictor_simple",
    )
    parser.add_argument("--drones", type=int, default=1, help="无人机数量（默认1）")
    parser.add_argument(
        "--no-visualization", action="store_true", help="禁用可视化（默认启用）"
    )
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
                logger.info(
                    "模型: 自动选择（best_model > weight_predictor_airsim > weight_predictor_simple）"
                )
        else:
            logger.info("模式: 固定权重")
        logger.info(f"可视化: {'禁用' if args.no_visualization else '启用'}")
        logger.info("=" * 60)

        # 创建服务器实例
        server = MultiDroneAlgorithmServer(
            drone_names=drone_names,
            use_learned_weights=args.use_learned_weights,
            model_path=args.model_path,
            enable_visualization=not args.no_visualization,
        )

        if server.start():
            server.start_mission()
            # 主循环保持运行
            while server.running:
                _time.sleep(1)
    except KeyboardInterrupt:
        logger.info("用户中断，停止服务")
    except Exception as e:
        logger.critical(f"服务运行出错: {str(e)}", exc_info=True)
    finally:
        if "server" in locals():
            server.stop()
        sys.exit(0)

