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

    def __init__(self, config_file: Optional[str] = None, drone_names: Optional[List[str]] = None, use_learned_weights: bool = False):
        """
        初始化服务器实例
        :param config_file: 算法配置文件路径（默认使用scanner_config.json）
        :param drone_names: 无人机名称列表（默认使用["UAV1", "UAV2", "UAV3"]）
        :param use_learned_weights: 是否使用学习的权重（DQN模型预测）
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
        
        # DQN权重预测（可选）
        self.use_learned_weights = use_learned_weights
        self.weight_model = None
        if self.use_learned_weights:
            self._init_weight_predictor()
        
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

    def _init_weight_predictor(self):
        """初始化权重预测器（DDPG模型）"""
        try:
            logger.info("初始化权重预测器...")
            from stable_baselines3 import DDPG
            
            model_path = os.path.join(os.path.dirname(__file__), 'DQN', 'models', 'weight_predictor_simple')
            
            if os.path.exists(model_path + '.zip'):
                self.weight_model = DDPG.load(model_path)
                logger.info(f"✓ 权重预测模型加载成功: {model_path}")
            else:
                logger.warning(f"权重预测模型不存在: {model_path}.zip")
                logger.warning("将使用配置文件中的固定权重")
                self.use_learned_weights = False
                
        except ImportError:
            logger.error("stable-baselines3未安装，无法使用权重预测")
            logger.info("安装方法: pip install stable-baselines3")
            self.use_learned_weights = False
        except Exception as e:
            logger.error(f"权重预测器初始化失败: {str(e)}")
            self.use_learned_weights = False
    
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
            
            # 2. 启动算法处理线程
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

            logger.info("所有无人机任务启动完成")
            return True
        logger.warning("任务已在运行中")
        return False

    def _takeoff_all(self) -> bool:
        """控制所有无人机起飞"""
        logger.info("开始所有无人机起飞流程")
        all_success = True
        for drone_name in self.drone_names:
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
    
    def _predict_weights(self, drone_name: str) -> Dict[str, float]:
        """使用模型预测权重并进行平衡处理"""
        if not self.weight_model:
            return None
        
        try:
            state = self._get_state_for_prediction(drone_name)
            action, _ = self.weight_model.predict(state, deterministic=True)
            
            # 权重范围限制 [0.5, 5.0]
            action = np.clip(action, 0.5, 5.0)
            
            # 权重平衡处理：避免某个权重过高
            action_mean = np.mean(action)
            action_std = np.std(action)
            
            # 如果标准差过大，进行平滑
            if action_std > 1.5:
                action = action_mean + (action - action_mean) * 0.7
                action = np.clip(action, 0.5, 5.0)
            
            # 确保最大权重不超过最小权重的5倍
            min_weight = np.min(action)
            max_weight = np.max(action)
            if max_weight > min_weight * 5:
                scale = (min_weight * 5) / max_weight
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
        logger.info(f"无人机{drone_name}算法线程启动")
        while self.running:
            try:
                # 检查数据就绪状态
                has_grid = bool(self.grid_data.cells)
                has_runtime = bool(self.unity_runtime_data[drone_name].position)

                if not (has_grid and has_runtime):
                    time.sleep(1)
                    continue

                # 如果启用权重预测，更新APF权重
                if self.use_learned_weights:
                    predicted_weights = self._predict_weights(drone_name)
                    if predicted_weights:
                        self.algorithms[drone_name].set_coefficients(predicted_weights)
                
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

        if not success:
            logger.error(f"无人机{drone_name}移动指令发送失败")
        else:
            logger.debug(
                f"无人机{drone_name}移动: Unity方向{direction} -> "
                f"水平{horizontal_direction} + 垂直{vertical_direction} -> "
                f"AirSim速度{velocity_airsim} (水平:{horizontal_speed_airsim:.2f}, 垂直:{abs(velocity_airsim.z):.2f})"
            )

    def _check_drone_stuck(self, drone_name: str, current_pos: Vector3) -> None:
        """检查无人机是否卡住（位置长时间不变）"""
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
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='多无人机算法服务器')
    parser.add_argument('--use-learned-weights', action='store_true', 
                        help='使用DQN学习的权重（需要先训练模型）')
    parser.add_argument('--drones', type=int, default=2,
                        help='无人机数量（默认2）')
    args = parser.parse_args()
    
    try:
        # 生成无人机名称列表
        drone_names = [f"UAV{i}" for i in range(1, args.drones + 1)]
        
        logger.info("=" * 60)
        logger.info(f"启动多无人机系统 - {args.drones}台无人机")
        logger.info(f"无人机列表: {drone_names}")
        if args.use_learned_weights:
            logger.info("模式: DQN权重预测")
        else:
            logger.info("模式: 固定权重")
        logger.info("=" * 60)
        
        # 创建服务器实例
        server = MultiDroneAlgorithmServer(
            drone_names=drone_names,
            use_learned_weights=args.use_learned_weights
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
