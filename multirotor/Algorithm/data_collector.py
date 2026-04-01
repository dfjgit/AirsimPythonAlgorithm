"""
数据采集模块
功能：独立的数据采集系统，定期统计AOI区域内栅格的侦察状态和权重值
"""
import time
import threading
import csv
import json
import logging
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, Callable

logger = logging.getLogger("DataCollector")


class DataCollector:
    """数据采集器类，负责采集和记录扫描数据"""
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        collection_interval: float = 1.0,
        enable_debug_print: bool = False,
        training_prefix: str = "ddpg",
        experiment_id: str = "",
        stage_name: str = "",
        stage_index: int = 1,
        is_resume: bool = False,
        source_model: str = "",
    ):
        """
        初始化数据采集器
        :param data_dir: 数据保存目录（默认使用 DDPG_Weight/airsim_training_logs）
        :param collection_interval: 采集间隔（秒，默认1.0）
        :param enable_debug_print: 是否启用DEBUG打印（默认False，训练时应设置为True）
        :param training_prefix: 训练数据文件名前缀（默认 "ddpg"）
        """
        self.collection_interval = collection_interval
        self.training_prefix = training_prefix
        self.running = False
        self.collection_thread: Optional[threading.Thread] = None
        self.csv_file = None
        self.csv_writer = None
        self.training_csv_file = None  # 新增：训练数据 CSV 文件
        self.training_csv_writer = None  # 新增：训练数据 CSV writer
        self.global_start_time = time.time()
        self.start_time = self.global_start_time
        self.episode_start_time = self.global_start_time
        self.header_written = False  # 表头是否已写入
        self.training_header_written = False  # 新增：训练数据表头是否已写入
        self.drone_names_list = []  # 无人机名称列表（用于确定列顺序）
        self.enable_debug_print = enable_debug_print  # 控制DEBUG打印开关
        self.experiment_id = str(experiment_id or "").strip()
        self.stage_name = str(stage_name or "").strip()
        self.stage_index = max(int(stage_index or 1), 1)
        self.is_resume = bool(is_resume)
        self.source_model = str(source_model or "").strip()
        
        # 外部数据记录
        self.external_data = {}
        self.external_data_lock = threading.Lock()
        
        # 训练数据统计
        self.current_episode = 0
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.current_episode_weights = []  # 记录当前 episode 的权重用于取平均
        self.current_episode_elapsed_time = 0.0
        self.last_episode = -1  # 用于检测 episode 切换
        self.last_step = -1     # 用于检测 step 切换
        self.last_scanned_count = 0  # 记录最近一次扫描数
        self.last_global_scanned_count = 0  # 记录最近一次全局扫描数
        self.last_scan_episode = -1  # 用于检测 scan_data 的 episode 切换
        self.terminal_scan_episode = None  # 记录已写入 terminal 帧的 episode
        self.terminal_scan_step = None  # 记录 terminal 帧对应 step
        self.last_written_scan_episode = None  # 记录最近一次已写入的 episode
        self.last_written_scan_step = None  # 记录最近一次已写入的 step
        self.terminal_episode_meta = {}  # 按 episode 锁存终止元数据，避免 episode 切换时写串
        self.episode_scan_summary = {}  # 按 episode 锁存最后一帧扫描统计，避免下一轮首帧覆盖
        
        # 初始化CSV文件
        self._init_csv_file(data_dir)
    
    def set_external_data(self, key: str, value: Any):
        """设置外部数据（如训练奖励、步数等），将在下一次采集时记录"""
        with self.external_data_lock:
            self.external_data[key] = value

    def set_run_stage_meta(
        self,
        experiment_id: str,
        stage_name: str,
        stage_index: int,
        is_resume: bool = False,
        source_model: str = "",
    ):
        """Update stage metadata for rows written after initialization."""
        self.experiment_id = str(experiment_id or "").strip()
        self.stage_name = str(stage_name or "").strip()
        self.stage_index = max(int(stage_index or 1), 1)
        self.is_resume = bool(is_resume)
        self.source_model = str(source_model or "").strip()

    def _get_run_stage_meta(self):
        """Return current run-stage metadata with external overrides when present."""
        with self.external_data_lock:
            experiment_id = str(
                self.external_data.get("experiment_id", self.experiment_id)
            ).strip()
            stage_name = str(
                self.external_data.get("stage_name", self.stage_name)
            ).strip()
            stage_index = int(
                self.external_data.get("stage_index", self.stage_index) or self.stage_index
            )
            is_resume = bool(
                self.external_data.get("is_resume", self.is_resume)
            )
            source_model = str(
                self.external_data.get("source_model", self.source_model)
            ).strip()
        return experiment_id, stage_name, stage_index, is_resume, source_model
    
    def _init_csv_file(self, data_dir: Optional[str] = None):
        """初始化CSV文件并写入表头"""
        try:
            # 创建数据采集目录
            if data_dir:
                data_path = Path(data_dir)
            else:
                # 默认输出到 DDPG_Weight/airsim_training_logs 目录
                data_path = Path(__file__).parent.parent / "DDPG_Weight" / "airsim_training_logs"
            
            data_path.mkdir(parents=True, exist_ok=True)
            
            # 生成CSV文件名（带时间戳）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.experiment_id:
                file_token = f"{self.experiment_id}_stage{self.stage_index:02d}"
                csv_filename = data_path / f"scan_data_{file_token}_{timestamp}.csv"
                training_csv_filename = (
                    data_path / f"{self.training_prefix}_training_{file_token}_{timestamp}.csv"
                )
            else:
                csv_filename = data_path / f"scan_data_{timestamp}.csv"
                training_csv_filename = data_path / f"{self.training_prefix}_training_{timestamp}.csv"
            
            # 打开 scan_data CSV 文件（表头将在第一次采集数据时写入）
            self.csv_file = open(csv_filename, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_filename = csv_filename
            
            # 打开 training CSV 文件并写入表头
            self.training_csv_file = open(training_csv_filename, 'w', newline='', encoding='utf-8')
            self.training_csv_writer = csv.writer(self.training_csv_file)
            self.training_csv_filename = training_csv_filename
                    
            # 写入训练数据表头 (新增元数据字段以支持跨算法比较)
            header = ['episode', 'reward', 'length', 'scanned_cells', 'global_scanned_cells', 'timestep', 'train_timestep_end', 'elapsed_time', 'episode_elapsed_time', 'timestamp', 'scan_efficiency',
                      'avg_repulsion', 'avg_entropy', 'avg_distance', 'avg_leader', 'avg_direction',
                      'reset_reason', 'collision_count', 'collision_count_final', 'out_of_range_count', 'out_of_range_count_final', 'max_out_of_range_duration_sec', 'terminal_battery_voltage', 'success_flag', 'final_global_scan_ratio', 'max_global_scan_ratio', 'final_global_avg_entropy', 'min_global_avg_entropy',
                      'collision_object_name', 'collision_penetration_depth', 'collision_position', 'recent_trajectory',
                      'algorithm_type', 'env_type', 'control_mode',
                      'experiment_id', 'stage_name', 'stage_index', 'is_resume', 'source_model']
            self.training_csv_writer.writerow(header)
            self.training_csv_file.flush()
            self.training_header_written = True
            
            logger.info(f"数据采集系统初始化完成")
            logger.info(f"  - 扫描数据: {csv_filename}")
            logger.info(f"  - 训练数据: {training_csv_filename}")
        except Exception as e:
            logger.error(f"数据采集系统初始化失败: {str(e)}")
            self.csv_file = None
            self.csv_writer = None
            self.training_csv_file = None
            self.training_csv_writer = None

    def _calc_entropy_distribution(self, entropies, bin_size: int = 5, max_entropy: int = 100):
        """计算熵值直方图和CDF（用于CSV输出）"""
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
        cdf = []
        running = 0
        for count in hist:
            running += count
            cdf.append(running / total)

        return bins, hist, cdf
    
    def start(self, 
              get_grid_data_func: Callable,
              get_runtime_data_func: Callable,
              get_algorithms_func: Callable,
              get_drone_names_func: Callable,
              get_battery_data_func: Callable,  
              get_training_data_func: Optional[Callable] = None,  # 新增：获取训练数据的函数
              data_lock: Optional[threading.Lock] = None,
              grid_lock: Optional[threading.Lock] = None):
        """
        启动数据采集线程
        :param get_grid_data_func: 获取网格数据的函数
        :param get_runtime_data_func: 获取运行时数据的函数（返回Dict[str, ScannerRuntimeData]）
        :param get_algorithms_func: 获取算法实例的函数（返回Dict[str, ScannerAlgorithm]）
        :param get_drone_names_func: 获取无人机名称列表的函数
        :param get_battery_data_func: 获取电量数据的函数
        :param get_training_data_func: 获取训练数据的函数（返回Dict[str, Any]）
        :param data_lock: 数据锁
        :param grid_lock: 网格锁
        """
        if self.running:
            logger.warning("数据采集线程已在运行")
            return
        
        self.running = True
        self.global_start_time = time.time()
        self.start_time = self.global_start_time
        self.episode_start_time = self.global_start_time
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.current_episode_weights = []
        self.current_episode_elapsed_time = 0.0
        self.last_episode = -1
        self.last_step = -1
        self.last_scanned_count = 0
        self.last_global_scanned_count = 0
        self.terminal_episode_meta = {}
        self.episode_scan_summary = {}
        
        # 兼容旧版本调用（如果没有传锁）
        if data_lock is None:
            data_lock = threading.Lock()
        if grid_lock is None:
            grid_lock = threading.Lock()
            
        self.collection_thread = threading.Thread(
            target=self._collection_thread,
            args=(
                get_grid_data_func,
                get_runtime_data_func,
                get_algorithms_func,
                get_drone_names_func,
                get_battery_data_func,
                get_training_data_func,  # 新增
                data_lock,
                grid_lock
            ),
            daemon=True
        )
        self.collection_thread.start()
        logger.info("数据采集线程已启动")
    
    def stop(self):
        """停止数据采集线程并关闭文件"""
        if not self.running:
            return
            
        logger.info("停止数据采集线程...")
        self.running = False
            
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=2.0)
            logger.info("数据采集线程已停止")
            
        # 强制写入最后一个 episode 的数据
        self._flush_training_data()
            
        # 关闭 CSV 文件
        if self.csv_file:
            try:
                self.csv_file.close()
                logger.info(f"扫描数据文件已关闭: {self.csv_filename}")
            except Exception as e:
                logger.error(f"关闭扫描数据文件失败: {str(e)}")
            
        # 关闭训练数据 CSV 文件
        if self.training_csv_file:
            try:
                self.training_csv_file.close()
                logger.info(f"训练数据文件已关闭: {self.training_csv_filename}")
            except Exception as e:
                logger.error(f"关闭训练数据文件失败: {str(e)}")
    
    def _flush_training_data(self):
        """???????????????"""
        if self.training_csv_writer and self.last_episode >= 0 and self.current_episode_length > 0:
            try:
                elapsed_time = time.time() - self.global_start_time
                episode_elapsed_time = float(self.current_episode_elapsed_time)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # 统一口径：扫描效率使用“格子/步(Cell/Step)”而不是“格子/秒”
                # 这样才能与图表标题和跨算法对比含义保持一致。
                scan_efficiency = self.last_global_scanned_count / max(self.current_episode_length, 1)

                avg_weights = [0.0] * 5
                if self.current_episode_weights:
                    avg_weights = np.mean(self.current_episode_weights, axis=0).tolist()

                with self.external_data_lock:
                    self._capture_external_terminal_meta_locked()
                    algo_type = self.external_data.get('algorithm_type', '')
                    env_type = self.external_data.get('env_type', '')
                    ctrl_mode = self.external_data.get('control_mode', '')
                experiment_id, stage_name, stage_index, is_resume, source_model = self._get_run_stage_meta()
                terminal_meta = self._consume_terminal_meta(self.last_episode)
                scan_summary = self._consume_episode_scan_summary(self.last_episode) or {}
                final_scanned_count = int(scan_summary.get('scanned_count', self.last_scanned_count))
                final_global_scanned_count = int(scan_summary.get('global_scanned_count', self.last_global_scanned_count))
                final_step = int(scan_summary.get('step', max(self.current_episode_length, 0)))

                final_global_scan_ratio = float(
                    scan_summary.get(
                        'global_scan_ratio',
                        float(self.external_data.get('global_scan_ratio', 0.0) or 0.0),
                    )
                )
                final_global_avg_entropy = float(
                    scan_summary.get(
                        'global_avg_entropy',
                        float(self.external_data.get('global_avg_entropy', 100.0) or 100.0),
                    )
                )
                episode_max_global_scan_ratio = float(
                    scan_summary.get(
                        'max_global_scan_ratio',
                        (terminal_meta or {}).get(
                            'max_global_scan_ratio',
                            self.external_data.get('max_global_scan_ratio', final_global_scan_ratio),
                        ),
                    )
                )
                episode_min_global_avg_entropy = float(
                    scan_summary.get(
                        'min_global_avg_entropy',
                        (terminal_meta or {}).get(
                            'min_global_avg_entropy',
                            self.external_data.get('min_global_avg_entropy', final_global_avg_entropy),
                        ),
                    )
                )

                final_collision_count = int(
                    (terminal_meta or {}).get(
                        'collision_count',
                        self.external_data.get('collision_count', 0),
                    )
                )
                final_out_of_range_count = int(
                    (terminal_meta or {}).get(
                        'out_of_range_count',
                        self.external_data.get('out_of_range_count', 0),
                    )
                )
                episode_max_oob_duration = float(
                    scan_summary.get(
                        'max_out_of_range_duration_sec',
                        float(
                            self.external_data.get(
                                'max_out_of_range_duration_sec',
                                self.external_data.get('out_of_range_duration_sec', 0.0),
                            )
                            or 0.0
                        ),
                    )
                )

                terminal_battery_voltage = float(
                    scan_summary.get(
                        'terminal_battery_voltage',
                        float(self.external_data.get('terminal_battery_voltage', 0.0) or 0.0),
                    )
                )
                target_scan_ratio = float(self.external_data.get('target_scan_ratio', 0.0) or 0.0)
                success_flag = int(bool(target_scan_ratio > 0.0 and (final_global_scan_ratio / 100.0) >= target_scan_ratio))

                training_row = [
                    self.last_episode,
                    f"{self.current_episode_reward:.2f}",
                    self.current_episode_length,
                    final_scanned_count,
                    final_global_scanned_count,
                    final_step,
                    final_step,
                    f"{elapsed_time:.2f}",
                    f"{episode_elapsed_time:.2f}",
                    timestamp,
                    f"{final_global_scanned_count / max(self.current_episode_length, 1):.2f}",
                    f"{avg_weights[0]:.3f}",
                    f"{avg_weights[1]:.3f}",
                    f"{avg_weights[2]:.3f}",
                    f"{avg_weights[3]:.3f}",
                    f"{avg_weights[4]:.3f}",
                    (terminal_meta or {}).get('reset_reason', self.external_data.get('reset_reason', '')),
                    final_collision_count,
                    final_collision_count,
                    final_out_of_range_count,
                    final_out_of_range_count,
                    f"{episode_max_oob_duration:.3f}",
                    f"{terminal_battery_voltage:.3f}",
                    success_flag,
                    f"{final_global_scan_ratio:.2f}%",
                    f"{episode_max_global_scan_ratio:.2f}%",
                    f"{final_global_avg_entropy:.2f}",
                    f"{episode_min_global_avg_entropy:.2f}",
                    (terminal_meta or {}).get('collision_object_name', self.external_data.get('collision_object_name', '')),
                    f"{float((terminal_meta or {}).get('collision_penetration_depth', self.external_data.get('collision_penetration_depth', 0.0))):.3f}",
                    (terminal_meta or {}).get('collision_position', self.external_data.get('collision_position', '')),
                    (terminal_meta or {}).get('recent_trajectory', self.external_data.get('recent_trajectory', '')),
                    algo_type,
                    env_type,
                    ctrl_mode,
                    experiment_id,
                    stage_name,
                    int(stage_index),
                    int(bool(is_resume)),
                    source_model,
                ]
                self.training_csv_writer.writerow(training_row)
                self.training_csv_file.flush()
                logger.info(
                    f"??? Episode {self.last_episode} ???? (??: {self.current_episode_reward:.2f}, ??: {self.current_episode_length})"
                )
                self.last_episode = -1
                self.current_episode_length = 0
                self.current_episode_elapsed_time = 0.0
                self.current_episode_weights = []
            except Exception as e:
                logger.error(f"????????: {e}")

    def _capture_external_terminal_meta_locked(self):
        """从 external_data 中抓取并锁存终止元数据，避免 episode 切换时被下一轮覆盖。"""
        terminal_reason = str(self.external_data.get('terminal_reset_reason', '')).strip()
        terminal_episode = self.external_data.get('terminal_episode', -1)
        try:
            terminal_episode = int(terminal_episode)
        except (TypeError, ValueError):
            terminal_episode = -1

        if terminal_episode < 0 or not terminal_reason:
            return

        self.terminal_episode_meta[terminal_episode] = {
            'reset_reason': terminal_reason,
            'collision_count': int(self.external_data.get('terminal_collision_count', 0) or 0),
            'out_of_range_count': int(self.external_data.get('terminal_out_of_range_count', 0) or 0),
            'max_global_scan_ratio': float(self.external_data.get('terminal_max_global_scan_ratio', 0.0) or 0.0),
            'min_global_avg_entropy': float(self.external_data.get('terminal_min_global_avg_entropy', 100.0) or 100.0),
            'collision_object_name': self.external_data.get('terminal_collision_object_name', ''),
            'collision_penetration_depth': float(self.external_data.get('terminal_collision_penetration_depth', 0.0) or 0.0),
            'collision_position': self.external_data.get('terminal_collision_position', ''),
            'recent_trajectory': self.external_data.get('terminal_recent_trajectory', ''),
        }
        self.external_data['terminal_episode'] = -1
        self.external_data['terminal_reset_reason'] = ''
        self.external_data['terminal_collision_count'] = 0
        self.external_data['terminal_out_of_range_count'] = 0
        self.external_data['terminal_max_global_scan_ratio'] = 0.0
        self.external_data['terminal_min_global_avg_entropy'] = 100.0
        self.external_data['terminal_collision_object_name'] = ''
        self.external_data['terminal_collision_penetration_depth'] = 0.0
        self.external_data['terminal_collision_position'] = ''
        self.external_data['terminal_recent_trajectory'] = ''
        self.external_data['reset_reason'] = ''

    def _consume_terminal_meta(self, episode: int):
        """按 episode 取出锁存的终止元数据。"""
        try:
            episode = int(episode)
        except (TypeError, ValueError):
            return None
        return self.terminal_episode_meta.pop(episode, None)

    def _consume_episode_scan_summary(self, episode: int):
        """按 episode 取出锁存的最后一帧扫描统计。"""
        try:
            episode = int(episode)
        except (TypeError, ValueError):
            return None
        return self.episode_scan_summary.pop(episode, None)

    def _collection_thread(self,
                          get_grid_data_func,
                          get_runtime_data_func,
                          get_algorithms_func,
                          get_drone_names_func,
                          get_battery_data_func,
                          get_training_data_func,  # 新增
                          data_lock,
                          grid_lock):
        """数据采集线程主循环"""
        logger.info("数据采集线程启动")
        
        while self.running:
            try:
                # 等待采集间隔
                time.sleep(self.collection_interval)
                
                # 获取数据
                grid_data = get_grid_data_func()
                runtime_data_dict = get_runtime_data_func()
                algorithms_dict = get_algorithms_func()
                drone_names = get_drone_names_func()
                
                # 获取训练数据
                training_data = {}
                if get_training_data_func:
                    try:
                        training_data = get_training_data_func()
                    except:
                        pass
                
                # 获取外部手动设置的数据
                with self.external_data_lock:
                    training_data.update(self.external_data)
                
                # 检查数据是否就绪
                first_drone_name = None
                leader_pos = None
                leader_radius = None
                
                with data_lock:
                    if not runtime_data_dict or not drone_names:
                        continue
                    
                    # 更新无人机列表（如果发生变化）
                    if not self.drone_names_list or set(self.drone_names_list) != set(drone_names):
                        self.drone_names_list = sorted(drone_names)  # 按名称排序以保持一致性
                    
                    # 获取第一个无人机的运行时数据（所有无人机应该有相同的leader信息）
                    first_drone_name = drone_names[0]
                    runtime_data = runtime_data_dict.get(first_drone_name)
                    
                    if not runtime_data or not runtime_data.leader_position:
                        continue
                    
                    leader_pos = runtime_data.leader_position
                    leader_radius = runtime_data.leader_scan_radius
                
                # 获取所有无人机的坐标和姿态
                drone_states = {}
                with data_lock:
                    for drone_name in self.drone_names_list:
                        runtime_data = runtime_data_dict.get(drone_name)
                        if runtime_data:
                            drone_states[drone_name] = {
                                'x': runtime_data.position.x,
                                'y': runtime_data.position.y,
                                'z': runtime_data.position.z,
                                'roll': runtime_data.orientation.x if hasattr(runtime_data, 'orientation') else 0.0,
                                'pitch': runtime_data.orientation.y if hasattr(runtime_data, 'orientation') else 0.0,
                                'yaw': runtime_data.orientation.z if hasattr(runtime_data, 'orientation') else 0.0
                            }
                        else:
                            drone_states[drone_name] = {
                                'x': 0.0, 'y': 0.0, 'z': 0.0,
                                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
                            }
                
                # 获取权重值（从第一个无人机的算法实例）
                weights = {}
                if first_drone_name and first_drone_name in algorithms_dict:
                    algorithm = algorithms_dict[first_drone_name]
                    if hasattr(algorithm, 'get_current_coefficients'):
                        weights = algorithm.get_current_coefficients()
                    elif hasattr(algorithm, 'config'):
                        # 直接从config获取
                        config = algorithm.config
                        weights = {
                            'repulsionCoefficient': config.repulsionCoefficient,
                            'entropyCoefficient': config.entropyCoefficient,
                            'distanceCoefficient': config.distanceCoefficient,
                            'leaderRangeCoefficient': config.leaderRangeCoefficient,
                            'directionRetentionCoefficient': config.directionRetentionCoefficient
                        }
                
                # 获取所有无人机的电量数据
                battery_data = {}
                try:
                    battery_data = get_battery_data_func()
                except Exception as e:
                    logger.debug(f"获取电量数据失败: {str(e)}")
                    battery_data = {}
                
                # 统计AOI区域内的栅格状态和全局统计
                with grid_lock:
                    if not grid_data or not hasattr(grid_data, 'cells'):
                        continue
                    
                    scanned_count = 0
                    unscanned_count = 0
                    total_count = 0
                    
                    # 全局统计变量
                    global_scanned_count = 0
                    global_total_count = 0
                    total_entropy = 0.0
                    entropies = []
                    
                    for cell in grid_data.cells:
                        # 全局统计：所有栅格
                        global_total_count += 1
                        total_entropy += cell.entropy
                        entropies.append(cell.entropy)
                        
                        # 判断是否已侦察：entropy < 30 表示已侦察
                        if cell.entropy < 30:
                            global_scanned_count += 1
                        
                        # 计算栅格中心到Leader的距离
                        cell_center = cell.center
                        distance = (cell_center - leader_pos).magnitude()
                        
                        # 判断是否在AOI区域内（Leader扫描半径内）
                        if distance <= leader_radius:
                            total_count += 1
                            # 判断是否已侦察：entropy < 30 表示已侦察（计为0），否则未侦察（计为1）
                            if cell.entropy < 30:
                                scanned_count += 1
                            else:
                                unscanned_count += 1
                    
                    # 计算AOI区域扫描比例
                    scan_ratio = (scanned_count / total_count * 100) if total_count > 0 else 0.0
                    
                    # 计算全局平均熵值
                    global_avg_entropy = (total_entropy / global_total_count) if global_total_count > 0 else 0.0
                    
                    # 计算全局采集百分比
                    global_scan_ratio = (global_scanned_count / global_total_count * 100) if global_total_count > 0 else 0.0
                
                # 记录最近的扫描数，用于训练数据输出
                self.last_scanned_count = scanned_count
                self.last_global_scanned_count = global_scanned_count

                # 如果表头未写入，先写入表头
                if self.csv_writer and not self.header_written:
                    header = [
                        'episode',
                        'timestamp',
                        'elapsed_time',
                        'episode_elapsed_time',
                        'episode_step',
                        'step_reward',
                        'episode_reward',
                        'scanned_count',
                        'unscanned_count',
                        'total_count',
                        'global_scanned_count',
                        'global_total_count',
                        'scan_ratio',
                        'local_scan_ratio',
                        'global_avg_entropy',
                        'global_scan_ratio',
                        'reset_reason',
                        'collision_count',
                        'out_of_range_count',
                        'max_global_scan_ratio',
                        'min_global_avg_entropy',
                        'collision_object_name',
                        'collision_penetration_depth',
                        'collision_position',
                        'recent_trajectory',
                        'entropy_bins',
                        'entropy_hist',
                        'entropy_cdf',
                        'repulsion_coefficient',
                        'entropy_coefficient',
                        'distance_coefficient',
                        'leader_range_coefficient',
                        'direction_retention_coefficient',
                        'hl_action',
                        'hl_goal_x',
                        'hl_goal_y',
                        'hl_goal_z',
                        'algorithm_type',
                        'env_type',
                        'control_mode',
                        'current_drone',
                        'current_action',
                        'current_leader_distance',
                        'current_is_out_of_range',
                        'current_out_of_range_steps',
                        'current_out_of_range_duration_sec',
                        'current_out_of_range_count',
                        'current_drone_reward',
                        'experiment_id',
                        'stage_name',
                        'stage_index',
                        'is_resume',
                        'source_model',
                    ]
                    for drone_name in self.drone_names_list:
                        header.append(f'{drone_name}_action')
                        header.append(f'{drone_name}_leader_distance')
                        header.append(f'{drone_name}_is_out_of_range')
                        header.append(f'{drone_name}_out_of_range_steps')
                        header.append(f'{drone_name}_out_of_range_duration_sec')
                        header.append(f'{drone_name}_out_of_range_count')
                        header.append(f'{drone_name}_reward')
                    # ?????????????????
                    for drone_name in self.drone_names_list:
                        header.append(f'{drone_name}_x')
                        header.append(f'{drone_name}_y')
                        header.append(f'{drone_name}_z')
                        header.append(f'{drone_name}_roll')   # 新增
                        header.append(f'{drone_name}_pitch')  # 新增
                        header.append(f'{drone_name}_yaw')    # 新增
                    
                    # 为每个无人机添加电量列
                    for drone_name in self.drone_names_list:
                        header.append(f'{drone_name}_battery_voltage')
                    
                    self.csv_writer.writerow(header)
                    self.csv_file.flush()
                    self.header_written = True
                    logger.info(f"CSV表头已写入，包含 {len(self.drone_names_list)} 个无人机的坐标列")
                
                # 记录到 scan_data CSV 文件（不包含训练数据）
                if self.csv_writer:
                    current_time = time.time()
                    elapsed_time = current_time - self.global_start_time
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    bins, hist, cdf = self._calc_entropy_distribution(entropies)

                    # 提取元数据 (用于 scan_data 行写入)
                    with self.external_data_lock:
                        self._capture_external_terminal_meta_locked()
                        hl_action_val = self.external_data.get('hl_action', '')
                        hl_goal_x_val = self.external_data.get('hl_goal_x', '')
                        hl_goal_y_val = self.external_data.get('hl_goal_y', '')
                        hl_goal_z_val = self.external_data.get('hl_goal_z', '')
                        algo_type = self.external_data.get('algorithm_type', '')
                        env_type = self.external_data.get('env_type', '')
                        ctrl_mode = self.external_data.get('control_mode', '')
                        current_drone = training_data.get('drone_name', self.external_data.get('drone_name', ''))
                        current_action = training_data.get('last_action', self.external_data.get('last_action', ''))
                        current_leader_distance = training_data.get('leader_distance', self.external_data.get('leader_distance', ''))
                        current_is_out_of_range = training_data.get('is_out_of_range', self.external_data.get('is_out_of_range', ''))
                        current_out_of_range_steps = training_data.get('out_of_range_steps', self.external_data.get('out_of_range_steps', 0))
                        current_out_of_range_duration_sec = training_data.get('out_of_range_duration_sec', self.external_data.get('out_of_range_duration_sec', 0.0))
                        current_out_of_range_count = training_data.get('out_of_range_count', self.external_data.get('out_of_range_count', 0))
                        current_drone_reward = training_data.get('current_drone_reward', self.external_data.get('current_drone_reward', 0.0))
                        per_drone_actions = training_data.get('per_drone_actions', self.external_data.get('per_drone_actions', {})) or {}
                        reset_reason = self.external_data.get('reset_reason', '')
                        collision_count = int(self.external_data.get('collision_count', 0))
                        out_of_range_count = int(self.external_data.get('out_of_range_count', 0))
                        max_global_scan_ratio = float(self.external_data.get('max_global_scan_ratio', 0.0))
                        min_global_avg_entropy = float(self.external_data.get('min_global_avg_entropy', global_avg_entropy))
                        collision_object_name = self.external_data.get('collision_object_name', '')
                        collision_penetration_depth = float(self.external_data.get('collision_penetration_depth', 0.0))
                        collision_position = self.external_data.get('collision_position', '')
                        recent_trajectory = self.external_data.get('recent_trajectory', '')

                    experiment_id, stage_name, stage_index, is_resume, source_model = self._get_run_stage_meta()

                    # 获取当前episode（从training_data或external_data）
                    current_episode = training_data.get('episode', self.external_data.get('episode', -1))
                    current_step = training_data.get('step', self.external_data.get('step', -1))
                    episode_elapsed_time = float(
                        training_data.get('episode_elapsed_time', self.external_data.get('episode_elapsed_time', 0.0))
                    )
                    step_reward = float(training_data.get('step_reward', training_data.get('reward', 0.0)))
                    episode_reward = float(
                        training_data.get('episode_reward', training_data.get('total_reward', self.external_data.get('episode_reward', 0.0)))
                    )

                    try:
                        current_episode_int = int(current_episode)
                    except (TypeError, ValueError):
                        current_episode_int = -1
                    try:
                        current_step_int = int(current_step)
                    except (TypeError, ValueError):
                        current_step_int = -1

                    if current_episode_int != self.last_scan_episode:
                        self.last_scan_episode = current_episode_int
                        self.terminal_scan_episode = None
                        self.terminal_scan_step = None
                        self.last_written_scan_episode = None
                        self.last_written_scan_step = None

                    valid_step_frame = current_episode_int >= 0 and current_step_int > 0
                    latched_terminal_meta = self.terminal_episode_meta.get(current_episode_int)
                    terminal_reason = (
                        str((latched_terminal_meta or {}).get('reset_reason', '')).strip()
                        if valid_step_frame
                        else ""
                    )
                    is_terminal_frame = bool(terminal_reason)
                    is_duplicate_step_frame = (
                        current_episode_int == self.last_written_scan_episode
                        and current_step_int == self.last_written_scan_step
                    )
                    should_skip_scan_row = (
                        not valid_step_frame
                        or is_duplicate_step_frame
                        or self.terminal_scan_episode == current_episode_int
                    )

                    row_reset_reason = terminal_reason if is_terminal_frame else ""
                    row_collision_count = int((latched_terminal_meta or {}).get('collision_count', collision_count)) if is_terminal_frame else 0
                    row_out_of_range_count = int((latched_terminal_meta or {}).get('out_of_range_count', out_of_range_count)) if is_terminal_frame else 0
                    row_collision_object_name = (latched_terminal_meta or {}).get('collision_object_name', collision_object_name) if is_terminal_frame else ""
                    row_collision_penetration_depth = float((latched_terminal_meta or {}).get('collision_penetration_depth', collision_penetration_depth)) if is_terminal_frame else 0.0
                    row_collision_position = (latched_terminal_meta or {}).get('collision_position', collision_position) if is_terminal_frame else ""
                    row_recent_trajectory = (latched_terminal_meta or {}).get('recent_trajectory', recent_trajectory) if is_terminal_frame else ""

                    if not should_skip_scan_row:
                        row = [
                            current_episode,
                            timestamp,
                            f"{elapsed_time:.2f}",
                            f"{episode_elapsed_time:.2f}",
                            current_step,
                            f"{step_reward:.4f}",
                            f"{episode_reward:.4f}",
                            scanned_count,
                            unscanned_count,
                            total_count,
                            global_scanned_count,
                            global_total_count,
                            f"{scan_ratio:.2f}%",
                            f"{scan_ratio:.2f}%",
                            f"{global_avg_entropy:.2f}",
                            f"{global_scan_ratio:.2f}%",
                            row_reset_reason,
                            row_collision_count,
                            row_out_of_range_count,
                            f"{max_global_scan_ratio:.2f}%",
                            f"{min_global_avg_entropy:.2f}",
                            row_collision_object_name,
                            f"{row_collision_penetration_depth:.3f}",
                            row_collision_position,
                            row_recent_trajectory,
                            json.dumps(bins, ensure_ascii=False),
                            json.dumps(hist, ensure_ascii=False),
                            json.dumps(cdf, ensure_ascii=False),
                            weights.get('repulsionCoefficient', 0.0),
                            weights.get('entropyCoefficient', 0.0),
                            weights.get('distanceCoefficient', 0.0),
                            weights.get('leaderRangeCoefficient', 0.0),
                            weights.get('directionRetentionCoefficient', 0.0),
                            hl_action_val,
                            hl_goal_x_val,
                            hl_goal_y_val,
                            hl_goal_z_val,
                            algo_type,
                            env_type,
                            ctrl_mode,
                            current_drone,
                            current_action,
                            f"{float(current_leader_distance):.3f}" if current_leader_distance not in (None, "") else "",
                            int(bool(current_is_out_of_range)) if current_is_out_of_range not in ("", None) else "",
                            int(current_out_of_range_steps or 0),
                            f"{float(current_out_of_range_duration_sec or 0.0):.3f}",
                            int(current_out_of_range_count or 0),
                            f"{float(current_drone_reward):.4f}",
                            experiment_id,
                            stage_name,
                            int(stage_index),
                            int(bool(is_resume)),
                            source_model,
                        ]

                        for drone_name in self.drone_names_list:
                            row_action = per_drone_actions.get(drone_name, {}) if isinstance(per_drone_actions, dict) else {}
                            row.append(row_action.get('last_action', ''))
                            leader_distance = row_action.get('leader_distance', '')
                            row.append(f"{float(leader_distance):.3f}" if leader_distance not in (None, "") else "")
                            is_oor = row_action.get('is_out_of_range', '')
                            row.append(int(bool(is_oor)) if is_oor not in ("", None) else "")
                            row.append(int(row_action.get('out_of_range_steps', 0) or 0))
                            row.append(f"{float(row_action.get('out_of_range_duration_sec', 0.0) or 0.0):.3f}")
                            row.append(int(row_action.get('out_of_range_count', 0) or 0))
                            row.append(f"{float(row_action.get('current_drone_reward', 0.0) or 0.0):.4f}")
                        
                        # 添加所有无人机的坐标和姿态
                        for drone_name in self.drone_names_list:
                            state = drone_states.get(drone_name, {'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0})
                            row.append(f"{state['x']:.3f}")
                            row.append(f"{state['y']:.3f}")
                            row.append(f"{state['z']:.3f}")
                            row.append(f"{state['roll']:.2f}")
                            row.append(f"{state['pitch']:.2f}")
                            row.append(f"{state['yaw']:.2f}")
                        
                        # 添加所有无人机的电量
                        for drone_name in self.drone_names_list:
                            drone_battery = battery_data.get(drone_name, {})
                            voltage = drone_battery.get('voltage', 0.0)
                            row.append(f"{voltage:.3f}")
                        
                        self.csv_writer.writerow(row)
                        self.csv_file.flush()  # 立即刷新到文件
                        self.last_written_scan_episode = current_episode_int
                        self.last_written_scan_step = current_step_int
                        prev_scan_summary = self.episode_scan_summary.get(current_episode_int, {})
                        battery_voltage_values = []
                        for drone_name in self.drone_names_list:
                            drone_battery = battery_data.get(drone_name, {})
                            voltage = drone_battery.get('voltage', None)
                            if voltage not in (None, ""):
                                try:
                                    battery_voltage_values.append(float(voltage))
                                except (TypeError, ValueError):
                                    pass

                        self.episode_scan_summary[current_episode_int] = {
                            'step': int(current_step_int),
                            'scanned_count': int(scanned_count),
                            'global_scanned_count': int(global_scanned_count),
                            'scan_ratio': float(scan_ratio),
                            'global_scan_ratio': float(global_scan_ratio),
                            'global_avg_entropy': float(global_avg_entropy),
                            'terminal_battery_voltage': (
                                min(battery_voltage_values) if battery_voltage_values else 0.0
                            ),
                            'max_out_of_range_duration_sec': max(
                                float(prev_scan_summary.get('max_out_of_range_duration_sec', 0.0)),
                                float(current_out_of_range_duration_sec or 0.0),
                            ),
                            'max_global_scan_ratio': max(
                                float(prev_scan_summary.get('max_global_scan_ratio', global_scan_ratio)),
                                float(global_scan_ratio),
                            ),
                            'min_global_avg_entropy': min(
                                float(prev_scan_summary.get('min_global_avg_entropy', global_avg_entropy)),
                                float(global_avg_entropy),
                            ),
                        }
                        self.last_scanned_count = int(scanned_count)
                        self.last_global_scanned_count = int(global_scanned_count)

                        if is_terminal_frame:
                            self.terminal_scan_episode = current_episode_int
                            self.terminal_scan_step = current_step_int
                    
                    # 仅在启用DEBUG打印时输出（训练时启用）
                    if self.enable_debug_print and not should_skip_scan_row:
                        logger.debug(
                            f"数据采集: 时间={elapsed_time:.1f}s, "
                            f"已侦察={scanned_count}, 未侦察={unscanned_count}, "
                            f"总数={total_count}, 扫描比例={scan_ratio:.2f}%, "
                            f"全局平均熵值={global_avg_entropy:.2f}, 全局采集比例={global_scan_ratio:.2f}%, "
                            f"权重={weights}, 无人机数={len(self.drone_names_list)}"
                        )
                
                # 写入训练数据（每个 episode 完成时）
                if self.training_csv_writer and training_data:
                    current_episode = int(training_data.get('episode', 0))
                    current_step = int(training_data.get('step', -1))
                    step_reward = float(training_data.get('step_reward', training_data.get('reward', 0.0)))
                    episode_elapsed_time = float(training_data.get('episode_elapsed_time', 0.0))

                    current_weights = [
                        weights.get('repulsionCoefficient', 0.0),
                        weights.get('entropyCoefficient', 0.0),
                        weights.get('distanceCoefficient', 0.0),
                        weights.get('leaderRangeCoefficient', 0.0),
                        weights.get('directionRetentionCoefficient', 0.0)
                    ]

                    if current_episode != self.last_episode:
                        if self.last_episode >= 0 and self.current_episode_length > 0:
                            avg_weights = [0.0] * 5
                            if self.current_episode_weights:
                                avg_weights = np.mean(self.current_episode_weights, axis=0).tolist()

                            with self.external_data_lock:
                                self._capture_external_terminal_meta_locked()
                                algo_type = self.external_data.get('algorithm_type', '')
                                env_type = self.external_data.get('env_type', '')
                                ctrl_mode = self.external_data.get('control_mode', '')
                                reset_reason = self.external_data.get('reset_reason', '')
                                collision_count = int(self.external_data.get('collision_count', 0))
                                out_of_range_count = int(self.external_data.get('out_of_range_count', 0))
                                max_global_scan_ratio = float(self.external_data.get('max_global_scan_ratio', 0.0))
                                min_global_avg_entropy = float(self.external_data.get('min_global_avg_entropy', 100.0))
                                collision_object_name = self.external_data.get('collision_object_name', '')
                                collision_penetration_depth = float(self.external_data.get('collision_penetration_depth', 0.0))
                                collision_position = self.external_data.get('collision_position', '')
                                recent_trajectory = self.external_data.get('recent_trajectory', '')
                            experiment_id, stage_name, stage_index, is_resume, source_model = self._get_run_stage_meta()
                            terminal_meta = self._consume_terminal_meta(self.last_episode)
                            scan_summary = self._consume_episode_scan_summary(self.last_episode) or {}
                            elapsed_time = time.time() - self.global_start_time
                            previous_episode_elapsed = float(self.current_episode_elapsed_time)
                            # 统一口径：训练 CSV 中的 scan_efficiency 始终表示 Cell/Step。
                            final_scanned_count = int(scan_summary.get('scanned_count', self.last_scanned_count))
                            final_global_scanned_count = int(scan_summary.get('global_scanned_count', self.last_global_scanned_count))
                            final_step = int(scan_summary.get('step', max(self.current_episode_length, 0)))
                            scan_efficiency = final_global_scanned_count / max(self.current_episode_length, 1)

                            final_global_scan_ratio = float(
                                scan_summary.get(
                                    'global_scan_ratio',
                                    float(self.external_data.get('global_scan_ratio', 0.0) or 0.0),
                                )
                            )
                            final_global_avg_entropy = float(
                                scan_summary.get(
                                    'global_avg_entropy',
                                    float(self.external_data.get('global_avg_entropy', 100.0) or 100.0),
                                )
                            )
                            episode_max_global_scan_ratio = float(
                                scan_summary.get(
                                    'max_global_scan_ratio',
                                    (terminal_meta or {}).get('max_global_scan_ratio', max_global_scan_ratio),
                                )
                            )
                            episode_min_global_avg_entropy = float(
                                scan_summary.get(
                                    'min_global_avg_entropy',
                                    (terminal_meta or {}).get('min_global_avg_entropy', min_global_avg_entropy),
                                )
                            )

                            final_collision_count = int(
                                (terminal_meta or {}).get('collision_count', collision_count)
                            )
                            final_out_of_range_count = int(
                                (terminal_meta or {}).get('out_of_range_count', out_of_range_count)
                            )
                            episode_max_oob_duration = float(
                                scan_summary.get(
                                    'max_out_of_range_duration_sec',
                                    float(
                                        self.external_data.get(
                                            'max_out_of_range_duration_sec',
                                            self.external_data.get('out_of_range_duration_sec', 0.0),
                                        )
                                        or 0.0
                                    ),
                                )
                            )

                            terminal_battery_voltage = float(
                                scan_summary.get(
                                    'terminal_battery_voltage',
                                    float(self.external_data.get('terminal_battery_voltage', 0.0) or 0.0),
                                )
                            )
                            target_scan_ratio = float(self.external_data.get('target_scan_ratio', 0.0) or 0.0)
                            success_flag = int(
                                bool(
                                    target_scan_ratio > 0.0
                                    and (final_global_scan_ratio / 100.0) >= target_scan_ratio
                                )
                            )

                            training_row = [
                                self.last_episode,
                                f"{self.current_episode_reward:.2f}",
                                self.current_episode_length,
                                final_scanned_count,
                                final_global_scanned_count,
                                final_step,
                                final_step,
                                f"{elapsed_time:.2f}",
                                f"{previous_episode_elapsed:.2f}",
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                f"{scan_efficiency:.2f}",
                                f"{avg_weights[0]:.3f}",
                                f"{avg_weights[1]:.3f}",
                                f"{avg_weights[2]:.3f}",
                                f"{avg_weights[3]:.3f}",
                                f"{avg_weights[4]:.3f}",
                                (terminal_meta or {}).get('reset_reason', reset_reason),
                                final_collision_count,
                                final_collision_count,
                                final_out_of_range_count,
                                final_out_of_range_count,
                                f"{episode_max_oob_duration:.3f}",
                                f"{terminal_battery_voltage:.3f}",
                                success_flag,
                                f"{final_global_scan_ratio:.2f}%",
                                f"{episode_max_global_scan_ratio:.2f}%",
                                f"{final_global_avg_entropy:.2f}",
                                f"{episode_min_global_avg_entropy:.2f}",
                                (terminal_meta or {}).get('collision_object_name', collision_object_name),
                                f"{float((terminal_meta or {}).get('collision_penetration_depth', collision_penetration_depth)):.3f}",
                                (terminal_meta or {}).get('collision_position', collision_position),
                                (terminal_meta or {}).get('recent_trajectory', recent_trajectory),
                                algo_type,
                                env_type,
                                ctrl_mode,
                                experiment_id,
                                stage_name,
                                int(stage_index),
                                int(bool(is_resume)),
                                source_model
                            ]
                            self.training_csv_writer.writerow(training_row)
                            self.training_csv_file.flush()
                            logger.info(f"??? Episode {self.last_episode} ???? (??: {self.current_episode_reward:.2f}, ??: {self.current_episode_length})")

                        self.last_episode = current_episode
                        self.current_episode_elapsed_time = episode_elapsed_time
                        self.last_step = current_step
                        if current_step > 0:
                            self.current_episode_reward = step_reward
                            self.current_episode_length = 1
                            self.current_episode_weights = [current_weights]
                        else:
                            self.current_episode_reward = 0.0
                            self.current_episode_length = 0
                            self.current_episode_weights = []
                    else:
                        self.current_episode_elapsed_time = episode_elapsed_time
                        if current_step > 0 and current_step != self.last_step:
                            self.current_episode_reward += step_reward
                            self.current_episode_length += 1
                            self.last_step = current_step
                            self.current_episode_weights.append(current_weights)
            except Exception as e:
                logger.error(f"数据采集线程出错: {str(e)}")
                logger.debug(traceback.format_exc())
                time.sleep(1)  # 出错后等待1秒再继续
        
        logger.info("数据采集线程已停止")



