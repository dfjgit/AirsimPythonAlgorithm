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
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, Callable

logger = logging.getLogger("DataCollector")


class DataCollector:
    """数据采集器类，负责采集和记录扫描数据"""
    
    def __init__(self, data_dir: Optional[str] = None, collection_interval: float = 1.0, enable_debug_print: bool = False):
        """
        初始化数据采集器
        :param data_dir: 数据保存目录（默认使用当前目录下的data_logs）
        :param collection_interval: 采集间隔（秒，默认1.0）
        :param enable_debug_print: 是否启用DEBUG打印（默认False，训练时应设置为True）
        """
        self.collection_interval = collection_interval
        self.running = False
        self.collection_thread: Optional[threading.Thread] = None
        self.csv_file = None
        self.csv_writer = None
        self.start_time = time.time()
        self.header_written = False  # 表头是否已写入
        self.drone_names_list = []  # 无人机名称列表（用于确定列顺序）
        self.enable_debug_print = enable_debug_print  # 控制DEBUG打印开关
        
        # 初始化CSV文件
        self._init_csv_file(data_dir)
    
    def _init_csv_file(self, data_dir: Optional[str] = None):
        """初始化CSV文件并写入表头"""
        try:
            # 创建数据采集目录
            if data_dir:
                data_path = Path(data_dir)
            else:
                data_path = Path(__file__).parent.parent / "data_logs"
            
            data_path.mkdir(exist_ok=True)
            
            # 生成CSV文件名（带时间戳）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = data_path / f"scan_data_{timestamp}.csv"
            
            # 打开CSV文件（表头将在第一次采集数据时写入）
            self.csv_file = open(csv_filename, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_filename = csv_filename
            
            logger.info(f"数据采集系统初始化完成，输出文件: {csv_filename}")
        except Exception as e:
            logger.error(f"数据采集系统初始化失败: {str(e)}")
            self.csv_file = None
            self.csv_writer = None

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
              get_battery_data_func: Callable,  # 新增：获取电量数据的函数
              data_lock: threading.Lock,
              grid_lock: threading.Lock):
        """
        启动数据采集线程
        :param get_grid_data_func: 获取网格数据的函数
        :param get_runtime_data_func: 获取运行时数据的函数（返回Dict[str, ScannerRuntimeData]）
        :param get_algorithms_func: 获取算法实例的函数（返回Dict[str, ScannerAlgorithm]）
        :param get_drone_names_func: 获取无人机名称列表的函数
        :param data_lock: 数据锁
        :param grid_lock: 网格锁
        """
        if self.running:
            logger.warning("数据采集线程已在运行")
            return
        
        self.running = True
        self.start_time = time.time()
        
        self.collection_thread = threading.Thread(
            target=self._collection_thread,
            args=(
                get_grid_data_func,
                get_runtime_data_func,
                get_algorithms_func,
                get_drone_names_func,
                get_battery_data_func,  # 新增
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
        
        # 关闭CSV文件
        if self.csv_file:
            try:
                self.csv_file.close()
                logger.info("数据采集文件已关闭")
            except Exception as e:
                logger.error(f"关闭数据采集文件失败: {str(e)}")
    
    def _collection_thread(self,
                          get_grid_data_func,
                          get_runtime_data_func,
                          get_algorithms_func,
                          get_drone_names_func,
                          get_battery_data_func,  # 新增
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
                
                # 获取所有无人机的坐标
                drone_positions = {}
                with data_lock:
                    for drone_name in self.drone_names_list:
                        runtime_data = runtime_data_dict.get(drone_name)
                        if runtime_data and runtime_data.position:
                            drone_positions[drone_name] = {
                                'x': runtime_data.position.x,
                                'y': runtime_data.position.y,
                                'z': runtime_data.position.z
                            }
                        else:
                            drone_positions[drone_name] = {'x': 0.0, 'y': 0.0, 'z': 0.0}
                
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
                
                # 如果表头未写入，先写入表头
                if self.csv_writer and not self.header_written:
                    header = [
                        'timestamp', 
                        'elapsed_time', 
                        'scanned_count', 
                        'unscanned_count', 
                        'total_count', 
                        'scan_ratio',
                        'global_avg_entropy',
                        'global_scan_ratio',
                        'entropy_bins',
                        'entropy_hist',
                        'entropy_cdf',
                        'repulsion_coefficient',
                        'entropy_coefficient',
                        'distance_coefficient',
                        'leader_range_coefficient',
                        'direction_retention_coefficient'
                    ]
                    # 为每个无人机添加坐标列
                    for drone_name in self.drone_names_list:
                        header.append(f'{drone_name}_x')
                        header.append(f'{drone_name}_y')
                        header.append(f'{drone_name}_z')
                    
                    # 为每个无人机添加电量列
                    for drone_name in self.drone_names_list:
                        header.append(f'{drone_name}_battery_voltage')
                    
                    self.csv_writer.writerow(header)
                    self.csv_file.flush()
                    self.header_written = True
                    logger.info(f"CSV表头已写入，包含 {len(self.drone_names_list)} 个无人机的坐标列")
                
                # 记录到CSV文件
                if self.csv_writer:
                    current_time = time.time()
                    elapsed_time = current_time - self.start_time
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    bins, hist, cdf = self._calc_entropy_distribution(entropies)

                    row = [
                        timestamp,
                        f"{elapsed_time:.2f}",
                        scanned_count,
                        unscanned_count,
                        total_count,
                        f"{scan_ratio:.2f}%",
                        f"{global_avg_entropy:.2f}",
                        f"{global_scan_ratio:.2f}%",
                        json.dumps(bins, ensure_ascii=False),
                        json.dumps(hist, ensure_ascii=False),
                        json.dumps(cdf, ensure_ascii=False),
                        weights.get('repulsionCoefficient', 0.0),
                        weights.get('entropyCoefficient', 0.0),
                        weights.get('distanceCoefficient', 0.0),
                        weights.get('leaderRangeCoefficient', 0.0),
                        weights.get('directionRetentionCoefficient', 0.0)
                    ]
                    
                    # 添加所有无人机的坐标
                    for drone_name in self.drone_names_list:
                        pos = drone_positions.get(drone_name, {'x': 0.0, 'y': 0.0, 'z': 0.0})
                        row.append(f"{pos['x']:.3f}")
                        row.append(f"{pos['y']:.3f}")
                        row.append(f"{pos['z']:.3f}")
                    
                    # 添加所有无人机的电量
                    for drone_name in self.drone_names_list:
                        drone_battery = battery_data.get(drone_name, {})
                        voltage = drone_battery.get('voltage', 0.0)
                        row.append(f"{voltage:.3f}")
                    
                    self.csv_writer.writerow(row)
                    self.csv_file.flush()  # 立即刷新到文件
                    
                    # 仅在启用DEBUG打印时输出（训练时启用）
                    if self.enable_debug_print:
                        logger.debug(
                            f"数据采集: 时间={elapsed_time:.1f}s, "
                            f"已侦察={scanned_count}, 未侦察={unscanned_count}, "
                            f"总数={total_count}, 扫描比例={scan_ratio:.2f}%, "
                            f"全局平均熵值={global_avg_entropy:.2f}, 全局采集比例={global_scan_ratio:.2f}%, "
                            f"权重={weights}, 无人机数={len(self.drone_names_list)}"
                        )
                
            except Exception as e:
                logger.error(f"数据采集线程出错: {str(e)}")
                logger.debug(traceback.format_exc())
                time.sleep(1)  # 出错后等待1秒再继续
        
        logger.info("数据采集线程已停止")

