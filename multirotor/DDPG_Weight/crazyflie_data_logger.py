"""
Crazyflie 实体无人机数据记录器

功能说明：
    - 在实体无人机训练时实时记录飞行数据
    - 支持记录位置、速度、加速度、姿态、电池等完整状态信息
    - 训练结束后保存为 JSON 或 CSV 文件
    - 记录训练权重变化历史
    - 支持多无人机数据同步记录

使用场景：
    - Crazyflie 在线训练（train_with_crazyflie_online.py）
    - 虚实融合训练（train_with_hybrid.py）
    
日期：2026-01-26
"""

import csv
import json
import os
import time
import threading
from typing import Dict, List, Optional
from dataclasses import asdict

# 导入 Crazyflie 日志数据类
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Crazyswarm.crazyflie_logging_data import CrazyflieLoggingData


class CrazyflieDataLogger:
    """
    Crazyflie 实体无人机数据记录器
    
    功能：
        - 实时记录实体无人机的飞行数据
        - 记录训练权重的变化历史
        - 支持多种输出格式（JSON、CSV）
        - 线程安全的数据收集
    """
    
    def __init__(self, drone_names: List[str], output_dir: str = "crazyflie_logs"):
        """
        初始化数据记录器
        
        参数：
            drone_names: 需要记录的无人机名称列表（如 ["UAV1", "UAV2"]）
            output_dir: 输出目录，相对于当前目录
        """
        self.drone_names = drone_names
        self.output_dir = output_dir
        self.is_recording = False
        
        # 数据存储
        self.flight_data: Dict[str, List[Dict]] = {name: [] for name in drone_names}
        self.weight_history: List[Dict] = []  # 权重变化历史
        self.episode_data: List[Dict] = []  # Episode 统计信息
        
        # 线程锁，保护数据访问
        self.data_lock = threading.Lock()
        
        # 记录开始时间
        self.start_time = None
        self.session_id = None
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"✅ CrazyflieDataLogger 初始化完成")
        print(f"   记录无人机: {', '.join(drone_names)}")
        print(f"   输出目录: {os.path.abspath(self.output_dir)}")
    
    def start_recording(self):
        """开始记录数据"""
        with self.data_lock:
            if self.is_recording:
                print("⚠️  数据记录器已经在运行中")
                return False
            
            self.is_recording = True
            self.start_time = time.time()
            self.session_id = time.strftime("%Y%m%d_%H%M%S")
            
            # 清空之前的数据
            self.flight_data = {name: [] for name in self.drone_names}
            self.weight_history = []
            self.episode_data = []
            
            print(f"🎬 开始记录实体无人机数据")
            print(f"   Session ID: {self.session_id}")
            print(f"   开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            return True
    
    def stop_recording(self):
        """停止记录数据"""
        with self.data_lock:
            if not self.is_recording:
                print("⚠️  数据记录器未运行")
                return False
            
            self.is_recording = False
            duration = time.time() - self.start_time if self.start_time else 0
            
            print(f"⏹️  停止记录实体无人机数据")
            print(f"   记录时长: {duration:.2f} 秒")
            print(f"   记录数据点:")
            for drone_name in self.drone_names:
                print(f"     - {drone_name}: {len(self.flight_data[drone_name])} 条")
            print(f"   权重变化记录: {len(self.weight_history)} 条")
            print(f"   Episode 记录: {len(self.episode_data)} 条")
            return True
    
    def record_flight_data(self, drone_name: str, logging_data: CrazyflieLoggingData):
        """
        记录单个无人机的飞行数据
        
        参数：
            drone_name: 无人机名称
            logging_data: CrazyflieLoggingData 实例
        """
        if not self.is_recording:
            return
        
        if drone_name not in self.drone_names:
            return
        
        if logging_data is None:
            return
        
        with self.data_lock:
            # 计算相对时间
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            
            # 转换为字典并添加时间戳
            data_dict = logging_data.to_dict()
            data_dict['elapsed_time'] = elapsed_time
            data_dict['session_id'] = self.session_id
            data_dict['drone_name'] = drone_name
            
            self.flight_data[drone_name].append(data_dict)
    
    def record_weights(self, drone_name: str, weights: Dict[str, float], episode: int = None, step: int = None):
        """
        记录训练权重变化
        
        参数：
            drone_name: 无人机名称
            weights: 权重字典，包含 5 个 APF 系数
            episode: Episode 编号（可选）
            step: 步数（可选）
        """
        if not self.is_recording:
            return
        
        with self.data_lock:
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            
            weight_record = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'elapsed_time': elapsed_time,
                'session_id': self.session_id,
                'drone_name': drone_name,
                'episode': episode,
                'step': step,
                'repulsionCoefficient': weights.get('repulsionCoefficient', 0.0),
                'entropyCoefficient': weights.get('entropyCoefficient', 0.0),
                'distanceCoefficient': weights.get('distanceCoefficient', 0.0),
                'leaderRangeCoefficient': weights.get('leaderRangeCoefficient', 0.0),
                'directionRetentionCoefficient': weights.get('directionRetentionCoefficient', 0.0)
            }
            
            self.weight_history.append(weight_record)
    
    def record_episode_stats(self, episode: int, reward: float, length: int, **kwargs):
        """
        记录 Episode 统计信息
        
        参数：
            episode: Episode 编号
            reward: Episode 总奖励
            length: Episode 步数
            **kwargs: 其他自定义统计信息
        """
        if not self.is_recording:
            return
        
        with self.data_lock:
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            
            episode_record = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'elapsed_time': elapsed_time,
                'session_id': self.session_id,
                'episode': episode,
                'reward': reward,
                'length': length
            }
            
            # 添加自定义统计信息
            episode_record.update(kwargs)
            
            self.episode_data.append(episode_record)
    
    def save_to_json(self, filename: str = None) -> str:
        """
        保存数据为 JSON 格式
        
        参数：
            filename: 自定义文件名（不含扩展名），None 则使用 session_id
            
        返回：
            保存的文件路径
        """
        with self.data_lock:
            if filename is None:
                filename = f"crazyflie_training_log_{self.session_id}"
            
            filepath = os.path.join(self.output_dir, f"{filename}.json")
            
            # 构建完整的数据结构
            full_data = {
                'metadata': {
                    'session_id': self.session_id,
                    'start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time)) if self.start_time else None,
                    'duration_seconds': time.time() - self.start_time if self.start_time else 0,
                    'drone_names': self.drone_names,
                    'total_episodes': len(self.episode_data),
                    'data_format': 'crazyflie_training_log_v1.0'
                },
                'flight_data': self.flight_data,
                'weight_history': self.weight_history,
                'episode_stats': self.episode_data
            }
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(full_data, f, ensure_ascii=False, indent=2)
                
                file_size = os.path.getsize(filepath) / 1024  # KB
                print(f"💾 数据已保存为 JSON 格式")
                print(f"   文件: {os.path.abspath(filepath)}")
                print(f"   大小: {file_size:.2f} KB")
                return filepath
            except Exception as e:
                print(f"❌ 保存 JSON 文件失败: {e}")
                return None
    
    def save_flight_data_to_csv(self, drone_name: str = None, filename: str = None) -> str:
        """
        保存飞行数据为 CSV 格式
        
        参数：
            drone_name: 指定无人机名称，None 则保存所有无人机的数据
            filename: 自定义文件名（不含扩展名），None 则使用 session_id
            
        返回：
            保存的文件路径
        """
        with self.data_lock:
            if drone_name and drone_name not in self.drone_names:
                print(f"⚠️  无人机 {drone_name} 不在记录列表中")
                return None
            
            # 确定要保存的无人机
            drones_to_save = [drone_name] if drone_name else self.drone_names
            
            for drone in drones_to_save:
                if len(self.flight_data[drone]) == 0:
                    print(f"⚠️  {drone} 没有飞行数据，跳过")
                    continue
                
                if filename is None:
                    csv_filename = f"crazyflie_flight_{drone}_{self.session_id}.csv"
                else:
                    csv_filename = f"{filename}_{drone}.csv"
                
                filepath = os.path.join(self.output_dir, csv_filename)
                
                try:
                    # 获取所有字段名（使用第一条数据的键）
                    fieldnames = list(self.flight_data[drone][0].keys())
                    
                    with open(filepath, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(self.flight_data[drone])
                    
                    file_size = os.path.getsize(filepath) / 1024  # KB
                    print(f"💾 {drone} 飞行数据已保存为 CSV 格式")
                    print(f"   文件: {os.path.abspath(filepath)}")
                    print(f"   大小: {file_size:.2f} KB")
                    print(f"   记录数: {len(self.flight_data[drone])} 条")
                
                except Exception as e:
                    print(f"❌ 保存 {drone} CSV 文件失败: {e}")
                    return None
            
            return filepath
    
    def save_weight_history_to_csv(self, filename: str = None) -> str:
        """
        保存权重历史为 CSV 格式
        
        参数：
            filename: 自定义文件名（不含扩展名），None 则使用 session_id
            
        返回：
            保存的文件路径
        """
        with self.data_lock:
            if len(self.weight_history) == 0:
                print("⚠️  没有权重历史数据")
                return None
            
            if filename is None:
                filename = f"crazyflie_weights_{self.session_id}"
            
            filepath = os.path.join(self.output_dir, f"{filename}.csv")
            
            try:
                fieldnames = list(self.weight_history[0].keys())
                
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.weight_history)
                
                file_size = os.path.getsize(filepath) / 1024  # KB
                print(f"💾 权重历史已保存为 CSV 格式")
                print(f"   文件: {os.path.abspath(filepath)}")
                print(f"   大小: {file_size:.2f} KB")
                print(f"   记录数: {len(self.weight_history)} 条")
                return filepath
            
            except Exception as e:
                print(f"❌ 保存权重历史 CSV 文件失败: {e}")
                return None
    
    def save_all(self, base_filename: str = None):
        """
        保存所有数据（JSON + CSV）
        
        参数：
            base_filename: 基础文件名（不含扩展名），None 则使用 session_id
        """
        print(f"\n{'='*60}")
        print(f"📊 保存实体无人机训练数据")
        print(f"{'='*60}")
        
        # 保存完整的 JSON 数据
        self.save_to_json(base_filename)
        
        print()
        
        # 保存每个无人机的飞行数据 CSV
        for drone_name in self.drone_names:
            self.save_flight_data_to_csv(drone_name, base_filename)
        
        print()
        
        # 保存权重历史 CSV
        self.save_weight_history_to_csv(base_filename)
        
        print(f"{'='*60}")
        print(f"✅ 所有数据保存完成")
        print(f"{'='*60}\n")
    
    def get_statistics(self) -> Dict:
        """
        获取记录统计信息
        
        返回：
            统计信息字典
        """
        with self.data_lock:
            stats = {
                'session_id': self.session_id,
                'is_recording': self.is_recording,
                'duration_seconds': time.time() - self.start_time if self.start_time else 0,
                'drone_names': self.drone_names,
                'flight_data_points': {name: len(self.flight_data[name]) for name in self.drone_names},
                'weight_history_points': len(self.weight_history),
                'episode_count': len(self.episode_data),
                'total_data_points': sum(len(self.flight_data[name]) for name in self.drone_names)
            }
            return stats
