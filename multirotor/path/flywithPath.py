#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无人机路径飞行和比较脚本
使用AirSim单无人机从Path1的起点到终点飞行直线，
并按照Path1的时间戳采样记录实际位置，对比预期路径与实际飞行路径

使用速度控制方式（moveByVelocityAsync）而非位置控制（moveToPositionAsync）
以避免高度方向的固有偏差（约0.5m）
"""

import json
import time
import math
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import sys
import os

# 添加项目路径
current_dir = os.path.dirname(__file__)
multirotor_dir = os.path.dirname(current_dir)  # multirotor目录
project_dir = os.path.dirname(multirotor_dir)  # AirsimAlgorithmPython目录
sys.path.append(project_dir)

# 导入AirSim
try:
    import airsim
except ImportError:
    # 如果无法导入airsim，尝试添加路径
    airsim_path = os.path.join(project_dir, 'airsim')
    if os.path.exists(airsim_path):
        sys.path.insert(0, project_dir)
        import airsim

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('path_flight.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PathFlight")

class PathFlightController:
    """无人机路径飞行控制器"""
    
    def __init__(self):
        self.client = None
        self.vehicle_name = "UAV1"  # 默认无人机名称
        self.min_speed = 0.1  # 最小飞行速度 m/s
        self.max_speed = 5.0  # 最大飞行速度 m/s
        self.default_speed = 2.0  # 默认飞行速度 m/s
        self.position_tolerance = 0.2  # 位置容差 m
        self.actual_path = []  # 记录实际飞行路径
        self.connected = False
        self.ground_z = 0.0  # 地面的Z坐标（NED）
        self.takeoff_z = 0.0  # 起飞后的Z坐标（NED）
    
    def calculate_appropriate_speed(self, distance: float, available_time: float = None) -> float:
        """
        根据移动距离和可用时间计算合适的飞行速度
        
        Args:
            distance: 移动距离（米）
            available_time: 可用时间（秒），如果为None则根据距离估算
        
        Returns:
            合适的飞行速度（m/s）
        """
        if distance <= 0:
            return self.min_speed
        
        if available_time is not None and available_time > 0:
            # 根据时间计算所需速度
            required_speed = distance / available_time
        else:
            # 根据距离估算合适的速度
            # 短距离用慢速度，长距离用快速度
            if distance < 0.5:
                required_speed = 0.5  # 很短距离，慢速
            elif distance < 1.0:
                required_speed = 1.0  # 短距离
            elif distance < 3.0:
                required_speed = 2.0  # 中等距离
            elif distance < 10.0:
                required_speed = 3.0  # 较长距离
            else:
                required_speed = 4.0  # 长距离
        
        # 限制在最小和最大速度之间
        speed = max(self.min_speed, min(self.max_speed, required_speed))
        
        return speed
        
    def load_path(self, path_file: str) -> List[Dict[str, float]]:
        """加载路径文件"""
        try:
            with open(path_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取路径点（假设路径在"1"键下）
            if "1" in data and isinstance(data["1"], list):
                path_points = data["1"]
                logger.info(f"✓ 加载路径文件，包含 {len(path_points)} 个路径点")
                return path_points
            else:
                logger.error(f"路径文件 {path_file} 格式不正确")
                return []
                
        except Exception as e:
            logger.error(f"加载路径文件 {path_file} 失败: {str(e)}")
            return []
    
    def connect_and_setup(self) -> bool:
        """连接AirSim并设置无人机"""
        try:
            # 创建AirSim客户端
            self.client = airsim.MultirotorClient()
            
            # 确认连接
            self.client.confirmConnection()
            self.connected = True
            logger.info("✓ 连接到AirSim")
            
            self.client.reset()
            # 启用API控制和解锁
            self.client.enableApiControl(True, self.vehicle_name)
            self.client.armDisarm(True, self.vehicle_name)
            
            # 起飞前记录地面Z坐标
            state_before_takeoff = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            pos_before = state_before_takeoff.kinematics_estimated.position
            self.ground_z = pos_before.z_val
            logger.info(f"地面Z坐标: {self.ground_z:.4f}m")
            
            # 起飞
            self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
            time.sleep(2)
            
            # 起飞后记录位置
            state_after_takeoff = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            pos_after = state_after_takeoff.kinematics_estimated.position
            self.takeoff_z = pos_after.z_val
            takeoff_height_from_ground = -(pos_after.z_val - self.ground_z)
            logger.info(f"✓ 起飞完成，离地高度: {takeoff_height_from_ground:.2f}m")
            logger.info(f"准备飞行...")
            return True
            
        except Exception as e:
            logger.error(f"无人机设置失败: {str(e)}")
            self.connected = False
            return False
    
    def fly_path(self, path_points: List[Dict[str, float]], path_name: str = "路径") -> bool:
        """按路径飞行（使用速度控制）"""
        if not path_points:
            logger.error("路径点为空，无法飞行")
            return False
        
        if not self.connected or not self.client:
            logger.error("未连接到AirSim，无法飞行")
            return False
        
        logger.info(f"开始飞行 {path_name}（使用速度控制），共 {len(path_points)} 个路径点")
        self.actual_path = []
        
        try:
            for i, point in enumerate(path_points):
                x, y, z = point['x'], point['y'], point['z']
                # 坐标系转换：使用地面Z作为参考
                airsim_z = self.ground_z - z
                
                if i % 5 == 0 or i == len(path_points) - 1:
                    logger.info(f"飞向路径点 {i+1}/{len(path_points)}: ({x:.2f}, {y:.2f}, {z:.2f})")
                
                # 使用速度控制飞到目标点
                max_wait_time = 30.0
                wait_start = time.time()
                last_report_time = wait_start
                
                while time.time() - wait_start < max_wait_time:
                    # 获取当前状态
                    current_state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
                    current_pos = current_state.kinematics_estimated.position
                    
                    # 计算距离和方向
                    dx = x - current_pos.x_val
                    dy = y - current_pos.y_val
                    dz = airsim_z - current_pos.z_val
                    distance = math.sqrt(dx**2 + dy**2 + dz**2)
                    
                    # 检查是否到达
                    if distance < self.position_tolerance:
                        self.client.moveByVelocityAsync(0, 0, 0, 0.5, vehicle_name=self.vehicle_name)
                        break
                    
                    # 计算合适的速度
                    speed = self.calculate_appropriate_speed(distance)
                    
                    # 计算速度向量
                    vx = (dx / distance) * speed
                    vy = (dy / distance) * speed
                    vz = (dz / distance) * speed
                    
                    # 发送速度控制指令
                    self.client.moveByVelocityAsync(vx, vy, vz, 0.5, vehicle_name=self.vehicle_name)
                    
                    time.sleep(0.5)
                
                # 记录实际位置
                state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
                position = state.kinematics_estimated.position
                actual_z = -(position.z_val - self.ground_z)
                self.actual_path.append({
                    'x': position.x_val,
                    'y': position.y_val, 
                    'z': actual_z,
                    'time': point.get('time', i * 0.2)
                })
                
                time.sleep(0.5)  # 在点上稳定一下
            
            logger.info(f"✓ {path_name} 飞行完成")
            return True
            
        except Exception as e:
            logger.error(f"飞行 {path_name} 时发生错误: {str(e)}")
            return False
    
    def fly_straight_with_sampling(self, path_points: List[Dict[str, float]], path_name: str = "路径") -> bool:
        """按起点到终点的直线飞行，但按照路径点的时间戳采样记录实际位置"""
        if not path_points or len(path_points) < 2:
            logger.error("路径点数量不足，无法飞行")
            return False
        
        if not self.connected or not self.client:
            logger.error("未连接到AirSim，无法飞行")
            return False
        
        # 获取起点和终点
        start_point = path_points[0]
        end_point = path_points[-1]
        
        start_x, start_y, start_z = start_point['x'], start_point['y'], start_point['z']
        end_x, end_y, end_z = end_point['x'], end_point['y'], end_point['z']
        
        # 转换为AirSim坐标系（z取负）
        # 重要：使用地面Z作为参考，确保高度是相对于地面的
        start_airsim_z = self.ground_z - start_z  # 地面Z - 目标高度 = 目标Z
        end_airsim_z = self.ground_z - end_z
        
        logger.info(f"开始直线飞行: 起点({start_x:.2f}, {start_y:.2f}, {start_z:.2f}) -> 终点({end_x:.2f}, {end_y:.2f}, {end_z:.2f})")
        logger.info(f"采样点数: {len(path_points)}")
        
        self.actual_path = []
        
        try:
            # 移动到起点
            logger.info("=" * 60)
            logger.info("步骤1: 移动到起点")
            
            # 获取当前位置
            current_state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            current_p = current_state.kinematics_estimated.position
            
            # 打印当前位置和目标位置（带详细信息）
            logger.info(f"当前位置(NED): X={current_p.x_val:.4f}, Y={current_p.y_val:.4f}, Z={current_p.z_val:.4f}")
            logger.info(f"目标起点(路径): X={start_x:.4f}, Y={start_y:.4f}, Z(高度)={start_z:.4f}")
            logger.info(f"目标起点(NED): X={start_x:.4f}, Y={start_y:.4f}, Z={start_airsim_z:.4f}")
            logger.info(f"地面Z参考: {self.ground_z:.4f}")
            
            # 计算距离
            distance_to_target = math.sqrt(
                (current_p.x_val - start_x)**2 +
                (current_p.y_val - start_y)**2 +
                (current_p.z_val - start_airsim_z)**2
            )
            
            logger.info(f"到起点3D距离: {distance_to_target:.4f}m")
            
            # 检查无人机状态
            logger.info("检查无人机状态...")
            is_api_enabled = self.client.isApiControlEnabled(self.vehicle_name)
            logger.info(f"  API控制状态: {is_api_enabled}")
            
            # 如果API控制未启用，重新启用
            if not is_api_enabled:
                logger.warning("⚠️ API控制未启用，正在重新启用...")
                self.client.enableApiControl(True, self.vehicle_name)
                time.sleep(0.5)
            
            # 如果距离太近，不需要移动
            if distance_to_target < 0.3:
                logger.info(f"✓ 已在起点附近 (距离={distance_to_target:.3f}m < 0.3m)，无需移动")
            else:
                # 使用速度控制飞行到起点
                logger.info("使用速度控制飞行到起点...")
                self.client.cancelLastTask(self.vehicle_name)
                time.sleep(0.3)
                
                max_wait_time = 60.0
                wait_start = time.time()
                last_report_time = wait_start
                flight_speed = 0.5  # 使用0.5 m/s的速度
                
                while time.time() - wait_start < max_wait_time:
                    # 获取当前状态
                    current_state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
                    current_pos = current_state.kinematics_estimated.position
                    
                    # 计算距离和方向
                    dx = start_x - current_pos.x_val
                    dy = start_y - current_pos.y_val
                    dz = start_airsim_z - current_pos.z_val
                    distance = math.sqrt(dx**2 + dy**2 + dz**2)
                    
                    # 检查是否到达
                    if distance < 0.3:
                        # 停止移动
                        self.client.moveByVelocityAsync(0, 0, 0, 1.0, vehicle_name=self.vehicle_name)
                        logger.info(f"✓ 到达起点，最终距离: {distance:.3f}m")
                        break
                    
                    # 计算速度向量
                    vx = (dx / distance) * flight_speed
                    vy = (dy / distance) * flight_speed
                    vz = (dz / distance) * flight_speed
                    
                    # 发送速度控制指令
                    self.client.moveByVelocityAsync(vx, vy, vz, 0.5, vehicle_name=self.vehicle_name)
                    
                    # 每2秒报告一次进度
                    if time.time() - last_report_time >= 2.0:
                        logger.info(f"  移动中... 距目标: {distance:.2f}m")
                        last_report_time = time.time()
                    
                    time.sleep(0.5)
                
                # 等待稳定
                time.sleep(1)
            
            # 验证起点位置
            state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            current_pos = state.kinematics_estimated.position
            actual_start_z = -(current_pos.z_val - self.ground_z)
            
            dx = current_pos.x_val - start_x
            dy = current_pos.y_val - start_y
            dz = actual_start_z - start_z
            distance_error = math.sqrt(dx**2 + dy**2 + dz**2)
            
            if distance_error <= self.position_tolerance:
                logger.info(f"✓ 到达起点，误差: {distance_error:.3f}m")
            else:
                logger.warning(f"⚠️ 起点偏差: {distance_error:.3f}m (ΔX={dx:.2f}, ΔY={dy:.2f}, ΔZ={dz:.2f})")
            
            logger.info("在起点稳定3秒...")
            time.sleep(3)
            
            # 获取起点和终点的时间
            start_time = path_points[0].get('time', 0)
            end_time = path_points[-1].get('time', len(path_points) * 0.2)
            flight_duration = end_time - start_time
            
            # 计算起点到终点的直线距离
            straight_distance = math.sqrt(
                (end_x - start_x)**2 +
                (end_y - start_y)**2 +
                (end_z - start_z)**2
            )
            
            # 根据距离和时间计算合适的速度
            flight_speed = self.calculate_appropriate_speed(straight_distance, flight_duration)
            theoretical_speed = straight_distance / flight_duration if flight_duration > 0 else 0
            
            logger.info("=" * 60)
            logger.info("步骤2: 使用速度控制直线飞行到终点")
            logger.info(f"终点: ({end_x:.2f}, {end_y:.2f}, {end_z:.2f})")
            logger.info(f"距离: {straight_distance:.2f}m, 时间: {flight_duration:.1f}s, 速度: {flight_speed:.2f}m/s")
            logger.info("=" * 60)
            
            # 取消之前的任务
            self.client.cancelLastTask(self.vehicle_name)
            time.sleep(0.3)
            
            # 记录飞行开始的实际时间
            actual_start_time = time.time()
            last_control_time = actual_start_time
            sample_index = 0
            
            # 持续飞行并采样，直到到达终点或采样完成
            while sample_index < len(path_points):
                current_time = time.time()
                
                # 获取当前状态
                state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
                position = state.kinematics_estimated.position
                
                # 计算到终点的距离和方向
                dx = end_x - position.x_val
                dy = end_y - position.y_val
                dz = end_airsim_z - position.z_val
                distance_to_end = math.sqrt(dx**2 + dy**2 + dz**2)
                
                # 检查是否需要采样
                point_time = path_points[sample_index].get('time', sample_index * 0.2)
                relative_time = point_time - start_time
                elapsed_time = current_time - actual_start_time
                
                if elapsed_time >= relative_time:
                    # 采样时间到了，记录当前位置
                    actual_z = -(position.z_val - self.ground_z)
                    self.actual_path.append({
                        'x': position.x_val,
                        'y': position.y_val,
                        'z': actual_z,
                        'time': point_time
                    })
                    
                    if sample_index % 20 == 0 or sample_index == len(path_points) - 1:
                        logger.info(f"采样 {sample_index+1}/{len(path_points)}: ({position.x_val:.2f}, {position.y_val:.2f}, {actual_z:.2f}), 距终点: {distance_to_end:.2f}m")
                    
                    sample_index += 1
                
                # 如果到达终点，停止移动
                if distance_to_end < 0.3:
                    self.client.moveByVelocityAsync(0, 0, 0, 1.0, vehicle_name=self.vehicle_name)
                    logger.info(f"✓ 到达终点，距离: {distance_to_end:.2f}m")
                    
                    # 继续采样剩余的点（在终点位置）
                    while sample_index < len(path_points):
                        point_time = path_points[sample_index].get('time', sample_index * 0.2)
                        actual_z = -(position.z_val - self.ground_z)
                        self.actual_path.append({
                            'x': position.x_val,
                            'y': position.y_val,
                            'z': actual_z,
                            'time': point_time
                        })
                        sample_index += 1
                    break
                
                # 每0.5秒更新一次速度控制
                if current_time - last_control_time >= 0.5:
                    # 计算速度向量
                    vx = (dx / distance_to_end) * flight_speed
                    vy = (dy / distance_to_end) * flight_speed
                    vz = (dz / distance_to_end) * flight_speed
                    
                    # 发送速度控制指令
                    self.client.moveByVelocityAsync(vx, vy, vz, 0.5, vehicle_name=self.vehicle_name)
                    last_control_time = current_time
                
                time.sleep(0.1)  # 小睡眠，避免CPU占用过高
            
            logger.info(f"✓ {path_name} 飞行完成，记录 {len(self.actual_path)} 个数据点")
            
            return True
            
        except Exception as e:
            logger.error(f"直线飞行 {path_name} 时发生错误: {str(e)}")
            return False
    
    # 注：此方法已弃用，改用速度控制方式，不再需要此等待函数
    # def _wait_for_position_reached(...) - 已删除，使用速度控制循环替代
    
    def land_and_disconnect(self) -> bool:
        """降落并断开连接"""
        try:
            if not self.connected or not self.client:
                logger.warning("未连接到AirSim，无需降落")
                return True
            
            # 降落
            self.client.landAsync(vehicle_name=self.vehicle_name).join()
            time.sleep(2)
            
            # 上锁并禁用API控制
            self.client.armDisarm(False, self.vehicle_name)
            self.client.enableApiControl(False, self.vehicle_name)
            
            self.connected = False
            logger.info("✓ 降落完成")
            return True
            
        except Exception as e:
            logger.error(f"降落操作失败: {str(e)}")
            return False

class PathComparator:
    """路径比较器"""
    
    def __init__(self):
        self.expected_path_data = []  # 预期路径
        self.actual_path_data = []    # 实际飞行路径
    
    def load_expected_path(self, path_file: str) -> bool:
        """加载预期路径文件"""
        try:
            with open(path_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.expected_path_data = data.get("1", [])
            
            logger.info(f"成功加载预期路径: {len(self.expected_path_data)} 个点")
            return True
            
        except Exception as e:
            logger.error(f"加载预期路径文件失败: {str(e)}")
            return False
    
    def set_actual_path(self, actual_path: List[Dict[str, float]]):
        """设置实际飞行路径"""
        self.actual_path_data = actual_path
        logger.info(f"✓ 设置实际路径: {len(self.actual_path_data)} 个点")
    
    def calculate_path_statistics(self, path_data: List[Dict[str, float]], path_name: str) -> Dict[str, float]:
        """计算路径统计信息"""
        if not path_data:
            return {}
        
        # 计算路径长度
        total_distance = 0.0
        for i in range(1, len(path_data)):
            p1 = path_data[i-1]
            p2 = path_data[i]
            distance = math.sqrt(
                (p2['x'] - p1['x'])**2 + 
                (p2['y'] - p1['y'])**2 + 
                (p2['z'] - p1['z'])**2
            )
            total_distance += distance
        
        # 计算高度变化
        heights = [point['z'] for point in path_data]
        min_height = min(heights)
        max_height = max(heights)
        height_range = max_height - min_height
        
        # 计算时间跨度
        times = [point.get('time', 0) for point in path_data]
        duration = max(times) - min(times) if times else 0
        
        stats = {
            'path_name': path_name,
            'point_count': len(path_data),
            'total_distance': total_distance,
            'min_height': min_height,
            'max_height': max_height,
            'height_range': height_range,
            'duration': duration,
            'avg_speed': total_distance / duration if duration > 0 else 0
        }
        
        return stats
    
    def compare_paths(self) -> Dict[str, Any]:
        """比较预期路径和实际飞行路径"""
        if not self.expected_path_data or not self.actual_path_data:
            logger.error("路径数据不完整，无法比较")
            return {}
        
        # 计算统计信息
        expected_stats = self.calculate_path_statistics(self.expected_path_data, "预期路径")
        actual_stats = self.calculate_path_statistics(self.actual_path_data, "实际路径")
        
        # 计算点对点的位置误差
        position_errors = self.calculate_position_errors()
        
        # 计算差异
        comparison = {
            'expected_stats': expected_stats,
            'actual_stats': actual_stats,
            'position_errors': position_errors,
            'differences': {
                'distance_diff': actual_stats['total_distance'] - expected_stats['total_distance'],
                'height_range_diff': actual_stats['height_range'] - expected_stats['height_range'],
                'duration_diff': actual_stats['duration'] - expected_stats['duration'],
                'speed_diff': actual_stats['avg_speed'] - expected_stats['avg_speed']
            }
        }
        
        return comparison
    
    def calculate_position_errors(self) -> Dict[str, Any]:
        """计算预期路径和实际路径的位置误差"""
        if not self.expected_path_data or not self.actual_path_data:
            return {}
        
        errors = []
        min_len = min(len(self.expected_path_data), len(self.actual_path_data))
        
        for i in range(min_len):
            expected = self.expected_path_data[i]
            actual = self.actual_path_data[i]
            
            # 计算3D距离误差
            error = math.sqrt(
                (actual['x'] - expected['x'])**2 +
                (actual['y'] - expected['y'])**2 +
                (actual['z'] - expected['z'])**2
            )
            
            errors.append({
                'point_index': i,
                'expected_position': (expected['x'], expected['y'], expected['z']),
                'actual_position': (actual['x'], actual['y'], actual['z']),
                'error': error,
                'x_error': actual['x'] - expected['x'],
                'y_error': actual['y'] - expected['y'],
                'z_error': actual['z'] - expected['z']
            })
        
        # 计算误差统计
        if errors:
            error_values = [e['error'] for e in errors]
            return {
                'point_errors': errors,
                'max_error': max(error_values),
                'min_error': min(error_values),
                'avg_error': sum(error_values) / len(error_values),
                'total_points_compared': min_len
            }
        
        return {}
    
    def print_path_comparison(self):
        """打印路径比较结果"""
        try:
            # 计算统计信息
            expected_stats = self.calculate_path_statistics(self.expected_path_data, "预期路径")
            actual_stats = self.calculate_path_statistics(self.actual_path_data, "实际路径")
            
            # 计算位置误差
            position_errors = self.calculate_position_errors()
            
            # 创建统计表格
            stats_text = f"""
========================================
   Path1 预期路径 vs 直线飞行 对比分析
========================================

说明: 对比Path1完整路径与从起点到终点的直线飞行

预期路径 (Path1完整路径):
  点数: {expected_stats.get('point_count', 0)}
  总距离: {expected_stats.get('total_distance', 0):.2f} m
  高度范围: {expected_stats.get('height_range', 0):.2f} m ({expected_stats.get('min_height', 0):.2f} ~ {expected_stats.get('max_height', 0):.2f})
  飞行时间: {expected_stats.get('duration', 0):.2f} s
  平均速度: {expected_stats.get('avg_speed', 0):.2f} m/s

实际飞行路径 (起点到终点直线):
  点数: {actual_stats.get('point_count', 0)}
  总距离: {actual_stats.get('total_distance', 0):.2f} m
  高度范围: {actual_stats.get('height_range', 0):.2f} m ({actual_stats.get('min_height', 0):.2f} ~ {actual_stats.get('max_height', 0):.2f})
  飞行时间: {actual_stats.get('duration', 0):.2f} s
  平均速度: {actual_stats.get('avg_speed', 0):.2f} m/s

路径统计差异:
  距离差: {actual_stats.get('total_distance', 0) - expected_stats.get('total_distance', 0):.2f} m
  高度范围差: {actual_stats.get('height_range', 0) - expected_stats.get('height_range', 0):.2f} m
  时间差: {actual_stats.get('duration', 0) - expected_stats.get('duration', 0):.2f} s
  速度差: {actual_stats.get('avg_speed', 0) - expected_stats.get('avg_speed', 0):.2f} m/s

位置误差统计 (各时间点的位置偏差):
  对比点数: {position_errors.get('total_points_compared', 0)}
  最大误差: {position_errors.get('max_error', 0):.4f} m
  最小误差: {position_errors.get('min_error', 0):.4f} m
  平均误差: {position_errors.get('avg_error', 0):.4f} m

========================================
            """
            
            print(stats_text)
            
            # 打印前10个点的详细误差
            if position_errors.get('point_errors'):
                print("\n前10个路径点的位置误差详情:")
                print("-" * 80)
                print(f"{'点序号':<8} {'预期位置 (x, y, z)':<30} {'实际位置 (x, y, z)':<30} {'误差 (m)':<10}")
                print("-" * 80)
                for error_data in position_errors['point_errors'][:10]:
                    idx = error_data['point_index']
                    exp_pos = error_data['expected_position']
                    act_pos = error_data['actual_position']
                    err = error_data['error']
                    print(f"{idx:<8} ({exp_pos[0]:>6.2f}, {exp_pos[1]:>6.2f}, {exp_pos[2]:>6.2f})   "
                          f"({act_pos[0]:>6.2f}, {act_pos[1]:>6.2f}, {act_pos[2]:>6.2f})   {err:>8.4f}")
                print("-" * 80)
            
        except Exception as e:
            logger.error(f"打印路径比较结果失败: {str(e)}")
    
    def save_path_data(self, filename: str = "path_comparison_data.json"):
        """保存路径比较数据到JSON文件"""
        try:
            expected_stats = self.calculate_path_statistics(self.expected_path_data, "预期路径")
            actual_stats = self.calculate_path_statistics(self.actual_path_data, "实际路径")
            position_errors = self.calculate_position_errors()
            
            comparison_data = {
                "expected_path_stats": expected_stats,
                "actual_path_stats": actual_stats,
                "expected_path_data": self.expected_path_data,
                "actual_path_data": self.actual_path_data,
                "position_errors": position_errors,
                "differences": {
                    "distance_diff": actual_stats.get('total_distance', 0) - expected_stats.get('total_distance', 0),
                    "height_range_diff": actual_stats.get('height_range', 0) - expected_stats.get('height_range', 0),
                    "duration_diff": actual_stats.get('duration', 0) - expected_stats.get('duration', 0),
                    "speed_diff": actual_stats.get('avg_speed', 0) - expected_stats.get('avg_speed', 0)
                }
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(comparison_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✓ 数据已保存: {filename}")
            
        except Exception as e:
            logger.error(f"保存路径比较数据失败: {str(e)}")

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("无人机路径飞行和比较程序")
    logger.info("=" * 60)
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path1_file = os.path.join(script_dir, "path1.json")
    
    if not os.path.exists(path1_file):
        logger.error(f"路径文件 {path1_file} 不存在")
        return
    
    # 创建路径比较器和飞行控制器
    comparator = PathComparator()
    if not comparator.load_expected_path(path1_file):
        return
    
    flight_controller = PathFlightController()
    
    try:
        # 连接并设置无人机
        if not flight_controller.connect_and_setup():
            return
        
        # 按照Path1的起点和终点飞行直线
        expected_path_points = comparator.expected_path_data
        if flight_controller.fly_straight_with_sampling(expected_path_points, "Path1"):
            comparator.set_actual_path(flight_controller.actual_path)
        else:
            logger.error("飞行失败")
            return
        
        # 降落
        flight_controller.land_and_disconnect()
        
        # 进行路径比较
        logger.info("=" * 60)
        logger.info("路径对比分析")
        logger.info("=" * 60)
        
        comparison_result = comparator.compare_paths()
        if comparison_result:
            logger.info(f"✓ 平均误差: {comparison_result['position_errors'].get('avg_error', 0):.3f}m, "
                       f"最大误差: {comparison_result['position_errors'].get('max_error', 0):.3f}m")
        
        # 打印比较结果
        comparator.print_path_comparison()
        
        # 保存比较数据
        output_file = os.path.join(script_dir, "path_comparison_data.json")
        comparator.save_path_data(output_file)
        
        logger.info("=" * 60)
        logger.info(f"✓ 程序完成，数据已保存: {output_file}")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("用户中断程序")
        flight_controller.land_and_disconnect()
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        flight_controller.land_and_disconnect()

if __name__ == "__main__":
    main()
