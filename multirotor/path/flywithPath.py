#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无人机路径飞行和比较脚本
使用AirSim单无人机从Path1的起点到终点飞行直线，
并按照Path1的时间戳采样记录实际位置，对比预期路径与实际飞行路径
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
                logger.info(f"成功加载路径文件 {path_file}，包含 {len(path_points)} 个路径点")
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
            logger.info("成功连接到AirSim模拟器")
            
            self.client.reset()
            # 启用API控制
            self.client.enableApiControl(True, self.vehicle_name)
            logger.info(f"无人机{self.vehicle_name}API控制已启用")
            
            # 解锁无人机
            self.client.armDisarm(True, self.vehicle_name)
            logger.info(f"无人机{self.vehicle_name}已解锁")
            
            # 起飞前记录位置（这是地面的Z坐标）
            state_before_takeoff = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            pos_before = state_before_takeoff.kinematics_estimated.position
            self.ground_z = pos_before.z_val  # 记录地面Z坐标
            logger.info(f"起飞前位置(NED): X={pos_before.x_val:.4f}, Y={pos_before.y_val:.4f}, Z={pos_before.z_val:.4f}")
            logger.info(f"🔵 地面Z坐标: {self.ground_z:.4f}m (这是地面的参考点)")
            
            # 起飞
            self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
            logger.info(f"无人机{self.vehicle_name}起飞完成")
            
            # 等待起飞稳定
            time.sleep(2)
            
            # 起飞后记录位置
            state_after_takeoff = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            pos_after = state_after_takeoff.kinematics_estimated.position
            self.takeoff_z = pos_after.z_val  # 记录起飞后Z坐标
            takeoff_height_from_ground = -(pos_after.z_val - self.ground_z)  # 相对地面的高度
            logger.info(f"起飞后位置(NED): X={pos_after.x_val:.4f}, Y={pos_after.y_val:.4f}, Z={pos_after.z_val:.4f}")
            logger.info(f"🔵 起飞后离地高度: {takeoff_height_from_ground:.4f}m")
            logger.info(f"⚠️ 重要：后续高度将相对于地面Z={self.ground_z:.4f}计算")
            
            logger.info("无人机设置完成，准备飞行")
            return True
            
        except Exception as e:
            logger.error(f"无人机设置失败: {str(e)}")
            self.connected = False
            return False
    
    def fly_path(self, path_points: List[Dict[str, float]], path_name: str = "路径") -> bool:
        """按路径飞行"""
        if not path_points:
            logger.error("路径点为空，无法飞行")
            return False
        
        if not self.connected or not self.client:
            logger.error("未连接到AirSim，无法飞行")
            return False
        
        logger.info(f"开始飞行 {path_name}，共 {len(path_points)} 个路径点")
        self.actual_path = []
        
        try:
            for i, point in enumerate(path_points):
                x, y, z = point['x'], point['y'], point['z']
                # 坐标系转换：使用地面Z作为参考
                # 目标Z = 地面Z - 目标高度
                airsim_z = self.ground_z - z
                
                # 获取当前位置
                current_state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
                current_pos = current_state.kinematics_estimated.position
                
                # 计算到目标点的距离
                distance = math.sqrt(
                    (x - current_pos.x_val)**2 +
                    (y - current_pos.y_val)**2 +
                    (airsim_z - current_pos.z_val)**2
                )
                
                # 计算合适的速度
                speed = self.calculate_appropriate_speed(distance)
                
                logger.info(f"飞行到路径点 {i+1}/{len(path_points)}: ({x:.3f}, {y:.3f}, {z:.3f}) -> AirSim坐标({x:.3f}, {y:.3f}, {airsim_z:.3f})")
                logger.info(f"  距离: {distance:.3f}m, 速度: {speed:.2f} m/s")
                
                # 移动到指定位置
                self.client.moveToPositionAsync(
                    x, y, airsim_z, speed, vehicle_name=self.vehicle_name
                ).join()
                
                # 等待到达目标点并稳定
                self._wait_for_position_reached(x, y, airsim_z)
                
                # 记录实际位置（转换为相对地面的高度）
                state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
                position = state.kinematics_estimated.position
                # 转换为相对地面的高度
                actual_z = -(position.z_val - self.ground_z)
                self.actual_path.append({
                    'x': position.x_val,
                    'y': position.y_val, 
                    'z': actual_z,  # 相对地面的高度
                    'time': point.get('time', i * 0.2)
                })
                
                logger.info(f"实际到达位置: ({position.x_val:.3f}, {position.y_val:.3f}, {actual_z:.3f})")
                
                # 短暂等待
                time.sleep(0.1)
            
            logger.info(f"{path_name} 飞行完成")
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
        
        logger.info(f"🔵 坐标转换:")
        logger.info(f"   地面Z参考: {self.ground_z:.4f}m")
        logger.info(f"   起点高度: {start_z:.4f}m → AirSim Z: {start_airsim_z:.4f}m")
        logger.info(f"   终点高度: {end_z:.4f}m → AirSim Z: {end_airsim_z:.4f}m")
        
        logger.info(f"开始直线飞行 {path_name}")
        logger.info(f"起点: ({start_x:.3f}, {start_y:.3f}, {start_z:.3f})")
        logger.info(f"终点: ({end_x:.3f}, {end_y:.3f}, {end_z:.3f})")
        logger.info(f"将按照 {len(path_points)} 个时间戳采样记录实际位置")
        
        self.actual_path = []
        
        try:
            # 移动到起点并校准位置
            logger.info("=" * 60)
            logger.info("第一步：移动到起点并校准位置")
            logger.info(f"目标起点 - X:{start_x:.4f}, Y:{start_y:.4f}, Z(高度):{start_z:.4f}")
            logger.info(f"AirSim坐标 - X:{start_x:.4f}, Y:{start_y:.4f}, Z(NED):{start_airsim_z:.4f}")
            logger.info(f"位置容差: {self.position_tolerance} m")
            logger.info("=" * 60)
 
            # 移动到起点
            # 记录当前位置
            current_state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            current_p = current_state.kinematics_estimated.position
            logger.info(f"当前位置(NED): X={current_p.x_val:.4f}, Y={current_p.y_val:.4f}, Z={current_p.z_val:.4f}")
            
            # 计算当前位置到目标起点的距离
            distance_to_target = math.sqrt(
                (current_p.x_val - start_x)**2 +
                (current_p.y_val - start_y)**2 +
                (current_p.z_val - start_airsim_z)**2
            )
            
            # 根据距离计算合适的速度
            appropriate_speed = self.calculate_appropriate_speed(distance_to_target)
            logger.info(f"到起点距离: {distance_to_target:.4f}m，使用速度: {appropriate_speed:.2f} m/s")
            
            
            self.client.moveToPositionAsync(
               start_x, start_y, -0.48, 0.5, vehicle_name=self.vehicle_name,lookahead=0.3
            )
            time.sleep(10)
            # 发送移动到起点的指令
            logger.info(f"发送移动指令: moveToPositionAsync(x={start_x:.4f}, y={start_y:.4f}, z={start_airsim_z:.4f}, speed={appropriate_speed:.2f})")
            move_task = self.client.moveToPositionAsync(
                start_x, start_y, -0.48, 0.5, vehicle_name=self.vehicle_name,lookahead=0.3
            )
            
            # 等待移动任务完成
            logger.info("等待移动任务完成...")
            move_task.join()
            logger.info("✓ 移动任务已完成")
            time.sleep(10)
            # 记录任务完成后的位置和速度
            after_move_state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            after_move_p = after_move_state.kinematics_estimated.position
            after_move_v = after_move_state.kinematics_estimated.linear_velocity
            after_move_speed = math.sqrt(after_move_v.x_val**2 + after_move_v.y_val**2 + after_move_v.z_val**2)
            logger.info(f"移动完成后位置(NED): X={after_move_p.x_val:.4f}, Y={after_move_p.y_val:.4f}, Z={after_move_p.z_val:.4f}")
            logger.info(f"移动完成后速度: {after_move_speed:.4f} m/s")
            
            # 验证最终位置
            logger.info("=" * 60)
            state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            current_pos = state.kinematics_estimated.position
            # 转换为相对地面的高度
            actual_start_z = -(current_pos.z_val - self.ground_z)
            
            # 计算位置误差
            dx = current_pos.x_val - start_x
            dy = current_pos.y_val - start_y
            dz = actual_start_z - start_z
            distance_error = math.sqrt(dx**2 + dy**2 + dz**2)
            
            logger.info("起点位置验证:")
            logger.info(f"  目标位置(路径坐标): X={start_x:.4f}, Y={start_y:.4f}, Z={start_z:.4f}")
            logger.info(f"  AirSim返回位置(NED): X={current_pos.x_val:.4f}, Y={current_pos.y_val:.4f}, Z={current_pos.z_val:.4f}")
            logger.info(f"  实际位置(转换后): X={current_pos.x_val:.4f}, Y={current_pos.y_val:.4f}, Z={actual_start_z:.4f}")
            logger.info(f"  位置偏差: ΔX={dx:.4f}, ΔY={dy:.4f}, ΔZ={dz:.4f}")
            logger.info(f"  3D距离误差: {distance_error:.4f} m")
            
            # 检查位置误差（仅作为参考，不影响继续执行）
            if distance_error <= self.position_tolerance:
                logger.info(f"✓ 起点到位精确！误差 {distance_error:.4f}m ≤ 容差 {self.position_tolerance}m")
            else:
                logger.warning(f"⚠️ 起点有偏差！误差 {distance_error:.4f}m > 容差 {self.position_tolerance}m")
                
                # 特别检查高度偏差
                if abs(dz) > 0.3:
                    logger.warning(f"⚠️ 特别注意：高度偏差很大 (ΔZ={dz:.4f}m)！")
                    logger.warning(f"   目标高度(路径坐标): {start_z:.4f}m")
                    logger.warning(f"   目标Z(AirSim NED): {start_airsim_z:.4f}m")
                    logger.warning(f"   实际Z(AirSim NED): {current_pos.z_val:.4f}m")
                    logger.warning(f"   实际高度(转换后): {actual_start_z:.4f}m")
            
            logger.info("=" * 60)
            
            # 在起点停稳3秒
            logger.info("\n在起点停稳3秒...")
            time.sleep(3)
            logger.info("停稳完成，准备开始飞行")
            
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
            logger.info("第二步：开始直线飞行到终点")
            logger.info(f"起点到终点直线距离: {straight_distance:.4f}m")
            logger.info(f"预计飞行时间: {flight_duration:.2f}秒")
            logger.info(f"理论所需速度: {theoretical_speed:.2f} m/s")
            logger.info(f"实际使用速度: {flight_speed:.2f} m/s (限制在 {self.min_speed}-{self.max_speed} m/s)")
            logger.info(f"终点位置: X={end_x:.4f}, Y={end_y:.4f}, Z(高度)={end_z:.4f}")
            logger.info(f"发送移动指令: moveToPositionAsync(x={end_x:.4f}, y={end_y:.4f}, z={end_airsim_z:.4f}, speed={flight_speed:.2f})")
            logger.info("=" * 60)
            
            # 开始异步飞行到终点
            flight_task = self.client.moveToPositionAsync(
                end_x, end_y, end_airsim_z, flight_speed, vehicle_name=self.vehicle_name
            )
            
            # 记录飞行开始的实际时间
            actual_start_time = time.time()
            
            # 按照path_points的时间戳采样
            for i, point in enumerate(path_points):
                point_time = point.get('time', i * 0.2)
                relative_time = point_time - start_time
                
                # 等待到达采样时间点
                elapsed_time = time.time() - actual_start_time
                wait_time = relative_time - elapsed_time
                
                if wait_time > 0:
                    time.sleep(wait_time)
                
                # 记录当前实际位置
                state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
                position = state.kinematics_estimated.position
                # 转换为相对地面的高度
                actual_z = -(position.z_val - self.ground_z)  # 实际高度 = -(当前Z - 地面Z)
                
                self.actual_path.append({
                    'x': position.x_val,
                    'y': position.y_val,
                    'z': actual_z,  # 相对地面的高度
                    'time': point_time
                })
                
                if i % 10 == 0 or i == len(path_points) - 1:
                    logger.info(f"采样点 {i+1}/{len(path_points)}: "
                              f"时间={point_time:.3f}s, "
                              f"AirSim位置(NED)=({position.x_val:.3f}, {position.y_val:.3f}, {position.z_val:.3f}), "
                              f"转换后位置=({position.x_val:.3f}, {position.y_val:.3f}, {actual_z:.3f})")
            
            # 等待飞行任务完成（设置超时）
            logger.info("等待飞行到终点任务完成...")
            logger.info(f"(采样已完成，共记录 {len(self.actual_path)} 个位置数据点)")
            
            try:
                # 使用超时等待，避免无限期卡住
                # 计算预期剩余时间：如果还没到终点，给足够的时间
                max_wait_time = 30.0  # 最多等待30秒
                logger.info(f"最多等待 {max_wait_time} 秒...")
                
                wait_start = time.time()
                while time.time() - wait_start < max_wait_time:
                    # 检查是否接近终点
                    state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
                    pos = state.kinematics_estimated.position
                    distance_to_end = math.sqrt(
                        (pos.x_val - end_x)**2 +
                        (pos.y_val - end_y)**2 +
                        (pos.z_val - end_airsim_z)**2
                    )
                    
                    if distance_to_end < 0.5:  # 距离终点小于0.5米
                        logger.info(f"✓ 已接近终点，距离: {distance_to_end:.3f}m")
                        break
                    
                    # 每秒输出一次进度
                    if int(time.time() - wait_start) % 2 == 0:
                        logger.info(f"等待中...距离终点: {distance_to_end:.3f}m")
                    
                    time.sleep(0.5)
                
                # 尝试 join，但不要永久等待
                logger.info("正在完成飞行任务...")
                # 由于可能还在移动，这里直接继续，不强制等待join完成
                
            except Exception as e:
                logger.warning(f"等待飞行任务时出错: {str(e)}")
            
            logger.info(f"{path_name} 直线飞行和采样完成")
            logger.info(f"实际记录了 {len(self.actual_path)} 个位置数据点")
            
            return True
            
        except Exception as e:
            logger.error(f"直线飞行 {path_name} 时发生错误: {str(e)}")
            return False
    
    def _wait_for_position_reached(self, target_x: float, target_y: float, target_z: float, timeout: float = 10.0) -> bool:
        """等待无人机到达目标位置"""
        start_time = time.time()
        tolerance = self.position_tolerance
        stable_count = 0  # 稳定计数器
        required_stable_count = 5  # 需要连续5次检查都稳定
        
        logger.info(f"等待到达目标位置(AirSim NED坐标): X={target_x:.4f}, Y={target_y:.4f}, Z={target_z:.4f}, 容差={tolerance}m")
        
        while time.time() - start_time < timeout:
            try:
                state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
                position = state.kinematics_estimated.position
                velocity = state.kinematics_estimated.linear_velocity
                
                # 计算距离
                distance = math.sqrt(
                    (position.x_val - target_x)**2 + 
                    (position.y_val - target_y)**2 + 
                    (position.z_val - target_z)**2
                )
                
                # 计算速度
                speed = math.sqrt(
                    velocity.x_val**2 + velocity.y_val**2 + velocity.z_val**2
                )
                
                # 检查是否到达并稳定
                if distance <= tolerance and speed < 0.2:
                    stable_count += 1
                    logger.debug(f"稳定检查 {stable_count}/{required_stable_count}: 距离={distance:.4f}m, 速度={speed:.4f}m/s")
                    
                    if stable_count >= required_stable_count:
                        logger.info(f"✓ 已到达目标位置并稳定: 距离={distance:.4f}m, 速度={speed:.4f}m/s")
                        return True
                else:
                    stable_count = 0  # 重置计数器
                    if (time.time() - start_time) % 2 < 0.1:  # 每2秒输出一次进度
                        logger.debug(f"移动中... 距离={distance:.4f}m, 速度={speed:.4f}m/s")
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"检查位置时出错: {str(e)}")
                time.sleep(0.1)
        
        # 超时，记录最终位置
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        position = state.kinematics_estimated.position
        final_distance = math.sqrt(
            (position.x_val - target_x)**2 + 
            (position.y_val - target_y)**2 + 
            (position.z_val - target_z)**2
        )
        logger.warning(f"⚠️ 等待超时！")
        logger.warning(f"   目标位置(NED): X={target_x:.4f}, Y={target_y:.4f}, Z={target_z:.4f}")
        logger.warning(f"   当前位置(NED): X={position.x_val:.4f}, Y={position.y_val:.4f}, Z={position.z_val:.4f}")
        logger.warning(f"   距离目标: {final_distance:.4f}m")
        return False
    
    def land_and_disconnect(self) -> bool:
        """降落并断开连接"""
        try:
            if not self.connected or not self.client:
                logger.warning("未连接到AirSim，无需降落")
                return True
            
            # 降落
            self.client.landAsync(vehicle_name=self.vehicle_name).join()
            logger.info(f"无人机{self.vehicle_name}降落完成")
            
            # 等待降落稳定
            time.sleep(2)
            
            # 上锁
            self.client.armDisarm(False, self.vehicle_name)
            logger.info(f"无人机{self.vehicle_name}已上锁")
            
            # 禁用API控制
            self.client.enableApiControl(False, self.vehicle_name)
            logger.info(f"无人机{self.vehicle_name}API控制已禁用")
            
            self.connected = False
            logger.info("无人机操作完成")
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
        logger.info(f"设置实际飞行路径: {len(self.actual_path_data)} 个点")
    
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
            logger.info("路径比较结果已打印")
            
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
            
            logger.info(f"路径比较数据已保存到: {filename}")
            
        except Exception as e:
            logger.error(f"保存路径比较数据失败: {str(e)}")

def main():
    """主函数"""
    logger.info("开始无人机路径飞行和比较程序")
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 文件路径
    path1_file = os.path.join(script_dir, "path1.json")
    
    # 检查文件是否存在
    if not os.path.exists(path1_file):
        logger.error(f"路径文件 {path1_file} 不存在")
        return
    
    # 创建路径比较器
    comparator = PathComparator()
    if not comparator.load_expected_path(path1_file):
        logger.error("加载预期路径文件失败")
        return
    
    # 创建飞行控制器
    flight_controller = PathFlightController()
    
    try:
        # 连接并设置无人机
        if not flight_controller.connect_and_setup():
            logger.error("无人机设置失败")
            return
        
        # 按照Path1的起点和终点飞行直线
        logger.info("=" * 50)
        logger.info("开始按照 Path1 的起点和终点飞行直线")
        logger.info("=" * 50)
        expected_path_points = comparator.expected_path_data
        if flight_controller.fly_straight_with_sampling(expected_path_points, "Path1"):
            # 保存实际飞行路径
            comparator.set_actual_path(flight_controller.actual_path)
            logger.info("Path1 直线飞行完成")
        else:
            logger.error("Path1 直线飞行失败")
            return
        
        # 降落
        flight_controller.land_and_disconnect()
        
        # 进行路径比较
        logger.info("=" * 50)
        logger.info("开始对比 Path1 预期路径和直线飞行实际路径")
        logger.info("=" * 50)
        
        comparison_result = comparator.compare_paths()
        if comparison_result:
            logger.info("路径比较完成")
            logger.info(f"预期路径 (Path1完整路径) 统计: {comparison_result['expected_stats']}")
            logger.info(f"实际路径 (起点到终点直线) 统计: {comparison_result['actual_stats']}")
            logger.info(f"位置误差统计: 平均={comparison_result['position_errors'].get('avg_error', 0):.4f}m, "
                       f"最大={comparison_result['position_errors'].get('max_error', 0):.4f}m")
            logger.info(f"差异分析: {comparison_result['differences']}")
        
        # 打印比较结果
        comparator.print_path_comparison()
        
        # 保存比较数据
        output_file = os.path.join(script_dir, "path_comparison_data.json")
        comparator.save_path_data(output_file)
        
        logger.info("=" * 50)
        logger.info("程序执行完成")
        logger.info(f"对比数据已保存到: {output_file}")
        logger.info(f"说明: 对比了Path1预期路径与从起点到终点的直线飞行实际路径")
        logger.info("=" * 50)
        
    except KeyboardInterrupt:
        logger.info("用户中断程序")
        flight_controller.land_and_disconnect()
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        flight_controller.land_and_disconnect()

if __name__ == "__main__":
    main()
