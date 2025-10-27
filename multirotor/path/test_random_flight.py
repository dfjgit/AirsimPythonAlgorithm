#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试AirSim随机位置飞行
生成5个随机目标点，测试无人机能否到达

配置参数说明（在 RandomFlightTester 类中修改）：
- FLIGHT_SPEED: 飞行速度（m/s），默认0.5，越小越慢越稳定
- POSITION_TOLERANCE: 位置容差（m），默认0.3，越小越精确但耗时更长
  * 0.3m - 较精确（推荐）
  * 0.2m - 很精确
  * 0.1m - 极精确（可能需要很长时间）
"""

import os
import sys
import time
import math
import random
import logging

# 添加项目路径
current_dir = os.path.dirname(__file__)
multirotor_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(multirotor_dir)
sys.path.append(project_dir)

import airsim

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_random_flight.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RandomFlightTest")

class RandomFlightTester:
    """随机飞行测试器"""
    
    # 配置参数
    FLIGHT_SPEED = 0.5  # 飞行速度 m/s
    POSITION_TOLERANCE = 0.3  # 位置容差 m（调小可以更精确，但可能需要更长时间）
    
    def __init__(self):
        self.client = None
        self.vehicle_name = "UAV1"
        self.test_results = []
        
    def generate_random_points(self, center_x=0, center_y=0, center_z=-2, radius=10, count=5):
        """
        生成随机测试点
        
        Args:
            center_x, center_y, center_z: 中心点坐标（NED）
            radius: 半径（米）
            count: 生成点数量
        """
        points = []
        logger.info(f"生成 {count} 个随机测试点，半径 {radius}m")
        
        for i in range(count):
            # 在圆形区域内随机生成点
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(2, radius)  # 最小距离2米
            
            x = center_x + r * math.cos(angle)
            y = center_y + r * math.sin(angle)
            z = center_z + random.uniform(-1, 1)  # Z轴稍微变化
            
            points.append({
                'id': i + 1,
                'x': x,
                'y': y,
                'z': z
            })
            
            logger.info(f"  点{i+1}: ({x:.2f}, {y:.2f}, {z:.2f})")
        
        return points
    
    def connect_and_setup(self):
        """连接并设置无人机"""
        try:
            logger.info("=" * 60)
            logger.info("连接到AirSim")
            logger.info("=" * 60)
            
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            logger.info("✓ 连接成功")
            
            # 重置
            self.client.reset()
            time.sleep(1)
            
            # 启用API控制
            self.client.enableApiControl(True, self.vehicle_name)
            logger.info("✓ API控制已启用")
            
            # 解锁
            self.client.armDisarm(True, self.vehicle_name)
            logger.info("✓ 无人机已解锁")
            
            # 起飞
            logger.info("起飞中...")
            self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
            time.sleep(2)
            
            # 获取起飞后位置
            state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            pos = state.kinematics_estimated.position
            logger.info(f"✓ 起飞完成，当前位置: ({pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f})")
            
            return True
            
        except Exception as e:
            logger.error(f"设置失败: {str(e)}")
            return False
    
    def fly_to_point(self, point, timeout=60):
        """
        飞到指定点
        
        Args:
            point: 目标点字典 {'id', 'x', 'y', 'z'}
            timeout: 超时时间（秒）
        
        Returns:
            dict: 测试结果
        """
        point_id = point['id']
        target_x = point['x']
        target_y = point['y']
        target_z = point['z']
        
        logger.info("=" * 60)
        logger.info(f"测试点 {point_id}/5")
        logger.info("=" * 60)
        
        # 获取当前位置
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        start_pos = state.kinematics_estimated.position
        
        logger.info(f"当前位置: ({start_pos.x_val:.2f}, {start_pos.y_val:.2f}, {start_pos.z_val:.2f})")
        logger.info(f"目标位置: ({target_x:.2f}, {target_y:.2f}, {target_z:.2f})")
        
        # 计算初始距离
        initial_distance = math.sqrt(
            (target_x - start_pos.x_val)**2 +
            (target_y - start_pos.y_val)**2 +
            (target_z - start_pos.z_val)**2
        )
        logger.info(f"初始距离: {initial_distance:.2f}m")
        
        # 检查API控制状态
        is_api_enabled = self.client.isApiControlEnabled(self.vehicle_name)
        if not is_api_enabled:
            logger.warning("⚠️ API控制未启用，重新启用...")
            self.client.enableApiControl(True, self.vehicle_name)
            time.sleep(0.5)
        
        # 取消之前的任务
        self.client.cancelLastTask(self.vehicle_name)
        time.sleep(0.3)
        
        # 发送移动指令
        logger.info(f"发送移动指令（速度: {self.FLIGHT_SPEED} m/s，容差: {self.POSITION_TOLERANCE} m）...")
        start_time = time.time()
        
        self.client.moveToPositionAsync(
            target_x, target_y, target_z, self.FLIGHT_SPEED,
            vehicle_name=self.vehicle_name
        )
        
        # 监控移动过程
        logger.info("监控移动过程...")
        min_distance = initial_distance
        last_report_time = start_time
        no_movement_count = 0
        last_pos = start_pos
        used_velocity_control = False
        
        while time.time() - start_time < timeout:
            # 获取当前状态
            state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            current_pos = state.kinematics_estimated.position
            velocity = state.kinematics_estimated.linear_velocity
            
            # 计算距离目标的距离
            distance = math.sqrt(
                (target_x - current_pos.x_val)**2 +
                (target_y - current_pos.y_val)**2 +
                (target_z - current_pos.z_val)**2
            )
            
            # 更新最小距离
            if distance < min_distance:
                min_distance = distance
            
            # 计算速度
            speed = math.sqrt(velocity.x_val**2 + velocity.y_val**2 + velocity.z_val**2)
            
            # 检查位置变化
            position_change = math.sqrt(
                (current_pos.x_val - last_pos.x_val)**2 +
                (current_pos.y_val - last_pos.y_val)**2 +
                (current_pos.z_val - last_pos.z_val)**2
            )
            
            if position_change < 0.01:
                no_movement_count += 1
            else:
                no_movement_count = 0
                last_pos = current_pos
            
            # 如果10次检查都没移动，使用速度控制
            if no_movement_count >= 10 and not used_velocity_control:
                logger.warning("⚠️ moveToPositionAsync可能失效，切换到速度控制...")
                dx = target_x - current_pos.x_val
                dy = target_y - current_pos.y_val
                dz = target_z - current_pos.z_val
                dist = math.sqrt(dx**2 + dy**2 + dz**2)
                
                if dist > 0.1:
                    vx = (dx / dist) * self.FLIGHT_SPEED
                    vy = (dy / dist) * self.FLIGHT_SPEED
                    vz = (dz / dist) * self.FLIGHT_SPEED
                    duration = dist / self.FLIGHT_SPEED
                    
                    logger.info(f"使用速度控制: 方向({vx:.2f}, {vy:.2f}, {vz:.2f}), 持续{duration:.1f}秒")
                    self.client.moveByVelocityAsync(vx, vy, vz, duration, vehicle_name=self.vehicle_name)
                    used_velocity_control = True
                    no_movement_count = 0
            
            # 每2秒报告一次
            if time.time() - last_report_time >= 2.0:
                logger.info(f"  进度: 距离={distance:.2f}m, 速度={speed:.2f}m/s, "
                           f"位置=({current_pos.x_val:.2f}, {current_pos.y_val:.2f}, {current_pos.z_val:.2f})")
                last_report_time = time.time()
            
            # 检查是否到达
            if distance < self.POSITION_TOLERANCE:
                # 计算向量差
                dx = current_pos.x_val - target_x
                dy = current_pos.y_val - target_y
                dz = current_pos.z_val - target_z
                
                logger.info(f"✓ 到达目标！最终距离: {distance:.2f}m")
                logger.info(f"  向量差: ΔX={dx:.3f}m, ΔY={dy:.3f}m, ΔZ={dz:.3f}m")
                elapsed_time = time.time() - start_time
                
                result = {
                    'point_id': point_id,
                    'target': (target_x, target_y, target_z),
                    'final_position': (current_pos.x_val, current_pos.y_val, current_pos.z_val),
                    'reached': True,
                    'final_distance': distance,
                    'min_distance': min_distance,
                    'vector_error': (dx, dy, dz),
                    'time_taken': elapsed_time,
                    'used_velocity_control': used_velocity_control
                }
                
                # 稳定2秒
                time.sleep(2)
                return result
            
            time.sleep(0.2)
        
        # 超时 - 获取最终位置
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        final_pos = state.kinematics_estimated.position
        
        # 计算向量差
        dx = final_pos.x_val - target_x
        dy = final_pos.y_val - target_y
        dz = final_pos.z_val - target_z
        
        final_distance = math.sqrt(dx**2 + dy**2 + dz**2)
        
        logger.warning(f"⚠️ 超时！最小接近距离: {min_distance:.2f}m")
        logger.warning(f"  最终位置: ({final_pos.x_val:.2f}, {final_pos.y_val:.2f}, {final_pos.z_val:.2f})")
        logger.warning(f"  向量差: ΔX={dx:.3f}m, ΔY={dy:.3f}m, ΔZ={dz:.3f}m")
        elapsed_time = time.time() - start_time
        
        result = {
            'point_id': point_id,
            'target': (target_x, target_y, target_z),
            'final_position': (final_pos.x_val, final_pos.y_val, final_pos.z_val),
            'reached': False,
            'final_distance': final_distance,
            'min_distance': min_distance,
            'vector_error': (dx, dy, dz),
            'time_taken': elapsed_time,
            'used_velocity_control': used_velocity_control
        }
        
        return result
    
    def run_test(self, test_points):
        """运行测试"""
        logger.info("=" * 60)
        logger.info("开始随机位置飞行测试")
        logger.info("=" * 60)
        logger.info(f"配置: 飞行速度={self.FLIGHT_SPEED} m/s, 位置容差={self.POSITION_TOLERANCE} m")
        logger.info("=" * 60)
        
        for point in test_points:
            result = self.fly_to_point(point)
            self.test_results.append(result)
        
        # 打印测试总结
        self.print_summary()
    
    def print_summary(self):
        """打印测试总结"""
        logger.info("")
        logger.info("=" * 60)
        logger.info("测试总结")
        logger.info("=" * 60)
        
        success_count = sum(1 for r in self.test_results if r['reached'])
        total_count = len(self.test_results)
        
        logger.info(f"总测试点数: {total_count}")
        logger.info(f"成功到达: {success_count}")
        logger.info(f"失败: {total_count - success_count}")
        logger.info(f"成功率: {success_count/total_count*100:.1f}%")
        logger.info("")
        
        logger.info("详细结果:")
        logger.info("-" * 60)
        for result in self.test_results:
            status = "✓ 成功" if result['reached'] else "✗ 失败"
            velocity_used = "是" if result['used_velocity_control'] else "否"
            vec_err = result['vector_error']
            
            logger.info(f"点{result['point_id']}: {status}")
            logger.info(f"  目标位置: ({result['target'][0]:.2f}, {result['target'][1]:.2f}, {result['target'][2]:.2f})")
            logger.info(f"  实际位置: ({result['final_position'][0]:.2f}, {result['final_position'][1]:.2f}, {result['final_position'][2]:.2f})")
            logger.info(f"  向量偏差: ΔX={vec_err[0]:+.3f}m, ΔY={vec_err[1]:+.3f}m, ΔZ={vec_err[2]:+.3f}m")
            logger.info(f"  最终距离: {result['final_distance']:.3f}m")
            logger.info(f"  最小距离: {result['min_distance']:.3f}m")
            logger.info(f"  耗时: {result['time_taken']:.1f}秒")
            logger.info(f"  使用速度控制: {velocity_used}")
            logger.info("")
        
        logger.info("=" * 60)
    
    def land_and_disconnect(self):
        """降落并断开"""
        try:
            logger.info("降落中...")
            self.client.landAsync(vehicle_name=self.vehicle_name).join()
            time.sleep(2)
            
            self.client.armDisarm(False, self.vehicle_name)
            self.client.enableApiControl(False, self.vehicle_name)
            
            logger.info("✓ 测试完成")
            
        except Exception as e:
            logger.error(f"降落失败: {str(e)}")

def main():
    """主函数"""
    tester = RandomFlightTester()
    
    try:
        # 连接并设置
        if not tester.connect_and_setup():
            return
        
        # 获取起飞后位置作为中心点
        state = tester.client.getMultirotorState(vehicle_name=tester.vehicle_name)
        center_pos = state.kinematics_estimated.position
        
        # 生成随机测试点
        test_points = tester.generate_random_points(
            center_x=center_pos.x_val,
            center_y=center_pos.y_val,
            center_z=center_pos.z_val,
            radius=10,
            count=5
        )
        
        # 运行测试
        tester.run_test(test_points)
        
        # 降落
        tester.land_and_disconnect()
        
    except KeyboardInterrupt:
        logger.info("用户中断测试")
        tester.land_and_disconnect()
    except Exception as e:
        logger.error(f"测试出错: {str(e)}")
        import traceback
        traceback.print_exc()
        tester.land_and_disconnect()

if __name__ == "__main__":
    main()

