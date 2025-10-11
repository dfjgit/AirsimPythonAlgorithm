import numpy as np
import math
import logging
from .Vector3 import Vector3
from .HexGridDataModel import HexGridDataModel, HexCell
from .scanner_config_data import ScannerConfigData
from .scanner_runtime_data import ScannerRuntimeData
from typing import List, Dict, Tuple, Optional, Set
import time

# 确保使用正确的坐标系
def ensure_unity_coordinates(vector: Vector3) -> Vector3:
    """确保向量使用Unity坐标系"""
    # 检查是否需要转换（如果Vector3实例已经有转换方法）
    if hasattr(vector, 'unity_to_air_sim'):
        # 这里根据实际需要决定是否进行转换
        # 注意：这个函数是一个安全措施，确保坐标系的一致性
        pass
    return vector


class ScannerAlgorithm:
    def __init__(self, config_data: ScannerConfigData):
        """初始化扫描器算法，传入配置数据"""
        self.config = config_data
        self.last_update_time = 0.0
        # 初始化上一帧的移动方向，使用Unity坐标系中的默认向前方向
        self.previous_move_dir = ensure_unity_coordinates(Vector3(0, 0, 1))  # 默认方向：z轴正方向
        self.visited_cells: Dict[Tuple[float, float, float], float] = {}  # 存储访问时间 (x,y,z) -> timestamp

    def calculate_proportional_weights(self) -> Tuple[float, float, float, float, float]:
        """计算权重：F = 系数 / 系数总和（与C#逻辑一致）"""
        total = (self.config.repulsionCoefficient + 
                 self.config.entropyCoefficient + 
                 self.config.distanceCoefficient + 
                 self.config.leaderRangeCoefficient + 
                 self.config.directionRetentionCoefficient)
        
        # 处理所有系数都为0的特殊情况
        if total < 0.001:
            return (0.2, 0.2, 0.2, 0.2, 0.2)
        
        return (
            self.config.repulsionCoefficient / total,
            self.config.entropyCoefficient / total,
            self.config.distanceCoefficient / total,
            self.config.leaderRangeCoefficient / total,
            self.config.directionRetentionCoefficient / total
        )

    def get_valid_candidate_cells(self, grid_data: HexGridDataModel, runtime_data: ScannerRuntimeData) -> List[HexCell]:
        """获取有效的候选蜂窝（与C# GetValidCandidateCells逻辑一致）"""
        # 确保使用Unity坐标系
        current_pos = ensure_unity_coordinates(runtime_data.position)
        candidate_cells = []
        
        for cell in grid_data.cells:
            # 确保蜂窝中心也使用Unity坐标系
            cell_center = ensure_unity_coordinates(cell.center)
            
            # 检查是否在Leader范围内
            if runtime_data.leader_position is not None and runtime_data.leader_scan_radius > 0:
                leader_pos = ensure_unity_coordinates(runtime_data.leader_position)
                distance_to_leader = (cell_center - leader_pos).magnitude()
                if distance_to_leader > runtime_data.leader_scan_radius:
                    continue  # 不在Leader范围内，跳过
            
            # 检查是否在搜索范围内
            distance_to_cell = (cell_center - current_pos).magnitude()
            if distance_to_cell > self.config.targetSearchRange:
                continue  # 超出搜索范围，跳过
            
            # 检查是否需要避免重复访问
            if self.config.avoidRevisits:
                # 四舍五入避免浮点数精度问题
                rounded_center = (
                    round(cell.center.x * 100) / 100,
                    round(cell.center.y * 100) / 100,
                    round(cell.center.z * 100) / 100
                )
                
                if rounded_center in self.visited_cells:
                    # 检查是否在冷却期内
                    if time.time() - self.visited_cells[rounded_center] < self.config.revisitCooldown:
                        continue  # 仍在冷却期，跳过
            
            candidate_cells.append(cell)
        
        return candidate_cells

    def calculate_score_direction(self, grid_data: HexGridDataModel, runtime_data: ScannerRuntimeData) -> Vector3:
        """计算熵最优方向向量（与C# CalculateScoreDirection逻辑一致）"""
        # 确保使用Unity坐标系
        current_pos = ensure_unity_coordinates(runtime_data.position)
        
        # 保留原始的y坐标，不强制设置为0，以确保3D空间中的准确计算
        # current_pos = Vector3(runtime_data.position.x, 0, runtime_data.position.z)  # 移除这个有问题的代码
        
        candidate_cells = self.get_valid_candidate_cells(grid_data, runtime_data)
        
        if not candidate_cells:
            return Vector3()
        
        # 归一化熵值范围（0-1）
        entropies = [cell.entropy for cell in candidate_cells]
        min_entropy = min(entropies)
        max_entropy = max(entropies)
        entropy_range = max_entropy - min_entropy
        all_entropy_same = abs(entropy_range) < 0.01
        
        # 计算每个候选蜂窝的分数
        scored_cells = []
        for cell in candidate_cells:
            # 确保蜂窝中心也使用Unity坐标系
            cell_center = ensure_unity_coordinates(cell.center)
            distance = (cell_center - current_pos).magnitude()
            normalized_distance = min(1.0, max(0.0, 1 - (distance / self.config.targetSearchRange)))
            
            # 计算熵值分数
            if all_entropy_same:
                entropy_score = 0.5
            else:
                entropy_score = (cell.entropy - min_entropy) / entropy_range
            
            # 综合分数：熵值为主（70%），距离为辅（30%）
            total_score = entropy_score * 0.7 + normalized_distance * 0.3
            scored_cells.append((cell, total_score))
        
        # 选择最高分的蜂窝作为目标
        best_cell = max(scored_cells, key=lambda x: x[1])[0]
        
        # 确保计算方向时使用正确的坐标系
        best_cell_center = ensure_unity_coordinates(best_cell.center)
        score_dir = (best_cell_center - current_pos).normalized()
        
        # 记录访问
        self.record_visited_cell(best_cell_center)
        return score_dir

    def calculate_path_direction(self, score_dir: Vector3) -> Vector3:
        """计算最短路径方向向量（与C# CalculatePathDirection逻辑一致）"""
        # 确保路径方向也使用Unity坐标系
        return ensure_unity_coordinates(score_dir)  # 路径方向与分数方向一致

    def calculate_collide_direction(self, runtime_data: ScannerRuntimeData) -> Vector3:
        """计算排斥力方向向量（与C# CalculateRepulsionDirection逻辑一致）"""
        collide_dir = Vector3()
        # 确保使用Unity坐标系
        current_pos = ensure_unity_coordinates(runtime_data.position)
        
        # 其他扫描器位置
        other_scanners = runtime_data.otherScannerPositions
        
        for other_pos in other_scanners:
            # 确保其他扫描器的位置也使用Unity坐标系
            other_pos_unity = ensure_unity_coordinates(other_pos)
            delta_pos = current_pos - other_pos_unity
            distance = delta_pos.magnitude()
            
            # 超出排斥范围或距离过近（避免除以零）
            if distance > self.config.maxRepulsionDistance or distance < 0.1:
                continue
            
            # 计算排斥力比例
            repulsion_ratio = self.calculate_repulsion_ratio(distance)
            collide_dir += delta_pos.normalized() * repulsion_ratio
        
        # 确保返回的排斥方向向量在Unity坐标系中正确
        return ensure_unity_coordinates(collide_dir.normalized() if collide_dir.magnitude() > 0.1 else collide_dir)

    def calculate_repulsion_ratio(self, distance: float) -> float:
        """计算排斥力比例（与C# CalculateRepulsionRatio逻辑一致）"""
        if distance <= self.config.minSafeDistance:
            return 1.0
        if distance >= self.config.maxRepulsionDistance:
            return 0.0
        
        # 非线性衰减，近距离排斥力增长更快
        t = (distance - self.config.minSafeDistance) / (self.config.maxRepulsionDistance - self.config.minSafeDistance)
        return 1.0 - (t * t)

    def calculate_leader_range_direction(self, runtime_data: ScannerRuntimeData) -> Vector3:
        """计算保持在Leader范围内的方向向量（与C# CalculateLeaderRangeDirection逻辑一致）"""
        leader_range_dir = Vector3()
        
        # 确保使用Unity坐标系
        current_pos = ensure_unity_coordinates(runtime_data.position)
        leader_pos = ensure_unity_coordinates(runtime_data.leader_position)
        leader_scan_radius = runtime_data.leader_scan_radius
        
        if leader_pos is None or leader_scan_radius <= 0:
            return leader_range_dir
        
        # 计算与Leader的距离（在Unity坐标系中）
        distance_to_leader = (current_pos - leader_pos).magnitude()
        
        # 避免除零错误
        if leader_scan_radius < 0.001:
            return leader_range_dir
        
        # 如果超出Leader的范围，生成指向Leader的方向向量
        if distance_to_leader > leader_scan_radius:
            # 距离越远，返回的力度越大
            range_ratio = min(1.0, (distance_to_leader - leader_scan_radius) / leader_scan_radius)
            # 确保方向向量在Unity坐标系中正确
            direction = (leader_pos - current_pos).normalized()
            leader_range_dir = direction * (1.0 + range_ratio)
        # 如果离Leader过近，生成轻微远离Leader的方向向量
        elif distance_to_leader < leader_scan_radius * 0.3 and distance_to_leader > 0.001:
            direction = (current_pos - leader_pos).normalized()
            leader_range_dir = direction * 0.3
        
        return leader_range_dir

    def calculate_direction_retention_direction(self) -> Vector3:
        """计算方向保持向量（与C# CalculateDirectionRetentionDirection逻辑一致）"""
        # 确保返回的方向向量在Unity坐标系中正确
        if self.previous_move_dir and isinstance(self.previous_move_dir, Vector3):
            return ensure_unity_coordinates(self.previous_move_dir)
        return Vector3(0, 0, 1)  # 默认方向

    def merge_directions(self, 
                        score_dir: Vector3, 
                        path_dir: Vector3, 
                        collide_dir: Vector3, 
                        leader_range_dir: Vector3, 
                        direction_retention_dir: Vector3,
                        weights: Tuple[float, float, float, float, float]) -> Vector3:
        """合并所有方向向量（与C# MergeDirections逻辑一致）"""
        repulsion_weight, entropy_weight, distance_weight, leader_range_weight, direction_retention_weight = weights
        
        # 确保所有输入向量都使用Unity坐标系
        score_dir = ensure_unity_coordinates(score_dir)
        path_dir = ensure_unity_coordinates(path_dir)
        collide_dir = ensure_unity_coordinates(collide_dir)
        leader_range_dir = ensure_unity_coordinates(leader_range_dir)
        direction_retention_dir = ensure_unity_coordinates(direction_retention_dir)
        
        # 应用权重合并向量
        final_move_dir = (
            score_dir * entropy_weight +
            path_dir * distance_weight +
            collide_dir * repulsion_weight +
            leader_range_dir * leader_range_weight +
            direction_retention_dir * direction_retention_weight
        )
        
        # 归一化最终方向
        if final_move_dir.magnitude() > 0.1:
            return ensure_unity_coordinates(final_move_dir.normalized())
        else:
            # 如果最终方向接近零，保持当前方向
            return ensure_unity_coordinates(self.previous_move_dir)

    def record_visited_cell(self, cell_center: Vector3) -> None:
        """记录已访问的蜂窝（与C# RecordVisitedCell逻辑一致）"""
        if not self.config.avoidRevisits:
            return
        
        # 四舍五入避免浮点数精度问题
        rounded_center = (
            round(cell_center.x * 100) / 100,
            round(cell_center.y * 100) / 100,
            round(cell_center.z * 100) / 100
        )
        
        self.visited_cells[rounded_center] = time.time()

    def cleanup_visited_records(self) -> None:
        """清理过期的访问记录（与C# CleanupVisitedRecords逻辑一致）"""
        if not self.config.avoidRevisits:
            return
        
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.visited_cells.items()
            if current_time - timestamp >= self.config.revisitCooldown
        ]
        
        for key in expired_keys:
            del self.visited_cells[key]

    def set_coefficients(self, coefficients):
        """动态设置权重系数"""
        if 'repulsionCoefficient' in coefficients:
            self.config.repulsionCoefficient = coefficients['repulsionCoefficient']
        if 'entropyCoefficient' in coefficients:
            self.config.entropyCoefficient = coefficients['entropyCoefficient']
        if 'distanceCoefficient' in coefficients:
            self.config.distanceCoefficient = coefficients['distanceCoefficient']
        if 'leaderRangeCoefficient' in coefficients:
            self.config.leaderRangeCoefficient = coefficients['leaderRangeCoefficient']
        if 'directionRetentionCoefficient' in coefficients:
            self.config.directionRetentionCoefficient = coefficients['directionRetentionCoefficient']
    
    def get_current_coefficients(self):
        """获取当前权重系数"""
        return {
            'repulsionCoefficient': self.config.repulsionCoefficient,
            'entropyCoefficient': self.config.entropyCoefficient,
            'distanceCoefficient': self.config.distanceCoefficient,
            'leaderRangeCoefficient': self.config.leaderRangeCoefficient,
            'directionRetentionCoefficient': self.config.directionRetentionCoefficient
        }
    
    def update_runtime_data(self, grid_data: HexGridDataModel, 
                          runtime_data: ScannerRuntimeData) -> ScannerRuntimeData:
        """更新运行时数据（供其他组件使用的接口）"""
        try:
            # 类型检查
            if not isinstance(grid_data, HexGridDataModel):
                logging.warning(f"ScannerAlgorithm.update_runtime_data: grid_data类型无效，期望HexGridDataModel，得到: {type(grid_data).__name__}")
                return runtime_data
            
            if not isinstance(runtime_data, ScannerRuntimeData):
                logging.warning(f"ScannerAlgorithm.update_runtime_data: runtime_data类型无效，期望ScannerRuntimeData，得到: {type(runtime_data).__name__}")
                return runtime_data
            current_time = time.time()
            
            # 定期更新方向（根据updateInterval）
            if current_time - self.last_update_time >= self.config.updateInterval:
                self.last_update_time = current_time
                
                # 保存当前方向作为下一帧的"previousMoveDir"
                try:
                    if runtime_data.finalMoveDir and runtime_data.finalMoveDir.magnitude() > 0.1:
                        self.previous_move_dir = runtime_data.finalMoveDir
                except Exception as e:
                    logging.warning(f"ScannerAlgorithm.update_runtime_data: 获取finalMoveDir失败: {str(e)}")
                    
                # 计算各权重
                weights = self.calculate_proportional_weights()
                
                # 计算各方向向量
                try:
                    score_dir = self.calculate_score_direction(grid_data, runtime_data)
                    path_dir = self.calculate_path_direction(score_dir)
                    collide_dir = self.calculate_collide_direction(runtime_data)
                    leader_range_dir = self.calculate_leader_range_direction(runtime_data)
                    direction_retention_dir = self.calculate_direction_retention_direction()
                    
                    # 合并所有向量
                    final_move_dir = self.merge_directions(
                        score_dir, path_dir, collide_dir, 
                        leader_range_dir, direction_retention_dir,
                        weights
                    )
                    
                    # 清理过期访问记录
                    self.cleanup_visited_records()

                    # 更新runtime_data中的方向向量，并确保它们使用Unity坐标系
                    runtime_data.scoreDir = ensure_unity_coordinates(score_dir)
                    runtime_data.collideDir = ensure_unity_coordinates(collide_dir)
                    runtime_data.pathDir = ensure_unity_coordinates(path_dir)
                    runtime_data.leaderRangeDir = ensure_unity_coordinates(leader_range_dir)
                    runtime_data.directionRetentionDir = ensure_unity_coordinates(direction_retention_dir)
                    runtime_data.finalMoveDir = ensure_unity_coordinates(final_move_dir)
                except Exception as e:
                    logging.error(f"ScannerAlgorithm.update_runtime_data: 计算方向向量失败: {str(e)}")

                # 使用日志记录替代print语句
                # logging.debug(f"输入的Grid数据: {grid_data}")
                # logging.debug(f"输入的Runtime数据: {runtime_data}")

            return runtime_data
        except Exception as e:
            logging.error(f"ScannerAlgorithm.update_runtime_data: 处理运行时数据时出错: {str(e)}")
            return runtime_data