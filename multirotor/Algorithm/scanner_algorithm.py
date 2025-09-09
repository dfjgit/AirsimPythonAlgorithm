import numpy as np
import math
import numpy as np
from Vector3 import Vector3
from HexGridDataModel import HexGridDataModel, HexCell
from scanner_config_data import ScannerConfigData
from scanner_runtime_data import ScannerRuntimeData
from typing import List, Dict, Tuple, Optional, Set


class ScannerAlgorithm:
    def __init__(self, config_data: ScannerConfigData):
        """初始化扫描器算法，传入配置数据
        
        Args:
            config_data: 包含算法参数的配置数据对象
        """
        # 导入配置参数并映射到算法所需的属性
        self.weight_coef = config_data.entropyCoefficient  # 权重系数映射到熵系数
        self.entropy_coef = config_data.entropyCoefficient  # 熵系数
        self.neighbor_coef = config_data.leaderRangeCoefficient  # 邻居系数映射到领导者范围系数
        self.angle_coef = config_data.directionRetentionCoefficient  # 角度系数映射到方向保持系数
        self.visited_coef = 1.0  # 默认值，配置中没有直接对应项
        self.distance_coef = config_data.distanceCoefficient  # 距离系数
        self.current_pos_coef = 0.5  # 默认值，配置中没有直接对应项
        self.max_velocity = config_data.moveSpeed  # 最大速度映射到移动速度
        self.max_angle_velocity = math.radians(config_data.rotationSpeed)  # 最大角速度（转换为弧度）
        self.rotation_radius = 1.0  # 默认值，配置中没有直接对应项
        self.max_move_distance = config_data.moveSpeed  # 最大移动距离映射到移动速度
        self.min_move_distance = 0.1  # 默认值，配置中没有直接对应项
        
        # 已访问记录清理参数
        self.revisit_cooldown = config_data.revisitCooldown
        self.avoid_revisits = config_data.avoidRevisits
    
    def calculate_weights(self, grid_data: HexGridDataModel, runtime_data: ScannerRuntimeData) -> Dict[Tuple[float, float], float]:
        """计算各个蜂窝单元的权重
        
        Args:
            grid_data: 网格数据对象
            runtime_data: 运行时数据对象
        
        Returns:
            以(x,z)坐标为键，权重值为值的字典
        """
        weights = {}
        current_heading = runtime_data.direction.normalized() if runtime_data.direction.magnitude() > 1e-4 else Vector3(1, 0, 0)
        current_position = runtime_data.position
        visited_cells = runtime_data.visited_cells
        
        for cell in grid_data.cells:
            cell_center = cell.center
            cell_entropy = cell.entropy
            
            # 检查是否已访问
            cell_key = (round(cell_center.x, 2), round(cell_center.z, 2))
            is_visited = cell_key in visited_cells
            
            # 计算从当前位置到单元中心的向量
            pos_to_cell = Vector3(
                cell_center.x - current_position.x,
                0.0,  # 假设为2D平面
                cell_center.z - current_position.z
            )
            distance = pos_to_cell.magnitude()
            
            # 归一化位置到单元中心的向量
            if distance > 0.001:
                pos_to_cell_normalized = pos_to_cell.normalized()
            else:
                pos_to_cell_normalized = Vector3()
            
            # 计算与当前航向的夹角
            angle_diff = math.acos(max(-1, min(1, current_heading.dot(pos_to_cell_normalized)))) if distance > 0.001 else 0
            
            # 计算邻居单元的熵值总和
            neighbor_entropy_sum = self._calculate_neighbor_entropy(grid_data, cell, visited_cells)
            
            # 计算权重
            weight = (
                self.weight_coef * cell_entropy +
                self.neighbor_coef * neighbor_entropy_sum -
                self.angle_coef * angle_diff -
                self.visited_coef * is_visited -
                self.distance_coef * distance -
                self.current_pos_coef * distance
            )
            
            weights[cell_key] = weight
        
        return weights
    
    def _calculate_neighbor_entropy(self, grid_data: HexGridDataModel, cell: HexCell, visited_cells: Set[Tuple[float, float]]) -> float:
        """计算邻居单元的熵值总和
        
        Args:
            grid_data: 网格数据对象
            cell: 当前蜂窝单元
            visited_cells: 已访问单元集合
        
        Returns:
            邻居单元熵值总和
        """
        neighbor_entropy_sum = 0.0
        cell_center = cell.center
        
        # 六边形网格的6个方向
        directions = [
            Vector3(1, 0, 0), Vector3(0.5, 0, np.sqrt(3)/2),
            Vector3(-0.5, 0, np.sqrt(3)/2), Vector3(-1, 0, 0),
            Vector3(-0.5, 0, -np.sqrt(3)/2), Vector3(0.5, 0, -np.sqrt(3)/2)
        ]
        
        # 假设网格间距为1，实际应用中需要根据实际网格调整
        grid_spacing = 1.0
        
        for direction in directions:
            neighbor_pos = Vector3(
                cell_center.x + direction.x * grid_spacing,
                0.0,
                cell_center.z + direction.z * grid_spacing
            )
            
            # 查找邻居单元
            neighbor_cell = self._find_cell_at_position(grid_data, neighbor_pos)
            if neighbor_cell:
                neighbor_key = (round(neighbor_cell.center.x, 2), round(neighbor_cell.center.z, 2))
                if neighbor_key not in visited_cells:
                    neighbor_entropy_sum += neighbor_cell.entropy
        
        return neighbor_entropy_sum
    
    def _find_cell_at_position(self, grid_data: HexGridDataModel, position: Vector3) -> Optional[HexCell]:
        """在网格中查找指定位置的单元
        
        Args:
            grid_data: 网格数据对象
            position: 目标位置
        
        Returns:
            找到的蜂窝单元，未找到则返回None
        """
        # 使用简单的距离检查，实际应用中可能需要更高效的查找算法
        for cell in grid_data.cells:
            dx = cell.center.x - position.x
            dz = cell.center.z - position.z
            if dx*dx + dz*dz < 0.5:  # 阈值可能需要根据网格大小调整
                return cell
        return None
    
    def calculate_score_direction(self, weights: Dict[Tuple[float, float], float], 
                                  current_position: Vector3, 
                                  grid_data: HexGridDataModel, 
                                  runtime_data: ScannerRuntimeData) -> Vector3:
        """计算不同方向的得分
        
        Args:
            weights: 各单元权重字典
            current_position: 当前位置
            grid_data: 网格数据对象
            runtime_data: 运行时数据对象
        
        Returns:
            得分最高的方向向量
        """
        direction_scores = []
        current_heading = runtime_data.direction.normalized() if runtime_data.direction.magnitude() > 1e-4 else Vector3(1, 0, 0)
        
        # 生成多个方向候选
        num_directions = 12  # 方向候选数量
        for i in range(num_directions):
            angle = 2 * math.pi * i / num_directions
            candidate_direction = Vector3(math.cos(angle), 0, math.sin(angle))
            
            # 计算该方向的得分
            score = self._calculate_direction_score(candidate_direction, current_position, weights, grid_data)
            
            # 考虑与当前航向的一致性
            direction_consistency = current_heading.dot(candidate_direction)
            adjusted_score = score * direction_consistency
            
            direction_scores.append((candidate_direction, adjusted_score))
        
        # 找到得分最高的方向
        if direction_scores:
            best_direction, best_score = max(direction_scores, key=lambda x: x[1])
            return best_direction
        
        # 如果没有有效的方向，返回当前航向
        return current_heading
    
    def _calculate_direction_score(self, direction: Vector3, current_position: Vector3, 
                                  weights: Dict[Tuple[float, float], float], 
                                  grid_data: HexGridDataModel) -> float:
        """计算指定方向的得分
        
        Args:
            direction: 候选方向
            current_position: 当前位置
            weights: 各单元权重字典
            grid_data: 网格数据对象
        
        Returns:
            该方向的得分
        """
        # 沿方向生成一条线，计算线上的权重积分
        score = 0.0
        samples = 10  # 采样点数
        max_line_length = self.max_move_distance  # 线的最大长度
        
        for i in range(1, samples + 1):
            distance = (i / samples) * max_line_length
            sample_point = Vector3(
                current_position.x + direction.x * distance,
                0.0,
                current_position.z + direction.z * distance
            )
            
            # 找到最近的单元
            nearest_cell = self._find_cell_at_position(grid_data, sample_point)
            if nearest_cell:
                cell_key = (round(nearest_cell.center.x, 2), round(nearest_cell.center.z, 2))
                if cell_key in weights:
                    # 距离越近权重越高
                    weight_contribution = weights[cell_key] * (1 - i / samples)
                    score += weight_contribution
        
        return score
    
    def update_heading(self, best_direction: Vector3, current_direction: Vector3) -> Vector3:
        """更新航向，考虑最大角速度限制
        
        Args:
            best_direction: 最佳方向向量
            current_direction: 当前方向向量
        
        Returns:
            更新后的方向向量
        """
        # 归一化方向向量
        best_dir_norm = best_direction.normalized()
        current_dir_norm = current_direction.normalized() if current_direction.magnitude() > 1e-4 else Vector3(1, 0, 0)
        
        # 计算方向之间的夹角
        dot_product = max(-1, min(1, current_dir_norm.dot(best_dir_norm)))  # 限制在[-1, 1]范围内
        angle = math.acos(dot_product)
        
        # 如果角度小于阈值，直接返回目标方向
        if angle < 0.001:
            return best_dir_norm
        
        # 计算旋转轴（叉积）
        rotation_axis = current_dir_norm.cross(best_dir_norm)
        rotation_axis = rotation_axis.normalized() if rotation_axis.magnitude() > 1e-4 else Vector3(0, 1, 0)
        
        # 限制旋转角度不超过最大角速度
        max_allowed_angle = self.max_angle_velocity  # 假设单位时间内的最大旋转角度
        actual_angle = min(angle, max_allowed_angle)
        
        # 使用罗德里格斯旋转公式进行旋转
        new_direction = self._rotate_vector(current_dir_norm, rotation_axis, actual_angle)
        return new_direction.normalized()
    
    def _rotate_vector(self, vector: Vector3, axis: Vector3, angle: float) -> Vector3:
        """使用罗德里格斯旋转公式旋转向量
        
        Args:
            vector: 待旋转向量
            axis: 旋转轴
            angle: 旋转角度（弧度）
        
        Returns:
            旋转后的向量
        """
        # 罗德里格斯旋转公式
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        
        rotated_vector = Vector3(
            vector.x * cos_angle + \
            axis.cross(vector).x * sin_angle + \
            axis.x * axis.dot(vector) * (1 - cos_angle),
            vector.y * cos_angle + \
            axis.cross(vector).y * sin_angle + \
            axis.y * axis.dot(vector) * (1 - cos_angle),
            vector.z * cos_angle + \
            axis.cross(vector).z * sin_angle + \
            axis.z * axis.dot(vector) * (1 - cos_angle)
        )
        
        return rotated_vector
    
    def calculate_movement(self, best_direction: Vector3, current_position: Vector3, 
                          current_velocity: Vector3, runtime_data: ScannerRuntimeData) -> Vector3:
        """计算移动距离和新位置
        
        Args:
            best_direction: 最佳方向向量
            current_position: 当前位置
            current_velocity: 当前速度
            runtime_data: 运行时数据对象
        
        Returns:
            新的位置向量
        """
        # 归一化方向
        direction_norm = best_direction.normalized()
        
        # 计算目标速度
        target_velocity = Vector3(
            direction_norm.x * self.max_velocity,
            direction_norm.y * self.max_velocity,
            direction_norm.z * self.max_velocity
        )
        
        # 平滑过渡到目标速度
        velocity_diff = Vector3(
            target_velocity.x - current_velocity.x,
            target_velocity.y - current_velocity.y,
            target_velocity.z - current_velocity.z
        )
        velocity_diff_mag = velocity_diff.magnitude()
        
        if velocity_diff_mag > 1e-4:
            # 限制速度变化量
            velocity_step = min(velocity_diff_mag, self.max_velocity * 0.1)  # 假设加速度限制
            new_velocity = Vector3(
                current_velocity.x + velocity_diff.x / velocity_diff_mag * velocity_step,
                current_velocity.y + velocity_diff.y / velocity_diff_mag * velocity_step,
                current_velocity.z + velocity_diff.z / velocity_diff_mag * velocity_step
            )
        else:
            new_velocity = current_velocity
        
        # 计算移动距离
        move_distance = new_velocity.magnitude()  # 假设单位时间
        
        # 确保移动距离在合理范围内
        move_distance = max(self.min_move_distance, min(move_distance, self.max_move_distance))
        
        # 计算新位置
        new_position = Vector3(
            current_position.x + direction_norm.x * move_distance,
            current_position.y,
            current_position.z + direction_norm.z * move_distance
        )
        
        return new_position
    
    def record_visited_cell(self, cell: HexCell, visited_cells: Set[Tuple[float, float]]) -> None:
        """记录已访问的蜂窝单元
        
        Args:
            cell: 蜂窝单元
            visited_cells: 已访问单元集合
        """
        if not self.avoid_revisits:
            return
        
        cell_key = (round(cell.center.x, 2), round(cell.center.z, 2))
        visited_cells.add(cell_key)
    
    def update_runtime_data(self, grid_data: HexGridDataModel, 
                          runtime_data: ScannerRuntimeData) -> ScannerRuntimeData:
        """更新运行时数据，主算法入口
        
        输入: 网格数据、配置数据(实例化时输入)、运行过程数据
        输出: 计算后的运行过程数据
        
        Args:
            grid_data: 网格数据对象
            runtime_data: 运行过程数据对象
        
        Returns:
            计算后的运行过程数据对象
        """
        # 1. 计算各个单元的权重
        weights = self.calculate_weights(grid_data, runtime_data)
        
        # 2. 计算最佳方向
        best_direction = self.calculate_score_direction(weights, runtime_data.position, grid_data, runtime_data)
        
        # 3. 更新航向
        new_direction = self.update_heading(best_direction, runtime_data.direction)
        
        # 4. 计算新位置
        new_position = self.calculate_movement(new_direction, runtime_data.position, runtime_data.velocity, runtime_data)
        
        # 5. 更新访问记录
        # 找到新位置对应的蜂窝单元
        nearest_cell = self._find_cell_at_position(grid_data, new_position)
        if nearest_cell and nearest_cell.entropy < 0.1:  # 假设熵值低于阈值表示已探索
            self.record_visited_cell(nearest_cell, runtime_data.visited_cells)
        
        # 6. 创建更新后的运行时数据
        updated_runtime_data = ScannerRuntimeData(
            direction=new_direction,
            position=new_position,
            velocity=Vector3(0, 0, 0),  # 简化处理，实际应该根据运动模型更新
            leader_position=runtime_data.leader_position,
            leader_velocity=runtime_data.leader_velocity,
            visited_cells=runtime_data.visited_cells.copy()
        )
        
        return updated_runtime_data