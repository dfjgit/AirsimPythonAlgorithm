import json
import os
import math
import time
from .Vector3 import Vector3
from .scannerData import ScannerData
from .HexGridDataModelData import HexGridDataModel, HexCellData

class ScannerAlgorithm:
    def __init__(self, grid_model=None, scanner_data=None):
        # 初始化算法参数
        # 存储已访问的蜂窝及其访问时间
        self.visited_cells = {}
        # 上一次移动方向，用于计算方向保持
        self.previous_move_dir = None
        # 蜂窝网格数据模型
        self.grid_model = grid_model
        # 网格数据文件路径
        self.grid_data_file = None
        # 当前扫描器数据（可选）
        self.scanner_data = scanner_data
    
    def calculate_weights(self, data):
        # 计算各权重
        total = (data.repulsionCoefficient + data.entropyCoefficient + data.distanceCoefficient + 
                data.leaderRangeCoefficient + data.directionRetentionCoefficient)
        
        # 处理所有系数都为0的特殊情况
        if total < 0.001:
            return 0.2, 0.2, 0.2, 0.2, 0.2
        
        repulsion_weight = data.repulsionCoefficient / total
        entropy_weight = data.entropyCoefficient / total
        distance_weight = data.distanceCoefficient / total
        leader_range_weight = data.leaderRangeCoefficient / total
        direction_retention_weight = data.directionRetentionCoefficient / total
        
        return repulsion_weight, entropy_weight, distance_weight, leader_range_weight, direction_retention_weight
    
    def calculate_repulsion_ratio(self, distance, min_safe_distance, max_repulsion_distance):
        # 计算排斥力比例
        if distance <= min_safe_distance:
            return 1.0
        if distance >= max_repulsion_distance:
            return 0.0
        
        # 非线性衰减，近距离排斥力增长更快
        t = (distance - min_safe_distance) / (max_repulsion_distance - min_safe_distance)
        return 1.0 - (t * t)
        
    def load_grid_model(self, file_path):
        """加载蜂窝网格数据模型"""
        self.grid_data_file = file_path
        self.grid_model = HexGridDataModel.deserialize_from_json_file(file_path)
        if self.grid_model:
            print(f"成功加载蜂窝网格数据模型，包含 {len(self.grid_model.cells)} 个蜂窝单元")
        else:
            print(f"未能加载蜂窝网格数据模型: {file_path}")
        return self.grid_model is not None
    
    def calculate_score_direction(self, data):
        """基于蜂窝数据计算熵最优方向向量"""
        if not self.grid_model or not self.grid_model.cells:
            # 如果没有网格数据，返回默认方向
            return data.scoreDir
        
        current_pos = data.position
        
        # 获取当前位置附近的蜂窝单元
        nearby_cells = self.grid_model.get_cells_in_radius(current_pos, data.scanRadius)
        
        if not nearby_cells:
            # 如果没有附近蜂窝，返回默认方向
            return data.scoreDir
        
        # 初始化最优方向和最大熵值
        best_direction = data.scoreDir
        max_entropy = -float('inf')
        
        # 遍历所有候选蜂窝单元，找出熵值最高且未被访问过的
        for cell in nearby_cells:
            # 检查蜂窝是否已被访问
            if self._is_cell_visited(cell, data.revisitCooldown, data.avoidRevisits):
                continue
            
            # 如果熵值更高，更新最优方向
            if cell.entropy > max_entropy:
                max_entropy = cell.entropy
                # 计算指向该蜂窝的方向向量
                best_direction = Vector3(cell.x - current_pos.x, 0, cell.z - current_pos.z)
                if best_direction.magnitude() > 0.01:
                    best_direction = best_direction.normalized()
        
        return best_direction
    
    def calculate_path_direction(self, score_dir):
        """基于最优方向计算路径方向向量"""
        # 可以根据网格数据进行更复杂的路径计算
        # 对于简单实现，我们直接使用score_dir作为path_dir
        return score_dir
    
    def calculate_repulsion_direction(self, data):
        """计算排斥力方向向量"""
        # 初始化排斥方向为零向量
        collide_dir = Vector3(0, 0, 0)
        
        # 检查ScannerData对象是否有otherScannerPositions属性
        if hasattr(data, 'otherScannerPositions') and data.otherScannerPositions:
            # 计算其他扫描器的排斥力
            for scanner_pos in data.otherScannerPositions:
                # 计算与其他扫描器的距离
                distance = (data.position - scanner_pos).magnitude()
                
                # 如果距离过近，添加排斥力
                if distance < data.minSafeDistance:
                    # 计算排斥方向（从其他扫描器指向当前扫描器）
                    repulse_vector = data.position - scanner_pos
                    if repulse_vector.magnitude() > 0.01:
                        repulse_vector = repulse_vector.normalized()
                        
                        # 计算排斥力大小（距离越近，排斥力越大）
                        repulsion_strength = self.calculate_repulsion_ratio(distance, 
                                                                          data.minSafeDistance, 
                                                                          data.maxRepulsionDistance)
                        
                        # 应用排斥力
                        collide_dir += repulse_vector * repulsion_strength
        
        # 如果计算结果有效，返回计算值，否则返回Unity发送的值
        if collide_dir.magnitude() > 0.01:
            return collide_dir.normalized()
        else:
            return data.collideDir
    
    def calculate_leader_range_direction(self, data):
        """计算保持在Leader范围内的方向向量"""
        leader_range_dir = Vector3(0, 0, 0)
        
        # 检查是否有Leader
        # Vector3对象始终有值，所以我们检查其坐标是否都为0
        if data.leaderPosition.x == 0 and data.leaderPosition.y == 0 and data.leaderPosition.z == 0:
            return leader_range_dir
        
        # 计算与Leader的距离
        distance_to_leader = (data.position - data.leaderPosition).magnitude()
        
        # 如果超出Leader的范围，生成指向Leader的方向向量
        if distance_to_leader > data.leaderScanRadius:
            # 距离越远，返回的力度越大
            range_ratio = self._inverse_lerp(data.leaderScanRadius, data.leaderScanRadius * 2, distance_to_leader)
            leader_range_dir = (data.leaderPosition - data.position).normalized() * (1 + range_ratio)
        # 如果离Leader过近，生成轻微远离Leader的方向向量
        elif distance_to_leader < data.leaderScanRadius * 0.3:
            leader_range_dir = (data.position - data.leaderPosition).normalized() * 0.3
        
        return leader_range_dir
    
    def calculate_direction_retention_direction(self, data):
        """计算方向保持向量（减少转弯）"""
        # 如果是第一次处理，设置previous_move_dir为当前forward方向
        if self.previous_move_dir is None:
            self.previous_move_dir = data.forward
            return data.forward
        
        # 方向保持向量与上一帧的移动方向一致
        return self.previous_move_dir
    
    def merge_directions(self, score_dir, path_dir, collide_dir, leader_range_dir, direction_retention_dir, 
                         entropy_weight, distance_weight, repulsion_weight, leader_range_weight, direction_retention_weight, data):
        """合并所有方向向量，计算最终移动方向"""
        # 应用权重合并向量，包含方向保持向量
        final_move_dir = (
            score_dir * entropy_weight +
            path_dir * distance_weight +
            collide_dir * repulsion_weight +
            leader_range_dir * leader_range_weight +
            direction_retention_dir * direction_retention_weight
        )
        
        # 归一化最终方向
        if final_move_dir.magnitude() > 0.1:
            final_move_dir = final_move_dir.normalized()
        else:
            # 如果最终方向接近零，保持当前方向
            final_move_dir = data.forward
        
        # 保存当前方向作为下一帧的previous_move_dir
        self.previous_move_dir = final_move_dir
        
        return final_move_dir
    
    def record_visited_cell(self, cell_center, revisit_cooldown, avoid_revisits):
        """记录已访问的蜂窝"""
        if not avoid_revisits:
            return
        
        # 四舍五入避免浮点数精度问题
        rounded_center = Vector3(
            round(cell_center.x * 100) / 100,
            0,
            round(cell_center.z * 100) / 100
        )
        
        # 使用字符串作为字典键，因为Vector3对象不能直接用作键
        cell_key = f"{rounded_center.x},{rounded_center.y},{rounded_center.z}"
        self.visited_cells[cell_key] = time.time()
        
    def get_visited_cells(self):
        """获取已访问的蜂窝列表，用于Unity可视化"""
        # 返回已访问蜂窝的坐标列表
        return list(self.visited_cells.keys())
    
    def cleanup_visited_records(self, revisit_cooldown, avoid_revisits):
        """清理过期的访问记录"""
        if not avoid_revisits:
            return
        
        current_time = time.time()
        expired_keys = [key for key, visit_time in self.visited_cells.items() 
                       if current_time - visit_time >= revisit_cooldown]
        
        for key in expired_keys:
            del self.visited_cells[key]
    
    def _inverse_lerp(self, a, b, value):
        """计算值在a和b之间的比例（逆Lerp）"""
        if b - a < 0.001:
            return 0.0
        return (value - a) / (b - a)
        
    def _is_cell_visited(self, cell, revisit_cooldown, avoid_revisits):
        """检查蜂窝是否已被访问"""
        if not avoid_revisits:
            return False
            
        # 创建蜂窝位置的键
        cell_key = f"{round(cell.x * 100) / 100},{0},{round(cell.z * 100) / 100}"
        
        # 检查是否存在访问记录
        if cell_key not in self.visited_cells:
            return False
            
        # 检查是否超过冷却时间
        current_time = time.time()
        return current_time - self.visited_cells[cell_key] < revisit_cooldown
    
    def process(self, data=None):
        """处理扫描器数据并计算新的移动方向
        
        Args:
            data: 扫描器数据对象，如果为None则使用初始化时提供的数据
        
        Returns:
            Vector3: 最终移动方向向量
        """
        # 如果没有提供数据，使用初始化时设置的数据
        if data is None:
            data = self.scanner_data
            
        # 确保有数据可用
        if data is None:
            raise ValueError("没有提供扫描器数据")
        
        # 计算权重
        repulsion_weight, entropy_weight, distance_weight, leader_range_weight, direction_retention_weight = self.calculate_weights(data)
        
        # 计算各方向向量
        new_score_dir = self.calculate_score_direction(data)
        new_path_dir = self.calculate_path_direction(new_score_dir)
        new_collide_dir = self.calculate_repulsion_direction(data)
        new_leader_range_dir = self.calculate_leader_range_direction(data)
        new_direction_retention_dir = self.calculate_direction_retention_direction(data)
        
        # 合并所有向量
        new_final_move_dir = self.merge_directions(
            new_score_dir, new_path_dir, new_collide_dir, new_leader_range_dir, new_direction_retention_dir,
            entropy_weight, distance_weight, repulsion_weight, leader_range_weight, direction_retention_weight,
            data
        )
        
        # 更新数据对象中的所有方向向量
        data.scoreDir = new_score_dir
        data.pathDir = new_path_dir
        data.collideDir = new_collide_dir
        data.leaderRangeDir = new_leader_range_dir
        data.directionRetentionDir = new_direction_retention_dir
        data.finalMoveDir = new_final_move_dir
        
        # 清理过期访问记录
        self.cleanup_visited_records(data.revisitCooldown, data.avoidRevisits)
        
        # 返回原始数据对象，包含所有计算后的方向向量
        return data

def main():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 输入输出文件路径
    input_file = os.path.join(current_dir, "testData", "python_input.json")
    output_file = os.path.join(current_dir, "testData", "python_output.json")
    grid_data_file = os.path.join(current_dir, "testData", "hex_grid_data.json")
    
    print(f"扫描器算法处理程序启动")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"网格数据文件: {grid_data_file}")
    
    # 创建算法实例
    algorithm = ScannerAlgorithm()
    
    # 尝试加载网格数据
    algorithm.load_grid_model(grid_data_file)
    
    try:
        # 读取Unity发送的数据
        with open(input_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            
        # 解析数据
        scanner_data = ScannerData(json_data)
        print(f"成功读取数据，位置: {scanner_data.position}")
        
        # 处理数据
        processed_data = algorithm.process(scanner_data)
        
        # 将处理后的数据转换为字典并保存到文件
        result_dict = processed_data.to_dict()
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        print(f"处理完成，结果已保存到 {output_file}")
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()