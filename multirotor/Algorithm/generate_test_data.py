import json
import os
import random
from Vector3 import Vector3
from HexGridDataModelData import HexGridDataModel, HexCellData
from scannerData import ScannerData

class TestDataGenerator:
    """
    测试数据生成器，用于生成蜂窝网格数据和扫描器配置数据
    """
    def __init__(self, output_dir):
        """初始化生成器"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_hex_grid_data(self, width=10, height=10, cell_size=1.0, output_file="hex_grid_data.json"):
        """生成蜂窝网格数据"""
        grid_model = HexGridDataModel()
        
        # 蜂窝网格的六边形布局计算
        for x in range(width):
            for z in range(height):
                # 计算蜂窝的世界坐标
                world_x = (x + (z % 2) * 0.5) * (cell_size * 1.5)
                world_z = z * (cell_size * math.sqrt(3) / 2)
                
                # 为每个蜂窝生成随机的熵值和访问次数
                entropy = random.uniform(0, 1)
                visited_count = random.randint(0, 3)
                
                # 创建蜂窝单元并添加到网格模型
                cell = HexCellData(
                    x=world_x,
                    z=world_z,
                    entropy=entropy
                )
                grid_model.cells.append(cell)
        
        # 保存为JSON文件
        output_path = os.path.join(self.output_dir, output_file)
        grid_model.serialize_to_json_file(output_path)
        
        print(f"已生成蜂窝网格数据: {output_path}")
        print(f"网格包含 {len(grid_model.cells)} 个蜂窝单元")
        
        return output_path
    
    def generate_scanner_data(self, output_file="scanner_data.json"):
        """生成扫描器配置数据"""
        # 创建扫描器数据对象
        scanner_data = ScannerData()
        
        # 设置各种属性
        scanner_data.position = Vector3(0, 10, 0)  # 初始位置
        scanner_data.forward = Vector3(1, 0, 0)    # 初始朝向
        scanner_data.scoreDir = Vector3(1, 0, 0)   # 初始评分方向
        scanner_data.pathDir = Vector3(1, 0, 0)    # 初始路径方向
        scanner_data.collideDir = Vector3(0, 0, 0) # 初始碰撞方向
        scanner_data.finalMoveDir = Vector3(1, 0, 0) # 初始移动方向
        scanner_data.targetPosition = Vector3(10, 10, 10) # 目标位置
        scanner_data.repulsionCoefficient = 0.2    # 排斥力系数
        scanner_data.entropyCoefficient = 0.5      # 熵系数
        scanner_data.distanceCoefficient = 0.1     # 距离系数
        scanner_data.leaderRangeCoefficient = 0.1  # Leader范围系数
        scanner_data.directionRetentionCoefficient = 0.1 # 方向保持系数
        scanner_data.minSafeDistance = 2.0         # 最小安全距离
        scanner_data.maxRepulsionDistance = 5.0    # 最大排斥距离
        scanner_data.scanRadius = 10.0             # 扫描半径
        scanner_data.moveSpeed = 3.0               # 移动速度
        scanner_data.revisitCooldown = 60.0        # 重访冷却时间
        scanner_data.avoidRevisits = True          # 是否避免重访
        scanner_data.minEntropy = 0.0              # 最小熵值
        scanner_data.maxEntropy = 1.0              # 最大熵值
        scanner_data.maxVisitedCount = 5           # 最大访问次数
        scanner_data.entropyWeight = 1.0           # 熵权重
        scanner_data.repulsionWeight = 1.0         # 排斥力权重
        scanner_data.leaderWeight = 1.0            # Leader权重
        scanner_data.directionPersistenceWeight = 1.0 # 方向持久化权重
        scanner_data.otherScannerPositions = []    # 其他扫描器位置
        scanner_data.leaderPosition = Vector3(5, 10, 5) # Leader位置
        scanner_data.leaderScanRadius = 8.0        # Leader扫描半径
        
        # 保存为JSON文件
        output_path = os.path.join(self.output_dir, output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(scanner_data.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"已生成扫描器数据: {output_path}")
        
        return output_path

# 主函数
if __name__ == "__main__":
    import sys
    import math
    
    # 获取输出目录，默认为当前目录
    output_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    
    # 创建生成器实例
    generator = TestDataGenerator(output_dir)
    
    # 生成测试数据
    generator.generate_hex_grid_data()
    generator.generate_scanner_data()
    
    print("所有测试数据生成完成！")