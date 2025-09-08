# // 数据模型类，用于 JSON 序列化
import json
from .Vector3 import Vector3

class HexCellData:
    """表示单个六边形蜂窝单元的数据类"""
    def __init__(self, x=0.0, z=0.0, entropy=0.0):
        self.x = x  # X坐标
        self.z = z  # Z坐标
        self.entropy = entropy  # 熵值
    
    def to_dict(self):
        """将对象转换为字典格式，用于JSON序列化"""
        return {
            'x': round(self.x, 3),  # 保留3位小数，与C#版本保持一致
            'z': round(self.z, 3),
            'entropy': round(self.entropy, 2)  # 保留2位小数
        }
    
    @classmethod
    def from_dict(cls, data):
        """从字典创建对象，用于JSON反序列化"""
        return cls(
            x=data.get('x', 0.0),
            z=data.get('z', 0.0),
            entropy=data.get('entropy', 0.0)
        )

class HexGridDataModel:
    """六边形网格数据模型类，用于存储和序列化网格数据"""
    def __init__(self, single_hex_radius=0.0, total_range=0.0, initial_entropy=0.0, 
                 min_entropy=0.0, color_reference_max=0.0):
        self.singleHexRadius = single_hex_radius  # 单个六边形半径
        self.totalRange = total_range  # 总范围
        self.initialEntropy = initial_entropy  # 初始熵值
        self.minEntropy = min_entropy  # 最小熵值
        self.colorReferenceMax = color_reference_max  # 颜色参考最大值
        self.cells = []  # 蜂窝单元列表
    
    def to_dict(self):
        """将对象转换为字典格式，用于JSON序列化"""
        return {
            'singleHexRadius': self.singleHexRadius,
            'totalRange': self.totalRange,
            'initialEntropy': self.initialEntropy,
            'minEntropy': self.minEntropy,
            'colorReferenceMax': self.colorReferenceMax,
            'cells': [cell.to_dict() for cell in self.cells]
        }
    
    @classmethod
    def from_dict(cls, data):
        """从字典创建对象，用于JSON反序列化"""
        model = cls(
            single_hex_radius=data.get('singleHexRadius', 0.0),
            total_range=data.get('totalRange', 0.0),
            initial_entropy=data.get('initialEntropy', 0.0),
            min_entropy=data.get('minEntropy', 0.0),
            color_reference_max=data.get('colorReferenceMax', 0.0)
        )
        
        # 从字典列表创建HexCellData对象
        if 'cells' in data:
            for cell_data in data['cells']:
                model.cells.append(HexCellData.from_dict(cell_data))
        
        return model
    
    def serialize_to_json(self):
        """将数据模型序列化为JSON字符串"""
        try:
            return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"序列化HexGridDataModel时出错: {str(e)}")
            return None
    
    def serialize_to_json_file(self, file_path):
        """将数据模型序列化为JSON文件"""
        try:
            json_str = self.serialize_to_json()
            if json_str is None:
                return False
            
            # 确保目录存在
            import os
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            # 写入文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
            
            print(f"六边形网格数据已成功保存到: {file_path}")
            return True
        except Exception as e:
            print(f"保存HexGridDataModel数据到文件时出错: {str(e)}")
            return False
    
    @classmethod
    def deserialize_from_json(cls, json_str):
        """从JSON字符串反序列化数据模型"""
        if not json_str:
            return None
        
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except Exception as e:
            print(f"反序列化HexGridDataModel时出错: {str(e)}")
            return None
    
    @classmethod
    def deserialize_from_json_file(cls, file_path):
        """从JSON文件反序列化数据模型"""
        try:
            import os
            if not os.path.exists(file_path):
                print(f"JSON文件不存在: {file_path}")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                json_str = f.read()
            
            return cls.deserialize_from_json(json_str)
        except Exception as e:
            print(f"读取JSON文件时出错: {str(e)}")
            return None

    def get_cell_at_position(self, position):
        """根据位置查找蜂窝单元"""
        # 简单实现：查找最近的蜂窝
        if not self.cells:
            return None
        
        closest_cell = None
        min_distance = float('inf')
        
        for cell in self.cells:
            # 计算距离（使用平面距离，忽略Y轴）
            distance = ((position.x - cell.x) ** 2 + (position.z - cell.z) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_cell = cell
        
        return closest_cell

    def get_cells_in_radius(self, position, radius):
        """获取指定半径内的所有蜂窝单元"""
        result = []
        radius_squared = radius ** 2
        
        for cell in self.cells:
            # 计算距离平方（避免开方运算）
            distance_squared = (position.x - cell.x) ** 2 + (position.z - cell.z) ** 2
            if distance_squared <= radius_squared:
                result.append(cell)
        
        return result