# 数据模型类，用于 JSON 序列化
import json
import os
from typing import List, Optional, Dict, Any
from .Vector3 import Vector3


class HexCell:
    """表示单个六边形蜂窝单元的数据类
    
    存储蜂窝的中心点和熵值，提供序列化和反序列化功能
    """
    center: Vector3
    entropy: float

    def __init__(self, center: Vector3 = None, entropy: float = 0.0):
        """初始化六边形蜂窝单元
        
        Args:
            center: 蜂窝的中心点坐标
            entropy: 蜂窝的熵值
        """
        self.center = center if center is not None else Vector3()
        self.entropy = entropy

    def to_dict(self) -> Dict[str, Any]:
        """将对象转换为字典格式，用于JSON序列化"""
        return {
            'center': self.center.to_dict() if self.center else Vector3().to_dict(),
            'entropy': self.entropy
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HexCell':
        """从字典创建对象，用于JSON反序列化
        
        Args:
            data: 包含蜂窝数据的字典，需包含'center'和'entropy'键
        """
        center_data = data.get('center', {})
        center = Vector3.from_dict(center_data) if isinstance(center_data, dict) else Vector3()
        entropy = float(data.get('entropy', 0.0))
        
        return cls(center=center, entropy=entropy)

    def __eq__(self, other: Any) -> bool:
        """重写相等性判断，用于单元格去重和比较"""
        if not isinstance(other, HexCell):
            return False
        
        return (self.center == other.center and 
                abs(self.entropy - other.entropy) < 1e-6)

    def __repr__(self) -> str:
        """提供清晰的对象字符串表示"""
        return f"HexCell(center={self.center}, entropy={self.entropy})"


class HexGridDataModel:
    """六边形网格数据模型类，用于存储和序列化网格数据"""
    cells: List[HexCell]

    def __init__(self):
        """初始化六边形网格模型"""
        self.cells = []  # 存储所有蜂窝单元

    def to_dict(self) -> Dict[str, Any]:
        """将对象转换为字典格式，用于JSON序列化"""
        return {
            'cells': [cell.to_dict() for cell in self.cells]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HexGridDataModel':
        """从字典创建对象，用于JSON反序列化"""
        model = cls()

        # 安全处理单元格数据
        if isinstance(data.get('cells'), list):
            model.cells = [
                HexCell.from_dict(cell_data) 
                for cell_data in data['cells'] 
                if isinstance(cell_data, dict)
            ]
        return model

    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """更新现有对象的数据，而不是创建新对象"""
        # 清空现有cells
        self.cells.clear()

        # 安全处理单元格数据
        if isinstance(data.get('cells'), list):
            self.cells = [
                HexCell.from_dict(cell_data)
                for cell_data in data['cells']
                if isinstance(cell_data, dict)
            ]
        print(f"更新HexGridDataModel: {self}")

    def serialize_to_json(self) -> Optional[str]:
        """将数据模型序列化为JSON字符串"""
        try:
            return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"序列化HexGridDataModel失败: {str(e)}")
            return None

    def serialize_to_json_file(self, file_path: str) -> bool:
        """将数据模型序列化为JSON文件"""
        json_str = self.serialize_to_json()
        if not json_str:
            return False

        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
            
            print(f"六边形网格数据已保存至: {file_path}")
            return True
        except Exception as e:
            print(f"保存HexGridDataModel至文件失败: {str(e)}")
            return False

    @classmethod
    def deserialize_from_json(cls, json_str: str) -> Optional['HexGridDataModel']:
        """从JSON字符串反序列化数据模型"""
        if not json_str:
            print("空JSON字符串，无法反序列化")
            return None

        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {str(e)}")
            return None
        except Exception as e:
            print(f"反序列化HexGridDataModel失败: {str(e)}")
            return None

    @classmethod
    def deserialize_from_json_file(cls, file_path: str) -> Optional['HexGridDataModel']:
        """从JSON文件反序列化数据模型"""
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_str = f.read()
            
            return cls.deserialize_from_json(json_str)
        except Exception as e:
            print(f"读取JSON文件失败: {str(e)}")
            return None

    def __len__(self) -> int:
        """返回蜂窝单元数量"""
        return len(self.cells)

    def __repr__(self) -> str:
        """提供清晰的对象字符串表示"""
        return f"HexGridDataModel(cell_count={len(self.cells)})"