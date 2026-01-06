from dataclasses import dataclass, asdict
from multirotor.Algorithm.Vector3 import Vector3
from typing import List
import json


# 1. 定义WayPoint
@dataclass
class WayPoint:
    """无人机航点类（示例结构，可根据实际WayPoint字段调整）"""
    x: float       # X轴坐标
    y: float       # Y轴坐标
    z: float       # Z轴坐标
    duration: float = 0.0  # 移动到航点可用时间

    @classmethod
    def Create(cls, x: float, y: float, z: float, duration: float = 0.0) -> "WayPoint":
        """
        静态创建WayPoint实例（C#风格的Create方法）
        :param x: X轴坐标
        :param y: Y轴坐标
        :param z: Z轴坐标
        :param duration: 停留时间（默认0秒）
        :return: WayPoint实例
        """
        return cls(x = x, y = y, z = z, duration = duration)
    
    @classmethod
    def CreateFromPosition(cls, position : Vector3, duration: float = 0.0) -> "WayPoint":
        """
        静态创建WayPoint实例（C#风格的Create方法）
        :param position: 三维坐标
        :param duration: 停留时间（默认0秒）
        :return: WayPoint实例
        """
        return cls(position.x, position.y, position.z, duration = duration)

    # 航点序列化：转成JSON字典
    def to_dict(self) -> dict:
        return asdict(self)  # 简单字段直接用asdict转换

    # 航点反序列化：从JSON字典转对象
    @classmethod
    def from_dict(cls, wp_dict: dict) -> "WayPoint":
        return cls(**wp_dict)


# 2. 定义WayPath（对应C#的WayPath类）
@dataclass
class WayPath:
    """无人机路径类（包含航点列表）"""
    points: List[WayPoint]  # 航点列表

    # ------------------- 对应C#的Create静态方法 -------------------
    @classmethod
    def Create(cls, points: List[WayPoint]) -> "WayPath":
        """
        静态创建WayPath实例（C#风格的Create方法）
        :param points: 航点列表（List[WayPoint]）
        :return: WayPath实例
        """
        return cls(points = points)
    
    # ------------------- 对应C#的Create静态方法 -------------------
    @classmethod
    def CreateFromPoint(cls, point: WayPoint) -> "WayPath":
        """
        静态创建WayPath实例（接收单个WayPoint，自动包装为列表）
        :param point: 单个航点（WayPoint类型）
        :return: WayPath实例
        """
        # 把单个WayPoint包装成列表，匹配WayPath的points字段类型
        return cls(points=[point])

    # 路径序列化：转成JSON字典
    def to_dict(self) -> dict:
        return {
            "points": [wp.to_dict() for wp in self.points]  # 航点列表转JSON数组
        }

    # 路径反序列化：从JSON字典转对象
    @classmethod
    def from_dict(cls, path_dict: dict) -> "WayPath":
        # 解析points数组，转成WayPoint列表
        way_points = [WayPoint.from_dict(wp) for wp in path_dict["points"]]
        return cls(points=way_points)

    # 快捷方法：转JSON字符串
    def to_json(self, indent: int = 4) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    # 快捷方法：从JSON字符串恢复对象
    @classmethod
    def from_json(cls, json_str: str) -> "WayPath":
        path_dict = json.loads(json_str)
        return cls.from_dict(path_dict)