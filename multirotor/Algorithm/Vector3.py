import math
import logging
from typing import Dict


class Vector3:
    """三维向量类，提供向量运算功能"""
    x: float
    y: float
    z: float

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self) -> str:
        return f"Vector3({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

    def magnitude(self) -> float:
        """返回向量的模长"""
        return math.sqrt(self.squared_magnitude())

    def squared_magnitude(self) -> float:
        """返回向量模长的平方（性能更优）"""
        return self.x **2 + self.y** 2 + self.z ** 2

    def normalized(self):
        """返回单位向量（归一化）"""
        mag = self.magnitude()
        if mag < 0.001:
            return Vector3()
        return Vector3(self.x / mag, self.y / mag, self.z / mag)

    def __sub__(self, other):
        if not isinstance(other, Vector3):
            raise TypeError("Operand must be a Vector3")
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, other):
        if not isinstance(other, Vector3):
            raise TypeError("Operand must be a Vector3")
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __mul__(self, scalar: float):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __rmul__(self, scalar: float):
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: float):
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def dot(self, other):
        """计算点积"""
        if not isinstance(other, Vector3):
            raise TypeError("Operand must be a Vector3")
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        """计算叉积"""
        if not isinstance(other, Vector3):
            raise TypeError("Operand must be a Vector3")
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典格式"""
        return {"x": self.x, "y": self.y, "z": self.z}

    @classmethod
    def from_dict(cls, data):
        """从字典创建Vector3实例，添加类型检查以防止TypeError"""
        if isinstance(data, dict):
            return cls(
                x=data.get("x", 0.0),
                y=data.get("y", 0.0),
                z=data.get("z", 0.0)
            )
        elif hasattr(data, 'x') and hasattr(data, 'y') and hasattr(data, 'z'):
            # 如果是对象类型且有x,y,z属性
            return cls(x=data.x, y=data.y, z=data.z)
        else:
            # 记录错误信息但不抛出异常，避免程序崩溃
            logging.warning(f"Vector3.from_dict: 数据类型无效: {type(data).__name__}, 数据值: {data}")
            return cls()

    def unity_to_air_sim(self):
        """ Unity坐标转换为AirSim坐标 """
        return Vector3(self.x, self.y, -self.z)

