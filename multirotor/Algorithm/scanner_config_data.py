import json
from typing import Dict, Any
from .Vector3 import Vector3


class ScannerConfigData:
    """扫描器配置数据类，对应C#中的ScannerConfigData"""
    # 系数设置(Python配置)
    repulsionCoefficient: float
    entropyCoefficient: float
    distanceCoefficient: float
    leaderRangeCoefficient: float
    directionRetentionCoefficient: float
    updateInterval: float

    # 基础参数（Python配置）
    moveSpeed: float
    rotationSpeed: float
    scanRadius: float
    altitude: float

    # 排斥力参数（Python配置）
    maxRepulsionDistance: float
    minSafeDistance: float

    # 目标选择策略（Python配置）
    avoidRevisits: bool
    targetSearchRange: float
    revisitCooldown: float

    def __init__(self, config_file: str = None):
        # 设置默认值
        self._set_default_values()
        # 如果提供了配置文件路径，则加载配置
        if config_file:
            self.load_from_file(config_file)

    def _set_default_values(self) -> None:
        """设置所有属性的默认值"""
        # 系数默认值
        self.repulsionCoefficient = 2.0
        self.entropyCoefficient = 3.0
        self.distanceCoefficient = 2.0
        self.leaderRangeCoefficient = 3.0
        self.directionRetentionCoefficient = 2.0
        self.updateInterval = 0.2

        # 运动参数默认值
        self.moveSpeed = 2.0
        self.rotationSpeed = 120.0
        self.scanRadius = 5.0
        self.altitude = 10.0  # 默认高度为10米

        # 距离参数默认值
        self.maxRepulsionDistance = 5.0
        self.minSafeDistance = 2.0

        # 目标选择策略默认值
        self.avoidRevisits = True
        self.targetSearchRange = 20.0
        self.revisitCooldown = 60.0

    def parse_json_data(self, json_data: Dict[str, Any]) -> None:
        """从JSON字典解析数据到对象属性"""
        # 解析基础参数
        self.repulsionCoefficient = self._get_float(json_data, 'repulsionCoefficient', 2.0)
        self.entropyCoefficient = self._get_float(json_data, 'entropyCoefficient', 3.0)
        self.distanceCoefficient = self._get_float(json_data, 'distanceCoefficient', 2.0)
        self.leaderRangeCoefficient = self._get_float(json_data, 'leaderRangeCoefficient', 3.0)
        self.directionRetentionCoefficient = self._get_float(json_data, 'directionRetentionCoefficient', 2.0)
        self.updateInterval = self._get_float(json_data, 'updateInterval', 0.2)

        self.moveSpeed = self._get_float(json_data, 'moveSpeed', 2.0)
        self.rotationSpeed = self._get_float(json_data, 'rotationSpeed', 120.0)
        self.scanRadius = self._get_float(json_data, 'scanRadius', 5.0)
        self.altitude = self._get_float(json_data, 'altitude', 10.0)

        self.maxRepulsionDistance = self._get_float(json_data, 'maxRepulsionDistance', 5.0)
        self.minSafeDistance = self._get_float(json_data, 'minSafeDistance', 2.0)

        self.avoidRevisits = json_data.get('avoidRevisits', True)
        self.targetSearchRange = self._get_float(json_data, 'targetSearchRange', 20.0)
        self.revisitCooldown = self._get_float(json_data, 'revisitCooldown', 60.0)

    @staticmethod
    def _get_float(data_dict: Dict[str, Any], key: str, default: float) -> float:
        """安全地从字典获取浮点数值"""
        value = data_dict.get(key, default)
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典以便序列化"""
        return {
            # 系数参数
            'repulsionCoefficient': self.repulsionCoefficient,
            'entropyCoefficient': self.entropyCoefficient,
            'distanceCoefficient': self.distanceCoefficient,
            'leaderRangeCoefficient': self.leaderRangeCoefficient,
            'directionRetentionCoefficient': self.directionRetentionCoefficient,
            'updateInterval': self.updateInterval,

            # 运动参数
            'moveSpeed': self.moveSpeed,
            'rotationSpeed': self.rotationSpeed,
            'scanRadius': self.scanRadius,
            'altitude': self.altitude,

            # 距离参数
            'maxRepulsionDistance': self.maxRepulsionDistance,
            'minSafeDistance': self.minSafeDistance,

            # 目标选择策略
            'avoidRevisits': self.avoidRevisits,
            'targetSearchRange': self.targetSearchRange,
            'revisitCooldown': self.revisitCooldown
        }

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def validate(self) -> bool:
        """验证数据有效性"""
        if self.moveSpeed <= 0:
            return False
        if self.scanRadius <= 0:
            return False
        if self.minSafeDistance < 0:
            return False
        if self.maxRepulsionDistance < self.minSafeDistance:
            return False
        if self.updateInterval <= 0:
            return False
        # 验证系数非负
        for coeff in [
            self.repulsionCoefficient,
            self.entropyCoefficient,
            self.distanceCoefficient,
            self.leaderRangeCoefficient,
            self.directionRetentionCoefficient
        ]:
            if coeff < 0:
                return False
        return True

    def copy(self):
        """创建对象的深拷贝"""
        new_data = ScannerConfigData()
        new_data.__dict__.update(self.__dict__)
        return new_data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScannerConfigData':
        """从字典创建ScannerConfigData实例"""
        instance = cls()
        instance.parse_json_data(data)
        return instance
    
    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """从字典更新ScannerConfigData实例的属性"""
        self.parse_json_data(data)
        
    def load_from_file(self, config_file: str) -> None:
        """
        从配置文件加载数据
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.parse_json_data(data)
        except Exception as e:
            print(f"加载配置文件失败: {str(e)}")
            # 加载失败时保持默认值
            self._set_default_values()

    def __repr__(self) -> str:
        return f"ScannerConfigData:ScanRadius:{self.scanRadius}"
