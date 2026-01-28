import json
from typing import Dict, Any, Optional
from .Vector3 import Vector3


class ScannerConfigData:
    """扫描器配置数据类，对应C#中的ScannerConfigData"""
    # 系数设置(Python配置)
    repulsionCoefficient: float
    entropyCoefficient: float
    distanceCoefficient: float
    leaderRangeCoefficient: float
    directionRetentionCoefficient: float
    groundRepulsionCoefficient: float
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

    # 新增字段：无人机配置（原isCrazyflieMirror升级为按无人机区分）
    droneSettings: Dict[str, Dict[str, bool]]
    # 新增字段：配置名称和隐藏标志
    name: str
    hideFlags: int

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
        self.groundRepulsionCoefficient = 0.2

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

        # 新增字段默认值
        self.droneSettings = {
            "UAV1": {"isCrazyflieMirror": False},
            "UAV2": {"isCrazyflieMirror": False}
        }
        self.name = "ScannerConfigData"
        self.hideFlags = 0

    def parse_json_data(self, json_data: Dict[str, Any]) -> None:
        """从JSON字典解析数据到对象属性"""
        # 解析基础参数
        self.repulsionCoefficient = self._get_float(json_data, 'repulsionCoefficient', 2.0)
        self.entropyCoefficient = self._get_float(json_data, 'entropyCoefficient', 3.0)
        self.distanceCoefficient = self._get_float(json_data, 'distanceCoefficient', 2.0)
        self.leaderRangeCoefficient = self._get_float(json_data, 'leaderRangeCoefficient', 3.0)
        self.directionRetentionCoefficient = self._get_float(json_data, 'directionRetentionCoefficient', 2.0)
        self.groundRepulsionCoefficient = self._get_float(json_data, 'groundRepulsionCoefficient', 2.0)

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

        # 解析新增字段
        # 解析无人机配置（兼容空值/非字典情况）
        self.droneSettings = json_data.get('droneSettings', {})
        # 确保droneSettings中每个无人机配置都有isCrazyflieMirror字段
        for uav_key in self.droneSettings:
            if 'isCrazyflieMirror' not in self.droneSettings[uav_key]:
                self.droneSettings[uav_key]['isCrazyflieMirror'] = False
        
        self.name = json_data.get('name', "ScannerConfigData")
        self.hideFlags = self._get_int(json_data, 'hideFlags', 0)

    @staticmethod
    def _get_float(data_dict: Dict[str, Any], key: str, default: float) -> float:
        """安全地从字典获取浮点数值"""
        value = data_dict.get(key, default)
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _get_int(data_dict: Dict[str, Any], key: str, default: int) -> int:
        """安全地从字典获取整数值"""
        value = data_dict.get(key, default)
        try:
            return int(value)
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
            'groundRepulsionCoefficient': self.groundRepulsionCoefficient,
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
            'revisitCooldown': self.revisitCooldown,

            # 新增字段
            'droneSettings': self.droneSettings,
            'name': self.name,
            'hideFlags': self.hideFlags
        }

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def validate(self) -> bool:
        """验证数据有效性"""
        # 基础参数验证
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
        
        # 系数非负验证
        for coeff in [
            self.repulsionCoefficient,
            self.entropyCoefficient,
            self.distanceCoefficient,
            self.leaderRangeCoefficient,
            self.directionRetentionCoefficient,
            self.groundRepulsionCoefficient
        ]:
            if coeff < 0:
                return False
        
        # 新增字段验证
        # 确保droneSettings是字典且非空（可选，根据业务需求调整）
        if not isinstance(self.droneSettings, dict):
            return False
        # 验证每个无人机配置的isCrazyflieMirror是布尔值
        for uav_key, uav_config in self.droneSettings.items():
            if not isinstance(uav_config.get('isCrazyflieMirror', False), bool):
                return False
        
        # 验证hideFlags是非负整数
        if self.hideFlags < 0:
            return False
        
        return True

    def copy(self):
        """创建对象的深拷贝"""
        new_data = ScannerConfigData()
        # 深拷贝droneSettings（避免浅拷贝导致的引用问题）
        new_data.droneSettings = {k: v.copy() for k, v in self.droneSettings.items()}
        # 拷贝其他属性
        new_data.__dict__.update({
            k: v for k, v in self.__dict__.items() if k != 'droneSettings'
        })
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
        return f"ScannerConfigData(name={self.name}, ScanRadius={self.scanRadius}, UAVs={list(self.droneSettings.keys())})"

    # 新增便捷方法：获取指定无人机的isCrazyflieMirror配置
    def get_uav_crazyflie_mirror(self, uav_id: str) -> bool:
        """
        获取指定无人机的isCrazyflieMirror配置
        :param uav_id: 无人机ID（如UAV1、UAV2）
        :return: 是否为Crazyflie镜像
        """
        return self.droneSettings.get(uav_id, {}).get('isCrazyflieMirror', False)

    # 新增便捷方法：设置指定无人机的isCrazyflieMirror配置
    def set_uav_crazyflie_mirror(self, uav_id: str, is_mirror: bool) -> None:
        """
        设置指定无人机的isCrazyflieMirror配置
        :param uav_id: 无人机ID（如UAV1、UAV2）
        :param is_mirror: 是否为Crazyflie镜像
        """
        if uav_id not in self.droneSettings:
            self.droneSettings[uav_id] = {}
        self.droneSettings[uav_id]['isCrazyflieMirror'] = is_mirror
    
    def get_drone_list(self) -> list:
        """
        获取配置文件中所有无人机的名称列表
        :return: 无人机名称列表（如['UAV1', 'UAV2', 'UAV3']）
        """
        return list(self.droneSettings.keys())