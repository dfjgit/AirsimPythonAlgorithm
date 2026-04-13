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

    # 新增字段：配置名称和隐藏标志
    name: str
    hideFlags: int

    # 新增字段：统一环境配置（物理规则与RL解耦）
    env_config: Dict[str, Any]
    paper_benchmark: Dict[str, Any]

    def __init__(self, config_file: str = None):
        # 设置默认值
        self._set_default_values()
        # 如果提供了配置文件路径，则加载配置
        if config_file:
            self.load_from_file(config_file)

    def _set_default_values(self) -> None:
        """设置所有属性的默认值（与 system_config.json 保持一致）"""
        # 系数默认值
        self.repulsionCoefficient = 2.0
        self.entropyCoefficient = 2.0
        self.distanceCoefficient = 2.0
        self.leaderRangeCoefficient = 1.5
        self.directionRetentionCoefficient = 0.5
        self.updateInterval = 0.5
        self.groundRepulsionCoefficient = 2.0

        # 运动参数默认值
        self.moveSpeed = 1.0
        self.rotationSpeed = 120.0
        self.scanRadius = 2.0
        self.altitude = 2.0

        # 距离参数默认值
        self.maxRepulsionDistance = 3.0
        self.minSafeDistance = 1.0

        # 目标选择策略默认值
        self.avoidRevisits = True
        self.targetSearchRange = 20.0
        self.revisitCooldown = 10.0

        # 新增字段默认值
        self.name = "ScannerConfigData"
        self.hideFlags = 0

        # 统一环境配置默认值
        self.env_config = {
            "termination": {
                "target_scan_ratio": 0.25,
                "max_collision_count": 6,
                "max_elapsed_time_sec": 300.0,
                "stagnation_timeout_sec": 30.0,
                "out_of_range_reset_enabled": True,
                "out_of_range_continuous_count": 12
            },
            "battery": {
                "low_threshold": 3.5,
                "optimal_min": 3.7,
                "optimal_max": 4.1
            },
            "base_rewards": {
                "scan_reward": 10.0,
                "out_of_range_penalty": -30.0,
                "battery_low_penalty": -10.0,
                "battery_optimal_reward": 2.0,
                "collision_penalty": -50.0,
                "step_penalty": -0.1
            }
        }
        self.paper_benchmark = {
            "seeds": [20260413, 20260414, 20260415],
            "eval_episodes_per_seed": 10,
            "termination": {
                "target_scan_ratio": 0.25,
                "max_collision_count": 6,
                "max_elapsed_time_sec": 300.0,
                "stagnation_timeout_sec": 30.0,
                "out_of_range_reset_enabled": True,
                "out_of_range_continuous_count": 12,
            },
            "random_apf": {
                "weight_min": 0.5,
                "weight_max": 5.0,
                "sampling_mode": "uniform",
            },
        }

    def parse_json_data(self, json_data: Dict[str, Any]) -> None:
        """从JSON字典解析数据到对象属性"""
        # 解析基础参数
        self.repulsionCoefficient = self._get_float(json_data, 'repulsionCoefficient', 2.0)
        self.entropyCoefficient = self._get_float(json_data, 'entropyCoefficient', 2.0)
        self.distanceCoefficient = self._get_float(json_data, 'distanceCoefficient', 2.0)
        self.leaderRangeCoefficient = self._get_float(json_data, 'leaderRangeCoefficient', 1.5)
        self.directionRetentionCoefficient = self._get_float(json_data, 'directionRetentionCoefficient', 0.5)
        self.groundRepulsionCoefficient = self._get_float(json_data, 'groundRepulsionCoefficient', 2.0)

        self.updateInterval = self._get_float(json_data, 'updateInterval', 0.5)

        self.moveSpeed = self._get_float(json_data, 'moveSpeed', 1.0)
        self.rotationSpeed = self._get_float(json_data, 'rotationSpeed', 120.0)
        self.scanRadius = self._get_float(json_data, 'scanRadius', 2.0)
        self.altitude = self._get_float(json_data, 'altitude', 2.0)

        self.maxRepulsionDistance = self._get_float(json_data, 'maxRepulsionDistance', 3.0)
        self.minSafeDistance = self._get_float(json_data, 'minSafeDistance', 1.0)

        self.avoidRevisits = json_data.get('avoidRevisits', True)
        self.targetSearchRange = self._get_float(json_data, 'targetSearchRange', 20.0)
        self.revisitCooldown = self._get_float(json_data, 'revisitCooldown', 10.0)

        # 解析统一环境配置
        self.env_config = json_data.get('env_config', self.env_config)
        self.paper_benchmark = json_data.get('paper_benchmark', self.paper_benchmark)

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
            'name': self.name,
            'hideFlags': self.hideFlags,
            'env_config': self.env_config,
            'paper_benchmark': self.paper_benchmark
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

        return True

    def copy(self):
        """创建对象的深拷贝"""
        new_data = ScannerConfigData()
        # 深拷贝env_config（避免浅拷贝导致的引用问题）
        new_data.env_config = {k: v.copy() if isinstance(v, dict) else v for k, v in self.env_config.items()}
        new_data.paper_benchmark = {
            k: v.copy() if isinstance(v, dict) else list(v) if isinstance(v, list) else v
            for k, v in self.paper_benchmark.items()
        }
        # 拷贝其他属性
        new_data.__dict__.update({
            k: v for k, v in self.__dict__.items() if k not in {'env_config', 'paper_benchmark'}
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
        支持两种格式：
        - 新格式（system_config.json）：从 algorithm 和 environment 节读取
        - 旧格式（apf_algorithm_config.json）：直接从顶层读取
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 新格式：system_config.json 包含 algorithm 和 environment 顶层键
            if 'algorithm' in data and 'environment' in data:
                self.parse_json_data(data['algorithm'])
                self.env_config = data.get('environment', self.env_config)
                self.paper_benchmark = data.get('paper_benchmark', self.paper_benchmark)
            # 旧格式：apf_algorithm_config.json 直接包含 APF 参数和 env_config
            elif 'repulsionCoefficient' in data or 'env_config' in data:
                self.parse_json_data(data)
            else:
                print(f"未识别的配置文件格式: {config_file}")
                self._set_default_values()
        except Exception as e:
            print(f"加载配置文件失败: {str(e)}")
            # 加载失败时保持默认值
            self._set_default_values()

    def __repr__(self) -> str:
        return f"ScannerConfigData(name={self.name}, ScanRadius={self.scanRadius}, Altitude={self.altitude})"
