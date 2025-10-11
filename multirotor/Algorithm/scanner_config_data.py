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

    # DQN配置
    dqn_enabled: bool
    dqn_learning_rate: float
    dqn_gamma: float
    dqn_epsilon: float
    dqn_epsilon_min: float
    dqn_epsilon_decay: float
    dqn_batch_size: int
    dqn_target_update: int
    dqn_memory_capacity: int
    dqn_model_save_interval: int

    # 奖励函数配置
    reward_exploration_weight: float
    reward_efficiency_weight: float
    reward_collision_penalty: float
    reward_boundary_penalty: float
    reward_energy_penalty: float
    reward_completion_reward: float

    # 学习环境配置
    learning_env_coefficient_step: float
    learning_env_coefficient_ranges: dict

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

        # DQN配置默认值
        self.dqn_enabled = False
        self.dqn_learning_rate = 0.001
        self.dqn_gamma = 0.99
        self.dqn_epsilon = 1.0
        self.dqn_epsilon_min = 0.01
        self.dqn_epsilon_decay = 0.995
        self.dqn_batch_size = 64
        self.dqn_target_update = 10
        self.dqn_memory_capacity = 10000
        self.dqn_model_save_interval = 1000

        # 奖励函数配置默认值
        self.reward_exploration_weight = 1.0
        self.reward_efficiency_weight = 0.5
        self.reward_collision_penalty = -5.0
        self.reward_boundary_penalty = -2.0
        self.reward_energy_penalty = -0.1
        self.reward_completion_reward = 100.0

        # 学习环境配置默认值
        self.learning_env_coefficient_step = 0.5
        self.learning_env_coefficient_ranges = {
            'repulsionCoefficient': (0.1, 10.0),
            'entropyCoefficient': (0.1, 10.0),
            'distanceCoefficient': (0.1, 10.0),
            'leaderRangeCoefficient': (0.1, 10.0),
            'directionRetentionCoefficient': (0.1, 10.0)
        }

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

        # 解析DQN配置
        dqn_config = json_data.get('dqn', {})
        self.dqn_enabled = dqn_config.get('enabled', False)
        self.dqn_learning_rate = self._get_float(dqn_config, 'learning_rate', 0.001)
        self.dqn_gamma = self._get_float(dqn_config, 'gamma', 0.99)
        self.dqn_epsilon = self._get_float(dqn_config, 'epsilon', 1.0)
        self.dqn_epsilon_min = self._get_float(dqn_config, 'epsilon_min', 0.01)
        self.dqn_epsilon_decay = self._get_float(dqn_config, 'epsilon_decay', 0.995)
        self.dqn_batch_size = self._get_int(dqn_config, 'batch_size', 64)
        self.dqn_target_update = self._get_int(dqn_config, 'target_update', 10)
        self.dqn_memory_capacity = self._get_int(dqn_config, 'memory_capacity', 10000)
        self.dqn_model_save_interval = self._get_int(dqn_config, 'model_save_interval', 1000)

        # 解析奖励函数配置
        reward_config = json_data.get('reward', {})
        self.reward_exploration_weight = self._get_float(reward_config, 'exploration_weight', 1.0)
        self.reward_efficiency_weight = self._get_float(reward_config, 'efficiency_weight', 0.5)
        self.reward_collision_penalty = self._get_float(reward_config, 'collision_penalty', -5.0)
        self.reward_boundary_penalty = self._get_float(reward_config, 'boundary_penalty', -2.0)
        self.reward_energy_penalty = self._get_float(reward_config, 'energy_penalty', -0.1)
        self.reward_completion_reward = self._get_float(reward_config, 'completion_reward', 100.0)

        # 解析学习环境配置
        learning_env_config = json_data.get('learning_env', {})
        self.learning_env_coefficient_step = self._get_float(learning_env_config, 'coefficient_step', 0.5)
        
        # 解析权重范围配置
        coefficient_ranges_config = learning_env_config.get('coefficient_ranges', {})
        self.learning_env_coefficient_ranges = {}
        for key, default_range in [
            ('repulsionCoefficient', (0.1, 10.0)),
            ('entropyCoefficient', (0.1, 10.0)),
            ('distanceCoefficient', (0.1, 10.0)),
            ('leaderRangeCoefficient', (0.1, 10.0)),
            ('directionRetentionCoefficient', (0.1, 10.0))
        ]:
            range_config = coefficient_ranges_config.get(key, default_range)
            if isinstance(range_config, list) and len(range_config) == 2:
                self.learning_env_coefficient_ranges[key] = (float(range_config[0]), float(range_config[1]))
            else:
                self.learning_env_coefficient_ranges[key] = default_range

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

            # DQN配置
            'dqn': {
                'enabled': self.dqn_enabled,
                'learning_rate': self.dqn_learning_rate,
                'gamma': self.dqn_gamma,
                'epsilon': self.dqn_epsilon,
                'epsilon_min': self.dqn_epsilon_min,
                'epsilon_decay': self.dqn_epsilon_decay,
                'batch_size': self.dqn_batch_size,
                'target_update': self.dqn_target_update,
                'memory_capacity': self.dqn_memory_capacity,
                'model_save_interval': self.dqn_model_save_interval
            },

            # 奖励函数配置
            'reward': {
                'exploration_weight': self.reward_exploration_weight,
                'efficiency_weight': self.reward_efficiency_weight,
                'collision_penalty': self.reward_collision_penalty,
                'boundary_penalty': self.reward_boundary_penalty,
                'energy_penalty': self.reward_energy_penalty,
                'completion_reward': self.reward_completion_reward
            },

            # 学习环境配置
            'learning_env': {
                'coefficient_step': self.learning_env_coefficient_step,
                'coefficient_ranges': {
                    key: list(range_tuple) for key, range_tuple in self.learning_env_coefficient_ranges.items()
                }
            }
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
