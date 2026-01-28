"""
无人机配置管理类
统一管理所有无人机的配置信息
"""
import json
import os
from typing import List, Dict, Optional
from pathlib import Path


class DronesConfig:
    """无人机配置管理类"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化无人机配置
        :param config_file: 配置文件路径，如果为None则使用默认路径
        """
        if config_file is None:
            # 默认路径：multirotor/drones_config.json
            default_path = Path(__file__).parent.parent / "drones_config.json"
            config_file = str(default_path)
        
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        """加载配置文件"""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"无人机配置文件不存在: {self.config_file}")
        
        with open(self.config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_all_drones(self) -> List[str]:
        """
        获取所有无人机名称列表
        :return: 无人机名称列表
        """
        return list(self.config.get('drones', {}).keys())
    
    def get_enabled_drones(self) -> List[str]:
        """
        获取所有启用的无人机名称列表
        :return: 启用的无人机名称列表
        """
        drones = self.config.get('drones', {})
        return [name for name, info in drones.items() if info.get('enabled', True)]
    
    def get_drone_info(self, drone_name: str) -> Optional[Dict]:
        """
        获取指定无人机的详细信息
        :param drone_name: 无人机名称
        :return: 无人机信息字典，如果不存在返回None
        """
        return self.config.get('drones', {}).get(drone_name)
    
    def is_crazyflie_mirror(self, drone_name: str) -> bool:
        """
        判断指定无人机是否为Crazyflie实体无人机镜像
        :param drone_name: 无人机名称
        :return: True=实体无人机，False=虚拟无人机
        """
        drone_info = self.get_drone_info(drone_name)
        if drone_info is None:
            return False
        return drone_info.get('isCrazyflieMirror', False)
    
    def get_drone_type(self, drone_name: str) -> str:
        """
        获取无人机类型
        :param drone_name: 无人机名称
        :return: 'virtual'=虚拟无人机, 'physical'=实体无人机
        """
        drone_info = self.get_drone_info(drone_name)
        if drone_info is None:
            return 'unknown'
        return drone_info.get('type', 'virtual')
    
    def is_enabled(self, drone_name: str) -> bool:
        """
        判断指定无人机是否启用
        :param drone_name: 无人机名称
        :return: True=启用，False=禁用
        """
        drone_info = self.get_drone_info(drone_name)
        if drone_info is None:
            return False
        return drone_info.get('enabled', True)
    
    def get_training_drones(self, algorithm: str = 'dqn') -> List[str]:
        """
        获取指定算法训练使用的无人机列表
        :param algorithm: 算法名称，'dqn' 或 'ddpg'
        :return: 无人机名称列表
        """
        training_config = self.config.get('training', {}).get(algorithm, {})
        use_all = training_config.get('use_all_drones', False)
        
        if use_all:
            # 使用所有启用的无人机
            return self.get_enabled_drones()
        else:
            # 使用指定的无人机列表
            drone_list = training_config.get('drone_list', [])
            # 验证指定的无人机是否存在且启用
            valid_drones = []
            for drone in drone_list:
                if drone in self.get_all_drones():
                    if self.is_enabled(drone):
                        valid_drones.append(drone)
                    else:
                        print(f"警告: 无人机 {drone} 已禁用，将被跳过")
                else:
                    print(f"警告: 无人机 {drone} 不存在于配置中")
            return valid_drones
    
    def save_config(self):
        """保存配置到文件"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def __str__(self):
        """字符串表示"""
        enabled = self.get_enabled_drones()
        return f"DronesConfig(total={len(self.get_all_drones())}, enabled={len(enabled)}, drones={enabled})"


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试 DronesConfig - 无人机配置管理")
    print("=" * 60)
    
    # 创建配置对象
    config = DronesConfig()
    
    print(f"\n配置对象: {config}")
    print(f"\n所有无人机: {config.get_all_drones()}")
    print(f"启用的无人机: {config.get_enabled_drones()}")
    
    # 测试单个无人机信息
    drone = "UAV1"
    print(f"\n{drone} 信息:")
    print(f"  类型: {config.get_drone_type(drone)}")
    print(f"  是否Crazyflie: {config.is_crazyflie_mirror(drone)}")
    print(f"  是否启用: {config.is_enabled(drone)}")
    
    # 测试训练无人机获取
    print(f"\nDQN训练无人机: {config.get_training_drones('dqn')}")
    print(f"DDPG训练无人机: {config.get_training_drones('ddpg')}")
    
    print("\n" + "=" * 60)
    print("[OK] 配置测试通过！")
    print("=" * 60)
