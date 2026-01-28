import json
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from Algorithm.scanner_config_data import ScannerConfigData
from Algorithm.drones_config import DronesConfig


class BatteryStatus(Enum):
    """电池状态枚举"""
    NORMAL = "normal"        # 正常 (4.0V - 4.2V)
    WARNING = "warning"      # 警告 (3.7V - 4.0V)
    LOW = "low"              # 低电量 (3.5V - 3.7V)
    CRITICAL = "critical"    # 严重 (3.0V - 3.5V)
    EMPTY = "empty"          # 耗尽 (< 3.0V)


@dataclass
class BatteryInfo:
    """无人机电池信息数据类"""
    voltage: float = 4.2                    # 当前电压 (V)
    initial_voltage: float = 4.2           # 初始电压 (V)
    consumption_rate: float = 0.0020       # 基础消耗率 (V/秒) - 基于Crazyflie实际续航
    last_update_time: float = None         # 最后更新时间戳
    status: BatteryStatus = BatteryStatus.NORMAL  # 电池状态
    crazyflieMirror: bool = False            # Crazyflie镜像标志
    
    def __post_init__(self):
        """初始化后处理"""
        if self.last_update_time is None:
            self.last_update_time = time.time()
        self._update_status()
    
    def _update_status(self):
        """根据电压更新电池状态"""
        if self.voltage >= 4.0:
            self.status = BatteryStatus.NORMAL
        elif self.voltage >= 3.7:
            self.status = BatteryStatus.WARNING
        elif self.voltage >= 3.5:
            self.status = BatteryStatus.LOW
        elif self.voltage >= 3.0:
            self.status = BatteryStatus.CRITICAL
        else:
            self.status = BatteryStatus.EMPTY
    
    def update_voltage(self, new_voltage: float) -> None:
        """更新电压并重新计算状态"""
        self.voltage = max(3.0, new_voltage)  # 不低于3.0V
        self.last_update_time = time.time()
        self._update_status()
    
    def get_remaining_percentage(self) -> float:
        """获取剩余电量百分比 (4.2V为100%, 3.0V为0%)"""
        return max(0.0, min(100.0, (self.voltage - 3.0) / (4.2 - 3.0) * 100))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典以便序列化"""
        return {
            'voltage': self.voltage,
            'initial_voltage': self.initial_voltage,
            'consumption_rate': self.consumption_rate,
            'last_update_time': self.last_update_time,
            'status': self.status.value,
            'crazyflieMirror': self.crazyflieMirror,
            'remaining_percentage': self.get_remaining_percentage()
        }
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> 'BatteryInfo':
        """从字典创建BatteryInfo实例"""
        # 处理状态枚举
        status_value = data_dict.get('status', 'normal')
        status = BatteryStatus(status_value)
        
        return cls(
            voltage=data_dict.get('voltage', 4.2),
            initial_voltage=data_dict.get('initial_voltage', 4.2),
            consumption_rate=data_dict.get('consumption_rate', 0.01),
            last_update_time=data_dict.get('last_update_time', time.time()),
            status=status,
            crazyflieMirror=data_dict.get('crazyflieMirror', False)
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BatteryInfo':
        """从JSON字符串创建BatteryInfo实例"""
        data_dict = json.loads(json_str)
        return cls.from_dict(data_dict)


class BatteryManager:
    """电池管理器类，管理多无人机的电池数据"""
    
    def __init__(self, configData: ScannerConfigData, drones_config: Optional[DronesConfig] = None):
        """初始化电池管理器
        :param configData: APF算法配置（兼容参数）
        :param drones_config: 无人机配置，如果为None则自动创建
        """
        self.battery_data: Dict[str, BatteryInfo] = {}
        self.lock = threading.Lock()  # 线程安全锁
        
        # 使用新的DronesConfig加载无人机配置
        if drones_config is None:
            drones_config = DronesConfig()
        
        for drone_name in drones_config.get_all_drones():
            is_crazyflie = drones_config.is_crazyflie_mirror(drone_name)
            self.add_drone(drone_name, 4.2, 0.0020, is_crazyflie)
    
    def add_drone(self, drone_name: str, initial_voltage: float = 4.2, 
                  consumption_rate: float = 0.0020, crazyflie_mirror: bool = False) -> BatteryInfo:
        """添加无人机电池数据"""
        with self.lock:
            battery_info = BatteryInfo(
                voltage = initial_voltage,
                initial_voltage = initial_voltage,
                consumption_rate = consumption_rate,
                crazyflieMirror = crazyflie_mirror
            )
            self.battery_data[drone_name] = battery_info
            return battery_info
    
    def get_voltage(self, drone_name: str) -> float:
        """获取指定无人机的当前电压"""
        with self.lock:
            if drone_name in self.battery_data:
                return self.battery_data[drone_name].voltage
            else:
                logging.warning(f"无人机 {drone_name} 的电量数据不存在")
                return 4.2  # 返回默认电压
    
    def update_voltage(self, drone_name: str, action_intensity: float = 0.0, 
                      real_battery_voltage: Optional[float] = None) -> float:
        """更新指定无人机的电量消耗"""
        with self.lock:
            if drone_name not in self.battery_data:
                logging.warning(f"无人机 {drone_name} 的电量数据不存在，初始化电量数据")
                self.add_drone(drone_name)
            
            battery_info = self.battery_data[drone_name]
            
            # 如果是实体无人机镜像，直接使用实体无人机的电池数据
            if battery_info.crazyflieMirror:
                if real_battery_voltage is None:
                    real_battery_voltage = 4.2
                # 使用实体无人机的真实电池电压
                new_voltage = max(3.0, real_battery_voltage)
                battery_info.update_voltage(new_voltage)
                logging.debug(f"实体无人机 {drone_name} 电量更新: {battery_info.voltage:.2f}V "
                             f"(来自实体无人机数据, 状态: {battery_info.status.value})")
                return new_voltage
            
            # 虚拟无人机：使用模拟电量消耗（优化为匹配Crazyflie真实耗电特性）
            current_time = time.time()
            time_elapsed = current_time - battery_info.last_update_time
            
            # 基础消耗（悬停状态）
            # Crazyflie 2.x: 240mAh电池，悬停功耗约1A，全电压差1.2V，理论续航约10分钟
            # 优化公式：0.0020V/秒 = 1.2V / (10分钟 * 60秒) = 0.002V/s
            base_consumption = battery_info.consumption_rate * time_elapsed
            
            # 动作强度影响（高速飞行、急转等会增加功耗，最多增加50%）
            # 全速飞行时功耗约1.5A，相当于悬停的1.5倍
            action_consumption = action_intensity * 0.0010 * time_elapsed  # 最大额外50%消耗
            
            # 总消耗（悬停10分钟，激烈飞行约6-7分钟）
            total_consumption = base_consumption + action_consumption
            
            # 更新电压（不低于3.0V）
            new_voltage = max(3.0, battery_info.voltage - total_consumption)
            battery_info.update_voltage(new_voltage)
            
            logging.debug(f"虚拟无人机 {drone_name} 电量更新: {battery_info.voltage:.3f}V "
                         f"(消耗: {total_consumption:.4f}V, 时间: {time_elapsed:.2f}s, "
                         f"动作强度: {action_intensity:.2f}, 状态: {battery_info.status.value})")
            return new_voltage
    
    def reset_voltage(self, drone_name: str) -> float:
        """重置指定无人机的电量为初始值"""
        with self.lock:
            if drone_name in self.battery_data:
                initial_voltage = self.battery_data[drone_name].initial_voltage
                self.battery_data[drone_name].update_voltage(initial_voltage)
                logging.info(f"无人机 {drone_name} 电量已重置为: {initial_voltage:.2f}V")
                return initial_voltage
            else:
                logging.warning(f"无人机 {drone_name} 的电量数据不存在，初始化电量数据")
                return self.add_drone(drone_name).voltage
    
    def get_all_battery_data(self) -> Dict[str, Dict[str, Any]]:
        """获取所有无人机的电量数据（转换为字典格式）"""
        with self.lock:
            return {
                drone_name: battery_info.to_dict()
                for drone_name, battery_info in self.battery_data.items()
            }
    
    def set_consumption_rate(self, drone_name: str, rate: float) -> None:
        """设置指定无人机的电量消耗率"""
        with self.lock:
            if drone_name in self.battery_data:
                self.battery_data[drone_name].consumption_rate = rate
                logging.info(f"无人机 {drone_name} 电量消耗率设置为: {rate:.4f}V/秒")
            else:
                logging.warning(f"无人机 {drone_name} 的电量数据不存在，初始化电量数据")
                self.add_drone(drone_name, consumption_rate=rate)
    
    def get_battery_info(self, drone_name: str) -> Optional[BatteryInfo]:
        """获取指定无人机的完整电池信息"""
        with self.lock:
            return self.battery_data.get(drone_name)
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """转换为字典以便序列化"""
        return self.get_all_battery_data()
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data_dict: Dict[str, Dict[str, Any]]) -> 'BatteryManager':
        """从字典创建BatteryManager实例"""
        manager = cls()
        for drone_name, battery_dict in data_dict.items():
            manager.battery_data[drone_name] = BatteryInfo.from_dict(battery_dict)
        return manager
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BatteryManager':
        """从JSON字符串创建BatteryManager实例"""
        data_dict = json.loads(json_str)
        return cls.from_dict(data_dict)


# 导入线程模块
import threading
import logging