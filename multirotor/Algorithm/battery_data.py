import json
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from Algorithm.drones_config import DronesConfig
from Algorithm.scanner_config_data import ScannerConfigData


class BatteryStatus(Enum):
    """Battery status bands shared by DDPG and DQN."""

    NORMAL = "normal"      # 4.0V - 4.2V
    WARNING = "warning"    # 3.7V - 4.0V
    LOW = "low"            # 3.5V - 3.7V
    CRITICAL = "critical"  # 3.2V - 3.5V
    EMPTY = "empty"        # < 3.2V


@dataclass
class BatteryInfo:
    voltage: float = 4.2
    initial_voltage: float = 4.2
    consumption_rate: float = 0.0020
    last_update_time: Optional[float] = None
    status: BatteryStatus = BatteryStatus.NORMAL
    crazyflieMirror: bool = False

    def __post_init__(self):
        if self.last_update_time is None:
            self.last_update_time = time.time()
        self._update_status()

    def _update_status(self):
        if self.voltage >= 4.0:
            self.status = BatteryStatus.NORMAL
        elif self.voltage >= 3.7:
            self.status = BatteryStatus.WARNING
        elif self.voltage >= 3.5:
            self.status = BatteryStatus.LOW
        elif self.voltage >= 3.2:
            self.status = BatteryStatus.CRITICAL
        else:
            self.status = BatteryStatus.EMPTY

    def update_voltage(self, new_voltage: float) -> None:
        # Keep the simulated battery above the hard-empty cutoff used by training.
        self.voltage = max(3.2, float(new_voltage))
        self.last_update_time = time.time()
        self._update_status()

    def get_remaining_percentage(self) -> float:
        # Map 4.2V -> 100%, 3.2V -> 0% for Crazyflie-oriented experiments.
        return max(0.0, min(100.0, (self.voltage - 3.2) / (4.2 - 3.2) * 100.0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "voltage": self.voltage,
            "initial_voltage": self.initial_voltage,
            "consumption_rate": self.consumption_rate,
            "last_update_time": self.last_update_time,
            "status": self.status.value,
            "crazyflieMirror": self.crazyflieMirror,
            "remaining_percentage": self.get_remaining_percentage(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> "BatteryInfo":
        status_value = data_dict.get("status", BatteryStatus.NORMAL.value)
        status = BatteryStatus(status_value)
        return cls(
            voltage=float(data_dict.get("voltage", 4.2)),
            initial_voltage=float(data_dict.get("initial_voltage", 4.2)),
            consumption_rate=float(data_dict.get("consumption_rate", 0.0020)),
            last_update_time=data_dict.get("last_update_time", time.time()),
            status=status,
            crazyflieMirror=bool(data_dict.get("crazyflieMirror", False)),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "BatteryInfo":
        return cls.from_dict(json.loads(json_str))


class BatteryManager:
    """Manage battery data for all drones."""

    def __init__(
        self,
        configData: Optional[ScannerConfigData] = None,
        drones_config: Optional[DronesConfig] = None,
    ):
        self.battery_data: Dict[str, BatteryInfo] = {}
        self.lock = threading.Lock()

        if drones_config is None:
            drones_config = DronesConfig()

        for drone_name in drones_config.get_all_drones():
            is_crazyflie = drones_config.is_crazyflie_mirror(drone_name)
            self.add_drone(drone_name, 4.2, 0.0020, is_crazyflie)

    def add_drone(
        self,
        drone_name: str,
        initial_voltage: float = 4.2,
        consumption_rate: float = 0.0020,
        crazyflie_mirror: bool = False,
    ) -> BatteryInfo:
        with self.lock:
            battery_info = BatteryInfo(
                voltage=initial_voltage,
                initial_voltage=initial_voltage,
                consumption_rate=consumption_rate,
                crazyflieMirror=crazyflie_mirror,
            )
            self.battery_data[drone_name] = battery_info
            return battery_info

    def get_voltage(self, drone_name: str) -> float:
        with self.lock:
            if drone_name in self.battery_data:
                return self.battery_data[drone_name].voltage
            logging.warning(f"无人机 {drone_name} 的电量数据不存在")
            return 4.2

    def update_voltage(
        self,
        drone_name: str,
        action_intensity: float = 0.0,
        real_battery_voltage: Optional[float] = None,
    ) -> float:
        with self.lock:
            if drone_name not in self.battery_data:
                logging.warning(f"无人机 {drone_name} 的电量数据不存在，初始化电量数据")
                self.add_drone(drone_name)

            battery_info = self.battery_data[drone_name]

            if battery_info.crazyflieMirror:
                if real_battery_voltage is None:
                    real_battery_voltage = 4.2
                new_voltage = max(3.2, float(real_battery_voltage))
                battery_info.update_voltage(new_voltage)
                logging.debug(
                    f"实体无人机 {drone_name} 电量更新: {battery_info.voltage:.2f}V "
                    f"(来自实体无人机数据, 状态: {battery_info.status.value})"
                )
                return new_voltage

            current_time = time.time()
            time_elapsed = current_time - float(battery_info.last_update_time or current_time)
            base_consumption = battery_info.consumption_rate * time_elapsed
            action_consumption = float(action_intensity) * 0.0010 * time_elapsed
            total_consumption = base_consumption + action_consumption
            new_voltage = max(3.2, battery_info.voltage - total_consumption)
            battery_info.update_voltage(new_voltage)

            logging.debug(
                f"虚拟无人机 {drone_name} 电量更新: {battery_info.voltage:.3f}V "
                f"(消耗 {total_consumption:.4f}V, 时间 {time_elapsed:.2f}s, "
                f"动作强度 {action_intensity:.2f}, 状态 {battery_info.status.value})"
            )
            return new_voltage

    def reset_voltage(self, drone_name: str) -> float:
        with self.lock:
            if drone_name in self.battery_data:
                initial_voltage = self.battery_data[drone_name].initial_voltage
                self.battery_data[drone_name].update_voltage(initial_voltage)
                logging.info(f"无人机 {drone_name} 电量已重置为: {initial_voltage:.2f}V")
                return initial_voltage

            logging.warning(f"无人机 {drone_name} 的电量数据不存在，初始化电量数据")
            return self.add_drone(drone_name).voltage

    def get_all_battery_data(self) -> Dict[str, Dict[str, Any]]:
        with self.lock:
            return {
                drone_name: battery_info.to_dict()
                for drone_name, battery_info in self.battery_data.items()
            }

    def set_consumption_rate(self, drone_name: str, rate: float) -> None:
        with self.lock:
            if drone_name in self.battery_data:
                self.battery_data[drone_name].consumption_rate = float(rate)
                logging.info(f"无人机 {drone_name} 电量消耗率设置为: {rate:.4f}V/秒")
            else:
                logging.warning(f"无人机 {drone_name} 的电量数据不存在，初始化电量数据")
                self.add_drone(drone_name, consumption_rate=float(rate))

    def get_battery_info(self, drone_name: str) -> Optional[BatteryInfo]:
        with self.lock:
            return self.battery_data.get(drone_name)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return self.get_all_battery_data()

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Dict[str, Any]]) -> "BatteryManager":
        manager = cls()
        for drone_name, battery_dict in data_dict.items():
            manager.battery_data[drone_name] = BatteryInfo.from_dict(battery_dict)
        return manager

    @classmethod
    def from_json(cls, json_str: str) -> "BatteryManager":
        return cls.from_dict(json.loads(json_str))
