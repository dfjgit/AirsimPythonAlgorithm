from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .system_config import SystemConfig


class DronesConfig:
    def __init__(self, config_file: Optional[str] = None):
        base_dir = Path(__file__).resolve().parent.parent
        self.config_file = Path(config_file) if config_file else base_dir / "system_config.json"
        self.system_config = SystemConfig(config_file=str(self.config_file))
        # 直接使用 system_config.config 作为内部数据，避免两个独立副本不同步
        self.config = self.system_config.config

    def get_all_drones(self) -> List[str]:
        return self.system_config.get_all_drones()

    def get_enabled_drones(self) -> List[str]:
        return self.system_config.get_enabled_drones()

    def get_drone_info(self, drone_name: str) -> Optional[Dict]:
        return self.system_config.get_drone_info(drone_name)

    def is_crazyflie_mirror(self, drone_name: str) -> bool:
        return self.system_config.is_crazyflie_mirror(drone_name)

    def get_drone_type(self, drone_name: str) -> str:
        info = self.get_drone_info(drone_name)
        return info.get("type", "unknown") if info else "unknown"

    def is_enabled(self, drone_name: str) -> bool:
        info = self.get_drone_info(drone_name)
        return bool(info and info.get("enabled", True))

    def get_training_drones(self, algorithm: str = "dqn") -> List[str]:
        training_config = self.config.get("training", {}).get(algorithm, {})
        use_all = training_config.get("use_all_drones", False)
        if use_all:
            return self.get_enabled_drones()
        result: List[str] = []
        for drone in training_config.get("drone_list", []):
            if drone in self.get_all_drones() and self.is_enabled(drone):
                result.append(drone)
        return result

    def get_drones_dict(self) -> dict:
        return {"drones": self.config.get("drones", {})}

    def save_config(self) -> None:
        """保存配置到 system_config.json（统一单文件）"""
        with self.config_file.open("w", encoding="utf-8") as handle:
            json.dump(self.config, handle, indent=2, ensure_ascii=False)
