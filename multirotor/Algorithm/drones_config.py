"""Helpers for loading and querying drone training configuration."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional


class DronesConfig:
    """Accessors around ``multirotor/drones_config.json``."""

    def __init__(self, config_file: Optional[str] = None):
        if config_file is None:
            default_path = Path(__file__).parent.parent / "drones_config.json"
            config_file = str(default_path)

        self.config_file = config_file
        self.config = self._load_config()

    def _load_config(self) -> dict:
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Drone config file not found: {self.config_file}")

        with open(self.config_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_all_drones(self) -> List[str]:
        return list(self.config.get("drones", {}).keys())

    def get_enabled_drones(self) -> List[str]:
        drones = self.config.get("drones", {})
        return [name for name, info in drones.items() if info.get("enabled", True)]

    def get_drone_info(self, drone_name: str) -> Optional[Dict]:
        return self.config.get("drones", {}).get(drone_name)

    def is_crazyflie_mirror(self, drone_name: str) -> bool:
        drone_info = self.get_drone_info(drone_name)
        if drone_info is None:
            return False
        return drone_info.get("isCrazyflieMirror", False)

    def get_drone_type(self, drone_name: str) -> str:
        drone_info = self.get_drone_info(drone_name)
        if drone_info is None:
            return "unknown"
        return drone_info.get("type", "virtual")

    def is_enabled(self, drone_name: str) -> bool:
        drone_info = self.get_drone_info(drone_name)
        if drone_info is None:
            return False
        return drone_info.get("enabled", True)

    def get_training_drones(self, algorithm: str = "dqn") -> List[str]:
        training_config = self.config.get("training", {}).get(algorithm, {})
        use_all = training_config.get("use_all_drones", False)

        if use_all:
            return self.get_enabled_drones()

        drone_list = training_config.get("drone_list", [])
        valid_drones: List[str] = []
        for drone in drone_list:
            if drone in self.get_all_drones():
                if self.is_enabled(drone):
                    valid_drones.append(drone)
                else:
                    print(f"Warning: drone {drone} is disabled and will be skipped")
            else:
                print(f"Warning: drone {drone} is not present in drones_config.json")
        return valid_drones

    def get_drones_dict(self) -> dict:
        return {"drones": self.config.get("drones", {})}

    def save_config(self) -> None:
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def __str__(self) -> str:
        enabled = self.get_enabled_drones()
        return (
            f"DronesConfig(total={len(self.get_all_drones())}, "
            f"enabled={len(enabled)}, drones={enabled})"
        )


if __name__ == "__main__":
    config = DronesConfig()
    print(config)
    print("All drones:", config.get_all_drones())
    print("Enabled drones:", config.get_enabled_drones())
    print("DQN drones:", config.get_training_drones("dqn"))
    print("DDPG drones:", config.get_training_drones("ddpg"))