from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Dict, List, Optional

from .system_config import SystemConfig


class DronesConfig:
    def __init__(self, config_file: Optional[str] = None, system_config_file: Optional[str] = None):
        base_dir = Path(__file__).resolve().parent.parent
        self.config_file = Path(config_file) if config_file else base_dir / "drones_config.json"
        payload = self._load_config_payload()
        self._source_payload = deepcopy(payload)
        self._legacy_override_mode = False
        self.system_config = self._load_system_config(payload, system_config_file)
        self.config = self._load_training_config(payload)

    def _load_config_payload(self) -> dict:
        if not self.config_file.exists():
            raise FileNotFoundError(f"Drone training config file not found: {self.config_file}")
        with self.config_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}

    def _load_system_config(self, payload: dict, system_config_file: Optional[str]) -> SystemConfig:
        system_config = SystemConfig(config_file=system_config_file)
        if system_config_file is None:
            legacy_drones = payload.get("drones")
            if isinstance(legacy_drones, dict):
                self._legacy_override_mode = True
                system_config.config_file = self.config_file
                system_config.config["drones"] = deepcopy(legacy_drones)
        return system_config

    def _load_training_config(self, payload: dict) -> dict:
        training = payload.get("training", {}) if isinstance(payload, dict) else {}
        return {"training": training if isinstance(training, dict) else {}}

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
        return info.get("type", "virtual") if info else "unknown"

    def is_enabled(self, drone_name: str) -> bool:
        info = self.get_drone_info(drone_name)
        if info is None:
            return False
        return bool(info.get("enabled", True))

    def get_training_drones(self, algorithm: str = "dqn") -> List[str]:
        training_config = self.config.get("training", {}).get(algorithm, {})
        use_all = training_config.get("use_all_drones", False)
        if use_all:
            return self.get_enabled_drones()
        all_drones = set(self.get_all_drones())
        result: List[str] = []
        for drone in training_config.get("drone_list", []):
            if drone in all_drones:
                if self.is_enabled(drone):
                    result.append(drone)
                else:
                    print(f"Warning: drone {drone} is disabled and will be skipped")
            else:
                print(f"Warning: drone {drone} is not present in shared system inventory config")
        return result

    def get_drones_dict(self) -> dict:
        return {"drones": self.system_config.config.get("drones", {})}

    def save_config(self) -> None:
        training_payload = {"training": self.config.get("training", {})}
        if self._legacy_override_mode:
            merged_payload = {
                key: value
                for key, value in self._source_payload.items()
                if key not in {"drones", "training"}
            }
            merged_payload["drones"] = self.system_config.config.get("drones", {})
            merged_payload["training"] = training_payload["training"]
            with self.config_file.open("w", encoding="utf-8") as handle:
                json.dump(merged_payload, handle, indent=2, ensure_ascii=False)
            self._source_payload = deepcopy(merged_payload)
            return

        with self.config_file.open("w", encoding="utf-8") as handle:
            json.dump(training_payload, handle, indent=2, ensure_ascii=False)
        with self.system_config.config_file.open("w", encoding="utf-8") as handle:
            json.dump(self.system_config.config, handle, indent=2, ensure_ascii=False)
