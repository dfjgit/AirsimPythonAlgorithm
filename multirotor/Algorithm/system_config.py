from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Dict, Optional


class SystemConfig:
    def __init__(
        self,
        config_file: Optional[str] = None,
        legacy_drones_file: Optional[str] = None,
        legacy_apf_file: Optional[str] = None,
    ):
        base_dir = Path(__file__).resolve().parent.parent
        self.config_file = Path(config_file) if config_file else base_dir / "system_config.json"
        self.legacy_drones_file = Path(legacy_drones_file) if legacy_drones_file else base_dir / "drones_config.json"
        self.legacy_apf_file = Path(legacy_apf_file) if legacy_apf_file else base_dir / "apf_algorithm_config.json"
        self.config = self._load_config()

    def _load_json(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}

    def _invalid_config(self, reason: str) -> ValueError:
        return ValueError(f"Invalid system config structure: {self.config_file} ({reason})")

    def _load_config(self) -> Dict[str, Any]:
        if self.config_file.exists():
            data = self._load_json(self.config_file)
            if "drones" not in data or "environment" not in data:
                raise self._invalid_config("missing 'drones' or 'environment'")

            drones = data.get("drones")
            if not isinstance(drones, dict):
                raise self._invalid_config("drones must be a dict")

            environment = data.get("environment")
            if not isinstance(environment, dict):
                raise self._invalid_config("environment must be a dict")

            for drone_name, drone_info in drones.items():
                if not isinstance(drone_info, dict):
                    raise self._invalid_config(f"drone entry '{drone_name}' must be a dict")
            return data

        drones = {}
        if self.legacy_drones_file.exists():
            drones = self._load_json(self.legacy_drones_file).get("drones", {})

        environment = {}
        if self.legacy_apf_file.exists():
            environment = self._load_json(self.legacy_apf_file).get("env_config", {})

        return {"drones": drones, "environment": environment}

    def get_all_drones(self):
        return list(self.config.get("drones", {}).keys())

    def get_enabled_drones(self):
        drones = self.config.get("drones", {})
        return [name for name, info in drones.items() if info.get("enabled", True)]

    def get_drone_info(self, drone_name: str):
        return self.config.get("drones", {}).get(drone_name)

    def is_crazyflie_mirror(self, drone_name: str) -> bool:
        info = self.get_drone_info(drone_name)
        return bool(info and info.get("isCrazyflieMirror", False))

    def get_environment_rules(self) -> Dict[str, Any]:
        return deepcopy(self.config.get("environment", {}))


def load_environment_rules(source: SystemConfig) -> Dict[str, Any]:
    return source.get_environment_rules()
