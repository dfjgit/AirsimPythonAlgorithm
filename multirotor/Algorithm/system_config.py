from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


class SystemConfig:
    def __init__(self, config_file: Optional[str] = None):
        base_dir = Path(__file__).resolve().parent.parent
        self.config_file = Path(config_file) if config_file else base_dir / "system_config.json"
        self.config = self._load_config()

    def _load_json(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise self._invalid_config(path, "top-level JSON root must be a dict")
        return payload

    def _invalid_config(self, path: Path, reason: str) -> ValueError:
        return ValueError(f"Invalid system config structure: {path} ({reason})")

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_file.exists():
            raise FileNotFoundError(f"System config not found: {self.config_file}")

        data = self._load_json(self.config_file)

        if "drones" not in data:
            raise self._invalid_config(self.config_file, "missing 'drones'")

        if "environment" not in data:
            raise self._invalid_config(self.config_file, "missing 'environment'")

        drones = data.get("drones")
        if not isinstance(drones, dict):
            raise self._invalid_config(self.config_file, "drones must be a dict")

        environment = data.get("environment")
        if not isinstance(environment, dict):
            raise self._invalid_config(self.config_file, "environment must be a dict")

        for drone_name, drone_info in drones.items():
            if not isinstance(drone_info, dict):
                raise self._invalid_config(
                    self.config_file,
                    f"drone entry '{drone_name}' must be a dict",
                )

        return data

    @classmethod
    def from_legacy_sources(
        cls,
        legacy_drones_file: Optional[str] = None,
        legacy_apf_file: Optional[str] = None,
        shared_system_config_file: Optional[str] = None,
    ) -> "SystemConfig":
        instance = cls.__new__(cls)
        base_dir = Path(__file__).resolve().parent.parent
        instance.config_file = (
            Path(shared_system_config_file)
            if shared_system_config_file
            else base_dir / "system_config.json"
        )
        instance.legacy_drones_file = Path(legacy_drones_file) if legacy_drones_file else base_dir / "drones_config.json"
        instance.legacy_apf_file = Path(legacy_apf_file) if legacy_apf_file else base_dir / "apf_algorithm_config.json"
        if instance.config_file.exists():
            instance.config = instance._load_json(instance.config_file)
        else:
            instance.config = {"drones": {}, "environment": {}}

        drones_payload: Dict[str, Any] = {}
        if instance.legacy_drones_file.exists():
            drones_payload = instance._load_json(instance.legacy_drones_file)
            if isinstance(drones_payload.get("drones"), dict):
                instance.config["drones"] = deepcopy(drones_payload["drones"])
            if isinstance(drones_payload.get("training"), dict):
                instance.config["training"] = deepcopy(drones_payload["training"])

        if instance.legacy_apf_file.exists():
            apf_payload = instance._load_json(instance.legacy_apf_file)
            legacy_environment = None
            if isinstance(apf_payload.get("env_config"), dict):
                legacy_environment = apf_payload["env_config"]
            elif isinstance(apf_payload.get("environment"), dict):
                legacy_environment = apf_payload["environment"]
            if legacy_environment:
                current_environment = instance.config.get("environment", {})
                if not isinstance(current_environment, dict):
                    current_environment = {}
                instance.config["environment"] = _deep_merge_dict(current_environment, legacy_environment)

            if isinstance(apf_payload.get("algorithm"), dict):
                current_algorithm = instance.config.get("algorithm", {})
                if not isinstance(current_algorithm, dict):
                    current_algorithm = {}
                instance.config["algorithm"] = _deep_merge_dict(current_algorithm, apf_payload["algorithm"])

        if "drones" not in instance.config or not isinstance(instance.config.get("drones"), dict):
            instance.config["drones"] = {}
        if "environment" not in instance.config or not isinstance(instance.config.get("environment"), dict):
            instance.config["environment"] = {}
        return instance

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

    def get_algorithm_params(self) -> Dict[str, Any]:
        """返回 APF 算法参数节"""
        return deepcopy(self.config.get("algorithm", {}))


def overlay_environment_rules(scanner_config: Any, environment_rules: Optional[Dict[str, Any]]) -> Any:
    """Overlay shared termination/battery rules without replacing scanner reward defaults."""
    if not environment_rules:
        return scanner_config

    current_env_config = getattr(scanner_config, "env_config", {})
    if not isinstance(current_env_config, dict):
        current_env_config = {}

    merged_env_config = deepcopy(current_env_config)
    base_rewards = deepcopy(merged_env_config.get("base_rewards"))

    for section_name in ("termination", "battery"):
        section_rules = environment_rules.get(section_name)
        if not isinstance(section_rules, dict):
            continue
        section_config = merged_env_config.get(section_name, {})
        if not isinstance(section_config, dict):
            section_config = {}
        merged_section = deepcopy(section_config)
        merged_section.update(deepcopy(section_rules))
        merged_env_config[section_name] = merged_section

    if base_rewards is not None:
        merged_env_config["base_rewards"] = base_rewards

    scanner_config.env_config = merged_env_config
    return scanner_config


def load_environment_rules(source: SystemConfig) -> Dict[str, Any]:
    return source.get_environment_rules()
