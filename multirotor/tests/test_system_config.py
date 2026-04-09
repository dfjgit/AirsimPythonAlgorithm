import json
import tempfile
import unittest
from pathlib import Path

from multirotor.Algorithm.system_config import SystemConfig, load_environment_rules


class SystemConfigTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def _write_json(self, relative_path: str, payload: dict) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def test_prefers_new_system_config_when_present(self):
        system_path = self._write_json(
            "system_config.json",
            {
                "drones": {
                    "UAV1": {"enabled": True, "type": "virtual", "isCrazyflieMirror": False},
                    "CF1": {"enabled": True, "type": "physical", "isCrazyflieMirror": True},
                },
                "environment": {
                    "termination": {"target_scan_ratio": 0.25},
                    "battery": {"low_threshold": 3.5, "optimal_min": 3.7, "optimal_max": 4.1},
                },
            },
        )

        config = SystemConfig(config_file=str(system_path))

        self.assertEqual(config.get_all_drones(), ["UAV1", "CF1"])
        self.assertEqual(config.get_enabled_drones(), ["UAV1", "CF1"])
        self.assertTrue(config.is_crazyflie_mirror("CF1"))
        self.assertEqual(config.get_environment_rules()["termination"]["target_scan_ratio"], 0.25)

    def test_falls_back_to_legacy_files_when_system_config_is_missing(self):
        drones_path = self._write_json(
            "drones_config.json",
            {
                "drones": {
                    "UAV1": {"enabled": True, "type": "virtual", "isCrazyflieMirror": False}
                }
            },
        )
        apf_path = self._write_json(
            "apf_algorithm_config.json",
            {
                "repulsionCoefficient": 2.0,
                "env_config": {
                    "termination": {"target_scan_ratio": 0.4},
                    "battery": {"low_threshold": 3.4, "optimal_min": 3.7, "optimal_max": 4.1},
                },
            },
        )

        config = SystemConfig(
            config_file=str(self.root / "missing_system_config.json"),
            legacy_drones_file=str(drones_path),
            legacy_apf_file=str(apf_path),
        )

        self.assertEqual(config.get_all_drones(), ["UAV1"])
        self.assertEqual(load_environment_rules(config), config.get_environment_rules())
        self.assertEqual(config.get_environment_rules()["termination"]["target_scan_ratio"], 0.4)


if __name__ == "__main__":
    unittest.main()
