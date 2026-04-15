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

    def test_loads_drones_and_environment_from_system_config(self):
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

    def test_reads_algorithm_params(self):
        system_path = self._write_json(
            "system_config.json",
            {
                "drones": {"UAV1": {"enabled": True, "type": "virtual"}},
                "environment": {
                    "termination": {"target_scan_ratio": 0.25},
                    "battery": {"low_threshold": 3.5},
                },
                "algorithm": {
                    "repulsionCoefficient": 2.0,
                    "scanRadius": 5.0,
                    "moveSpeed": 1.0,
                },
            },
        )

        config = SystemConfig(config_file=str(system_path))
        algo = config.get_algorithm_params()

        self.assertEqual(algo["repulsionCoefficient"], 2.0)
        self.assertEqual(algo["scanRadius"], 5.0)
        self.assertEqual(algo["moveSpeed"], 1.0)

    def test_raises_file_not_found_when_missing(self):
        with self.assertRaises(FileNotFoundError):
            SystemConfig(config_file=str(self.root / "missing.json"))

    def test_environment_rules_are_isolated_from_internal_state(self):
        system_path = self._write_json(
            "system_config.json",
            {
                "drones": {
                    "UAV1": {"enabled": True, "type": "virtual", "isCrazyflieMirror": False},
                },
                "environment": {
                    "termination": {"target_scan_ratio": 0.25},
                    "battery": {"low_threshold": 3.5, "optimal_min": 3.7, "optimal_max": 4.1},
                },
            },
        )

        config = SystemConfig(config_file=str(system_path))
        rules = config.get_environment_rules()
        rules["termination"]["target_scan_ratio"] = 0.99
        rules["battery"]["low_threshold"] = 3.2

        self.assertEqual(config.get_environment_rules()["termination"]["target_scan_ratio"], 0.25)
        self.assertEqual(config.get_environment_rules()["battery"]["low_threshold"], 3.5)

    def test_rejects_malformed_system_config_shapes(self):
        cases = [
            (
                "drones_not_dict",
                {"drones": [], "environment": {}},
                "drones must be a dict",
            ),
            (
                "environment_not_dict",
                {"drones": {}, "environment": []},
                "environment must be a dict",
            ),
            (
                "drone_entry_not_dict",
                {"drones": {"UAV1": 1}, "environment": {}},
                "drone entry 'UAV1' must be a dict",
            ),
        ]

        for relative_path, payload, expected_message in cases:
            with self.subTest(case=relative_path):
                config_path = self._write_json(f"{relative_path}.json", payload)
                with self.assertRaisesRegex(ValueError, expected_message):
                    SystemConfig(config_file=str(config_path))

    def test_load_environment_rules_helper(self):
        system_path = self._write_json(
            "system_config.json",
            {
                "drones": {"UAV1": {"enabled": True}},
                "environment": {
                    "termination": {"target_scan_ratio": 0.3},
                    "battery": {"low_threshold": 3.4},
                },
            },
        )

        config = SystemConfig(config_file=str(system_path))
        rules = load_environment_rules(config)
        self.assertEqual(rules["termination"]["target_scan_ratio"], 0.3)


if __name__ == "__main__":
    unittest.main()
