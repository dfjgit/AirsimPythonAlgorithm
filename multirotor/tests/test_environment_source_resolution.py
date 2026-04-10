import json
import tempfile
import unittest
from pathlib import Path

from multirotor.Algorithm.scanner_config_data import ScannerConfigData
from multirotor.Algorithm.system_config import SystemConfig, overlay_environment_rules


class EnvironmentSourceResolutionTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_overlay_environment_rules_prefers_system_config(self):
        system_path = self.root / "system_config.json"
        system_path.write_text(
            json.dumps(
                {
                    "drones": {},
                    "environment": {
                        "termination": {"target_scan_ratio": 0.33},
                        "battery": {"low_threshold": 3.6, "optimal_min": 3.7, "optimal_max": 4.1},
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        config = ScannerConfigData()
        overlay_environment_rules(config, SystemConfig(config_file=str(system_path)).get_environment_rules())
        self.assertEqual(config.env_config["termination"]["target_scan_ratio"], 0.33)
        self.assertEqual(config.env_config["battery"]["low_threshold"], 3.6)


if __name__ == "__main__":
    unittest.main()
