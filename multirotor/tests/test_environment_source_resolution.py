import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MULTIROTOR_DIR = PROJECT_ROOT / "multirotor"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MULTIROTOR_DIR) not in sys.path:
    sys.path.insert(0, str(MULTIROTOR_DIR))

if "msgpackrpc" not in sys.modules:
    msgpackrpc_stub = types.ModuleType("msgpackrpc")
    msgpackrpc_stub.Client = object
    msgpackrpc_stub.Address = object
    sys.modules["msgpackrpc"] = msgpackrpc_stub
if "msgpack" not in sys.modules:
    sys.modules["msgpack"] = types.ModuleType("msgpack")

from multirotor.AlgorithmServer import MultiDroneAlgorithmServer
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

    def test_algorithm_server_custom_config_preserves_legacy_environment_override(self):
        apf_path = self.root / "custom_apf.json"
        apf_path.write_text(
            json.dumps(
                {
                    "repulsionCoefficient": 2.0,
                    "entropyCoefficient": 3.0,
                    "distanceCoefficient": 2.0,
                    "leaderRangeCoefficient": 3.0,
                    "directionRetentionCoefficient": 2.0,
                    "groundRepulsionCoefficient": 0.2,
                    "updateInterval": 0.2,
                    "moveSpeed": 2.0,
                    "rotationSpeed": 120.0,
                    "scanRadius": 5.0,
                    "altitude": 10.0,
                    "maxRepulsionDistance": 5.0,
                    "minSafeDistance": 2.0,
                    "avoidRevisits": True,
                    "targetSearchRange": 20.0,
                    "revisitCooldown": 60.0,
                    "env_config": {
                        "termination": {"target_scan_ratio": 0.42},
                        "battery": {"low_threshold": 3.55, "optimal_min": 3.75, "optimal_max": 4.05},
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        server = MultiDroneAlgorithmServer.__new__(MultiDroneAlgorithmServer)
        server.config_path = str(apf_path)
        server.system_config_path = None
        server._config_file_provided = True

        config = server._load_config()

        self.assertEqual(config.env_config["termination"]["target_scan_ratio"], 0.42)
        self.assertEqual(config.env_config["battery"]["low_threshold"], 3.55)


if __name__ == "__main__":
    unittest.main()
