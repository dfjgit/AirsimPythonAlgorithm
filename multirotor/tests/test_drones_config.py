import json
import tempfile
import unittest
from pathlib import Path

from multirotor.Algorithm.drones_config import DronesConfig


class DronesConfigFacadeTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        (self.root / "system_config.json").write_text(
            json.dumps(
                {
                    "drones": {
                        "UAV1": {"enabled": True, "type": "virtual", "isCrazyflieMirror": False},
                        "UAV2": {"enabled": False, "type": "virtual", "isCrazyflieMirror": False},
                    },
                    "environment": {"termination": {}, "battery": {}},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (self.root / "drones_config.json").write_text(
            json.dumps(
                {
                    "training": {
                        "dqn": {"use_all_drones": False, "drone_list": ["UAV1", "UAV2"]},
                        "ddpg": {"use_all_drones": True},
                    }
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_uses_system_config_for_inventory_and_training_config_for_selection(self):
        config = DronesConfig(
            config_file=str(self.root / "drones_config.json"),
            system_config_file=str(self.root / "system_config.json"),
        )

        self.assertEqual(config.get_all_drones(), ["UAV1", "UAV2"])
        self.assertEqual(config.get_enabled_drones(), ["UAV1"])
        self.assertEqual(config.get_training_drones("dqn"), ["UAV1"])
        self.assertEqual(config.get_training_drones("ddpg"), ["UAV1"])

    def test_save_config_persists_training_and_drone_updates_to_their_own_files(self):
        config = DronesConfig(
            config_file=str(self.root / "drones_config.json"),
            system_config_file=str(self.root / "system_config.json"),
        )

        config.get_drone_info("UAV1")["isCrazyflieMirror"] = True
        config.config["training"]["dqn"] = {"use_all_drones": True}
        config.save_config()

        saved_training = json.loads((self.root / "drones_config.json").read_text(encoding="utf-8"))
        saved_system = json.loads((self.root / "system_config.json").read_text(encoding="utf-8"))
        self.assertTrue(saved_training["training"]["dqn"]["use_all_drones"])
        self.assertTrue(saved_system["drones"]["UAV1"]["isCrazyflieMirror"])


if __name__ == "__main__":
    unittest.main()
