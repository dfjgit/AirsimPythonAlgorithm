import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
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

    def test_mixed_legacy_shape_only_persists_training_block_to_drones_config_file(self):
        (self.root / "drones_config.json").write_text(
            json.dumps(
                {
                    "drones": {
                        "STALE_DRONE": {"enabled": False, "type": "real"},
                    },
                    "training": {
                        "dqn": {"use_all_drones": False, "drone_list": ["UAV1"]},
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        config = DronesConfig(
            config_file=str(self.root / "drones_config.json"),
            system_config_file=str(self.root / "system_config.json"),
        )
        config.get_drone_info("UAV1")["isCrazyflieMirror"] = True
        config.save_config()

        saved_training = json.loads((self.root / "drones_config.json").read_text(encoding="utf-8"))
        saved_system = json.loads((self.root / "system_config.json").read_text(encoding="utf-8"))
        self.assertEqual(set(saved_training.keys()), {"training"})
        self.assertEqual(saved_training["training"]["dqn"]["drone_list"], ["UAV1"])
        self.assertTrue(saved_system["drones"]["UAV1"]["isCrazyflieMirror"])

    def test_get_drone_type_defaults_to_virtual_when_type_is_missing(self):
        config = DronesConfig(
            config_file=str(self.root / "drones_config.json"),
            system_config_file=str(self.root / "system_config.json"),
        )
        config.system_config.config["drones"]["UAV_NO_TYPE"] = {"enabled": True}

        self.assertEqual(config.get_drone_type("UAV_NO_TYPE"), "virtual")

    def test_get_training_drones_prints_warnings_for_missing_and_disabled(self):
        (self.root / "drones_config.json").write_text(
            json.dumps(
                {
                    "training": {
                        "dqn": {
                            "use_all_drones": False,
                            "drone_list": ["UAV1", "UAV2", "UAV_MISSING"],
                        }
                    }
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        config = DronesConfig(
            config_file=str(self.root / "drones_config.json"),
            system_config_file=str(self.root / "system_config.json"),
        )

        output = StringIO()
        with redirect_stdout(output):
            training_drones = config.get_training_drones("dqn")

        self.assertEqual(training_drones, ["UAV1"])
        self.assertIn("Warning: drone UAV2 is disabled and will be skipped", output.getvalue())
        self.assertIn(
            "Warning: drone UAV_MISSING is not present in shared system inventory config",
            output.getvalue(),
        )

    def test_empty_drone_entry_defaults_to_enabled_and_is_selected(self):
        config = DronesConfig(
            config_file=str(self.root / "drones_config.json"),
            system_config_file=str(self.root / "system_config.json"),
        )
        config.system_config.config["drones"]["UAV_EMPTY"] = {}
        config.config["training"]["dqn"] = {"use_all_drones": False, "drone_list": ["UAV_EMPTY"]}

        self.assertTrue(config.is_enabled("UAV_EMPTY"))
        self.assertEqual(config.get_training_drones("dqn"), ["UAV_EMPTY"])

    def test_config_file_override_without_system_config_uses_mixed_legacy_inventory_and_persistence(self):
        legacy_path = self.root / "legacy_override.json"
        legacy_path.write_text(
            json.dumps(
                {
                    "drones": {
                        "LEGACY_ONLY_1": {"enabled": True, "type": "real", "isCrazyflieMirror": False},
                        "LEGACY_ONLY_2": {"enabled": False, "type": "virtual", "isCrazyflieMirror": False},
                    },
                    "training": {
                        "dqn": {"use_all_drones": False, "drone_list": ["LEGACY_ONLY_1"]},
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        config = DronesConfig(config_file=str(legacy_path))

        self.assertEqual(config.get_all_drones(), ["LEGACY_ONLY_1", "LEGACY_ONLY_2"])
        self.assertEqual(config.get_enabled_drones(), ["LEGACY_ONLY_1"])
        self.assertEqual(config.get_training_drones("dqn"), ["LEGACY_ONLY_1"])
        self.assertEqual(config.get_drone_info("LEGACY_ONLY_1")["type"], "real")
        self.assertEqual(config.system_config.config_file, legacy_path)

        config.get_drone_info("LEGACY_ONLY_1")["isCrazyflieMirror"] = True
        config.save_config()

        saved_override = json.loads(legacy_path.read_text(encoding="utf-8"))
        self.assertTrue(saved_override["drones"]["LEGACY_ONLY_1"]["isCrazyflieMirror"])
        self.assertEqual(saved_override["training"]["dqn"]["drone_list"], ["LEGACY_ONLY_1"])


if __name__ == "__main__":
    unittest.main()
