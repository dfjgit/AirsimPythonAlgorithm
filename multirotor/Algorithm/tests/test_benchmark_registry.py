import json
import os
import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmark_registry import (
    load_benchmark_registry,
    recommend_family_memberships,
    resolve_algorithm_registration,
)


class BenchmarkRegistryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / "_tmp_benchmark_registry"
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.root / "benchmark_registry.json"

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_registry(self, payload):
        self.registry_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def test_load_registry_validates_known_algorithms(self):
        self._write_registry(
            {
                "registry_version": 1,
                "families": [
                    {
                        "family_id": "apf_family",
                        "display_name": "APF Family",
                        "analysis_template": "apf_common",
                        "description": "APF variants",
                        "enabled": True,
                    },
                    {
                        "family_id": "learning_family",
                        "display_name": "Learning Family",
                        "analysis_template": "learning_common",
                        "description": "Trainable variants",
                        "enabled": True,
                    },
                ],
                "algorithms": [
                    {
                        "algorithm_type": "ddpg_apf",
                        "display_name": "DDPG+APF",
                        "primary_family": "apf_family",
                        "family_memberships": ["apf_family", "learning_family"],
                        "comparison_profiles": ["global_benchmark"],
                        "is_trainable": True,
                        "control_mode": "apf",
                        "enabled": True,
                    }
                ],
            }
        )

        registry = load_benchmark_registry(self.registry_path)

        self.assertIn("apf_family", registry.families)
        self.assertIn("ddpg_apf", registry.algorithms)
        self.assertEqual(
            registry.algorithms["ddpg_apf"].family_memberships,
            ["apf_family", "learning_family"],
        )

    def test_unknown_algorithm_gets_global_benchmark_fallback_and_recommendation(self):
        self._write_registry(
            {
                "registry_version": 1,
                "families": [
                    {
                        "family_id": "apf_family",
                        "display_name": "APF Family",
                        "analysis_template": "apf_common",
                        "description": "APF variants",
                        "enabled": True,
                    },
                    {
                        "family_id": "learning_family",
                        "display_name": "Learning Family",
                        "analysis_template": "learning_common",
                        "description": "Trainable variants",
                        "enabled": True,
                    },
                ],
                "algorithms": [],
            }
        )

        registry = load_benchmark_registry(self.registry_path)
        resolved = resolve_algorithm_registration(
            "ppo_scan",
            registry,
            control_mode="dqn",
            apf_weight_mode="fixed",
            is_trainable=True,
        )

        self.assertEqual(resolved.algorithm_type, "ppo_scan")
        self.assertEqual(resolved.comparison_profiles, ["global_benchmark"])
        self.assertEqual(resolved.family_memberships, [])
        self.assertEqual(resolved.recommended_family_memberships, ["learning_family"])

    def test_apf_trainable_algorithm_gets_dual_family_recommendation(self):
        self.assertEqual(
            recommend_family_memberships(
                control_mode="apf",
                apf_weight_mode="learned",
                is_trainable=True,
            ),
            ["apf_family", "learning_family"],
        )

    def test_unknown_family_reference_fails_validation(self):
        self._write_registry(
            {
                "registry_version": 1,
                "families": [
                    {
                        "family_id": "learning_family",
                        "display_name": "Learning Family",
                        "analysis_template": "learning_common",
                        "description": "Trainable variants",
                        "enabled": True,
                    }
                ],
                "algorithms": [
                    {
                        "algorithm_type": "random_apf",
                        "display_name": "Random APF",
                        "primary_family": "apf_family",
                        "family_memberships": ["apf_family"],
                        "comparison_profiles": ["global_benchmark"],
                        "is_trainable": False,
                        "control_mode": "apf",
                        "enabled": True,
                    }
                ],
            }
        )

        with self.assertRaises(ValueError):
            load_benchmark_registry(self.registry_path)


if __name__ == "__main__":
    unittest.main()
