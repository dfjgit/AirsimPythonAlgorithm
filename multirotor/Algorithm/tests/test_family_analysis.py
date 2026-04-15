import json
import os
import shutil
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _test_temp_paths import make_temp_dir
from benchmark_registry import load_benchmark_registry
from family_analysis import _localized_text, generate_family_reports


class FamilyAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.root = make_temp_dir("family_analysis")
        self.registry_path = self.root / "benchmark_registry.json"
        self.output_dir = self.root / "out"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.eval_csv = self.root / "four_group_eval_episodes.csv"

        self.registry_path.write_text(
            json.dumps(
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
                            "algorithm_type": "fixed_apf",
                            "display_name": "Fixed APF",
                            "primary_family": "apf_family",
                            "family_memberships": ["apf_family"],
                            "comparison_profiles": ["global_benchmark"],
                            "is_trainable": False,
                            "control_mode": "apf",
                            "enabled": True,
                        },
                        {
                            "algorithm_type": "random_apf",
                            "display_name": "Random APF",
                            "primary_family": "apf_family",
                            "family_memberships": ["apf_family"],
                            "comparison_profiles": ["global_benchmark"],
                            "is_trainable": False,
                            "control_mode": "apf",
                            "enabled": True,
                        },
                        {
                            "algorithm_type": "ddpg_apf",
                            "display_name": "DDPG+APF",
                            "primary_family": "apf_family",
                            "family_memberships": ["apf_family", "learning_family"],
                            "comparison_profiles": ["global_benchmark"],
                            "is_trainable": True,
                            "control_mode": "apf",
                            "enabled": True,
                        },
                        {
                            "algorithm_type": "pure_dqn",
                            "display_name": "Pure DQN",
                            "primary_family": "learning_family",
                            "family_memberships": ["learning_family"],
                            "comparison_profiles": ["global_benchmark"],
                            "is_trainable": True,
                            "control_mode": "dqn",
                            "enabled": True,
                        },
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        pd.DataFrame(
            [
                {
                    "algorithm_type": "fixed_apf",
                    "seed": 1,
                    "episode": 1,
                    "success_flag": 1,
                    "final_global_scan_ratio": 25.0,
                    "final_global_avg_entropy": 70.0,
                    "scan_efficiency": 1.5,
                    "collision_count": 0,
                },
                {
                    "algorithm_type": "random_apf",
                    "seed": 1,
                    "episode": 1,
                    "success_flag": 0,
                    "final_global_scan_ratio": 20.0,
                    "final_global_avg_entropy": 75.0,
                    "scan_efficiency": 1.2,
                    "collision_count": 1,
                },
                {
                    "algorithm_type": "ddpg_apf",
                    "seed": 1,
                    "episode": 1,
                    "success_flag": 1,
                    "final_global_scan_ratio": 35.0,
                    "final_global_avg_entropy": 60.0,
                    "scan_efficiency": 2.1,
                    "collision_count": 0,
                },
                {
                    "algorithm_type": "pure_dqn",
                    "seed": 1,
                    "episode": 1,
                    "success_flag": 1,
                    "final_global_scan_ratio": 32.0,
                    "final_global_avg_entropy": 62.0,
                    "scan_efficiency": 1.9,
                    "collision_count": 0,
                },
            ]
        ).to_csv(self.eval_csv, index=False, encoding="utf-8-sig")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_localized_text_uses_chinese_when_ui_lang_is_zh(self):
        original_lang = os.environ.get("AIRSIM_UI_LANG")
        os.environ["AIRSIM_UI_LANG"] = "zh"
        try:
            self.assertEqual(_localized_text("中文", "English"), "中文")
        finally:
            if original_lang is None:
                os.environ.pop("AIRSIM_UI_LANG", None)
            else:
                os.environ["AIRSIM_UI_LANG"] = original_lang

    def test_localized_text_uses_english_by_default(self):
        original_lang = os.environ.get("AIRSIM_UI_LANG")
        os.environ.pop("AIRSIM_UI_LANG", None)
        try:
            self.assertEqual(_localized_text("中文", "English"), "English")
        finally:
            if original_lang is not None:
                os.environ["AIRSIM_UI_LANG"] = original_lang

    def test_generate_family_reports_filters_registered_members(self):
        registry = load_benchmark_registry(self.registry_path)

        generated = generate_family_reports(
            eval_csv_paths=[self.eval_csv],
            registry=registry,
            output_root=self.output_dir,
        )

        self.assertIn("apf_family", generated)
        self.assertIn("learning_family", generated)

        apf_summary = pd.read_csv(
            generated["apf_family"]["summary_csv"],
            encoding="utf-8-sig",
        )
        learning_summary = pd.read_csv(
            generated["learning_family"]["summary_csv"],
            encoding="utf-8-sig",
        )

        self.assertEqual(
            sorted(apf_summary["algorithm_type"].tolist()),
            ["ddpg_apf", "fixed_apf", "random_apf"],
        )
        self.assertEqual(
            sorted(learning_summary["algorithm_type"].tolist()),
            ["ddpg_apf", "pure_dqn"],
        )

    def test_unknown_family_reports_fail_fast(self):
        registry = load_benchmark_registry(self.registry_path)
        del registry.families["learning_family"]

        with self.assertRaises(ValueError):
            generate_family_reports(
                eval_csv_paths=[self.eval_csv],
                registry=registry,
                output_root=self.output_dir,
            )


if __name__ == "__main__":
    unittest.main()
