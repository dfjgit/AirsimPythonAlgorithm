import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _test_temp_paths import make_temp_dir
from apf_baseline_sim_runner import write_apf_baseline_outputs


class APFBaselineSimRunnerTests(unittest.TestCase):
    def setUp(self):
        self.root = make_temp_dir("apf_baseline_sim_runner")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_write_apf_baseline_outputs_emits_training_and_scan_csvs_for_both_apf_groups(self):
        rows = {
            "fixed_apf": [
                {
                    "algorithm_type": "fixed_apf",
                    "episode": 1,
                    "reward": 10.0,
                    "length": 20,
                    "global_scanned_cells": 80,
                    "scan_efficiency": 4.0,
                    "final_global_scan_ratio": 30.0,
                    "final_global_avg_entropy": 70.0,
                    "success_flag": 1,
                    "collision_count": 0,
                    "reset_reason": "timeout",
                    "episode_elapsed_time": 40.0,
                }
            ],
            "random_apf": [
                {
                    "algorithm_type": "random_apf",
                    "episode": 1,
                    "reward": 8.0,
                    "length": 18,
                    "global_scanned_cells": 60,
                    "scan_efficiency": 3.3,
                    "final_global_scan_ratio": 24.0,
                    "final_global_avg_entropy": 74.0,
                    "success_flag": 0,
                    "collision_count": 1,
                    "reset_reason": "collision",
                    "episode_elapsed_time": 36.0,
                }
            ],
        }

        outputs = write_apf_baseline_outputs(
            output_root=self.root,
            grouped_rows=rows,
            experiment_id="baseline-demo",
            stage_name="stage00_apf_baseline",
            stage_index=0,
        )

        self.assertIn("fixed_apf", outputs)
        self.assertIn("random_apf", outputs)

        fixed_training = pd.read_csv(outputs["fixed_apf"]["training_csv"], encoding="utf-8-sig")
        fixed_scan = pd.read_csv(outputs["fixed_apf"]["scan_csv"], encoding="utf-8-sig")
        self.assertIn("reward", fixed_training.columns)
        self.assertIn("scan_efficiency", fixed_training.columns)
        self.assertIn("scan_ratio", fixed_scan.columns)
        self.assertIn("global_avg_entropy", fixed_scan.columns)

    def test_write_apf_baseline_outputs_marks_experiment_metadata(self):
        rows = {
            "fixed_apf": [
                {
                    "algorithm_type": "fixed_apf",
                    "episode": 1,
                    "reward": 10.0,
                    "length": 20,
                    "global_scanned_cells": 80,
                    "scan_efficiency": 4.0,
                    "final_global_scan_ratio": 30.0,
                    "final_global_avg_entropy": 70.0,
                    "success_flag": 1,
                    "collision_count": 0,
                    "reset_reason": "timeout",
                    "episode_elapsed_time": 40.0,
                }
            ]
        }

        outputs = write_apf_baseline_outputs(
            output_root=self.root,
            grouped_rows=rows,
            experiment_id="baseline-demo",
            stage_name="stage00_apf_baseline",
            stage_index=0,
        )
        training = pd.read_csv(outputs["fixed_apf"]["training_csv"], encoding="utf-8-sig")
        self.assertEqual(training["experiment_id"].iloc[0], "baseline-demo")
        self.assertEqual(training["stage_name"].iloc[0], "stage00_apf_baseline")


if __name__ == "__main__":
    unittest.main()
