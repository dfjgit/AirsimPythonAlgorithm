import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _test_temp_paths import make_temp_dir
from apf_baseline_sim_runner import run_apf_baseline_simulation, write_apf_baseline_outputs


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

    def test_run_apf_baseline_simulation_flushes_incremental_outputs_after_each_completed_episode(self):
        summary_root = self.root / "workflow_outputs"
        raw_log_root = self.root / "workflow_outputs" / "logs"
        fixed_rows = [
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
            },
            {
                "algorithm_type": "fixed_apf",
                "episode": 2,
                "reward": 12.0,
                "length": 22,
                "global_scanned_cells": 90,
                "scan_efficiency": 4.1,
                "final_global_scan_ratio": 34.0,
                "final_global_avg_entropy": 68.0,
                "success_flag": 1,
                "collision_count": 0,
                "reset_reason": "timeout",
                "episode_elapsed_time": 44.0,
            },
        ]
        random_rows = [
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
        ]

        def fake_run_apf_algorithm(*, algorithm_type, data_log_dir=None, on_episode_complete=None, **kwargs):
            self.assertEqual(Path(data_log_dir), raw_log_root)
            emitted_rows = fixed_rows if algorithm_type == "fixed_apf" else random_rows
            for expected_count, row in enumerate(emitted_rows, start=1):
                on_episode_complete(dict(row))
                algo_dir = summary_root / algorithm_type
                training_files = list(algo_dir.glob(f"{algorithm_type}_training_workflow-demo_*.csv"))
                scan_files = list(algo_dir.glob(f"{algorithm_type}_scan_workflow-demo_*.csv"))
                self.assertEqual(len(training_files), 1)
                self.assertEqual(len(scan_files), 1)
                training_frame = pd.read_csv(training_files[0], encoding="utf-8-sig")
                scan_frame = pd.read_csv(scan_files[0], encoding="utf-8-sig")
                self.assertEqual(len(training_frame), expected_count)
                self.assertEqual(len(scan_frame), expected_count)
            return emitted_rows

        with patch("apf_baseline_sim_runner._run_apf_algorithm", side_effect=fake_run_apf_algorithm):
            outputs = run_apf_baseline_simulation(
                output_root=summary_root,
                raw_log_dir=raw_log_root,
                seeds=[20260413],
                episodes=2,
                experiment_id="workflow-demo",
                stage_name="stage00_apf_baseline",
                stage_index=0,
            )

        fixed_training = pd.read_csv(outputs["fixed_apf"]["training_csv"], encoding="utf-8-sig")
        random_training = pd.read_csv(outputs["random_apf"]["training_csv"], encoding="utf-8-sig")
        self.assertEqual(len(fixed_training), 2)
        self.assertEqual(len(random_training), 1)


if __name__ == "__main__":
    unittest.main()
