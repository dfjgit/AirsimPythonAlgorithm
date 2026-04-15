import os
import shutil
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _test_temp_paths import make_temp_dir
from two_stage_analysis_suite_builder import _build_stage02_normalized_frame
from visualize_scan_csv import RunData, _select_representative_episodes


class AnalysisDataFallbackTests(unittest.TestCase):
    def test_representative_episode_labels_use_chinese_descriptions(self):
        root = make_temp_dir("analysis_data_fallbacks")
        try:
            output_dir = root / "out"
            output_dir.mkdir(parents=True, exist_ok=True)

            training_csv = root / "training.csv"
            scan_csv = root / "scan.csv"

            pd.DataFrame(
                {
                    "episode": [1, 2, 3, 4],
                    "reward": [10, 12, 8, 6],
                    "length": [20, 20, 20, 20],
                    "max_global_scan_ratio": ["30%", "10%", "20%", "5%"],
                    "min_global_avg_entropy": [80, 90, 82, 95],
                    "reset_reason": ["达到时长上限", "碰撞", "", "达到时长上限"],
                    "collision_object_name": ["", "", "", ""],
                    "collision_position": ["", "", "", ""],
                }
            ).to_csv(training_csv, index=False, encoding="utf-8-sig")

            pd.DataFrame(
                {
                    "episode": [1, 2, 3, 4],
                    "step": [1, 1, 1, 1],
                    "episode_reward": [10, 12, 8, 6],
                    "global_scan_ratio": ["30%", "10%", "20%", "5%"],
                    "global_avg_entropy": [80, 90, 82, 95],
                    "reset_reason": ["达到时长上限", "碰撞", "", "达到时长上限"],
                }
            ).to_csv(scan_csv, index=False, encoding="utf-8-sig")

            run = RunData(scan_path=scan_csv, training_path=training_csv, output_dir=output_dir)
            labels = [label for label, _ in _select_representative_episodes(run, limit=4)]

            self.assertEqual(labels, ["最佳扫描", "最近回合", "代表性失败", "最近补充"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_run_data_uses_scan_metrics_when_training_metrics_are_placeholder_constants(self):
        root = make_temp_dir("analysis_data_fallbacks")
        try:
            output_dir = root / "out"
            output_dir.mkdir(parents=True, exist_ok=True)

            training_csv = root / "dqn_training_demo.csv"
            scan_csv = root / "scan_data_demo.csv"

            pd.DataFrame(
                {
                    "episode": [1, 2, 3],
                    "reward": [1, 2, 3],
                    "length": [10, 11, 12],
                    "max_global_scan_ratio": ["0%", "0%", "0%"],
                    "min_global_avg_entropy": [100, 100, 100],
                    "reset_reason": ["", "", ""],
                    "collision_object_name": ["", "", ""],
                    "collision_position": ["", "", ""],
                }
            ).to_csv(training_csv, index=False, encoding="utf-8-sig")

            pd.DataFrame(
                {
                    "episode": [1, 1, 2, 2, 3, 3],
                    "step": [1, 2, 1, 2, 1, 2],
                    "episode_step": [1, 2, 1, 2, 1, 2],
                    "episode_reward": [1, 2, 2, 3, 3, 4],
                    "global_scan_ratio": ["4%", "8%", "9%", "12%", "14%", "18%"],
                    "global_avg_entropy": [90, 86, 84, 81, 79, 76],
                    "reset_reason": ["", "timeout", "", "collision", "", "timeout"],
                }
            ).to_csv(scan_csv, index=False, encoding="utf-8-sig")

            run = RunData(scan_path=scan_csv, training_path=training_csv, output_dir=output_dir)

            self.assertEqual(run.episode_df["episode_scan_ratio"].tolist(), [8.0, 12.0, 18.0])
            self.assertEqual(run.episode_df["episode_min_entropy"].tolist(), [86.0, 81.0, 76.0])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_stage02_voltage_normalization_uses_episode_battery_drop_before_reset_row(self):
        root = make_temp_dir("analysis_data_fallbacks")
        try:
            training_csv = root / "training.csv"
            scan_csv = root / "scan.csv"

            pd.DataFrame(
                {
                    "episode": [1, 2, 3],
                    "length": [20, 20, 20],
                    "global_scanned_cells": [120, 140, 160],
                }
            ).to_csv(training_csv, index=False, encoding="utf-8-sig")

            pd.DataFrame(
                {
                    "episode": [1, 1, 2, 2, 3, 3],
                    "episode_step": [1, 2, 1, 2, 1, 2],
                    "reset_reason": ["", "达到时长上限", "", "碰撞", "", "达到时长上限"],
                    "UAV1_battery_voltage": [4.10, 4.20, 3.95, 4.20, 3.80, 4.20],
                    "UAV2_battery_voltage": [4.08, 4.20, 3.96, 4.20, 3.82, 4.20],
                    "UAV3_battery_voltage": [4.09, 4.20, 3.94, 4.20, 3.81, 4.20],
                }
            ).to_csv(scan_csv, index=False, encoding="utf-8-sig")

            frame = _build_stage02_normalized_frame(training_csv, scan_csv, seconds_per_step=2.0)

            self.assertEqual(frame["avg_scan_cells_per_volt_drop"].round(2).tolist(), [1000.0, 538.46, 400.0])
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
