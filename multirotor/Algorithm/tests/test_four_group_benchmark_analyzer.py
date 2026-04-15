import os
import shutil
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from four_group_benchmark_analyzer import generate_four_group_benchmark_report


class FourGroupBenchmarkAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / "_tmp_four_group_benchmark_analyzer"
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self.input_csv = self.root / "four_group_eval_episodes.csv"
        self.output_dir = self.root / "analysis_results" / "four_group_benchmark"

        pd.DataFrame(
            [
                {
                    "algorithm_type": "fixed_apf",
                    "seed": 20260413,
                    "episode": 1,
                    "success_flag": 1,
                    "final_global_scan_ratio": 23.0,
                    "final_global_avg_entropy": 72.0,
                    "scan_efficiency": 1.4,
                    "avg_scan_cells_per_second": 0.9,
                    "avg_scan_cells_per_volt_drop": 110.0,
                    "collision_count": 0,
                    "reset_reason": "timeout",
                },
                {
                    "algorithm_type": "random_apf",
                    "seed": 20260413,
                    "episode": 1,
                    "success_flag": 0,
                    "final_global_scan_ratio": 18.0,
                    "final_global_avg_entropy": 77.0,
                    "scan_efficiency": 1.1,
                    "avg_scan_cells_per_second": 0.7,
                    "avg_scan_cells_per_volt_drop": 95.0,
                    "collision_count": 1,
                    "reset_reason": "collision",
                },
                {
                    "algorithm_type": "ddpg_apf",
                    "seed": 20260413,
                    "episode": 1,
                    "success_flag": 1,
                    "final_global_scan_ratio": 34.0,
                    "final_global_avg_entropy": 61.0,
                    "scan_efficiency": 2.0,
                    "avg_scan_cells_per_second": 1.0,
                    "avg_scan_cells_per_volt_drop": 130.0,
                    "collision_count": 0,
                    "reset_reason": "timeout",
                },
                {
                    "algorithm_type": "pure_dqn",
                    "seed": 20260413,
                    "episode": 1,
                    "success_flag": 1,
                    "final_global_scan_ratio": 31.0,
                    "final_global_avg_entropy": 63.0,
                    "scan_efficiency": 1.8,
                    "avg_scan_cells_per_second": 1.2,
                    "avg_scan_cells_per_volt_drop": 140.0,
                    "collision_count": 0,
                    "reset_reason": "timeout",
                },
            ]
        ).to_csv(self.input_csv, index=False, encoding="utf-8-sig")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_generate_four_group_benchmark_report_writes_expected_outputs(self):
        generated = generate_four_group_benchmark_report(
            eval_csv_path=self.input_csv,
            output_dir=self.output_dir,
        )

        for expected_name in [
            "four_group_eval_episodes.csv",
            "four_group_eval_seed_summary.csv",
            "four_group_summary.csv",
            "scan_ratio_boxplot.png",
            "entropy_boxplot.png",
            "efficiency_bar.png",
            "safety_bar.png",
            "reset_reason_stacked_bar.png",
        ]:
            self.assertTrue((self.output_dir / expected_name).exists(), expected_name)

        self.assertIn("summary_csv", generated)
        summary = pd.read_csv(generated["summary_csv"], encoding="utf-8-sig")
        self.assertEqual(
            sorted(summary["algorithm_type"].tolist()),
            ["ddpg_apf", "fixed_apf", "pure_dqn", "random_apf"],
        )


if __name__ == "__main__":
    unittest.main()
