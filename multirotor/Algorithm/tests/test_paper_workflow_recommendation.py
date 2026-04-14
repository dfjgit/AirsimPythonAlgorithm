import shutil
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_workflow_recommendation import recommend_comparison_stage02


class ComparisonRecommendationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / "tmp_recommendation"
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_recommend_continue_when_recent_success_is_low(self):
        training_csv = self.root / "training.csv"
        benchmark_csv = self.root / "benchmark.csv"
        pd.DataFrame(
            {
                "episode": [1, 2, 3, 4],
                "success_flag": [0, 0, 0, 1],
                "scan_efficiency": [1.0, 1.01, 1.0, 1.01],
                "collision_rate": [2.0, 1.5, 1.0, 1.8],
            }
        ).to_csv(training_csv, index=False, encoding="utf-8-sig")
        pd.DataFrame(
            {
                "algorithm_type": ["ddpg_apf", "fixed_apf"],
                "success_flag": [0, 1],
                "final_global_scan_ratio": [30.0, 30.0],
            }
        ).to_csv(benchmark_csv, index=False, encoding="utf-8-sig")
        result = recommend_comparison_stage02(
            training_csv,
            benchmark_csv,
            algorithm_type="ddpg_apf",
            recent_window=4,
            min_recent_window=2,
        )
        self.assertEqual(result["decision"], "建议续训")
        self.assertEqual(len(result["reasons"]), 1)
        self.assertIn("recent success", result["reasons"][0])

    def test_recommend_stop_when_recent_window_is_stable_and_safe(self):
        training_csv = self.root / "training.csv"
        benchmark_csv = self.root / "benchmark.csv"
        pd.DataFrame(
            {
                "episode": [1, 2, 3, 4],
                "success_flag": [1, 1, 1, 1],
                "scan_efficiency": [2.0, 2.01, 2.02, 2.01],
                "collision_rate": [5.0, 4.0, 6.0, 4.0],
            }
        ).to_csv(training_csv, index=False, encoding="utf-8-sig")
        pd.DataFrame(
            {
                "algorithm_type": ["ddpg_apf", "fixed_apf"],
                "success_flag": [1, 1],
                "final_global_scan_ratio": [31.0, 30.0],
            }
        ).to_csv(benchmark_csv, index=False, encoding="utf-8-sig")
        result = recommend_comparison_stage02(
            training_csv,
            benchmark_csv,
            algorithm_type="ddpg_apf",
            recent_window=4,
            min_recent_window=2,
        )
        self.assertEqual(result["decision"], "当前可结束 stage01")

    def test_insufficient_samples_prompt_manual_review(self):
        training_csv = self.root / "training.csv"
        benchmark_csv = self.root / "benchmark.csv"
        pd.DataFrame(
            {"episode": [1, 2], "success_flag": [1, 0], "scan_efficiency": [1.0, 1.1], "collision_rate": [0.0, 0.0]}
        ).to_csv(training_csv, index=False, encoding="utf-8-sig")
        pd.DataFrame(
            {
                "algorithm_type": ["ddpg_apf", "fixed_apf"],
                "success_flag": [1, 1],
                "final_global_scan_ratio": [31.0, 30.0],
            }
        ).to_csv(benchmark_csv, index=False, encoding="utf-8-sig")
        result = recommend_comparison_stage02(
            training_csv,
            benchmark_csv,
            algorithm_type="ddpg_apf",
            recent_window=4,
            min_recent_window=3,
        )
        self.assertEqual(result["decision"], "可选续训")
        self.assertIn("样本不足", result["reasons"][0])

    def test_collision_data_without_collision_rate_triggers_continue(self):
        training_csv = self.root / "training.csv"
        benchmark_csv = self.root / "benchmark.csv"
        pd.DataFrame(
            {
                "episode": [1, 2, 3, 4, 5],
                "success_flag": [1, 1, 1, 1, 1],
                "scan_efficiency": [2.0, 2.01, 2.02, 2.03, 2.02],
                "reset_reason": ["", "collision", "", "collision", ""],
            }
        ).to_csv(training_csv, index=False, encoding="utf-8-sig")
        pd.DataFrame(
            {
                "algorithm_type": ["ddpg_apf", "fixed_apf"],
                "success_flag": [1, 1],
                "final_global_scan_ratio": [30.0, 30.0],
            }
        ).to_csv(benchmark_csv, index=False, encoding="utf-8-sig")
        result = recommend_comparison_stage02(
            training_csv,
            benchmark_csv,
            algorithm_type="ddpg_apf",
            recent_window=5,
            min_recent_window=3,
        )
        self.assertEqual(result["decision"], "建议续训")
        self.assertTrue(
            any("collision termination ratio" in reason for reason in result["reasons"])
        )

    def test_benchmark_gap_triggers_continue(self):
        training_csv = self.root / "training.csv"
        benchmark_csv = self.root / "benchmark.csv"
        pd.DataFrame(
            {
                "episode": [1, 2, 3, 4],
                "success_flag": [1, 1, 1, 1],
                "scan_efficiency": [1.5, 1.5, 1.5, 1.5],
                "collision_rate": [1.0, 1.0, 1.0, 1.0],
            }
        ).to_csv(training_csv, index=False, encoding="utf-8-sig")
        pd.DataFrame(
            {
                "algorithm_type": ["ddpg_apf", "fixed_apf"],
                "success_flag": [1, 1],
                "final_global_scan_ratio": [28.0, 30.0],
            }
        ).to_csv(benchmark_csv, index=False, encoding="utf-8-sig")
        result = recommend_comparison_stage02(
            training_csv,
            benchmark_csv,
            algorithm_type="ddpg_apf",
            recent_window=4,
            min_recent_window=2,
        )
        self.assertEqual(result["decision"], "建议续训")
        self.assertTrue(
            any("trails fixed_apf" in reason for reason in result["reasons"])
        )

    def test_missing_benchmark_rows_requests_manual_review(self):
        training_csv = self.root / "training.csv"
        benchmark_csv = self.root / "benchmark.csv"
        pd.DataFrame(
            {
                "episode": [1, 2, 3, 4],
                "success_flag": [1, 1, 1, 1],
                "scan_efficiency": [2.0, 2.0, 2.0, 2.0],
                "collision_rate": [1.0, 1.0, 1.0, 1.0],
            }
        ).to_csv(training_csv, index=False, encoding="utf-8-sig")
        pd.DataFrame(
            {
                "algorithm_type": ["fixed_apf"],
                "success_flag": [1],
                "final_global_scan_ratio": [30.0],
            }
        ).to_csv(benchmark_csv, index=False, encoding="utf-8-sig")
        result = recommend_comparison_stage02(
            training_csv,
            benchmark_csv,
            algorithm_type="ddpg_apf",
            recent_window=4,
            min_recent_window=2,
        )
        self.assertEqual(result["decision"], "可选续训")
        self.assertTrue(any("benchmark" in reason for reason in result["reasons"]))

    def test_malformed_benchmark_schema_requires_manual_review(self):
        training_csv = self.root / "training.csv"
        benchmark_csv = self.root / "benchmark.csv"
        pd.DataFrame(
            {
                "episode": [1, 2, 3, 4],
                "success_flag": [1, 1, 1, 1],
                "scan_efficiency": [2.0, 2.0, 2.0, 2.0],
                "collision_rate": [1.0, 1.0, 1.0, 1.0],
            }
        ).to_csv(training_csv, index=False, encoding="utf-8-sig")
        pd.DataFrame(
            {
                "algorithm_type": ["ddpg_apf"],
                "success_flag": [1],
            }
        ).to_csv(benchmark_csv, index=False, encoding="utf-8-sig")
        result = recommend_comparison_stage02(
            training_csv,
            benchmark_csv,
            algorithm_type="ddpg_apf",
            recent_window=4,
            min_recent_window=2,
        )
        self.assertEqual(result["decision"], "可选续训")
        self.assertTrue(
            any(
                "benchmark schema missing columns" in reason
                for reason in result["reasons"]
            )
        )
