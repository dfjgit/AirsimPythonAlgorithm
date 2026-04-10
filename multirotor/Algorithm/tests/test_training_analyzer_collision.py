import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training_analyzer import UnifiedTrainingAnalyzer


class TrainingAnalyzerCollisionTests(unittest.TestCase):
    def test_normalize_metrics_derives_collision_rate(self):
        output_dir = Path(__file__).resolve().parent / "_tmp_training_analyzer_collision_normalize"
        output_dir.mkdir(parents=True, exist_ok=True)
        analyzer = UnifiedTrainingAnalyzer(output_dir=str(output_dir))
        df = pd.DataFrame(
            {
                "episode": [1, 2, 3],
                "length": [10, 10, 10],
                "reward": [1, 2, 3],
                "reset_reason": ["collision", "达到时长上限", ""],
                "collision_count_final": [0, 0, 1],
            }
        )

        normalized = analyzer._normalize_metrics(df, data_type="training")
        self.assertEqual(normalized["collision_rate"].tolist(), [100.0, 0.0, 100.0])

    def test_plot_comparison_writes_collision_rate_png(self):
        root = Path(__file__).resolve().parent / "_tmp_training_analyzer_collision_plot"
        try:
            shutil.rmtree(root, ignore_errors=True)
            ddpg_dir = root / "ddpg"
            dqn_dir = root / "dqn"
            out_dir = root / "out"
            ddpg_dir.mkdir(parents=True, exist_ok=True)
            dqn_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                {
                    "episode": [1, 2],
                    "reward": [1, 2],
                    "length": [10, 10],
                    "reset_reason": ["collision", "达到时长上限"],
                    "collision_count_final": [0, 0],
                    "algorithm_type": ["ddpg_apf", "ddpg_apf"],
                }
            ).to_csv(ddpg_dir / "ddpg_training_demo.csv", index=False)

            pd.DataFrame(
                {
                    "episode": [1, 2],
                    "reward": [1, 2],
                    "length": [10, 10],
                    "reset_reason": ["达到时长上限", ""],
                    "collision_count_final": [0, 1],
                    "algorithm_type": ["pure_dqn", "pure_dqn"],
                }
            ).to_csv(dqn_dir / "dqn_training_demo.csv", index=False)

            analyzer = UnifiedTrainingAnalyzer(output_dir=str(out_dir))
            analyzer.load_data([str(ddpg_dir), str(dqn_dir)])
            analyzer.plot_comparison(metric="collision_rate", data_type="training", x_axis="episode")
            analyzer.generate_summary_report()

            self.assertTrue((out_dir / "comparison_training_collision_rate.png").exists())
            report = pd.read_csv(out_dir / "algorithm_comparison_report.csv", encoding="utf-8-sig")
            self.assertIn("平均碰撞终止占比(%)", report.columns)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_safe_to_markdown_fallback_only_for_import_error(self):
        analyzer = UnifiedTrainingAnalyzer(output_dir=str(Path(__file__).resolve().parent))
        df = pd.DataFrame({"算法名称": ["A"], "平均奖励": [1.0]})

        with patch.object(pd.DataFrame, "to_markdown", side_effect=ImportError("missing tabulate")):
            rendered = analyzer._safe_to_markdown(df, index=False)
        self.assertTrue(rendered.startswith("```text\n"))
        self.assertIn("算法名称", rendered)
        self.assertIn("平均奖励", rendered)
        self.assertTrue(rendered.endswith("\n```"))

        with patch.object(pd.DataFrame, "to_markdown", side_effect=ValueError("broken table")):
            with self.assertRaises(ValueError):
                analyzer._safe_to_markdown(df, index=False)

    def test_generate_recent_window_report_contains_collision_rate_column(self):
        root = Path(__file__).resolve().parent / "_tmp_training_analyzer_collision_recent"
        try:
            shutil.rmtree(root, ignore_errors=True)
            data_dir = root / "data"
            out_dir = root / "out"
            data_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                {
                    "episode": [1, 2, 3],
                    "reward": [1.0, 2.0, 3.0],
                    "length": [10, 10, 10],
                    "reset_reason": ["collision", "达到时长上限", "collision"],
                    "collision_count_final": [0, 0, 0],
                    "scan_efficiency": [0.1, 0.2, 0.3],
                    "algorithm_type": ["ddpg_apf", "ddpg_apf", "ddpg_apf"],
                }
            ).to_csv(data_dir / "ddpg_training_recent.csv", index=False)

            pd.DataFrame(
                {
                    "episode": [1, 2, 3],
                    "scan_ratio": ["10%", "20%", "30%"],
                    "global_avg_entropy": [0.8, 0.6, 0.4],
                    "elapsed_time": [5, 10, 15],
                    "algorithm_type": ["ddpg_apf", "ddpg_apf", "ddpg_apf"],
                }
            ).to_csv(data_dir / "scan_data_recent.csv", index=False)

            analyzer = UnifiedTrainingAnalyzer(output_dir=str(out_dir))
            analyzer.load_data([str(data_dir)])
            analyzer.generate_recent_window_report(tail_episodes=2, min_training_episodes=2)

            report = pd.read_csv(out_dir / "recent_window_algorithm_comparison_report.csv", encoding="utf-8-sig")
            self.assertIn("平均碰撞终止占比(%)", report.columns)
            self.assertEqual(report["平均碰撞终止占比(%)"].tolist(), [50.0])
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
