import os
import shutil
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training_analyzer import UnifiedTrainingAnalyzer


class TrainingAnalyzerCollisionCountTests(unittest.TestCase):
    def test_normalize_metrics_derives_collision_count(self):
        output_dir = Path(__file__).resolve().parent / "_tmp_training_analyzer_collision_count_normalize"
        output_dir.mkdir(parents=True, exist_ok=True)
        analyzer = UnifiedTrainingAnalyzer(output_dir=str(output_dir))
        df = pd.DataFrame(
            {
                "episode": [1, 2, 3],
                "length": [10, 10, 10],
                "reward": [1, 2, 3],
                "collision_count_final": [3, 1, 0],
            }
        )

        normalized = analyzer._normalize_metrics(df, data_type="training")
        self.assertEqual(normalized["collision_count"].tolist(), [3.0, 1.0, 0.0])

    def test_plot_comparison_writes_collision_count_png(self):
        root = Path(__file__).resolve().parent / "_tmp_training_analyzer_collision_count_plot"
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
                    "collision_count_final": [2, 1],
                    "algorithm_type": ["ddpg_apf", "ddpg_apf"],
                }
            ).to_csv(ddpg_dir / "ddpg_training_demo.csv", index=False)

            pd.DataFrame(
                {
                    "episode": [1, 2],
                    "reward": [1, 2],
                    "length": [10, 10],
                    "collision_count_final": [1, 0],
                    "algorithm_type": ["pure_dqn", "pure_dqn"],
                }
            ).to_csv(dqn_dir / "dqn_training_demo.csv", index=False)

            analyzer = UnifiedTrainingAnalyzer(output_dir=str(out_dir))
            analyzer.load_data([str(ddpg_dir), str(dqn_dir)])
            analyzer.plot_comparison(metric="collision_count", data_type="training", x_axis="episode")
            analyzer.generate_summary_report()

            self.assertTrue((out_dir / "comparison_training_collision_count.png").exists())
            report = pd.read_csv(out_dir / "algorithm_comparison_report.csv", encoding="utf-8-sig")
            self.assertIn("平均碰撞次数", report.columns)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
