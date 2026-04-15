import os
import shutil
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _test_temp_paths import make_temp_dir
from visualize_scan_csv import PLOT_PIPELINE, RunData, plot_collision_count_trend


class CollisionCountTrendPlotTests(unittest.TestCase):
    def test_plot_collision_count_trend_writes_png(self):
        root = make_temp_dir("collision_count_trend_tests")
        try:
            output_dir = root / "out"
            output_dir.mkdir(parents=True, exist_ok=True)

            training_csv = root / "ddpg_training_demo.csv"
            scan_csv = root / "scan_data_demo.csv"

            pd.DataFrame(
                {
                    "episode": [1, 2, 3],
                    "reward": [10, 11, 12],
                    "length": [20, 20, 20],
                    "max_global_scan_ratio": ["10%", "12%", "15%"],
                    "min_global_avg_entropy": [88, 84, 80],
                    "reset_reason": ["", "", ""],
                    "collision_count_final": [2, 1, 0],
                    "collision_object_name": ["", "", ""],
                    "collision_position": ["", "", ""],
                }
            ).to_csv(training_csv, index=False, encoding="utf-8-sig")

            pd.DataFrame({"episode": [1, 2, 3], "step": [1, 1, 1]}).to_csv(
                scan_csv, index=False, encoding="utf-8-sig"
            )

            run = RunData(scan_path=scan_csv, training_path=training_csv, output_dir=output_dir)
            plot_collision_count_trend(run)

            self.assertTrue((output_dir / "collision_count_trend.png").exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_plot_pipeline_registers_collision_count_trend(self):
        names = [plot_name for _, plot_name in PLOT_PIPELINE]
        self.assertIn("collision_count_trend", names)


if __name__ == "__main__":
    unittest.main()
