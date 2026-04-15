import os
import shutil
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithm_specific_analysis import generate_algorithm_specific_reports


class AlgorithmSpecificAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / "_tmp_algorithm_specific_analysis"
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self.eval_csv = self.root / "four_group_eval_episodes.csv"
        self.output_root = self.root / "analysis_results" / "algorithm_specific"

        pd.DataFrame(
            [
                {"algorithm_type": "fixed_apf", "seed": 1, "episode": 1, "final_global_scan_ratio": 20.0},
                {"algorithm_type": "fixed_apf", "seed": 1, "episode": 2, "final_global_scan_ratio": 21.0},
                {"algorithm_type": "pure_dqn", "seed": 1, "episode": 1, "final_global_scan_ratio": 30.0},
            ]
        ).to_csv(self.eval_csv, index=False, encoding="utf-8-sig")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_generate_algorithm_specific_reports_writes_per_algorithm_outputs(self):
        generated = generate_algorithm_specific_reports(
            eval_csv_paths=[self.eval_csv],
            output_root=self.output_root,
        )

        self.assertIn("fixed_apf", generated)
        self.assertIn("pure_dqn", generated)
        self.assertTrue((self.output_root / "fixed_apf" / "eval_episodes.csv").exists())
        self.assertTrue((self.output_root / "fixed_apf" / "summary.csv").exists())
        self.assertTrue((self.output_root / "pure_dqn" / "eval_episodes.csv").exists())


if __name__ == "__main__":
    unittest.main()
