import os
import shutil
import sys
import unittest
import importlib.util

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from _test_temp_paths import make_temp_dir
from paper_two_stage_analysis import build_two_stage_summary


class PaperTwoStageAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.root = make_temp_dir("paper_two_stage_analysis")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    @unittest.skipUnless(importlib.util.find_spec("tabulate") is not None, "tabulate is required for markdown export")
    def test_build_two_stage_summary_writes_before_after_csv_and_markdown(self):
        sim_csv = self.root / "sim.csv"
        refine_csv = self.root / "refine.csv"
        out_dir = self.root / "out"
        pd.DataFrame(
            {"episode": [1, 2], "scan_efficiency": [1.0, 1.2], "success_flag": [0, 1]}
        ).to_csv(sim_csv, index=False, encoding="utf-8-sig")
        pd.DataFrame(
            {"episode": [1, 2], "scan_efficiency": [1.4, 1.6], "success_flag": [1, 1]}
        ).to_csv(refine_csv, index=False, encoding="utf-8-sig")

        outputs = build_two_stage_summary(sim_csv, refine_csv, out_dir)
        self.assertEqual(set(outputs), {"summary_csv", "summary_md"})
        self.assertTrue(outputs["summary_csv"].exists())
        self.assertTrue(outputs["summary_md"].exists())
        self.assertEqual(outputs["summary_csv"].name, "two_stage_summary.csv")
        self.assertEqual(outputs["summary_md"].name, "two_stage_summary.md")

        summary_df = pd.read_csv(outputs["summary_csv"], encoding="utf-8-sig")
        self.assertListEqual(
            list(summary_df.columns),
            ["phase", "episodes", "avg_scan_efficiency", "success_rate"],
        )
        self.assertListEqual(
            summary_df["phase"].tolist(),
            ["sim_pretrain", "real_weighted_refine"],
        )
        self.assertListEqual(summary_df["episodes"].tolist(), [2, 2])

        self.assertAlmostEqual(summary_df.loc[0, "avg_scan_efficiency"], 1.1)
        self.assertAlmostEqual(summary_df.loc[1, "avg_scan_efficiency"], 1.5)
        self.assertAlmostEqual(summary_df.loc[0, "success_rate"], 0.5)
        self.assertAlmostEqual(summary_df.loc[1, "success_rate"], 1.0)

        summary_md = outputs["summary_md"].read_text(encoding="utf-8")
        expected_summary = pd.DataFrame(
            [
                {
                    "phase": "sim_pretrain",
                    "episodes": 2,
                    "avg_scan_efficiency": 1.1,
                    "success_rate": 0.5,
                },
                {
                    "phase": "real_weighted_refine",
                    "episodes": 2,
                    "avg_scan_efficiency": 1.5,
                    "success_rate": 1.0,
                },
            ]
        )
        expected_md = "# Two-Stage Summary\n\n" + expected_summary.to_markdown(index=False)
        self.assertEqual(summary_md, expected_md)
