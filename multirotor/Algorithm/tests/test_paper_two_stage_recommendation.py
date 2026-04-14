import os
import shutil
import sys
import unittest
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paper_two_stage_recommendation import (
    CAUTION_DECISION,
    CONTINUE_DECISION,
    STOP_DECISION,
    recommend_real_weighted_continue,
)


class PaperTwoStageRecommendationTests(unittest.TestCase):
    def setUp(self):
        workspace_root = Path(__file__).parents[3].resolve()
        self.root = workspace_root / "tmp_paper_two_stage_recommendation" / uuid.uuid4().hex
        self.root.mkdir(parents=True, exist_ok=False)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_summary(self, rows):
        summary_csv = self.root / "summary.csv"
        pd.DataFrame(rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")
        return summary_csv

    def test_recommend_continue_when_refine_gain_is_still_large(self):
        summary_csv = self._write_summary(
            [
                {"phase": "sim_pretrain", "avg_scan_efficiency": 1.0, "success_rate": 0.50},
                {"phase": "real_weighted_refine", "avg_scan_efficiency": 1.6, "success_rate": 0.70},
            ]
        )
        result = recommend_real_weighted_continue(summary_csv)
        self.assertEqual(result["decision"], CONTINUE_DECISION)

    def test_recommend_stop_when_refine_gain_is_small_and_success_is_high(self):
        summary_csv = self._write_summary(
            [
                {"phase": "sim_pretrain", "avg_scan_efficiency": 1.0, "success_rate": 0.85},
                {"phase": "real_weighted_refine", "avg_scan_efficiency": 1.02, "success_rate": 0.90},
            ]
        )
        result = recommend_real_weighted_continue(summary_csv)
        self.assertEqual(result["decision"], STOP_DECISION)

    def test_recommend_caution_for_moderate_gain(self):
        summary_csv = self._write_summary(
            [
                {"phase": "sim_pretrain", "avg_scan_efficiency": 1.0, "success_rate": 0.80},
                {"phase": "real_weighted_refine", "avg_scan_efficiency": 1.04, "success_rate": 0.88},
            ]
        )
        result = recommend_real_weighted_continue(summary_csv)
        self.assertEqual(result["decision"], CAUTION_DECISION)

    def test_never_stop_when_refinement_regresses(self):
        summary_csv = self._write_summary(
            [
                {"phase": "sim_pretrain", "avg_scan_efficiency": 1.2, "success_rate": 0.80},
                {"phase": "real_weighted_refine", "avg_scan_efficiency": 1.1, "success_rate": 0.75},
            ]
        )
        result = recommend_real_weighted_continue(summary_csv)
        self.assertEqual(result["decision"], CAUTION_DECISION)
        self.assertTrue(any("regression" in str(reason) for reason in result["reasons"]))

    def test_mixed_regression_still_caution(self):
        summary_csv = self._write_summary(
            [
                {"phase": "sim_pretrain", "avg_scan_efficiency": 1.0, "success_rate": 0.5},
                {"phase": "real_weighted_refine", "avg_scan_efficiency": 0.9, "success_rate": 0.7},
            ]
        )
        result = recommend_real_weighted_continue(summary_csv)
        self.assertEqual(result["decision"], CAUTION_DECISION)
        self.assertTrue(any("regression" in str(reason) for reason in result["reasons"]))

    def test_plateau_stop_with_low_success(self):
        summary_csv = self._write_summary(
            [
                {"phase": "sim_pretrain", "avg_scan_efficiency": 1.0, "success_rate": 0.70},
                {"phase": "real_weighted_refine", "avg_scan_efficiency": 1.01, "success_rate": 0.72},
            ]
        )
        result = recommend_real_weighted_continue(summary_csv)
        self.assertEqual(result["decision"], STOP_DECISION)
