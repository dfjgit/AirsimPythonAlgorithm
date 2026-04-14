import os
import shutil
import sys
import unittest
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paper_two_stage_analysis import build_two_stage_summary


class PaperTwoStageAnalysisTests(unittest.TestCase):
    def setUp(self):
        workspace_root = Path(__file__).parents[3].resolve()
        self.root = workspace_root / f"tmp_paper_two_stage_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=False)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

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
        self.assertTrue(outputs["summary_csv"].exists())
        self.assertTrue(outputs["summary_md"].exists())
