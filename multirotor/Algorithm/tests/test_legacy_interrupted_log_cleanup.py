import json
import os
import shutil
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _test_temp_paths import make_temp_dir
import legacy_interrupted_log_cleanup


class LegacyInterruptedLogCleanupTests(unittest.TestCase):
    def setUp(self):
        self.root = make_temp_dir("legacy_interrupted_cleanup")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_training_csv(self, name: str, rows: list[dict]) -> Path:
        path = self.root / name
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def _write_scan_csv(self, name: str, rows: list[dict]) -> Path:
        path = self.root / name
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def test_analyze_training_csv_detects_interrupted_last_episode_without_terminal_scan_row(self):
        training_csv = self._write_training_csv(
            "ddpg_training_demo_stage01_20260416_120000.csv",
            [
                {"episode": 1, "reward": 10, "length": 60, "reset_reason": "达到时长上限"},
                {"episode": 2, "reward": 12, "length": 17, "reset_reason": ""},
            ],
        )
        scan_csv = self._write_scan_csv(
            "scan_data_demo_stage01_20260416_120000.csv",
            [
                {"episode": 1, "step": 60, "reset_reason": "达到时长上限"},
                {"episode": 2, "step": 1, "reset_reason": ""},
                {"episode": 2, "step": 17, "reset_reason": ""},
            ],
        )

        result = legacy_interrupted_log_cleanup.analyze_training_csv(training_csv, scan_csv=scan_csv)

        self.assertEqual(result["status"], "interrupted_last_episode")
        self.assertEqual(result["candidate_episode"], 2)
        self.assertEqual(result["removed_rows"], 1)

    def test_analyze_training_csv_keeps_last_episode_when_terminal_scan_row_exists(self):
        training_csv = self._write_training_csv(
            "dqn_training_demo_stage01_20260416_120500.csv",
            [
                {"episode": 1, "reward": 10, "length": 60, "reset_reason": "达到时长上限"},
                {"episode": 2, "reward": 12, "length": 20, "reset_reason": ""},
            ],
        )
        scan_csv = self._write_scan_csv(
            "scan_data_demo_stage01_20260416_120500.csv",
            [
                {"episode": 1, "step": 60, "reset_reason": "达到时长上限"},
                {"episode": 2, "step": 20, "reset_reason": "碰撞"},
            ],
        )

        result = legacy_interrupted_log_cleanup.analyze_training_csv(training_csv, scan_csv=scan_csv)

        self.assertEqual(result["status"], "complete")
        self.assertIsNone(result["candidate_episode"])
        self.assertEqual(result["removed_rows"], 0)

    def test_main_dry_run_does_not_modify_training_csv(self):
        training_csv = self._write_training_csv(
            "ddpg_training_demo_stage01_20260416_121000.csv",
            [
                {"episode": 1, "reward": 10, "length": 60, "reset_reason": "达到时长上限"},
                {"episode": 2, "reward": 12, "length": 17, "reset_reason": ""},
            ],
        )
        scan_csv = self._write_scan_csv(
            "scan_data_demo_stage01_20260416_121000.csv",
            [
                {"episode": 1, "step": 60, "reset_reason": "达到时长上限"},
                {"episode": 2, "step": 17, "reset_reason": ""},
            ],
        )
        before = training_csv.read_text(encoding="utf-8-sig")

        exit_code = legacy_interrupted_log_cleanup.main(
            ["--csv", str(training_csv), "--scan-csv", str(scan_csv)]
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(training_csv.read_text(encoding="utf-8-sig"), before)
        self.assertFalse((self.root / "interrupted_runs" / training_csv.name).exists())

    def test_apply_cleanup_rewrites_training_csv_and_quarantines_removed_row(self):
        training_csv = self._write_training_csv(
            "ddpg_training_demo_stage01_20260416_121500.csv",
            [
                {"episode": 1, "reward": 10, "length": 60, "reset_reason": "达到时长上限"},
                {"episode": 2, "reward": 12, "length": 17, "reset_reason": ""},
            ],
        )
        scan_csv = self._write_scan_csv(
            "scan_data_demo_stage01_20260416_121500.csv",
            [
                {"episode": 1, "step": 60, "reset_reason": "达到时长上限"},
                {"episode": 2, "step": 17, "reset_reason": ""},
            ],
        )

        exit_code = legacy_interrupted_log_cleanup.main(
            ["--csv", str(training_csv), "--scan-csv", str(scan_csv), "--apply"]
        )

        self.assertEqual(exit_code, 0)
        cleaned = pd.read_csv(training_csv, encoding="utf-8-sig")
        self.assertEqual(cleaned["episode"].tolist(), [1])
        self.assertEqual(cleaned["episode_complete"].tolist(), [1])

        quarantined_csv = self.root / "interrupted_runs" / training_csv.name
        self.assertTrue(quarantined_csv.exists())
        quarantined = pd.read_csv(quarantined_csv, encoding="utf-8-sig")
        self.assertEqual(quarantined["episode"].tolist(), [2])
        self.assertEqual(quarantined["episode_complete"].tolist(), [0])

        backup_csv = training_csv.with_suffix(training_csv.suffix + ".bak")
        self.assertTrue(backup_csv.exists())


if __name__ == "__main__":
    unittest.main()
