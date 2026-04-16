import csv
import os
import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_collector import DataCollector


class DataCollectorBenchmarkMetadataTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / "_tmp_data_collector_benchmark_metadata"
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _close_collector_files(self, collector: DataCollector):
        for handle_name in (
            "csv_file",
            "training_csv_file",
            "interrupted_training_csv_file",
        ):
            handle = getattr(collector, handle_name, None)
            if handle and not handle.closed:
                handle.close()

    def test_training_header_contains_registry_and_family_columns(self):
        collector = DataCollector(data_dir=str(self.root))
        try:
            with collector.training_csv_filename.open("r", encoding="utf-8-sig", newline="") as f:
                header = next(csv.reader(f))
        finally:
            self._close_collector_files(collector)

        self.assertIn("seed", header)
        self.assertIn("run_kind", header)
        self.assertIn("primary_family", header)
        self.assertIn("family_memberships", header)
        self.assertIn("comparison_profiles", header)
        self.assertIn("is_trainable", header)
        self.assertIn("registry_version", header)
        self.assertIn("episode_complete", header)

    def test_custom_training_prefix_changes_training_csv_filename_prefix(self):
        collector = DataCollector(
            data_dir=str(self.root),
            training_prefix="apf",
            experiment_id="fixed_apf_seed_20260413",
            stage_index=1,
        )
        try:
            self.assertTrue(
                collector.training_csv_filename.name.startswith(
                    "apf_training_fixed_apf_seed_20260413_stage01_"
                )
            )
        finally:
            self._close_collector_files(collector)

    def test_stop_routes_interrupted_episode_to_diagnostics_not_training_csv(self):
        collector = DataCollector(data_dir=str(self.root))
        try:
            collector.running = True
            collector.last_episode = 7
            collector.current_episode_reward = 12.5
            collector.current_episode_length = 4
            collector.current_episode_elapsed_time = 8.0
            collector.current_episode_weights = [[1, 2, 3, 4, 5]] * 4
            collector.last_scanned_count = 4
            collector.last_global_scanned_count = 12
            collector.episode_scan_summary[7] = {
                "step": 4,
                "scanned_count": 4,
                "global_scanned_count": 12,
                "global_scan_ratio": 25.0,
                "global_avg_entropy": 70.0,
                "terminal_battery_voltage": 3.9,
                "max_out_of_range_duration_sec": 0.0,
                "max_global_scan_ratio": 25.0,
                "min_global_avg_entropy": 70.0,
            }

            collector.stop()

            with collector.training_csv_filename.open("r", encoding="utf-8-sig", newline="") as f:
                training_rows = list(csv.DictReader(f))
            self.assertEqual(training_rows, [])

            interrupted_csv = collector.interrupted_training_csv_filename
            with interrupted_csv.open("r", encoding="utf-8-sig", newline="") as f:
                interrupted_rows = list(csv.DictReader(f))
            self.assertEqual(len(interrupted_rows), 1)
            self.assertEqual(interrupted_rows[0]["episode_complete"], "0")
        finally:
            self._close_collector_files(collector)

    def test_stop_keeps_completed_episode_in_training_csv(self):
        collector = DataCollector(data_dir=str(self.root))
        try:
            collector.running = True
            collector.last_episode = 3
            collector.current_episode_reward = 20.0
            collector.current_episode_length = 5
            collector.current_episode_elapsed_time = 10.0
            collector.current_episode_weights = [[1, 1, 1, 1, 1]] * 5
            collector.last_scanned_count = 5
            collector.last_global_scanned_count = 15
            collector.terminal_episode_meta[3] = {
                "reset_reason": "达到时长上限",
                "collision_count": 0,
                "out_of_range_count": 0,
                "max_global_scan_ratio": 30.0,
                "min_global_avg_entropy": 65.0,
                "collision_object_name": "",
                "collision_penetration_depth": 0.0,
                "collision_position": "",
                "recent_trajectory": "",
            }
            collector.episode_scan_summary[3] = {
                "step": 5,
                "scanned_count": 5,
                "global_scanned_count": 15,
                "global_scan_ratio": 30.0,
                "global_avg_entropy": 65.0,
                "terminal_battery_voltage": 3.8,
                "max_out_of_range_duration_sec": 0.0,
                "max_global_scan_ratio": 30.0,
                "min_global_avg_entropy": 65.0,
            }

            collector.stop()

            with collector.training_csv_filename.open("r", encoding="utf-8-sig", newline="") as f:
                training_rows = list(csv.DictReader(f))
            self.assertEqual(len(training_rows), 1)
            self.assertEqual(training_rows[0]["episode_complete"], "1")

            interrupted_csv = collector.interrupted_training_csv_filename
            with interrupted_csv.open("r", encoding="utf-8-sig", newline="") as f:
                interrupted_rows = list(csv.DictReader(f))
            self.assertEqual(interrupted_rows, [])
        finally:
            self._close_collector_files(collector)


if __name__ == "__main__":
    unittest.main()
