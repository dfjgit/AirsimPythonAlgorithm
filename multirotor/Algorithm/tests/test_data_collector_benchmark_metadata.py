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

    def test_training_header_contains_registry_and_family_columns(self):
        collector = DataCollector(data_dir=str(self.root))
        try:
            with collector.training_csv_filename.open("r", encoding="utf-8-sig", newline="") as f:
                header = next(csv.reader(f))
        finally:
            if collector.csv_file:
                collector.csv_file.close()
            if collector.training_csv_file:
                collector.training_csv_file.close()

        self.assertIn("seed", header)
        self.assertIn("run_kind", header)
        self.assertIn("primary_family", header)
        self.assertIn("family_memberships", header)
        self.assertIn("comparison_profiles", header)
        self.assertIn("is_trainable", header)
        self.assertIn("registry_version", header)

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
            if collector.csv_file:
                collector.csv_file.close()
            if collector.training_csv_file:
                collector.training_csv_file.close()


if __name__ == "__main__":
    unittest.main()
