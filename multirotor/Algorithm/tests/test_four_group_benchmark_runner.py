import os
import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from four_group_benchmark_runner import (
    build_apf_action_vector,
    candidate_project_roots,
    choose_first_existing_model,
    summarize_episode_metrics,
)


class FourGroupBenchmarkRunnerHelperTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / "_tmp_four_group_benchmark_runner"
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_choose_first_existing_model_prefers_first_existing_candidate(self):
        first = self.root / "a.zip"
        second = self.root / "b.zip"
        second.write_text("ok", encoding="utf-8")

        chosen = choose_first_existing_model([first, second])

        self.assertEqual(chosen, second)

    def test_build_apf_action_vector_keeps_obstacle_defaults(self):
        action = build_apf_action_vector(
            {
                "repulsionCoefficient": 1.0,
                "entropyCoefficient": 2.0,
                "distanceCoefficient": 3.0,
                "leaderRangeCoefficient": 4.0,
                "directionRetentionCoefficient": 5.0,
                "obstacleRepulsionDistance": 15.0,
                "obstacleRepulsionCoefficient": 5.0,
            }
        )

        self.assertEqual(action.tolist(), [1.0, 2.0, 3.0, 4.0, 5.0, 15.0, 5.0])

    def test_candidate_project_roots_adds_primary_workspace_for_worktree_paths(self):
        current_root = Path(r"D:\repo\.worktrees\feature-x\multirotor")

        roots = candidate_project_roots(current_root)

        self.assertEqual(
            roots,
            [
                Path(r"D:\repo\.worktrees\feature-x\multirotor"),
                Path(r"D:\repo\multirotor"),
            ],
        )

    def test_summarize_episode_metrics_produces_required_fields(self):
        row = summarize_episode_metrics(
            algorithm_type="fixed_apf",
            seed=20260413,
            episode=2,
            total_reward=15.5,
            episode_elapsed_time=25.0,
            final_global_scan_ratio=31.0,
            final_global_avg_entropy=64.0,
            global_scanned_count=155,
            collision_count=1,
            out_of_range_count=2,
            reset_reason="timeout",
            terminal_battery_voltage=3.82,
        )

        self.assertEqual(row["algorithm_type"], "fixed_apf")
        self.assertEqual(row["seed"], 20260413)
        self.assertEqual(row["episode"], 2)
        self.assertEqual(row["success_flag"], 1)
        self.assertEqual(row["collision_termination_flag"], 0)
        self.assertAlmostEqual(row["avg_scan_cells_per_second"], 6.2)
        self.assertIn("avg_scan_cells_per_volt_drop", row)


if __name__ == "__main__":
    unittest.main()
