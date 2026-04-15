import os
import shutil
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _test_temp_paths import make_temp_dir
import visualize_training_data


class VisualizeTrainingDataCollisionCountTests(unittest.TestCase):
    @patch("visualize_training_data.UnifiedTrainingAnalyzer")
    def test_analyze_algorithm_comparison_requests_collision_count_plots(self, analyzer_cls):
        analyzer = analyzer_cls.return_value

        root = make_temp_dir("visualize_training_data_collision_count_tests")
        try:
            project_root = root
            (project_root / "multirotor" / "DDPG_Weight" / "airsim_training_logs").mkdir(parents=True)
            (project_root / "multirotor" / "DQN_Movement" / "logs" / "dqn_scan_data").mkdir(parents=True)

            result = visualize_training_data._analyze_algorithm_comparison(
                project_root=project_root,
                out_root=project_root / "out",
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)

        self.assertEqual(result, 0)
        analyzer.plot_comparison.assert_any_call(metric="collision_count", data_type="training", x_axis="episode")
        analyzer.plot_comparison.assert_any_call(
            metric="collision_count",
            data_type="training",
            x_axis="episode",
            latest_only=True,
            file_prefix="latest_comparison",
        )
        analyzer.plot_recent_window_comparison.assert_any_call(
            metric="collision_count",
            data_type="training",
            tail_episodes=50,
            min_training_episodes=20,
            file_prefix="recent_window_comparison",
        )


if __name__ == "__main__":
    unittest.main()
