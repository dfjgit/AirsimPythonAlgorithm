import unittest
from pathlib import Path


class TestTempPathUsageTests(unittest.TestCase):
    def test_workspace_root_is_not_used_for_test_temp_dirs(self):
        repo_root = Path(__file__).resolve().parents[3]
        target_files = [
            repo_root / "multirotor" / "Algorithm" / "tests" / "test_analysis_data_fallbacks.py",
            repo_root / "multirotor" / "Algorithm" / "tests" / "test_family_analysis.py",
            repo_root / "multirotor" / "Algorithm" / "tests" / "test_four_group_benchmark_analyzer.py",
            repo_root / "multirotor" / "Algorithm" / "tests" / "test_four_group_benchmark_runner.py",
            repo_root / "multirotor" / "Algorithm" / "tests" / "test_paper_two_stage_analysis.py",
            repo_root / "multirotor" / "Algorithm" / "tests" / "test_paper_two_stage_recommendation.py",
            repo_root / "multirotor" / "Algorithm" / "tests" / "test_paper_workflow_archive.py",
            repo_root / "multirotor" / "Algorithm" / "tests" / "test_paper_workflow_orchestrator.py",
            repo_root / "multirotor" / "Algorithm" / "tests" / "test_paper_workflow_state.py",
            repo_root / "multirotor" / "Algorithm" / "tests" / "test_visualize_scan_csv_collision_count.py",
            repo_root / "multirotor" / "Algorithm" / "tests" / "test_visualize_scan_csv_collision_plot.py",
            repo_root / "multirotor" / "Algorithm" / "tests" / "test_visualize_training_data_collision.py",
            repo_root / "multirotor" / "Algorithm" / "tests" / "test_visualize_training_data_collision_count.py",
        ]

        offenders = []
        for path in target_files:
            content = path.read_text(encoding="utf-8")
            if "Path.cwd()" in content or "parents[3].resolve()" in content or ' / "pwf_tmp"' in content:
                offenders.append(str(path.relative_to(repo_root)))

        self.assertEqual(
            offenders,
            [],
            msg="These tests still derive temp roots from the workspace: " + ", ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
