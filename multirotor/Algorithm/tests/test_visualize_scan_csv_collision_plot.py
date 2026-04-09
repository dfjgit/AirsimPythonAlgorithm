import os
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from visualize_scan_csv import PLOT_PIPELINE, RunData, plot_collision_stability


def _workspace_root_for_tmp() -> Path:
    this_file = Path(__file__).resolve()
    for parent in this_file.parents:
        if parent.name == ".worktrees":
            return parent.parent
    return Path.cwd()


class CollisionStabilityPlotTests(unittest.TestCase):
    def test_visualize_scan_csv_importable_from_package_path(self):
        repo_root = Path(__file__).resolve().parents[3]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root)
        proc = subprocess.run(
            [sys.executable, "-c", "import multirotor.Algorithm.visualize_scan_csv"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            proc.returncode,
            0,
            msg=f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}",
        )

    def test_plot_collision_stability_writes_png(self):
        root = _workspace_root_for_tmp() / ".tmp_collision_stability_tests" / uuid.uuid4().hex
        root.mkdir(parents=True, exist_ok=False)
        try:
            output_dir = root / "out"
            output_dir.mkdir(parents=True, exist_ok=True)

            training_csv = root / "ddpg_training_demo.csv"
            scan_csv = root / "scan_data_demo.csv"

            pd.DataFrame(
                {
                    "episode": [1, 2, 3],
                    "reward": [10, 11, 12],
                    "length": [20, 20, 20],
                    "max_global_scan_ratio": ["10%", "12%", "15%"],
                    "min_global_avg_entropy": [88, 84, 80],
                    "reset_reason": ["collision", "达到时长上限", ""],
                    "collision_count_final": [0, 0, 1],
                    "collision_object_name": ["", "", ""],
                    "collision_position": ["", "", ""],
                }
            ).to_csv(training_csv, index=False, encoding="utf-8-sig")

            pd.DataFrame({"episode": [1, 2, 3], "step": [1, 1, 1]}).to_csv(
                scan_csv, index=False, encoding="utf-8-sig"
            )

            run = RunData(scan_path=scan_csv, training_path=training_csv, output_dir=output_dir)
            plot_collision_stability(run)

            self.assertTrue((output_dir / "collision_stability.png").exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_plot_pipeline_registers_collision_stability(self):
        names = [plot_name for _, plot_name in PLOT_PIPELINE]
        self.assertIn("collision_stability", names)


if __name__ == "__main__":
    unittest.main()
