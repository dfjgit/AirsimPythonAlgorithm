import os
import shutil
import sys
import types
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _test_temp_paths import make_temp_dir
from four_group_benchmark_runner import (
    _benchmark_stage_plan_lines,
    _run_apf_algorithm,
    _make_server_kwargs,
    build_apf_action_vector,
    candidate_project_roots,
    choose_first_existing_model,
    summarize_episode_metrics,
)


class FourGroupBenchmarkRunnerHelperTests(unittest.TestCase):
    def setUp(self):
        self.root = make_temp_dir("four_group_benchmark_runner")

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

    def test_benchmark_stage_plan_lines_use_chinese_when_ui_lang_is_zh(self):
        with patch.dict(os.environ, {"AIRSIM_UI_LANG": "zh"}, clear=False):
            lines = _benchmark_stage_plan_lines()

        self.assertIn("本阶段将依次在 Unity/AirSim 中评测以下四组：", lines[0])
        self.assertIn("fixed APF（固定策略基线，不参加训练）", lines[1])
        self.assertIn("random APF（随机策略基线，不参加训练）", lines[2])
        self.assertIn("DDPG+APF（使用已训练模型，冻结策略）", lines[3])
        self.assertIn("Pure DQN（使用已训练模型，冻结策略）", lines[4])

    def test_benchmark_stage_plan_lines_use_english_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            lines = _benchmark_stage_plan_lines()

        self.assertIn("This stage evaluates the following four groups in Unity/AirSim:", lines[0])

    def test_make_server_kwargs_defaults_visualization_to_enabled(self):
        with patch.dict(os.environ, {}, clear=True):
            kwargs = _make_server_kwargs(
                seed=1,
                run_kind="test",
                experiment_id="exp",
                algorithm_type="fixed_apf",
            )

        self.assertTrue(kwargs["enable_visualization"])

    def test_make_server_kwargs_honors_quick_visualization_override(self):
        with patch.dict(os.environ, {"AIRSIM_QUICK_VISUALIZATION": "0"}, clear=True):
            kwargs = _make_server_kwargs(
                seed=1,
                run_kind="test",
                experiment_id="exp",
                algorithm_type="fixed_apf",
            )

        self.assertFalse(kwargs["enable_visualization"])

    def test_run_apf_algorithm_uses_apf_training_prefix_for_raw_logs(self):
        captured = {}

        fake_sb3 = types.ModuleType("stable_baselines3")
        fake_sb3.DDPG = type("FakeDDPG", (), {"load": staticmethod(lambda *args, **kwargs: object())})

        class FakeDataCollector:
            def set_external_data(self, key, value):
                return None

        class FakeAlgorithm:
            def get_current_coefficients(self):
                return {
                    "repulsionCoefficient": 2.0,
                    "entropyCoefficient": 2.0,
                    "distanceCoefficient": 2.0,
                    "leaderRangeCoefficient": 1.5,
                    "directionRetentionCoefficient": 0.5,
                    "obstacleRepulsionDistance": 15.0,
                    "obstacleRepulsionCoefficient": 5.0,
                }

        class FakeServer:
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)
                self.drone_names = ["UAV1"]
                self.algorithms = {"UAV1": FakeAlgorithm()}
                self.config_data = types.SimpleNamespace(paper_benchmark={"random_apf": {}})
                self.data_collector = FakeDataCollector()

            def start(self):
                return True

            def start_mission(self):
                return True

            def set_experiment_meta(self, **kwargs):
                return None

            def stop(self):
                return None

        fake_algorithm_server_module = types.ModuleType("multirotor.AlgorithmServer")
        fake_algorithm_server_module.MultiDroneAlgorithmServer = FakeServer

        class FakeEnv:
            def __init__(self, *args, **kwargs):
                self.step_count = 0
                self.step_duration = 5.0
                self.term_cfg = {"target_scan_ratio": 0.25}

            def reset(self, seed=None):
                return [0.0]

        fake_simple_weight_env_module = types.ModuleType("multirotor.DDPG_Weight.envs.simple_weight_env")
        fake_simple_weight_env_module.SimpleWeightEnv = FakeEnv

        with patch.dict(
            sys.modules,
            {
                "stable_baselines3": fake_sb3,
                "multirotor.AlgorithmServer": fake_algorithm_server_module,
                "multirotor.DDPG_Weight.envs.simple_weight_env": fake_simple_weight_env_module,
            },
            clear=False,
        ):
            rows = _run_apf_algorithm(
                algorithm_type="fixed_apf",
                seed=20260413,
                eval_episodes=0,
                system_config_path=Path("multirotor/system_config.json"),
                ddpg_model_path=None,
                output_dir=self.root,
                data_log_dir=self.root / "logs",
            )

        self.assertEqual(rows, [])
        self.assertEqual(captured["training_prefix"], "apf")


if __name__ == "__main__":
    unittest.main()
