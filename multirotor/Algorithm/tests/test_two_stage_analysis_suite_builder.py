import os
import shutil
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from two_stage_analysis_suite_builder import (
    LEGACY_SINGLE_METRIC_PLOTS,
    STAGE02_NORMALIZED_PLOTS,
    build_two_stage_analysis_suite,
)


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _write_placeholder_pngs(base: Path, names: list[str]) -> None:
    for name in names:
        (base / name).write_bytes(b"old")


def _write_training_csv(
    path: Path,
    *,
    algorithm: str,
    stage_index: int,
    rewards: list[float],
    lengths: list[int],
    scan_efficiency: list[float],
    scan_ratio: list[str],
    entropy: list[float],
    collisions: list[int],
    reset_reasons: list[str],
    terminal_battery_voltage: list[float] | None = None,
    out_of_range_count: list[int] | None = None,
    final_mode: bool = False,
) -> None:
    rows = {
        "episode": [1, 2, 3],
        "reward": rewards,
        "length": lengths,
        "global_scanned_cells": [120, 140, 160],
        "scan_efficiency": scan_efficiency,
        "algorithm_type": [algorithm] * 3,
        "experiment_id": [f"{algorithm}_exp"] * 3,
        "stage_name": [f"stage{stage_index:02d}"] * 3,
        "stage_index": [stage_index] * 3,
        "is_resume": [1 if stage_index > 1 else 0] * 3,
        "reset_reason": reset_reasons,
        "collision_count": collisions,
        "collision_count_final": collisions,
        "collision_object_name": ["", "", ""],
        "collision_position": ["", "", ""],
    }
    if final_mode:
        rows["final_global_scan_ratio"] = scan_ratio
        rows["final_global_avg_entropy"] = entropy
    else:
        rows["max_global_scan_ratio"] = scan_ratio
        rows["min_global_avg_entropy"] = entropy
    if terminal_battery_voltage is not None:
        rows["terminal_battery_voltage"] = terminal_battery_voltage
    if out_of_range_count is not None:
        rows["out_of_range_count"] = out_of_range_count
        rows["out_of_range_count_final"] = out_of_range_count
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _write_scan_csv(path: Path, *, algorithm: str, stage_index: int) -> None:
    pd.DataFrame(
        {
            "episode": [1, 1, 2, 2, 3, 3],
            "elapsed_time": [5, 10, 15, 20, 25, 30],
            "step": [1, 2, 1, 2, 1, 2],
            "episode_step": [1, 2, 1, 2, 1, 2],
            "scan_ratio": ["4%", "8%", "9%", "12%", "14%", "18%"],
            "global_scan_ratio": ["4%", "8%", "9%", "12%", "14%", "18%"],
            "global_avg_entropy": [90, 86, 84, 81, 79, 76],
            "reset_reason": ["", "timeout", "", "collision", "", "timeout"],
            "collision_count": [0, 0, 0, 1, 0, 0],
            "out_of_range_count": [0, 0, 0, 0, 1, 1],
            "algorithm_type": [algorithm] * 6,
            "experiment_id": [f"{algorithm}_exp"] * 6,
            "stage_name": [f"stage{stage_index:02d}"] * 6,
            "stage_index": [stage_index] * 6,
            "is_resume": [1 if stage_index > 1 else 0] * 6,
            "UAV1_x": [0, 1, 2, 3, 4, 5],
            "UAV1_z": [0, -1, -2, -3, -4, -5],
            "UAV1_battery_voltage": [4.20, 4.16, 4.12, 4.08, 4.03, 3.98],
            "UAV2_battery_voltage": [4.20, 4.15, 4.10, 4.06, 4.01, 3.96],
            "UAV3_battery_voltage": [4.20, 4.14, 4.09, 4.05, 4.00, 3.95],
        }
    ).to_csv(path, index=False, encoding="utf-8-sig")


class TwoStageAnalysisSuiteBuilderTests(unittest.TestCase):
    def test_plot_metadata_uses_chinese_titles(self):
        expected_legacy_titles = {
            "episode_reward": "单轮累计奖励变化",
            "episode_length": "单轮步长变化",
            "global_scan_ratio": "最终全局扫描率变化",
            "global_avg_entropy": "全局平均熵变化",
            "scan_efficiency": "扫描效率变化",
        }
        self.assertEqual(
            {item["filename"]: item["title"] for item in LEGACY_SINGLE_METRIC_PLOTS},
            expected_legacy_titles,
        )

        expected_stage02_titles = {
            "avg_scan_cells_per_second": "按时间归一化扫描产出对比",
            "avg_scan_cells_per_volt_drop": "按电量归一化扫描产出对比",
        }
        self.assertEqual(
            {item["metric"]: item["title"] for item in STAGE02_NORMALIZED_PLOTS},
            expected_stage02_titles,
        )

    def test_builder_regenerates_stage_suites_with_current_paper_style(self):
        root = Path(__file__).resolve().parent / "_tmp_two_stage_suite_builder"
        try:
            shutil.rmtree(root, ignore_errors=True)

            ddpg_stage01 = root / "analysis_results" / "stage01_analysis_suite" / "ddpg_stage01"
            ddpg_stage02 = root / "analysis_results" / "stage02_analysis_suite" / "ddpg_stage02"
            dqn_stage01 = root / "analysis_results" / "stage01_analysis_suite" / "dqn_stage01"
            dqn_stage02 = root / "analysis_results" / "stage02_analysis_suite" / "dqn_stage02"
            cmp_stage01 = root / "analysis_results" / "stage01_analysis_suite" / "comparison"
            cmp_stage02 = root / "analysis_results" / "stage02_analysis_suite" / "comparison"

            for path in [ddpg_stage01, ddpg_stage02, dqn_stage01, dqn_stage02, cmp_stage01, cmp_stage02]:
                path.mkdir(parents=True, exist_ok=True)

            _write_placeholder_pngs(
                ddpg_stage01,
                [
                    "episode_reward.png",
                    "episode_length.png",
                    "global_scan_ratio.png",
                    "global_avg_entropy.png",
                    "scan_efficiency.png",
                    "trajectories_xz.png",
                ],
            )
            _write_placeholder_pngs(
                dqn_stage01,
                [
                    "episode_reward.png",
                    "episode_length.png",
                    "global_scan_ratio.png",
                    "global_avg_entropy.png",
                    "scan_efficiency.png",
                    "trajectories_xz.png",
                ],
            )
            _write_placeholder_pngs(
                ddpg_stage02,
                [
                    "episode_reward.png",
                    "episode_length.png",
                    "collision_stability.png",
                    "collision_count_trend.png",
                    "global_scan_ratio.png",
                    "global_avg_entropy.png",
                    "scan_efficiency.png",
                    "trajectories_xz.png",
                ],
            )
            _write_placeholder_pngs(
                dqn_stage02,
                [
                    "episode_reward.png",
                    "episode_length.png",
                    "collision_stability.png",
                    "collision_count_trend.png",
                    "global_scan_ratio.png",
                    "global_avg_entropy.png",
                    "scan_efficiency.png",
                    "trajectories_xz.png",
                ],
            )
            _write_placeholder_pngs(
                cmp_stage01,
                [
                    "comparison_training_reward.png",
                    "comparison_training_scan_efficiency.png",
                    "comparison_scan_scan_ratio.png",
                    "comparison_scan_global_avg_entropy.png",
                ],
            )
            _write_placeholder_pngs(
                cmp_stage02,
                [
                    "comparison_training_reward.png",
                    "comparison_training_scan_efficiency.png",
                    "comparison_training_collision_rate.png",
                    "comparison_training_collision_count.png",
                    "comparison_scan_scan_ratio.png",
                    "comparison_scan_global_avg_entropy.png",
                    "comparison_scan_per_second.png",
                    "comparison_scan_per_volt_drop.png",
                ],
            )

            ddpg_logs = root / "multirotor" / "DDPG_Weight" / "airsim_training_logs"
            dqn_logs = root / "multirotor" / "DQN_Movement" / "logs" / "dqn_scan_data"
            ddpg_logs.mkdir(parents=True, exist_ok=True)
            dqn_logs.mkdir(parents=True, exist_ok=True)

            _write_training_csv(
                ddpg_logs / "ddpg_training_ddpg_apf_demo_stage01_20260326_234955.csv",
                algorithm="ddpg_apf",
                stage_index=1,
                rewards=[10, 12, 15],
                lengths=[20, 20, 20],
                scan_efficiency=[2.2, 2.5, 2.8],
                scan_ratio=["10%", "12%", "15%"],
                entropy=[88, 84, 80],
                collisions=[0, 1, 0],
                reset_reasons=["达到时长上限", "collision", "达到时长上限"],
            )
            _write_scan_csv(
                ddpg_logs / "scan_data_ddpg_apf_demo_stage01_20260326_234955.csv",
                algorithm="ddpg_apf",
                stage_index=1,
            )
            _write_training_csv(
                ddpg_logs / "ddpg_training_ddpg_apf_demo_stage02_20260331_003640.csv",
                algorithm="ddpg_apf",
                stage_index=2,
                rewards=[21, 23, 22],
                lengths=[24, 24, 24],
                scan_efficiency=[2.9, 3.0, 3.1],
                scan_ratio=["14%", "15%", "17%"],
                entropy=[79, 77, 75],
                collisions=[1, 0, 1],
                reset_reasons=["collision", "达到时长上限", "达到时长上限"],
                final_mode=True,
            )
            _write_scan_csv(
                ddpg_logs / "scan_data_ddpg_apf_demo_stage02_20260331_003640.csv",
                algorithm="ddpg_apf",
                stage_index=2,
            )

            _write_training_csv(
                dqn_logs / "dqn_training_pure_dqn_demo_stage01_20260330_005101.csv",
                algorithm="pure_dqn",
                stage_index=1,
                rewards=[-10, 3, 18],
                lengths=[30, 36, 40],
                scan_efficiency=[1.1, 1.4, 1.8],
                scan_ratio=["2%", "8%", "16%"],
                entropy=[98, 90, 81],
                collisions=[0, 0, 0],
                reset_reasons=["", "", ""],
            )
            _write_scan_csv(
                dqn_logs / "scan_data_pure_dqn_demo_stage01_20260330_005101.csv",
                algorithm="pure_dqn",
                stage_index=1,
            )
            _write_training_csv(
                dqn_logs / "dqn_training_pure_dqn_demo_stage02_20260402_005952.csv",
                algorithm="pure_dqn",
                stage_index=2,
                rewards=[31, 37, 35],
                lengths=[42, 44, 46],
                scan_efficiency=[1.7, 1.9, 2.0],
                scan_ratio=["18%", "21%", "23%"],
                entropy=[78, 74, 72],
                collisions=[0, 0, 0],
                reset_reasons=["Timeout", "Timeout", "Timeout"],
                terminal_battery_voltage=[3.6, 3.55, 3.5],
                out_of_range_count=[2, 1, 3],
                final_mode=True,
            )
            _write_scan_csv(
                dqn_logs / "scan_data_pure_dqn_demo_stage02_20260402_005952.csv",
                algorithm="pure_dqn",
                stage_index=2,
            )

            out_root = root / "analysis_results" / "two_stage_analysis_suite"
            out_root.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {"algorithm": "DDPG+APF", "stage": "stage01", "avg_reward": 1, "tail_reward": 1, "avg_length": 1, "tail_length": 1, "avg_scan_efficiency": 1, "tail_scan_efficiency": 1, "avg_scan_ratio_pct": 10, "tail_scan_ratio_pct": 11, "avg_entropy": 80, "tail_entropy": 79, "avg_collision_count": 1, "tail_collision_count": 1, "avg_out_of_range_count": 0, "tail_out_of_range_count": 0, "avg_scan_cells_per_second": "", "avg_scan_cells_per_volt_drop": ""},
                    {"algorithm": "DDPG+APF", "stage": "stage02", "avg_reward": 1, "tail_reward": 1, "avg_length": 1, "tail_length": 1, "avg_scan_efficiency": 1, "tail_scan_efficiency": 1, "avg_scan_ratio_pct": 11, "tail_scan_ratio_pct": 11, "avg_entropy": 79, "tail_entropy": "", "avg_collision_count": 1, "tail_collision_count": 1, "avg_out_of_range_count": 0, "tail_out_of_range_count": 0, "avg_scan_cells_per_second": 0.6, "avg_scan_cells_per_volt_drop": 100},
                    {"algorithm": "纯DQN", "stage": "stage01", "avg_reward": 1, "tail_reward": 1, "avg_length": 1, "tail_length": 1, "avg_scan_efficiency": 1, "tail_scan_efficiency": 1, "avg_scan_ratio_pct": 8, "tail_scan_ratio_pct": 19, "avg_entropy": 85, "tail_entropy": 77, "avg_collision_count": 0, "tail_collision_count": 0, "avg_out_of_range_count": 0, "tail_out_of_range_count": 0, "avg_scan_cells_per_second": "", "avg_scan_cells_per_volt_drop": ""},
                    {"algorithm": "纯DQN", "stage": "stage02", "avg_reward": 1, "tail_reward": 1, "avg_length": 1, "tail_length": 1, "avg_scan_efficiency": 1, "tail_scan_efficiency": 1, "avg_scan_ratio_pct": 20, "tail_scan_ratio_pct": 19, "avg_entropy": 75, "tail_entropy": "", "avg_collision_count": 0, "tail_collision_count": 0, "avg_out_of_range_count": 10, "tail_out_of_range_count": 9, "avg_scan_cells_per_second": 1.1, "avg_scan_cells_per_volt_drop": 200},
                ]
            ).to_csv(out_root / "two_stage_key_metrics.csv", index=False, encoding="utf-8-sig")

            build_two_stage_analysis_suite(root, out_root)

            self.assertTrue((ddpg_stage01 / "episode_reward.png").read_bytes().startswith(PNG_SIGNATURE))
            self.assertTrue((ddpg_stage02 / "collision_stability.png").read_bytes().startswith(PNG_SIGNATURE))
            self.assertTrue((cmp_stage01 / "comparison_training_reward.png").read_bytes().startswith(PNG_SIGNATURE))
            self.assertTrue((cmp_stage02 / "comparison_scan_per_volt_drop.png").read_bytes().startswith(PNG_SIGNATURE))

            self.assertTrue((out_root / "ddpg_two_stage" / "stage01" / "episode_reward.png").exists())
            self.assertTrue((out_root / "ddpg_two_stage" / "stage02" / "collision_stability.png").exists())
            self.assertTrue((out_root / "dqn_two_stage" / "stage02" / "collision_count_trend.png").exists())
            self.assertTrue((out_root / "comparison" / "stage01" / "comparison_training_reward.png").exists())
            self.assertTrue((out_root / "comparison" / "stage02" / "comparison_scan_per_volt_drop.png").exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
