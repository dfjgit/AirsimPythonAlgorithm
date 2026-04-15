import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from two_stage_analysis_plots import (
    ALGORITHM_TRANSITION_CHARTS,
    EFFICIENCY_SUBPLOT_TITLES,
    RESULT_COMPARISON_TITLES,
    _apply_summary_legend_layout,
    generate_two_stage_plots,
)


class TwoStageAnalysisPlotsTests(unittest.TestCase):
    def test_summary_plot_titles_use_chinese_labels(self):
        self.assertEqual(
            [item[0] for item in ALGORITHM_TRANSITION_CHARTS],
            ["奖励变化", "步长变化", "扫描率变化(%)", "平均熵变化"],
        )
        self.assertEqual(RESULT_COMPARISON_TITLES, ("平均最终扫描率", "平均最终全局熵"))
        self.assertEqual(
            EFFICIENCY_SUBPLOT_TITLES,
            ("平均扫描效率（格/步）", "第二阶段按时间归一化产出", "第二阶段按电量归一化产出"),
        )

    def test_summary_legend_layout_separates_title_and_legend(self):
        fig = Mock()
        handles = [object(), object()]
        labels = ["A", "B"]

        _apply_summary_legend_layout(fig, handles, labels)

        legend_kwargs = fig.legend.call_args.kwargs
        self.assertEqual(legend_kwargs["loc"], "upper center")
        self.assertLessEqual(legend_kwargs["bbox_to_anchor"][1], 0.955)

        tight_layout_kwargs = fig.tight_layout.call_args.kwargs
        self.assertLessEqual(tight_layout_kwargs["rect"][3], 0.90)

    def test_generate_two_stage_plots_writes_expected_pngs(self):
        root = Path(__file__).resolve().parent / "_tmp_two_stage_analysis_plots"
        try:
            shutil.rmtree(root, ignore_errors=True)
            out_dir = root / "analysis_results" / "two_stage_analysis_suite"
            out_dir.mkdir(parents=True, exist_ok=True)

            metrics_csv = out_dir / "two_stage_key_metrics.csv"
            pd.DataFrame(
                [
                    {
                        "algorithm": "DDPG+APF",
                        "stage": "stage01",
                        "avg_reward": 2311.0387,
                        "tail_reward": 2315.3795,
                        "avg_length": 59.0252,
                        "tail_length": 57.55,
                        "avg_scan_efficiency": 3.0255,
                        "tail_scan_efficiency": 3.339,
                        "avg_scan_ratio_pct": 10.9162,
                        "tail_scan_ratio_pct": 11.219,
                        "avg_entropy": 76.9585,
                        "tail_entropy": 76.719,
                        "avg_collision_count": 2.7983,
                        "tail_collision_count": 2.4,
                        "avg_out_of_range_count": 0.0,
                        "tail_out_of_range_count": 0.0,
                        "avg_scan_cells_per_second": "",
                        "avg_scan_cells_per_volt_drop": "",
                    },
                    {
                        "algorithm": "DDPG+APF",
                        "stage": "stage02",
                        "avg_reward": 2339.49,
                        "tail_reward": 2319.28,
                        "avg_length": 59.4685,
                        "tail_length": 60.0,
                        "avg_scan_efficiency": 3.0141,
                        "tail_scan_efficiency": 2.9645,
                        "avg_scan_ratio_pct": 11.06,
                        "tail_scan_ratio_pct": 11.03,
                        "avg_entropy": 77.02,
                        "tail_entropy": "",
                        "avg_collision_count": 2.8649,
                        "tail_collision_count": 2.7,
                        "avg_out_of_range_count": 0.0,
                        "tail_out_of_range_count": 0.0,
                        "avg_scan_cells_per_second": 0.6028,
                        "avg_scan_cells_per_volt_drop": 193.52,
                    },
                    {
                        "algorithm": "纯DQN",
                        "stage": "stage01",
                        "avg_reward": 772.7626,
                        "tail_reward": 11372.2365,
                        "avg_length": 91.6523,
                        "tail_length": 192.3,
                        "avg_scan_efficiency": 1.5112,
                        "tail_scan_efficiency": 1.6745,
                        "avg_scan_ratio_pct": 8.4315,
                        "tail_scan_ratio_pct": 19.725,
                        "avg_entropy": 82.0804,
                        "tail_entropy": 76.545,
                        "avg_collision_count": 0.0,
                        "tail_collision_count": 0.0,
                        "avg_out_of_range_count": 0.0,
                        "tail_out_of_range_count": 0.0,
                        "avg_scan_cells_per_second": "",
                        "avg_scan_cells_per_volt_drop": "",
                    },
                    {
                        "algorithm": "纯DQN",
                        "stage": "stage02",
                        "avg_reward": 11945.97,
                        "tail_reward": 10960.03,
                        "avg_length": 195.0809,
                        "tail_length": 184.9,
                        "avg_scan_efficiency": 1.6618,
                        "tail_scan_efficiency": 1.679,
                        "avg_scan_ratio_pct": 20.03,
                        "tail_scan_ratio_pct": 19.0,
                        "avg_entropy": 75.22,
                        "tail_entropy": "",
                        "avg_collision_count": 0.0,
                        "tail_collision_count": 0.0,
                        "avg_out_of_range_count": 15.0074,
                        "tail_out_of_range_count": 14.45,
                        "avg_scan_cells_per_second": 1.1083,
                        "avg_scan_cells_per_volt_drop": 366.63,
                    },
                ]
            ).to_csv(metrics_csv, index=False, encoding="utf-8-sig")

            generate_two_stage_plots(metrics_csv, out_dir)

            self.assertTrue((out_dir / "ddpg_two_stage" / "ddpg_stage_transition_summary.png").exists())
            self.assertTrue((out_dir / "dqn_two_stage" / "dqn_stage_transition_summary.png").exists())
            self.assertTrue((out_dir / "comparison" / "two_stage_result_comparison.png").exists())
            self.assertTrue((out_dir / "comparison" / "two_stage_stage_gain_comparison.png").exists())
            self.assertTrue((out_dir / "comparison" / "two_stage_efficiency_comparison.png").exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
