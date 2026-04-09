from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from visualize_scan_csv import (
    PLOT_PIPELINE,
    RunData,
    ensure_dir,
    paired_training_csv,
    plot_entropy_hist_snapshots,
    plot_selected_episode_trajectories,
    safe_plot,
    write_manifest,
)
from training_analyzer import UnifiedTrainingAnalyzer


LOGGER = logging.getLogger("training_data_visualizer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _resolve_scan_dir(dir_path: Path) -> list[Path]:
    if not dir_path.exists():
        return []
    if dir_path.is_file() and dir_path.name.startswith("scan_data_") and dir_path.suffix.lower() == ".csv":
        return [dir_path]
    return sorted(dir_path.rglob("scan_data_*.csv"))


def _collect_auto_scan_files(project_root: Path) -> list[Path]:
    candidates = [
        project_root / "multirotor" / "DQN_Movement" / "logs" / "dqn_scan_data",
        project_root / "multirotor" / "DDPG_Weight" / "airsim_training_logs",
    ]
    files: list[Path] = []
    for candidate in candidates:
        files.extend(_resolve_scan_dir(candidate))

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in files:
        resolved = str(path.resolve())
        if resolved not in seen:
            deduped.append(path)
            seen.add(resolved)
    return deduped


def _build_scan_inputs(args: argparse.Namespace, project_root: Path) -> list[Path]:
    if args.csv:
        return [Path(args.csv)]
    if args.dir:
        return _resolve_scan_dir(Path(args.dir))
    if args.auto:
        return _collect_auto_scan_files(project_root)
    return []


def _analyze_scan_csvs(scan_files: list[Path], out_root: Path, snapshots: int) -> int:
    if not scan_files:
        LOGGER.warning("未找到可分析的 scan_data CSV 文件。")
        return 1

    ensure_dir(out_root)
    failures = 0
    for scan_csv in scan_files:
        run_dir = out_root / scan_csv.stem
        ensure_dir(run_dir)
        training_csv = paired_training_csv(scan_csv)
        run = RunData(scan_path=scan_csv, training_path=training_csv, output_dir=run_dir)
        LOGGER.info("Analyzing %s", scan_csv.name)
        for plot_fn, plot_name in PLOT_PIPELINE:
            safe_plot(plot_fn, run, plot_name=plot_name)
        safe_plot(plot_entropy_hist_snapshots, run, snapshots, plot_name="entropy_hist_snapshots")
        safe_plot(plot_selected_episode_trajectories, run, plot_name="selected_episode_trajectories")
        write_manifest(run)
        LOGGER.info("Finished %s -> %s", scan_csv.name, run_dir)
    return failures


def _analyze_algorithm_comparison(project_root: Path, out_root: Path) -> int:
    compare_dirs = [
        project_root / "multirotor" / "DDPG_Weight" / "airsim_training_logs",
        project_root / "multirotor" / "DQN_Movement" / "logs" / "dqn_scan_data",
    ]
    analyzer = UnifiedTrainingAnalyzer(output_dir=str(out_root))
    analyzer.load_data([str(p) for p in compare_dirs])
    # 1. Historical full comparison
    analyzer.plot_comparison(metric="reward", data_type="training", x_axis="episode")
    analyzer.plot_comparison(metric="scan_efficiency", data_type="training", x_axis="episode")
    analyzer.plot_comparison(metric="collision_rate", data_type="training", x_axis="episode")
    analyzer.plot_comparison(metric="collision_count", data_type="training", x_axis="episode")
    analyzer.plot_comparison(metric="scan_ratio", data_type="scan", x_axis="elapsed_time")
    analyzer.plot_comparison(metric="global_avg_entropy", data_type="scan", x_axis="elapsed_time")
    analyzer.generate_summary_report()

    # 2. Latest-run comparison with the same four core views
    analyzer.plot_comparison(
        metric="reward",
        data_type="training",
        x_axis="episode",
        latest_only=True,
        file_prefix="latest_comparison",
    )
    analyzer.plot_comparison(
        metric="scan_efficiency",
        data_type="training",
        x_axis="episode",
        latest_only=True,
        file_prefix="latest_comparison",
    )
    analyzer.plot_comparison(
        metric="collision_rate",
        data_type="training",
        x_axis="episode",
        latest_only=True,
        file_prefix="latest_comparison",
    )
    analyzer.plot_comparison(
        metric="collision_count",
        data_type="training",
        x_axis="episode",
        latest_only=True,
        file_prefix="latest_comparison",
    )
    analyzer.plot_comparison(
        metric="scan_ratio",
        data_type="scan",
        x_axis="elapsed_time",
        latest_only=True,
        file_prefix="latest_comparison",
    )
    analyzer.plot_comparison(
        metric="global_avg_entropy",
        data_type="scan",
        x_axis="elapsed_time",
        latest_only=True,
        file_prefix="latest_comparison",
    )
    analyzer.generate_summary_report(latest_only=True, report_prefix="latest_algorithm_comparison")

    # 3. Recent substantial window comparison
    analyzer.plot_recent_window_comparison(
        metric="reward",
        data_type="training",
        tail_episodes=50,
        min_training_episodes=20,
        file_prefix="recent_window_comparison",
    )
    analyzer.plot_recent_window_comparison(
        metric="scan_efficiency",
        data_type="training",
        tail_episodes=50,
        min_training_episodes=20,
        file_prefix="recent_window_comparison",
    )
    analyzer.plot_recent_window_comparison(
        metric="collision_rate",
        data_type="training",
        tail_episodes=50,
        min_training_episodes=20,
        file_prefix="recent_window_comparison",
    )
    analyzer.plot_recent_window_comparison(
        metric="collision_count",
        data_type="training",
        tail_episodes=50,
        min_training_episodes=20,
        file_prefix="recent_window_comparison",
    )
    analyzer.plot_recent_window_comparison(
        metric="scan_ratio",
        data_type="scan",
        tail_episodes=50,
        min_training_episodes=20,
        file_prefix="recent_window_comparison",
    )
    analyzer.plot_recent_window_comparison(
        metric="global_avg_entropy",
        data_type="scan",
        tail_episodes=50,
        min_training_episodes=20,
        file_prefix="recent_window_comparison",
    )
    analyzer.generate_recent_window_report(
        tail_episodes=50,
        min_training_episodes=20,
        report_prefix="recent_window_algorithm_comparison",
    )
    LOGGER.info("Finished algorithm comparison -> %s", out_root)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="训练数据可视化工具（统一离线分析入口）")
    parser.add_argument("--auto", action="store_true", help="自动扫描常用日志目录中的 scan_data CSV")
    parser.add_argument("--json", type=str, help="兼容旧参数；当前入口不再处理 JSON，可改用 scan_data CSV")
    parser.add_argument("--csv", type=str, help="分析单个 scan_data CSV 文件")
    parser.add_argument("--dir", type=str, help="分析目录中的 scan_data CSV 文件")
    parser.add_argument(
        "--out",
        type=str,
        default="multirotor/DQN_Movement/logs/analysis_results",
        help="输出目录",
    )
    parser.add_argument("--show", action="store_true", help="兼容旧参数；当前离线模式下忽略")
    parser.add_argument("--compare", action="store_true", help="兼容旧参数；当前入口暂不单独处理")
    parser.add_argument("--compare-algorithms", action="store_true", help="兼容旧参数；当前入口暂不单独处理")
    parser.add_argument("--compare-algorithms-full", action="store_true", help="兼容旧参数；当前入口暂不单独处理")
    parser.add_argument("--snapshots", type=int, default=4, help="熵值快照数量")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = project_root / out_root

    if args.show:
        LOGGER.info("当前分析器使用离线 Agg 后端，--show 参数将被忽略。")

    if args.json:
        LOGGER.warning("visualize_training_data.py 已统一为 CSV 离线分析入口，JSON 分析已弃用。")
        LOGGER.warning("请优先使用对应的 scan_data_*.csv 文件。")

    if args.compare or args.compare_algorithms or args.compare_algorithms_full:
        compare_out = out_root / "algorithm_comparison"
        ensure_dir(compare_out)
        return _analyze_algorithm_comparison(project_root, compare_out)

    scan_files = _build_scan_inputs(args, project_root)
    if not scan_files:
        LOGGER.info("提示: 使用 --auto 自动扫描，或使用 --csv/--dir 指定 scan_data CSV。")
        return 1

    return _analyze_scan_csvs(scan_files, out_root, args.snapshots)


if __name__ == "__main__":
    sys.exit(main())
