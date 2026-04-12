from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from training_analyzer import UnifiedTrainingAnalyzer
from two_stage_analysis_plots import generate_two_stage_plots
from visualize_scan_csv import (
    RunData,
    ensure_dir,
    plot_collision_count_trend,
    plot_collision_stability,
    plot_mean_with_band,
    plot_trajectories_xy,
)


SINGLE_STAGE_ASSETS = {
    "stage01": "stage01_analysis_suite",
    "stage02": "stage02_analysis_suite",
}

ALGORITHM_STAGE_DIRS = {
    "ddpg_two_stage": {
        "stage01": "ddpg_stage01",
        "stage02": "ddpg_stage02",
    },
    "dqn_two_stage": {
        "stage01": "dqn_stage01",
        "stage02": "dqn_stage02",
    },
}

ALGO_STAGE_LEAVES = {
    "stage01": {
        "ddpg_apf": "ddpg_stage01",
        "pure_dqn": "dqn_stage01",
    },
    "stage02": {
        "ddpg_apf": "ddpg_stage02",
        "pure_dqn": "dqn_stage02",
    },
}

ALGO_LOG_DIRS = {
    "ddpg_apf": Path("multirotor") / "DDPG_Weight" / "airsim_training_logs",
    "pure_dqn": Path("multirotor") / "DQN_Movement" / "logs" / "dqn_scan_data",
}

ALGO_TRAINING_PREFIX = {
    "ddpg_apf": "ddpg_training_",
    "pure_dqn": "dqn_training_",
}

SECONDS_PER_STEP = {
    "ddpg_apf": 5.0,
    "pure_dqn": 1.5,
}
ROLLING_BAND_LABEL = "滑动均值 ± 1σ"
EPISODE_XLABEL = "训练轮次"

LEGACY_SINGLE_METRIC_PLOTS = [
    {
        "filename": "episode_reward",
        "column": "episode_reward",
        "title": "单轮累计奖励变化",
        "ylabel": "累计奖励",
        "color": "#1d3557",
    },
    {
        "filename": "episode_length",
        "column": "episode_length",
        "title": "单轮步长变化",
        "ylabel": "步长",
        "color": "#457b9d",
    },
    {
        "filename": "global_scan_ratio",
        "column": "scan_ratio",
        "title": "最终全局扫描率变化",
        "ylabel": "全局扫描率 (%)",
        "color": "#2a9d8f",
        "ylim": (-5, 105),
    },
    {
        "filename": "global_avg_entropy",
        "column": "entropy",
        "title": "全局平均熵变化",
        "ylabel": "全局平均熵",
        "color": "#e76f51",
    },
    {
        "filename": "scan_efficiency",
        "column": "scan_efficiency",
        "title": "扫描效率变化",
        "ylabel": "扫描效率（格/步）",
        "color": "#264653",
    },
]

STAGE02_NORMALIZED_PLOTS = [
    {
        "metric": "avg_scan_cells_per_second",
        "filename": "comparison_scan_per_second.png",
        "title": "按时间归一化扫描产出对比",
        "ylabel": "单位时间扫描产出（格/秒）",
    },
    {
        "metric": "avg_scan_cells_per_volt_drop",
        "filename": "comparison_scan_per_volt_drop.png",
        "title": "按电量归一化扫描产出对比",
        "ylabel": "单位电量扫描产出（格/伏）",
    },
]

STAGE_COMPARISON_METRICS = {
    "stage01": [
        ("reward", "training", "episode"),
        ("scan_efficiency", "training", "episode"),
        ("scan_ratio", "scan", "elapsed_time"),
        ("global_avg_entropy", "scan", "elapsed_time"),
    ],
    "stage02": [
        ("reward", "training", "episode"),
        ("scan_efficiency", "training", "episode"),
        ("collision_rate", "training", "episode"),
        ("collision_count", "training", "episode"),
        ("scan_ratio", "scan", "elapsed_time"),
        ("global_avg_entropy", "scan", "elapsed_time"),
    ],
}


def _latest_nonempty_file(directory: Path, pattern: str) -> Path:
    candidates = [path for path in directory.glob(pattern) if path.is_file() and path.stat().st_size > 0]
    if not candidates:
        raise FileNotFoundError(f"No non-empty files matched {pattern} in {directory}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _copy_stage_assets(source_dir: Path, target_dir: Path) -> None:
    ensure_dir(target_dir)
    for path in source_dir.iterdir():
        if not path.is_file():
            continue
        shutil.copy2(path, target_dir / path.name)


def _normalize_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(
        df[column].astype(str).str.replace("%", "", regex=False),
        errors="coerce",
    )


def _plot_metric_figure(
    *,
    episode_df: pd.DataFrame,
    y_series: pd.Series,
    output_path: Path,
    title: str,
    ylabel: str,
    color: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    x_series = _normalize_series(episode_df, "episode")
    mask = ~(x_series.isna() | y_series.isna())
    if not mask.any():
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    plot_mean_with_band(
        ax,
        x_series[mask],
        y_series[mask],
        label=ROLLING_BAND_LABEL,
        color=color,
        window=20,
    )
    ax.set_title(title)
    ax.set_xlabel(EPISODE_XLABEL)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_legacy_single_algorithm_suite(scan_csv: Path, training_csv: Path, output_dir: Path, stage_name: str) -> None:
    ensure_dir(output_dir)
    run = RunData(scan_path=scan_csv, training_path=training_csv, output_dir=output_dir)
    episode_df = run.episode_df
    if episode_df.empty:
        return

    scan_ratio_col = "final_global_scan_ratio" if "final_global_scan_ratio" in episode_df.columns else "episode_scan_ratio"
    entropy_col = "final_global_avg_entropy" if "final_global_avg_entropy" in episode_df.columns else "episode_min_entropy"

    efficiency = _normalize_series(episode_df, "scan_efficiency")
    if efficiency.empty or efficiency.dropna().empty:
        length_series = _normalize_series(episode_df, "episode_length").replace(0, np.nan)
        efficiency = _normalize_series(episode_df, scan_ratio_col) / length_series

    series_map = {
        "episode_reward": _normalize_series(episode_df, "episode_reward"),
        "episode_length": _normalize_series(episode_df, "episode_length"),
        "global_scan_ratio": _normalize_series(episode_df, scan_ratio_col),
        "global_avg_entropy": _normalize_series(episode_df, entropy_col),
        "scan_efficiency": efficiency,
    }
    for config in LEGACY_SINGLE_METRIC_PLOTS:
        _plot_metric_figure(
            episode_df=episode_df,
            y_series=series_map[config["filename"]],
            output_path=output_dir / f"{config['filename']}.png",
            title=config["title"],
            ylabel=config["ylabel"],
            color=config["color"],
            ylim=config.get("ylim"),
        )

    trajectory_tmp = output_dir / "trajectories_xy.png"
    if trajectory_tmp.exists():
        trajectory_tmp.unlink()
    plot_trajectories_xy(run)
    if trajectory_tmp.exists():
        trajectory_target = output_dir / "trajectories_xz.png"
        if trajectory_target.exists():
            trajectory_target.unlink()
        trajectory_tmp.replace(trajectory_target)

    if stage_name == "stage02":
        plot_collision_stability(run)
        plot_collision_count_trend(run)


def _copy_stage_inputs_to_temp(stage_paths: dict[str, dict[str, Path]], temp_root: Path) -> list[Path]:
    copied_dirs: list[Path] = []
    shutil.rmtree(temp_root, ignore_errors=True)
    for algo_id, paths in stage_paths.items():
        algo_dir = temp_root / algo_id
        algo_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths["training"], algo_dir / paths["training"].name)
        shutil.copy2(paths["scan"], algo_dir / paths["scan"].name)
        copied_dirs.append(algo_dir)
    return copied_dirs


def _episode_terminal_battery(scan_df: pd.DataFrame) -> pd.DataFrame:
    battery_cols = [column for column in scan_df.columns if column.endswith("_battery_voltage")]
    if scan_df.empty or not battery_cols or "episode" not in scan_df.columns:
        return pd.DataFrame(columns=["episode", "terminal_min_battery_voltage"])

    working = scan_df.copy()
    working["episode"] = pd.to_numeric(working["episode"], errors="coerce")
    working = working.dropna(subset=["episode"])
    if working.empty:
        return pd.DataFrame(columns=["episode", "terminal_min_battery_voltage"])

    battery_values = working[battery_cols].apply(pd.to_numeric, errors="coerce")
    working["row_min_battery_voltage"] = battery_values.min(axis=1)
    terminal = (
        working.groupby("episode", as_index=False)["row_min_battery_voltage"]
        .min()
        .rename(columns={"row_min_battery_voltage": "terminal_min_battery_voltage"})
    )
    return terminal[["episode", "terminal_min_battery_voltage"]]


def _build_stage02_normalized_frame(training_csv: Path, scan_csv: Path, *, seconds_per_step: float) -> pd.DataFrame:
    training_df = pd.read_csv(training_csv, encoding="utf-8-sig")
    if training_df.empty or "episode" not in training_df.columns:
        return pd.DataFrame(columns=["episode", "avg_scan_cells_per_second", "avg_scan_cells_per_volt_drop"])

    working = pd.DataFrame()
    working["episode"] = pd.to_numeric(training_df["episode"], errors="coerce")
    working["length"] = pd.to_numeric(training_df.get("length"), errors="coerce")
    if "global_scanned_cells" in training_df.columns:
        working["global_scanned_cells"] = pd.to_numeric(training_df["global_scanned_cells"], errors="coerce")
    else:
        working["global_scanned_cells"] = pd.to_numeric(training_df.get("scanned_cells"), errors="coerce")

    duration = working["length"] * float(seconds_per_step)
    duration = duration.where(duration > 0)
    working["avg_scan_cells_per_second"] = working["global_scanned_cells"] / duration

    scan_df = pd.read_csv(scan_csv, encoding="utf-8-sig")
    terminal_battery = _episode_terminal_battery(scan_df)
    if not terminal_battery.empty:
        working = working.merge(terminal_battery, on="episode", how="left")
    elif "terminal_battery_voltage" in training_df.columns:
        working["terminal_min_battery_voltage"] = pd.to_numeric(
            training_df["terminal_battery_voltage"],
            errors="coerce",
        )
    else:
        working["terminal_min_battery_voltage"] = np.nan

    voltage_drop = 4.2 - pd.to_numeric(working["terminal_min_battery_voltage"], errors="coerce")
    voltage_drop = voltage_drop.where(voltage_drop > 1e-6)
    working["avg_scan_cells_per_volt_drop"] = working["global_scanned_cells"] / voltage_drop
    return working[["episode", "avg_scan_cells_per_second", "avg_scan_cells_per_volt_drop"]]


def _plot_stage02_normalized_metric(
    analyzer: UnifiedTrainingAnalyzer,
    *,
    frames: dict[str, pd.DataFrame],
    metric: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    plotted = False
    for algo_id, frame in frames.items():
        curve_df = analyzer._build_curve_with_band([frame], "episode", metric)
        if curve_df.empty:
            continue
        style = analyzer._get_algo_style(algo_id)
        analyzer._plot_curve_with_band(
            ax,
            curve_df,
            "episode",
            label=analyzer.ALGO_NAME_MAP.get(algo_id, algo_id),
            color=style["color"],
            linestyle=style["linestyle"],
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel(EPISODE_XLABEL, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(
        title="算法类型",
        title_fontsize=13,
        fontsize=11,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _generate_stage02_normalized_comparison_plots(stage_paths: dict[str, dict[str, Path]], output_dir: Path) -> None:
    analyzer = UnifiedTrainingAnalyzer(output_dir=str(output_dir))
    frames = {}
    for algo_id, paths in stage_paths.items():
        frame = _build_stage02_normalized_frame(
            paths["training"],
            paths["scan"],
            seconds_per_step=SECONDS_PER_STEP[algo_id],
        )
        if not frame.empty:
            frames[algo_id] = frame

    for config in STAGE02_NORMALIZED_PLOTS:
        _plot_stage02_normalized_metric(
            analyzer,
            frames=frames,
            metric=config["metric"],
            output_path=output_dir / config["filename"],
            title=config["title"],
            ylabel=config["ylabel"],
        )


def _generate_comparison_suite(stage_name: str, stage_paths: dict[str, dict[str, Path]], output_dir: Path) -> None:
    ensure_dir(output_dir)
    temp_root = output_dir / f".tmp_{stage_name}_comparison_inputs"
    try:
        copied_dirs = _copy_stage_inputs_to_temp(stage_paths, temp_root)
        analyzer = UnifiedTrainingAnalyzer(output_dir=str(output_dir))
        analyzer.load_data([str(path) for path in copied_dirs])
        for metric, data_type, x_axis in STAGE_COMPARISON_METRICS[stage_name]:
            analyzer.plot_comparison(metric=metric, data_type=data_type, x_axis=x_axis)
        if stage_name == "stage02":
            _generate_stage02_normalized_comparison_plots(stage_paths, output_dir)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _project_stage_data_paths(project_root: Path) -> dict[str, dict[str, dict[str, Path]]]:
    result: dict[str, dict[str, dict[str, Path]]] = {"stage01": {}, "stage02": {}}
    for stage_name in ("stage01", "stage02"):
        for algo_id, log_dir_rel in ALGO_LOG_DIRS.items():
            log_dir = project_root / log_dir_rel
            result[stage_name][algo_id] = {
                "scan": _latest_nonempty_file(log_dir, f"scan_data_*{stage_name}*.csv"),
                "training": _latest_nonempty_file(
                    log_dir,
                    f"{ALGO_TRAINING_PREFIX[algo_id]}*{stage_name}*.csv",
                ),
            }
    return result


def _source_analysis_dirs(project_root: Path) -> dict[str, Path]:
    return {
        "ddpg_stage01": project_root / "analysis_results" / "stage01_analysis_suite" / "ddpg_stage01",
        "ddpg_stage02": project_root / "analysis_results" / "stage02_analysis_suite" / "ddpg_stage02",
        "dqn_stage01": project_root / "analysis_results" / "stage01_analysis_suite" / "dqn_stage01",
        "dqn_stage02": project_root / "analysis_results" / "stage02_analysis_suite" / "dqn_stage02",
        "comparison_stage01": project_root / "analysis_results" / "stage01_analysis_suite" / "comparison",
        "comparison_stage02": project_root / "analysis_results" / "stage02_analysis_suite" / "comparison",
    }


def build_two_stage_analysis_suite(project_root: str | Path, output_root: str | Path | None = None) -> None:
    project_root = Path(project_root)
    source_dirs = _source_analysis_dirs(project_root)
    output_root = Path(output_root) if output_root else project_root / "analysis_results" / "two_stage_analysis_suite"

    stage_data_paths = _project_stage_data_paths(project_root)

    for stage_name, algo_stage_leaves in ALGO_STAGE_LEAVES.items():
        for algo_id, leaf in algo_stage_leaves.items():
            source_dir = project_root / "analysis_results" / SINGLE_STAGE_ASSETS[stage_name] / leaf
            _plot_legacy_single_algorithm_suite(
                stage_data_paths[stage_name][algo_id]["scan"],
                stage_data_paths[stage_name][algo_id]["training"],
                source_dir,
                stage_name,
            )

        comparison_dir = project_root / "analysis_results" / SINGLE_STAGE_ASSETS[stage_name] / "comparison"
        _generate_comparison_suite(stage_name, stage_data_paths[stage_name], comparison_dir)

    ensure_dir(output_root)
    ensure_dir(output_root / "ddpg_two_stage")
    ensure_dir(output_root / "dqn_two_stage")
    ensure_dir(output_root / "comparison")

    for target_group, stages in ALGORITHM_STAGE_DIRS.items():
        for stage_name, source_leaf in stages.items():
            source_dir = project_root / "analysis_results" / SINGLE_STAGE_ASSETS[stage_name] / source_leaf
            target_dir = output_root / target_group / stage_name
            _copy_stage_assets(source_dir, target_dir)

    _copy_stage_assets(source_dirs["comparison_stage01"], output_root / "comparison" / "stage01")
    _copy_stage_assets(source_dirs["comparison_stage02"], output_root / "comparison" / "stage02")

    metrics_csv = output_root / "two_stage_key_metrics.csv"
    if metrics_csv.exists():
        generate_two_stage_plots(metrics_csv, output_root)


def main() -> int:
    parser = argparse.ArgumentParser(description="重建 stage01/stage02/two_stage 分析图，并统一为当前论文风格")
    parser.add_argument("--project-root", type=str, default=None, help="项目根目录")
    parser.add_argument("--out", type=str, default=None, help="two_stage_analysis_suite 输出目录")
    args = parser.parse_args()

    project_root = Path(args.project_root) if args.project_root else Path(__file__).resolve().parents[2]
    output_root = Path(args.out) if args.out else project_root / "analysis_results" / "two_stage_analysis_suite"
    if not output_root.is_absolute():
        output_root = project_root / output_root

    build_two_stage_analysis_suite(project_root, output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
