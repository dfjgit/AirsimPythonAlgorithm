from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STAGE_ORDER = ["stage01", "stage02"]
STAGE_LABELS = {"stage01": "第一阶段", "stage02": "第二阶段"}
ALGORITHM_COLORS = {
    "DDPG+APF": "#1f77b4",
    "纯DQN": "#d62728",
}
SERIES_COLORS = {
    "avg": "#4e79a7",
    "tail": "#f28e2b",
}
SUMMARY_LEGEND_Y = 0.955
SUMMARY_LAYOUT_TOP = 0.89
ALGORITHM_TRANSITION_CHARTS = [
    ("奖励变化", "avg_reward", "tail_reward", "平均奖励"),
    ("步长变化", "avg_length", "tail_length", "平均步长"),
    ("扫描率变化(%)", "avg_scan_ratio_pct", "tail_scan_ratio_pct", "平均扫描率 (%)"),
    ("平均熵变化", "avg_entropy", "tail_entropy", "平均熵"),
]
RESULT_COMPARISON_TITLES = ("平均最终扫描率", "平均最终全局熵")
EFFICIENCY_SUBPLOT_TITLES = (
    "平均扫描效率（格/步）",
    "第二阶段按时间归一化产出",
    "第二阶段按电量归一化产出",
)


def setup_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.max_open_warning"] = 0


setup_plot_style()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_metrics(metrics_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv, encoding="utf-8-sig")
    numeric_columns = [
        "episodes",
        "avg_reward",
        "tail_reward",
        "avg_length",
        "tail_length",
        "avg_scan_efficiency",
        "tail_scan_efficiency",
        "avg_scan_ratio_pct",
        "tail_scan_ratio_pct",
        "avg_entropy",
        "tail_entropy",
        "avg_collision_count",
        "tail_collision_count",
        "avg_out_of_range_count",
        "tail_out_of_range_count",
        "avg_scan_cells_per_second",
        "avg_scan_cells_per_volt_drop",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["stage"] = pd.Categorical(df["stage"], categories=STAGE_ORDER, ordered=True)
    return df.sort_values(["algorithm", "stage"]).reset_index(drop=True)


def _annotate_bars(ax, bars, fmt: str = "{:.2f}") -> None:
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _apply_summary_legend_layout(fig, handles, labels) -> None:
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, SUMMARY_LEGEND_Y),
    )
    fig.tight_layout(rect=[0, 0, 1, SUMMARY_LAYOUT_TOP])


def _plot_algorithm_transition(df: pd.DataFrame, algorithm: str, output_path: Path) -> None:
    algo_df = df[df["algorithm"] == algorithm].sort_values("stage")
    x = np.arange(len(algo_df))
    width = 0.34

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{algorithm} 二阶段阶段跃迁说明图", fontsize=16, fontweight="bold")

    charts = [
        (title, avg_col, tail_col, axes[row, col], ylabel)
        for (row, col), (title, avg_col, tail_col, ylabel) in zip(
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            ALGORITHM_TRANSITION_CHARTS,
        )
    ]

    for title, avg_col, tail_col, ax, ylabel in charts:
        avg_values = algo_df[avg_col].to_numpy(dtype=float)
        tail_values = algo_df[tail_col].to_numpy(dtype=float)
        avg_bars = ax.bar(x - width / 2, avg_values, width=width, color=SERIES_COLORS["avg"], label="全阶段平均")
        tail_bars = ax.bar(x + width / 2, tail_values, width=width, color=SERIES_COLORS["tail"], label="后20轮平均")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([STAGE_LABELS[str(stage)] for stage in algo_df["stage"]])
        ax.set_ylabel(ylabel)
        _annotate_bars(ax, avg_bars)
        _annotate_bars(ax, tail_bars)
        if title == "平均熵变化":
            ax.text(
                0.02,
                0.96,
                "说明: 熵越低越好",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="#444444",
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    _apply_summary_legend_layout(fig, handles, labels)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_result_comparison(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("二阶段结果对比说明图", fontsize=16, fontweight="bold")
    x = np.arange(len(STAGE_ORDER))

    for algorithm, algo_df in df.groupby("algorithm"):
        algo_df = algo_df.sort_values("stage")
        color = ALGORITHM_COLORS.get(algorithm, None)
        axes[0].plot(
            x,
            algo_df["avg_scan_ratio_pct"].to_numpy(dtype=float),
            marker="o",
            linewidth=2.6,
            color=color,
            label=algorithm,
        )
        axes[1].plot(
            x,
            algo_df["avg_entropy"].to_numpy(dtype=float),
            marker="o",
            linewidth=2.6,
            color=color,
            label=algorithm,
        )

        for idx, value in enumerate(algo_df["avg_scan_ratio_pct"].to_numpy(dtype=float)):
            axes[0].annotate(f"{value:.2f}", (x[idx], value), textcoords="offset points", xytext=(0, 6), ha="center")
        for idx, value in enumerate(algo_df["avg_entropy"].to_numpy(dtype=float)):
            axes[1].annotate(f"{value:.2f}", (x[idx], value), textcoords="offset points", xytext=(0, 6), ha="center")

    axes[0].set_title(RESULT_COMPARISON_TITLES[0])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([STAGE_LABELS[s] for s in STAGE_ORDER])
    axes[0].set_ylabel("扫描率 (%)")

    axes[1].set_title(RESULT_COMPARISON_TITLES[1])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([STAGE_LABELS[s] for s in STAGE_ORDER])
    axes[1].set_ylabel("全局平均熵")
    axes[1].text(0.02, 0.96, "说明: 熵越低越好", transform=axes[1].transAxes, ha="left", va="top", fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    _apply_summary_legend_layout(fig, handles, labels)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_stage_gain_comparison(df: pd.DataFrame, output_path: Path) -> None:
    rows = []
    for algorithm, algo_df in df.groupby("algorithm"):
        algo_df = algo_df.sort_values("stage")
        if len(algo_df) != 2:
            continue
        stage01 = algo_df.iloc[0]
        stage02 = algo_df.iloc[1]
        rows.append(
            {
                "algorithm": algorithm,
                "scan_ratio_gain": stage02["avg_scan_ratio_pct"] - stage01["avg_scan_ratio_pct"],
                "entropy_drop": stage01["avg_entropy"] - stage02["avg_entropy"],
                "reward_gain": stage02["avg_reward"] - stage01["avg_reward"],
                "length_gain": stage02["avg_length"] - stage01["avg_length"],
            }
        )
    gains = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("二阶段阶段增益对比说明图", fontsize=16, fontweight="bold")

    configs = [
        ("扫描率增益(百分点)", "scan_ratio_gain", axes[0, 0], "#59a14f"),
        ("熵下降量", "entropy_drop", axes[0, 1], "#e15759"),
        ("平均奖励增量", "reward_gain", axes[1, 0], "#4e79a7"),
        ("平均步长增量", "length_gain", axes[1, 1], "#f28e2b"),
    ]

    positions = np.arange(len(gains))
    labels = gains["algorithm"].tolist()
    for title, column, ax, color in configs:
        values = gains[column].to_numpy(dtype=float)
        bars = ax.bar(positions, values, color=[ALGORITHM_COLORS.get(label, color) for label in labels], width=0.58)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        _annotate_bars(ax, bars)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_efficiency_comparison(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    fig.suptitle("二阶段效率与归一化产出说明图", fontsize=16, fontweight="bold")
    x = np.arange(len(STAGE_ORDER))

    for algorithm, algo_df in df.groupby("algorithm"):
        algo_df = algo_df.sort_values("stage")
        color = ALGORITHM_COLORS.get(algorithm, None)
        axes[0].plot(
            x,
            algo_df["avg_scan_efficiency"].to_numpy(dtype=float),
            marker="o",
            linewidth=2.6,
            color=color,
            label=algorithm,
        )
        for idx, value in enumerate(algo_df["avg_scan_efficiency"].to_numpy(dtype=float)):
            axes[0].annotate(f"{value:.2f}", (x[idx], value), textcoords="offset points", xytext=(0, 6), ha="center")

    stage02_df = df[df["stage"] == "stage02"].sort_values("algorithm")
    positions = np.arange(len(stage02_df))
    bars_per_second = axes[1].bar(
        positions,
        stage02_df["avg_scan_cells_per_second"].to_numpy(dtype=float),
        color=[ALGORITHM_COLORS.get(name, "#999999") for name in stage02_df["algorithm"]],
        width=0.58,
    )
    bars_per_volt = axes[2].bar(
        positions,
        stage02_df["avg_scan_cells_per_volt_drop"].to_numpy(dtype=float),
        color=[ALGORITHM_COLORS.get(name, "#999999") for name in stage02_df["algorithm"]],
        width=0.58,
    )

    axes[0].set_title(EFFICIENCY_SUBPLOT_TITLES[0])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([STAGE_LABELS[s] for s in STAGE_ORDER])
    axes[0].set_ylabel("扫描效率（格/步）")

    axes[1].set_title(EFFICIENCY_SUBPLOT_TITLES[1])
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(stage02_df["algorithm"].tolist())
    axes[1].set_ylabel("单位时间扫描产出（格/秒）")
    _annotate_bars(axes[1], bars_per_second)

    axes[2].set_title(EFFICIENCY_SUBPLOT_TITLES[2])
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(stage02_df["algorithm"].tolist())
    axes[2].set_ylabel("单位电量扫描产出（格/伏）")
    _annotate_bars(axes[2], bars_per_volt)

    handles, labels = axes[0].get_legend_handles_labels()
    _apply_summary_legend_layout(fig, handles, labels)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def generate_two_stage_plots(metrics_csv: str | Path, output_root: str | Path) -> None:
    metrics_csv = Path(metrics_csv)
    output_root = Path(output_root)
    ensure_dir(output_root)
    ensure_dir(output_root / "ddpg_two_stage")
    ensure_dir(output_root / "dqn_two_stage")
    ensure_dir(output_root / "comparison")

    df = load_metrics(metrics_csv)
    _plot_algorithm_transition(df, "DDPG+APF", output_root / "ddpg_two_stage" / "ddpg_stage_transition_summary.png")
    _plot_algorithm_transition(df, "纯DQN", output_root / "dqn_two_stage" / "dqn_stage_transition_summary.png")
    _plot_result_comparison(df, output_root / "comparison" / "two_stage_result_comparison.png")
    _plot_stage_gain_comparison(df, output_root / "comparison" / "two_stage_stage_gain_comparison.png")
    _plot_efficiency_comparison(df, output_root / "comparison" / "two_stage_efficiency_comparison.png")


def main() -> int:
    parser = argparse.ArgumentParser(description="二阶段总分析说明图生成器")
    parser.add_argument("--metrics", type=str, help="二阶段汇总 CSV 路径")
    parser.add_argument("--out", type=str, help="输出目录")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    metrics_csv = Path(args.metrics) if args.metrics else project_root / "analysis_results" / "two_stage_analysis_suite" / "two_stage_key_metrics.csv"
    output_root = Path(args.out) if args.out else project_root / "analysis_results" / "two_stage_analysis_suite"
    if not output_root.is_absolute():
        output_root = project_root / output_root

    generate_two_stage_plots(metrics_csv, output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
