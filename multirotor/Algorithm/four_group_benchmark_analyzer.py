from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def configure_plot_fonts():
    fonts = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]
    plt.rcParams["font.sans-serif"] = fonts
    plt.rcParams["axes.unicode_minus"] = False
    return fonts


def _coerce_numeric(frame: pd.DataFrame, column: str) -> None:
    if column not in frame.columns:
        frame[column] = np.nan
        return
    frame[column] = pd.to_numeric(
        frame[column].astype(str).str.replace("%", "", regex=False),
        errors="coerce",
    )


def _ci95(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) <= 1:
        return 0.0
    return float(1.96 * clean.std(ddof=1) / np.sqrt(len(clean)))


def _write_boxplot(frame: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    plot_frame = frame[["algorithm_type", metric]].dropna()
    plt.figure(figsize=(9, 5))
    if plot_frame.empty:
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.axis("off")
    else:
        plot_frame.boxplot(column=metric, by="algorithm_type", grid=False)
        plt.suptitle("")
        plt.title(title)
        plt.xlabel("algorithm_type")
        plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _write_bar(summary: pd.DataFrame, column: str, output_path: Path, title: str) -> None:
    plt.figure(figsize=(9, 5))
    plt.bar(summary["algorithm_type"], summary[column], color="#457b9d")
    plt.title(title)
    plt.ylabel(column)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _write_reset_reason_stacked_bar(frame: pd.DataFrame, output_path: Path) -> None:
    reason_frame = frame.copy()
    if "reset_reason" not in reason_frame.columns:
        reason_frame["reset_reason"] = "unknown"
    reason_frame["reset_reason"] = (
        reason_frame["reset_reason"].fillna("").astype(str).str.strip().replace("", "unknown")
    )
    counts = pd.crosstab(reason_frame["algorithm_type"], reason_frame["reset_reason"])
    normalized = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    normalized.plot(kind="bar", stacked=True, figsize=(10, 5), colormap="tab20")
    plt.title("Reset Reason Distribution")
    plt.ylabel("ratio")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def generate_four_group_benchmark_report(
    *,
    eval_csv_path: str | Path,
    output_dir: str | Path,
) -> Dict[str, Path]:
    eval_csv_path = Path(eval_csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_fonts()

    eval_frame = pd.read_csv(eval_csv_path, encoding="utf-8-sig")
    if eval_frame.empty:
        raise ValueError("evaluation CSV is empty")

    for numeric_column in (
        "seed",
        "episode",
        "success_flag",
        "final_global_scan_ratio",
        "final_global_avg_entropy",
        "scan_efficiency",
        "avg_scan_cells_per_second",
        "avg_scan_cells_per_volt_drop",
        "collision_count",
    ):
        _coerce_numeric(eval_frame, numeric_column)

    episodes_csv = output_dir / "four_group_eval_episodes.csv"
    eval_frame.to_csv(episodes_csv, index=False, encoding="utf-8-sig")

    seed_summary = (
        eval_frame.groupby(["algorithm_type", "seed"], dropna=False)
        .agg(
            success_rate=("success_flag", "mean"),
            mean_final_global_scan_ratio=("final_global_scan_ratio", "mean"),
            mean_final_global_avg_entropy=("final_global_avg_entropy", "mean"),
            mean_scan_efficiency=("scan_efficiency", "mean"),
            mean_scan_cells_per_second=("avg_scan_cells_per_second", "mean"),
            mean_scan_cells_per_volt_drop=("avg_scan_cells_per_volt_drop", "mean"),
            mean_collision_count=("collision_count", "mean"),
        )
        .reset_index()
    )
    seed_summary_csv = output_dir / "four_group_eval_seed_summary.csv"
    seed_summary.to_csv(seed_summary_csv, index=False, encoding="utf-8-sig")

    summary = (
        seed_summary.groupby("algorithm_type", dropna=False)
        .agg(
            seed_count=("seed", "nunique"),
            success_rate_mean=("success_rate", "mean"),
            success_rate_std=("success_rate", "std"),
            final_global_scan_ratio_mean=("mean_final_global_scan_ratio", "mean"),
            final_global_scan_ratio_std=("mean_final_global_scan_ratio", "std"),
            final_global_scan_ratio_median=("mean_final_global_scan_ratio", "median"),
            final_global_scan_ratio_ci95=("mean_final_global_scan_ratio", _ci95),
            final_global_avg_entropy_mean=("mean_final_global_avg_entropy", "mean"),
            final_global_avg_entropy_std=("mean_final_global_avg_entropy", "std"),
            final_global_avg_entropy_median=("mean_final_global_avg_entropy", "median"),
            final_global_avg_entropy_ci95=("mean_final_global_avg_entropy", _ci95),
            scan_efficiency_mean=("mean_scan_efficiency", "mean"),
            scan_efficiency_std=("mean_scan_efficiency", "std"),
            collision_count_mean=("mean_collision_count", "mean"),
            collision_count_std=("mean_collision_count", "std"),
            avg_scan_cells_per_second_mean=("mean_scan_cells_per_second", "mean"),
            avg_scan_cells_per_volt_drop_mean=("mean_scan_cells_per_volt_drop", "mean"),
        )
        .reset_index()
    )
    summary_csv = output_dir / "four_group_summary.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    _write_boxplot(
        eval_frame,
        "final_global_scan_ratio",
        output_dir / "scan_ratio_boxplot.png",
        "Final Global Scan Ratio",
    )
    _write_boxplot(
        eval_frame,
        "final_global_avg_entropy",
        output_dir / "entropy_boxplot.png",
        "Final Global Avg Entropy",
    )
    _write_bar(
        summary,
        "scan_efficiency_mean",
        output_dir / "efficiency_bar.png",
        "Mean Scan Efficiency",
    )

    safety_summary = summary.copy()
    safety_summary["safety_score"] = (
        safety_summary["success_rate_mean"].fillna(0.0)
        - safety_summary["collision_count_mean"].fillna(0.0)
    )
    _write_bar(
        safety_summary,
        "safety_score",
        output_dir / "safety_bar.png",
        "Safety Score",
    )
    _write_reset_reason_stacked_bar(
        eval_frame,
        output_dir / "reset_reason_stacked_bar.png",
    )

    return {
        "episodes_csv": episodes_csv,
        "seed_summary_csv": seed_summary_csv,
        "summary_csv": summary_csv,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate four-group benchmark summary tables and plots.")
    parser.add_argument("--eval-csv", type=str, required=True, help="Input evaluation CSV path.")
    parser.add_argument("--out", type=str, required=True, help="Output directory.")
    args = parser.parse_args()
    generate_four_group_benchmark_report(
        eval_csv_path=args.eval_csv,
        output_dir=args.out,
    )


if __name__ == "__main__":
    main()
