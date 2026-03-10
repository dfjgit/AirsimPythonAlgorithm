from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("scan_csv_visualizer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def setup_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.max_open_warning"] = 0


setup_plot_style()


PERCENT_COLUMNS = {
    "scan_ratio",
    "local_scan_ratio",
    "global_scan_ratio",
    "max_global_scan_ratio",
}

WEIGHT_COLUMNS = [
    "repulsion_coefficient",
    "entropy_coefficient",
    "distance_coefficient",
    "leader_range_coefficient",
    "direction_retention_coefficient",
]


class RunData:
    def __init__(self, scan_path: Path, training_path: Path | None, output_dir: Path):
        self.scan_path = scan_path
        self.training_path = training_path
        self.output_dir = output_dir
        self.scan_df = self._load_csv(scan_path)
        self.training_df = self._load_csv(training_path) if training_path else pd.DataFrame()
        self.drones = detect_drones(self.scan_df.columns.tolist())
        self.entropy_bins = parse_json_column(self.scan_df, "entropy_bins")
        self.entropy_hist = parse_json_column(self.scan_df, "entropy_hist")
        self.entropy_cdf = parse_json_column(self.scan_df, "entropy_cdf")
        self.episode_df = self._build_episode_df()

    @staticmethod
    def _load_csv(path: Path | None) -> pd.DataFrame:
        if path is None or not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        df = pd.read_csv(path, encoding="utf-8-sig")
        if df.empty:
            return df
        for col in df.columns:
            if col in PERCENT_COLUMNS:
                df[col] = normalize_percent_series(df[col])
        if "elapsed_time" in df.columns:
            df["elapsed_time"] = pd.to_numeric(df["elapsed_time"], errors="coerce")
        if "episode_elapsed_time" in df.columns:
            df["episode_elapsed_time"] = pd.to_numeric(df["episode_elapsed_time"], errors="coerce")
        if "step" in df.columns:
            df["step"] = pd.to_numeric(df["step"], errors="coerce")
        if "episode" in df.columns:
            df["episode"] = pd.to_numeric(df["episode"], errors="coerce")
        if "reward" in df.columns:
            df["reward"] = pd.to_numeric(df["reward"], errors="coerce")
        if "length" in df.columns:
            df["length"] = pd.to_numeric(df["length"], errors="coerce")
        if "global_avg_entropy" in df.columns:
            df["global_avg_entropy"] = pd.to_numeric(df["global_avg_entropy"], errors="coerce")
        if "min_global_avg_entropy" in df.columns:
            df["min_global_avg_entropy"] = pd.to_numeric(df["min_global_avg_entropy"], errors="coerce")
        return df

    def _build_episode_df(self) -> pd.DataFrame:
        if not self.training_df.empty and {"episode", "reward", "length"}.issubset(self.training_df.columns):
            episode_df = self.training_df.copy()
            rename_map = {
                "reward": "episode_reward",
                "length": "episode_length",
                "max_global_scan_ratio": "episode_scan_ratio",
                "min_global_avg_entropy": "episode_min_entropy",
            }
            episode_df = episode_df.rename(columns=rename_map)
            episode_df["episode"] = pd.to_numeric(episode_df["episode"], errors="coerce")
            episode_df["episode_reward"] = pd.to_numeric(episode_df["episode_reward"], errors="coerce")
            episode_df["episode_length"] = pd.to_numeric(episode_df["episode_length"], errors="coerce")
            if "episode_scan_ratio" in episode_df.columns:
                episode_df["episode_scan_ratio"] = normalize_percent_series(episode_df["episode_scan_ratio"])
            if "episode_min_entropy" in episode_df.columns:
                episode_df["episode_min_entropy"] = pd.to_numeric(
                    episode_df["episode_min_entropy"], errors="coerce"
                )
            episode_df["reset_reason"] = episode_df.get("reset_reason", "").fillna("").astype(str).str.strip()
            episode_df["collision_object_name"] = (
                episode_df.get("collision_object_name", "").fillna("").astype(str).str.strip()
            )
            episode_df["collision_position"] = (
                episode_df.get("collision_position", "").fillna("").astype(str).str.strip()
            )
            episode_df = episode_df.sort_values("episode").dropna(subset=["episode"])
            return episode_df

        if self.scan_df.empty or "episode" not in self.scan_df.columns:
            return pd.DataFrame()

        working = self.scan_df.copy()
        working = working.dropna(subset=["episode"]) 
        working["episode"] = working["episode"].astype(int)
        if "step" in working.columns:
            working["step"] = pd.to_numeric(working["step"], errors="coerce").fillna(0)
        else:
            working["step"] = 0
        working["reset_reason"] = working.get("reset_reason", "").fillna("").astype(str).str.strip()
        terminal = working[(working["step"] > 0) & (working["reset_reason"] != "")].copy()
        if terminal.empty:
            terminal = working.sort_values(["episode", "step"]).groupby("episode", as_index=False).tail(1)

        grouped = working.groupby("episode", as_index=False).agg(
            episode_reward=("episode_reward", "max"),
            episode_length=("step", "max"),
            episode_scan_ratio=("global_scan_ratio", "max"),
            episode_min_entropy=("global_avg_entropy", "min"),
        )
        merge_cols = [c for c in ["episode", "reset_reason", "collision_object_name", "collision_position"] if c in terminal.columns]
        terminal = terminal[merge_cols].drop_duplicates(subset=["episode"], keep="last")
        episode_df = grouped.merge(terminal, on="episode", how="left")
        episode_df["reset_reason"] = episode_df.get("reset_reason", "").fillna("").astype(str).str.strip()
        episode_df["collision_object_name"] = episode_df.get("collision_object_name", "").fillna("")
        episode_df["collision_position"] = episode_df.get("collision_position", "").fillna("")
        return episode_df.sort_values("episode")


def normalize_percent_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace("%", "", regex=False), errors="coerce")


def parse_json_column(df: pd.DataFrame, column: str) -> list[list[float]]:
    if df.empty or column not in df.columns:
        return []
    result = []
    for value in df[column].fillna(""):
        try:
            parsed = json.loads(str(value))
            if isinstance(parsed, list):
                result.append([float(x) for x in parsed])
            else:
                result.append([])
        except Exception:
            result.append([])
    return result


def detect_drones(columns: list[str]) -> list[str]:
    drones = set()
    for col in columns:
        if col.endswith("_x"):
            drones.add(col[:-2])
    return sorted(drones)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def moving_average(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def parse_position(value: str) -> tuple[float, float] | None:
    text = str(value).strip()
    if not text:
        return None
    parts = text.split(",")
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def plot_episode_performance_summary(run: RunData) -> None:
    df = run.episode_df
    if df.empty:
        return
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    x = df["episode"]
    configs = [
        ("episode_reward", "Episode Reward", axes[0], "tab:blue"),
        ("episode_scan_ratio", "Max Global Scan Ratio (%)", axes[1], "tab:green"),
        ("episode_min_entropy", "Min Global Avg Entropy", axes[2], "tab:red"),
    ]
    for col, title, ax, color in configs:
        if col not in df.columns:
            continue
        ax.plot(x, df[col], alpha=0.25, color=color, linewidth=1.0, label="raw")
        ax.plot(x, moving_average(df[col]), color=color, linewidth=2.4, label="MA20")
        ax.set_ylabel(title)
        ax.legend(loc="best")
    axes[2].set_xlabel("Episode")
    fig.suptitle("Episode Performance Summary", fontsize=16)
    fig.tight_layout()
    fig.savefig(run.output_dir / "episode_performance_summary.png", dpi=160)
    plt.close(fig)


def plot_reset_reason_rolling_ratio(run: RunData) -> None:
    df = run.episode_df
    if df.empty or "reset_reason" not in df.columns:
        return
    reason_counts = df["reset_reason"].fillna("").astype(str).str.strip()
    reason_counts = reason_counts[reason_counts != ""].value_counts()
    reasons = reason_counts.head(4).index.tolist()
    if not reasons:
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    x = df["episode"]
    for reason in reasons:
        values = (df["reset_reason"] == reason).astype(float)
        ax.plot(x, moving_average(values, 20) * 100.0, linewidth=2.2, label=reason)
    ax.set_title("Reset Reason Rolling Ratio (20 episodes)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Ratio (%)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(run.output_dir / "reset_reason_rolling_ratio.png", dpi=160)
    plt.close(fig)


def plot_collision_hotspots(run: RunData) -> None:
    df = run.episode_df
    if df.empty or "collision_position" not in df.columns:
        return
    points = [parse_position(v) for v in df.loc[df["collision_position"].astype(str).str.len() > 0, "collision_position"]]
    points = [p for p in points if p is not None]
    if not points:
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(xs, ys, s=60, alpha=0.75, c=np.arange(len(xs)), cmap="Reds")
    ax.set_title("Collision Hotspots (XY)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(run.output_dir / "collision_hotspots_xy.png", dpi=160)
    plt.close(fig)


def plot_collision_object_breakdown(run: RunData) -> None:
    df = run.episode_df
    if df.empty or "collision_object_name" not in df.columns:
        return
    series = df.loc[df["collision_object_name"].astype(str).str.len() > 0, "collision_object_name"]
    if series.empty:
        return
    counts = series.value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(counts.index.astype(str), counts.values, color="#d55e00")
    ax.set_title("Collision Object Breakdown")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(run.output_dir / "collision_object_breakdown.png", dpi=160)
    plt.close(fig)


def plot_algorithm_weights_stability(run: RunData) -> None:
    df = run.scan_df
    if df.empty:
        return
    cols = [c for c in WEIGHT_COLUMNS if c in df.columns]
    if not cols or "elapsed_time" not in df.columns:
        return
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    for col in cols:
        numeric = pd.to_numeric(df[col], errors="coerce")
        axes[0].plot(df["elapsed_time"], moving_average(numeric, 50), linewidth=2, label=col)
        axes[1].plot(df["elapsed_time"], numeric.rolling(50, min_periods=5).std(), linewidth=1.8, label=col)
    axes[0].set_title("Weight Rolling Mean (window=50)")
    axes[1].set_title("Weight Rolling Std (window=50)")
    axes[1].set_xlabel("Elapsed Time (s)")
    axes[0].legend(loc="best", ncol=2)
    axes[1].legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(run.output_dir / "algorithm_weights_stability.png", dpi=160)
    plt.close(fig)


def _episode_xy(run: RunData, episode: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if run.scan_df.empty or "episode" not in run.scan_df.columns:
        return {}
    subset = run.scan_df[run.scan_df["episode"] == episode]
    result = {}
    for drone in run.drones:
        x_col = f"{drone}_x"
        y_col = f"{drone}_y"
        if x_col in subset.columns and y_col in subset.columns:
            x = pd.to_numeric(subset[x_col], errors="coerce").to_numpy()
            y = pd.to_numeric(subset[y_col], errors="coerce").to_numpy()
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.any():
                result[drone] = (x[mask], y[mask])
    return result


def plot_best_vs_recent_trajectory_comparison(run: RunData) -> None:
    df = run.episode_df
    if df.empty or not run.drones:
        return
    best_row = df.sort_values(["episode_scan_ratio", "episode_reward"], ascending=[False, False]).iloc[0]
    recent_row = df.iloc[-1]
    selections = [("best", int(best_row["episode"])), ("recent", int(recent_row["episode"]))]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    has_data = False
    for ax, (label, episode) in zip(axes, selections):
        xy = _episode_xy(run, episode)
        if not xy:
            continue
        has_data = True
        for drone, (x, y) in xy.items():
            ax.plot(x, y, linewidth=1.8, label=drone)
        ax.set_title(f"{label.title()} Episode #{episode}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.25)
    if has_data:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, loc="best")
        fig.tight_layout()
        fig.savefig(run.output_dir / "best_vs_recent_trajectory_comparison.png", dpi=160)
    plt.close(fig)


def plot_scan_progress(run: RunData) -> None:
    df = run.episode_df
    if df.empty or "episode_scan_ratio" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["episode"], df["episode_scan_ratio"], alpha=0.35, label="raw")
    ax.plot(df["episode"], moving_average(df["episode_scan_ratio"], 20), linewidth=2.4, label="MA20")
    ax.set_title("Episode Scan Progress")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Max Global Scan Ratio (%)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(run.output_dir / "scan_progress.png", dpi=160)
    plt.close(fig)


def plot_entropy_trend(run: RunData) -> None:
    df = run.episode_df
    if df.empty or "episode_min_entropy" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["episode"], df["episode_min_entropy"], alpha=0.35, label="raw")
    ax.plot(df["episode"], moving_average(df["episode_min_entropy"], 20), linewidth=2.4, label="MA20")
    ax.set_title("Episode Min Entropy Trend")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Min Global Avg Entropy")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(run.output_dir / "entropy_trend.png", dpi=160)
    plt.close(fig)


def plot_trajectories_xy(run: RunData) -> None:
    df = run.scan_df
    if df.empty or not run.drones:
        return
    fig, ax = plt.subplots(figsize=(8, 8))
    plotted = False
    for drone in run.drones:
        x_col = f"{drone}_x"
        y_col = f"{drone}_y"
        if x_col in df.columns and y_col in df.columns:
            x = pd.to_numeric(df[x_col], errors="coerce")
            y = pd.to_numeric(df[y_col], errors="coerce")
            ax.plot(x, y, linewidth=0.9, alpha=0.7, label=drone)
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_title("All Trajectories (XY)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(run.output_dir / "trajectories_xy.png", dpi=160)
    plt.close(fig)


def plot_trajectories_3d(run: RunData) -> None:
    df = run.scan_df
    if df.empty or not run.drones:
        return
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    plotted = False
    for drone in run.drones:
        x_col = f"{drone}_x"
        y_col = f"{drone}_y"
        z_col = f"{drone}_z"
        if all(c in df.columns for c in [x_col, y_col, z_col]):
            x = pd.to_numeric(df[x_col], errors="coerce")
            y = pd.to_numeric(df[y_col], errors="coerce")
            z = pd.to_numeric(df[z_col], errors="coerce")
            ax.plot(x, y, z, linewidth=0.9, alpha=0.7, label=drone)
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_title("All Trajectories (3D)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(run.output_dir / "trajectories_3d.png", dpi=160)
    plt.close(fig)


def plot_uncertainty_elimination_efficiency(run: RunData) -> None:
    df = run.episode_df
    if df.empty or not {"episode_scan_ratio", "episode_length"}.issubset(df.columns):
        return
    denom = df["episode_length"].replace(0, np.nan)
    efficiency = df["episode_scan_ratio"] / denom
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["episode"], efficiency, alpha=0.35, label="raw")
    ax.plot(df["episode"], moving_average(efficiency, 20), linewidth=2.4, label="MA20")
    ax.set_title("Uncertainty Elimination Efficiency")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Scan Ratio per Step")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(run.output_dir / "uncertainty_elimination_efficiency.png", dpi=160)
    plt.close(fig)


def pick_snapshot_indices(n_rows: int, max_snapshots: int) -> list[int]:
    if n_rows <= 0:
        return []
    max_snapshots = max(1, max_snapshots)
    return np.linspace(0, n_rows - 1, num=min(n_rows, max_snapshots), dtype=int).tolist()


def plot_entropy_hist_snapshots(run: RunData, snapshots: int) -> None:
    if not run.entropy_bins or not run.entropy_hist or run.scan_df.empty:
        return
    indices = pick_snapshot_indices(len(run.scan_df), snapshots)
    fig, ax = plt.subplots(figsize=(10, 6))
    drawn = False
    for idx in indices:
        bins = run.entropy_bins[idx] if idx < len(run.entropy_bins) else []
        hist = run.entropy_hist[idx] if idx < len(run.entropy_hist) else []
        if not bins or not hist:
            continue
        if len(bins) == len(hist) + 1:
            x = bins[:-1]
            width = bins[1] - bins[0] if len(bins) > 1 else 1.0
        else:
            x = np.arange(len(hist))
            width = 0.8
        elapsed = run.scan_df.iloc[idx].get("elapsed_time", idx)
        ax.bar(x, hist, width=width, alpha=0.25, align="edge", label=f"t={elapsed:.1f}s")
        drawn = True
    if not drawn:
        plt.close(fig)
        return
    ax.set_title("Entropy Histogram Snapshots")
    ax.set_xlabel("Entropy Bin")
    ax.set_ylabel("Cell Count")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(run.output_dir / "entropy_hist_snapshots.png", dpi=160)
    plt.close(fig)


def plot_selected_episode_trajectories(run: RunData) -> None:
    df = run.episode_df
    if df.empty or not run.drones:
        return
    traj_dir = run.output_dir / "episode_trajectories"
    ensure_dir(traj_dir)
    episodes = set()
    top = df.sort_values(["episode_scan_ratio", "episode_reward"], ascending=[False, False]).head(5)
    low = df.sort_values(["episode_scan_ratio", "episode_reward"], ascending=[True, True]).head(5)
    recent = df.tail(5)
    collisions = df[(df.get("collision_object_name", "").astype(str).str.len() > 0) | (df.get("collision_position", "").astype(str).str.len() > 0)].tail(5)
    for subset in [top, low, recent, collisions]:
        episodes.update(int(v) for v in subset["episode"].dropna().tolist())

    if not episodes:
        return

    for episode in sorted(episodes):
        xy = _episode_xy(run, episode)
        if not xy:
            continue
        fig, ax = plt.subplots(figsize=(7, 7))
        for drone, (x, y) in xy.items():
            ax.plot(x, y, linewidth=1.4, label=drone)
        ax.set_title(f"Episode {episode} Trajectory XY")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(traj_dir / f"episode_{episode:03d}_trajectory_xy.png", dpi=140)
        plt.close(fig)


def write_manifest(run: RunData) -> None:
    manifest = {
        "scan_csv": str(run.scan_path),
        "training_csv": str(run.training_path) if run.training_path else None,
        "episodes": int(len(run.episode_df)),
        "charts": sorted([p.name for p in run.output_dir.glob("*.png")]),
    }
    (run.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )


def safe_plot(plot_fn, *args, **kwargs) -> None:
    plot_name = kwargs.pop("plot_name", plot_fn.__name__)
    try:
        plot_fn(*args, **kwargs)
        LOGGER.info("[OK] %s", plot_name)
    except Exception as exc:
        LOGGER.exception("[FAILED] %s: %s", plot_name, exc)


PLOT_PIPELINE = [
    (plot_episode_performance_summary, "episode_performance_summary"),
    (plot_reset_reason_rolling_ratio, "reset_reason_rolling_ratio"),
    (plot_collision_hotspots, "collision_hotspots_xy"),
    (plot_collision_object_breakdown, "collision_object_breakdown"),
    (plot_algorithm_weights_stability, "algorithm_weights_stability"),
    (plot_best_vs_recent_trajectory_comparison, "best_vs_recent_trajectory_comparison"),
    (plot_scan_progress, "scan_progress"),
    (plot_entropy_trend, "entropy_trend"),
    (plot_trajectories_xy, "trajectories_xy"),
    (plot_trajectories_3d, "trajectories_3d"),
    (plot_uncertainty_elimination_efficiency, "uncertainty_elimination_efficiency"),
]


def collect_scan_files(args: argparse.Namespace) -> list[Path]:
    files: list[Path] = []
    if args.csv:
        files.append(Path(args.csv))
    if args.csv_dir:
        files.extend(sorted(Path(args.csv_dir).glob("scan_data_*.csv")))
    deduped = []
    seen = set()
    for path in files:
        resolved = str(path.resolve())
        if resolved not in seen:
            deduped.append(path)
            seen.add(resolved)
    return deduped


def paired_training_csv(scan_csv: Path) -> Path | None:
    stem = scan_csv.stem.replace("scan_data_", "")
    candidate = scan_csv.parent / f"ddpg_training_{stem}.csv"
    return candidate if candidate.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline scan/training CSV visualizer")
    parser.add_argument("--csv", help="Path to one scan_data CSV file")
    parser.add_argument("--csv-dir", help="Directory containing scan_data_*.csv files")
    parser.add_argument("--out", default="analysis_results", help="Output directory")
    parser.add_argument("--snapshots", type=int, default=4, help="Number of entropy snapshots")
    args = parser.parse_args()

    scan_files = collect_scan_files(args)
    if not scan_files:
        raise SystemExit("No scan_data CSV input found.")

    out_root = Path(args.out)
    ensure_dir(out_root)

    for scan_csv in scan_files:
        run_dir = out_root / scan_csv.stem
        ensure_dir(run_dir)
        training_csv = paired_training_csv(scan_csv)
        run = RunData(scan_path=scan_csv, training_path=training_csv, output_dir=run_dir)
        LOGGER.info("Analyzing %s", scan_csv.name)
        for plot_fn, plot_name in PLOT_PIPELINE:
            safe_plot(plot_fn, run, plot_name=plot_name)
        safe_plot(plot_entropy_hist_snapshots, run, args.snapshots, plot_name="entropy_hist_snapshots")
        safe_plot(plot_selected_episode_trajectories, run, plot_name="selected_episode_trajectories")
        write_manifest(run)
        LOGGER.info("Finished %s -> %s", scan_csv.name, run_dir)


if __name__ == "__main__":
    main()
