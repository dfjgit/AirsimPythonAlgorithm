from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .collision_analysis import collision_termination_rate_percent
except ImportError:  # script-mode fallback
    from collision_analysis import collision_termination_rate_percent

LOGGER = logging.getLogger("scan_csv_visualizer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
ROLLING_BAND_LABEL = "滑动均值 ± 1σ"


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

    @staticmethod
    def _text_series(df: pd.DataFrame, column: str) -> pd.Series:
        if column not in df.columns:
            return pd.Series([""] * len(df), index=df.index, dtype=object)
        return df[column].fillna("").astype(str).str.strip()

    def _scan_episode_summary(self) -> pd.DataFrame:
        if self.scan_df.empty or "episode" not in self.scan_df.columns:
            return pd.DataFrame()

        working = self.scan_df.copy()
        working = working.dropna(subset=["episode"])
        if working.empty:
            return pd.DataFrame()
        working["episode"] = pd.to_numeric(working["episode"], errors="coerce")
        working = working.dropna(subset=["episode"])
        if working.empty:
            return pd.DataFrame()
        working["episode"] = working["episode"].astype(int)
        if "step" in working.columns:
            working["step"] = pd.to_numeric(working["step"], errors="coerce").fillna(0)
        elif "episode_step" in working.columns:
            working["step"] = pd.to_numeric(working["episode_step"], errors="coerce").fillna(0)
        else:
            working["step"] = 0
        working["reset_reason"] = self._text_series(working, "reset_reason")
        terminal = working[(working["step"] > 0) & (working["reset_reason"] != "")].copy()
        if terminal.empty:
            terminal = working.sort_values(["episode", "step"]).groupby("episode", as_index=False).tail(1)

        if "episode_reward" not in working.columns:
            working["episode_reward"] = np.nan
        if "global_scan_ratio" not in working.columns:
            working["global_scan_ratio"] = np.nan
        if "global_avg_entropy" not in working.columns:
            working["global_avg_entropy"] = np.nan

        grouped = working.groupby("episode", as_index=False).agg(
            episode_reward=("episode_reward", "max"),
            episode_length=("step", "max"),
            episode_scan_ratio=("global_scan_ratio", "max"),
            episode_min_entropy=("global_avg_entropy", "min"),
        )
        merge_cols = [c for c in ["episode", "reset_reason", "collision_object_name", "collision_position"] if c in terminal.columns]
        terminal = terminal[merge_cols].drop_duplicates(subset=["episode"], keep="last")
        episode_df = grouped.merge(terminal, on="episode", how="left")
        episode_df["reset_reason"] = self._text_series(episode_df, "reset_reason")
        episode_df["collision_object_name"] = self._text_series(episode_df, "collision_object_name")
        episode_df["collision_position"] = self._text_series(episode_df, "collision_position")
        return episode_df.sort_values("episode")

    @staticmethod
    def _should_replace_placeholder_metric(training_series: pd.Series, scan_series: pd.Series, *, metric: str) -> bool:
        training = pd.to_numeric(training_series, errors="coerce").dropna()
        scan = pd.to_numeric(scan_series, errors="coerce").dropna()
        if scan.empty:
            return False
        if training.empty:
            return True
        if metric == "episode_scan_ratio":
            return training.nunique() <= 1 and float(training.max()) <= 0 and float(scan.max()) > 0
        if metric == "episode_min_entropy":
            return training.nunique() <= 1 and float(training.min()) >= 100 and float(scan.min()) < 100
        return False

    def _build_episode_df(self) -> pd.DataFrame:
        scan_summary = self._scan_episode_summary()
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
            episode_df["reset_reason"] = self._text_series(episode_df, "reset_reason")
            episode_df["collision_object_name"] = self._text_series(episode_df, "collision_object_name")
            episode_df["collision_position"] = self._text_series(episode_df, "collision_position")
            episode_df = episode_df.sort_values("episode").dropna(subset=["episode"]).copy()

            if not scan_summary.empty:
                scan_episode_scan_ratio = scan_summary.set_index("episode")["episode_scan_ratio"]
                scan_episode_min_entropy = scan_summary.set_index("episode")["episode_min_entropy"]
                scan_reset_reason = scan_summary.set_index("episode")["reset_reason"]
                scan_collision_object_name = scan_summary.set_index("episode")["collision_object_name"]
                scan_collision_position = scan_summary.set_index("episode")["collision_position"]

                if "episode_scan_ratio" not in episode_df.columns:
                    episode_df["episode_scan_ratio"] = episode_df["episode"].map(scan_episode_scan_ratio)
                elif self._should_replace_placeholder_metric(
                    episode_df["episode_scan_ratio"],
                    episode_df["episode"].map(scan_episode_scan_ratio),
                    metric="episode_scan_ratio",
                ):
                    episode_df["episode_scan_ratio"] = episode_df["episode"].map(scan_episode_scan_ratio)

                if "episode_min_entropy" not in episode_df.columns:
                    episode_df["episode_min_entropy"] = episode_df["episode"].map(scan_episode_min_entropy)
                elif self._should_replace_placeholder_metric(
                    episode_df["episode_min_entropy"],
                    episode_df["episode"].map(scan_episode_min_entropy),
                    metric="episode_min_entropy",
                ):
                    episode_df["episode_min_entropy"] = episode_df["episode"].map(scan_episode_min_entropy)

                blank_reset = episode_df["reset_reason"].eq("")
                episode_df.loc[blank_reset, "reset_reason"] = episode_df.loc[blank_reset, "episode"].map(scan_reset_reason).fillna("")

                blank_collision = episode_df["collision_object_name"].eq("")
                episode_df.loc[blank_collision, "collision_object_name"] = (
                    episode_df.loc[blank_collision, "episode"].map(scan_collision_object_name).fillna("")
                )

                blank_position = episode_df["collision_position"].eq("")
                episode_df.loc[blank_position, "collision_position"] = (
                    episode_df.loc[blank_position, "episode"].map(scan_collision_position).fillna("")
                )
            return episode_df

        return scan_summary


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


def rolling_std(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window=window, min_periods=2).std().fillna(0.0)


def plot_mean_with_band(
    ax,
    x_values,
    y_values,
    *,
    label: str | None = None,
    color: str | None = None,
    window: int = 20,
    linewidth: float = 2.4,
    linestyle: str = "-",
    band_alpha: float = 0.18,
) -> None:
    x_series = pd.to_numeric(pd.Series(x_values), errors="coerce")
    y_series = pd.to_numeric(pd.Series(y_values), errors="coerce")
    mask = ~(x_series.isna() | y_series.isna())
    if not mask.any():
        return

    x = x_series[mask].to_numpy(dtype=float)
    y = y_series[mask].reset_index(drop=True)
    mean = moving_average(y, window=window)
    std = rolling_std(y, window=window)

    line, = ax.plot(
        x,
        mean.to_numpy(dtype=float),
        label=label,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
    )
    band_color = line.get_color()
    lower = (mean - std).to_numpy(dtype=float)
    upper = (mean + std).to_numpy(dtype=float)
    ax.fill_between(x, lower, upper, color=band_color, alpha=band_alpha, linewidth=0)


def parse_topdown_position(value: str) -> tuple[float, float] | None:
    text = str(value).strip()
    if not text:
        return None
    parts = text.split(",")
    if len(parts) < 2:
        return None
    try:
        x = float(parts[0])
        z = float(parts[2]) if len(parts) >= 3 else float(parts[1])
        return x, z
    except ValueError:
        return None


def plot_episode_performance_summary(run: RunData) -> None:
    df = run.episode_df
    if df.empty:
        return
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    x = df["episode"]
    configs = [
        ("episode_reward", "单轮累计奖励", axes[0], "tab:blue"),
        ("episode_scan_ratio", "最大全局扫描率 (%)", axes[1], "tab:green"),
        ("episode_min_entropy", "最小全局平均熵", axes[2], "tab:red"),
    ]
    for col, title, ax, color in configs:
        if col not in df.columns:
            continue
        plot_mean_with_band(
            ax,
            x,
            df[col],
            color=color,
            label=ROLLING_BAND_LABEL,
            window=20,
        )
        ax.set_ylabel(title)
        ax.legend(loc="best")
    axes[2].set_xlabel("训练轮次")
    fig.suptitle("单轮性能概览", fontsize=16)
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
        values = (df["reset_reason"] == reason).astype(float) * 100.0
        plot_mean_with_band(ax, x, values, label=reason, window=20, linewidth=2.2)
    ax.set_title("重置原因滚动占比（20轮）")
    ax.set_xlabel("训练轮次")
    ax.set_ylabel("占比 (%)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(run.output_dir / "reset_reason_rolling_ratio.png", dpi=160)
    plt.close(fig)


def plot_collision_stability(run: RunData) -> None:
    df = run.episode_df
    if df.empty or "episode" not in df.columns:
        return

    collision_rate = collision_termination_rate_percent(df)
    if collision_rate.empty:
        return

    x = pd.to_numeric(df["episode"], errors="coerce")
    fig, ax = plt.subplots(figsize=(14, 5))
    plot_mean_with_band(
        ax,
        x,
        collision_rate,
        color="#e76f51",
        label=ROLLING_BAND_LABEL,
        window=20,
    )
    ax.set_title("碰撞稳定性变化")
    ax.set_xlabel("训练轮次")
    ax.set_ylabel("碰撞终止占比 (%)")
    ax.set_ylim(-5, 105)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run.output_dir / "collision_stability.png", dpi=160)
    plt.close(fig)

def plot_collision_count_trend(run: RunData) -> None:
    df = run.episode_df
    if df.empty or "episode" not in df.columns:
        return

    collision_count = None
    for column in ("collision_count_final", "collision_count"):
        if column in df.columns:
            collision_count = pd.to_numeric(df[column], errors="coerce")
            break
    if collision_count is None:
        return

    x = pd.to_numeric(df["episode"], errors="coerce")
    fig, ax = plt.subplots(figsize=(14, 5))
    plot_mean_with_band(
        ax,
        x,
        collision_count.fillna(0.0),
        color="#bc6c25",
        label=ROLLING_BAND_LABEL,
        window=20,
    )
    ax.set_title("碰撞次数变化")
    ax.set_xlabel("训练轮次")
    ax.set_ylabel("碰撞次数")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run.output_dir / "collision_count_trend.png", dpi=160)
    plt.close(fig)
def plot_collision_hotspots(run: RunData) -> None:
    df = run.episode_df
    if df.empty or "collision_position" not in df.columns:
        return
    points = [
        parse_topdown_position(v)
        for v in df.loc[df["collision_position"].astype(str).str.len() > 0, "collision_position"]
    ]
    points = [p for p in points if p is not None]
    if not points:
        return
    xs = [p[0] for p in points]
    zs = [p[1] for p in points]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(xs, zs, s=60, alpha=0.75, c=np.arange(len(xs)), cmap="Reds")
    ax.set_title("碰撞热点分布（XZ 俯视图）")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
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
    ax.set_title("碰撞对象构成")
    ax.set_ylabel("次数")
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
        plot_mean_with_band(
            axes[0],
            df["elapsed_time"],
            numeric,
            label=col,
            window=50,
            linewidth=2.0,
            band_alpha=0.15,
        )
        rolling_std_series = numeric.rolling(50, min_periods=5).std()
        plot_mean_with_band(
            axes[1],
            df["elapsed_time"],
            rolling_std_series,
            label=col,
            window=50,
            linewidth=1.8,
            band_alpha=0.15,
        )
    axes[0].set_title("权重滑动均值（窗口=50）")
    axes[1].set_title("权重滑动标准差（窗口=50）")
    axes[1].set_xlabel("运行时间（秒）")
    axes[0].legend(loc="best", ncol=2)
    axes[1].legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(run.output_dir / "algorithm_weights_stability.png", dpi=160)
    plt.close(fig)


def _episode_topdown_xz(run: RunData, episode: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if run.scan_df.empty or "episode" not in run.scan_df.columns:
        return {}
    subset = run.scan_df[run.scan_df["episode"] == episode]
    result = {}
    for drone in run.drones:
        x_col = f"{drone}_x"
        z_col = f"{drone}_z"
        if x_col in subset.columns and z_col in subset.columns:
            x = pd.to_numeric(subset[x_col], errors="coerce").to_numpy()
            z = pd.to_numeric(subset[z_col], errors="coerce").to_numpy()
            mask = ~(np.isnan(x) | np.isnan(z))
            if mask.any():
                result[drone] = (x[mask], z[mask])
    return result


def _select_representative_episodes(run: RunData, limit: int = 4) -> list[tuple[str, int]]:
    df = run.episode_df
    if df.empty or "episode" not in df.columns:
        return []

    working = df.copy()
    working["episode"] = pd.to_numeric(working["episode"], errors="coerce")
    working = working.dropna(subset=["episode"]).sort_values("episode")
    if working.empty:
        return []

    if "episode_scan_ratio" in working.columns:
        working["episode_scan_ratio"] = pd.to_numeric(
            working["episode_scan_ratio"], errors="coerce"
        )
    else:
        working["episode_scan_ratio"] = np.nan

    if "episode_reward" in working.columns:
        working["episode_reward"] = pd.to_numeric(
            working["episode_reward"], errors="coerce"
        )
    else:
        working["episode_reward"] = np.nan

    if "episode_min_entropy" in working.columns:
        working["episode_min_entropy"] = pd.to_numeric(
            working["episode_min_entropy"], errors="coerce"
        )
    else:
        working["episode_min_entropy"] = np.nan

    if "reset_reason" in working.columns:
        working["reset_reason"] = working["reset_reason"].fillna("").astype(str).str.strip()
    else:
        working["reset_reason"] = ""

    candidates: list[tuple[str, int]] = []

    def _append_from_row(label: str, row: pd.Series | None) -> None:
        if row is None or row.empty:
            return
        try:
            episode = int(row["episode"])
        except Exception:
            return
        candidates.append((label, episode))

    best_scan = working.sort_values(
        ["episode_scan_ratio", "episode_reward", "episode"],
        ascending=[False, False, False],
        na_position="last",
    )
    _append_from_row("最佳扫描", best_scan.iloc[0] if not best_scan.empty else None)

    best_entropy = working.sort_values(
        ["episode_min_entropy", "episode_scan_ratio", "episode"],
        ascending=[True, False, False],
        na_position="last",
    )
    _append_from_row(
        "最低熵值", best_entropy.iloc[0] if not best_entropy.empty else None
    )

    _append_from_row("最近回合", working.iloc[-1])

    failure_mask = (
        (working["reset_reason"] != "")
        & (~working["reset_reason"].isin(["达到时长上限", "扫描完成"]))
    )
    failure_rows = working[failure_mask]
    if not failure_rows.empty:
        _append_from_row("代表性失败", failure_rows.iloc[-1])

    worst_scan = working.sort_values(
        ["episode_scan_ratio", "episode", "episode_reward"],
        ascending=[True, False, True],
        na_position="last",
    )
    _append_from_row("最低扫描", worst_scan.iloc[0] if not worst_scan.empty else None)

    deduped: list[tuple[str, int]] = []
    seen: set[int] = set()
    for label, episode in candidates:
        if episode not in seen:
            deduped.append((label, episode))
            seen.add(episode)
        if len(deduped) >= limit:
            break

    if len(deduped) < limit:
        for _, row in working.iloc[::-1].iterrows():
            episode = int(row["episode"])
            if episode in seen:
                continue
            deduped.append(("最近补充", episode))
            seen.add(episode)
            if len(deduped) >= limit:
                break

    return deduped


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
        xz = _episode_topdown_xz(run, episode)
        if not xz:
            continue
        has_data = True
        for drone, (x, z) in xz.items():
            ax.plot(x, z, linewidth=1.8, label=drone)
        label_map = {"best": "最佳回合", "recent": "最近回合"}
        ax.set_title(f"{label_map.get(label, label)} 第 {episode} 轮")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
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
    plot_mean_with_band(
        ax,
        df["episode"],
        df["episode_scan_ratio"],
        label=ROLLING_BAND_LABEL,
        window=20,
    )
    ax.set_title("单轮扫描进展")
    ax.set_xlabel("训练轮次")
    ax.set_ylabel("最大全局扫描率 (%)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(run.output_dir / "scan_progress.png", dpi=160)
    plt.close(fig)


def plot_entropy_trend(run: RunData) -> None:
    df = run.episode_df
    if df.empty or "episode_min_entropy" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    plot_mean_with_band(
        ax,
        df["episode"],
        df["episode_min_entropy"],
        label=ROLLING_BAND_LABEL,
        window=20,
    )
    ax.set_title("单轮最小熵变化")
    ax.set_xlabel("训练轮次")
    ax.set_ylabel("最小全局平均熵")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(run.output_dir / "entropy_trend.png", dpi=160)
    plt.close(fig)


def plot_trajectories_xy(run: RunData) -> None:
    if run.scan_df.empty or not run.drones:
        return
    selected = _select_representative_episodes(run, limit=4)
    if not selected:
        return

    n = len(selected)
    ncols = 2 if n > 1 else 1
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7 * ncols, 6 * nrows),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    plotted = False

    axes_list = axes.flatten().tolist()

    for ax, (label, episode) in zip(axes_list, selected):
        xz = _episode_topdown_xz(run, episode)
        if not xz:
            ax.set_visible(False)
            continue
        plotted = True
        for drone, (x, z) in xz.items():
            ax.plot(x, z, linewidth=1.6, alpha=0.9, label=drone)
        ax.set_title(f"{label} | 第 {episode} 轮")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    for ax in axes_list[n:]:
        ax.set_visible(False)

    if not plotted:
        plt.close(fig)
        return

    fig.suptitle("代表性轨迹对比（XZ 俯视图）", fontsize=16)
    fig.tight_layout()
    fig.savefig(run.output_dir / "trajectories_xy.png", dpi=160)
    plt.close(fig)


def plot_trajectories_3d(run: RunData) -> None:
    df = run.scan_df
    if df.empty or not run.drones:
        return
    selected = _select_representative_episodes(run, limit=4)
    if not selected:
        return

    n = len(selected)
    ncols = 2 if n > 1 else 1
    nrows = math.ceil(n / ncols)
    fig = plt.figure(figsize=(7 * ncols, 6 * nrows))
    plotted = False

    for index, (label, episode) in enumerate(selected, start=1):
        ax = fig.add_subplot(nrows, ncols, index, projection="3d")
        subset = df[df["episode"] == episode]
        episode_plotted = False
        for drone in run.drones:
            x_col = f"{drone}_x"
            y_col = f"{drone}_y"
            z_col = f"{drone}_z"
            if all(c in subset.columns for c in [x_col, y_col, z_col]):
                x = pd.to_numeric(subset[x_col], errors="coerce")
                y = pd.to_numeric(subset[y_col], errors="coerce")
                z = pd.to_numeric(subset[z_col], errors="coerce")
                mask = ~(x.isna() | y.isna() | z.isna())
                if not mask.any():
                    continue
                ax.plot(
                    x[mask].to_numpy(),
                    z[mask].to_numpy(),
                    y[mask].to_numpy(),
                    linewidth=1.3,
                    alpha=0.9,
                    label=drone,
                )
                episode_plotted = True
                plotted = True
        if episode_plotted:
            ax.set_title(f"{label} | 第 {episode} 轮")
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
            ax.set_zlabel("Y（高度）")
            ax.legend(loc="best")
        else:
            ax.set_visible(False)

    if not plotted:
        plt.close(fig)
        return

    fig.suptitle("代表性轨迹对比（三维，高度轴为 Y）", fontsize=16)
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
    plot_mean_with_band(
        ax,
        df["episode"],
        efficiency,
        label=ROLLING_BAND_LABEL,
        window=20,
    )
    ax.set_title("不确定性消减效率")
    ax.set_xlabel("训练轮次")
    ax.set_ylabel("单步扫描率")
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
    ax.set_title("熵直方图快照")
    ax.set_xlabel("熵分箱")
    ax.set_ylabel("网格数量")
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
    selected = _select_representative_episodes(run, limit=6)
    if not selected:
        return

    for label, episode in selected:
        xz = _episode_topdown_xz(run, episode)
        if not xz:
            continue
        fig, ax = plt.subplots(figsize=(7, 7))
        for drone, (x, z) in xz.items():
            ax.plot(x, z, linewidth=1.4, label=drone)
        ax.set_title(f"{label} | 第 {episode} 轮轨迹（XZ 俯视图）")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        safe_label = (
            label.lower()
            .replace(" ", "_")
            .replace("|", "_")
            .replace("/", "_")
        )
        fig.savefig(
            traj_dir / f"episode_{episode:03d}_{safe_label}_trajectory_xy.png",
            dpi=140,
        )
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
    (plot_collision_stability, "collision_stability"),
    (plot_collision_count_trend, "collision_count_trend"),
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
    candidates = [
        scan_csv.parent / f"dqn_training_{stem}.csv",
        scan_csv.parent / f"ddpg_training_{stem}.csv",
        scan_csv.parent / f"training_data_{stem}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline scan/training CSV visualizer")
    parser.add_argument("--csv", help="Path to one scan_data CSV file")
    parser.add_argument("--csv-dir", help="Directory containing scan_data_*.csv files")
    parser.add_argument(
        "--out",
        default="multirotor/DQN_Movement/logs/analysis_results",
        help="Output directory",
    )
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
