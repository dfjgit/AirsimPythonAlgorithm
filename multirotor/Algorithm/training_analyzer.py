"""
Unified cross-algorithm training analyzer.

This module loads historical training/scan CSV files from multiple algorithms,
normalizes key metrics, and produces comparison plots and summary reports.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .benchmark_registry import load_benchmark_registry, resolve_algorithm_registration
except ImportError:
    from benchmark_registry import load_benchmark_registry, resolve_algorithm_registration

try:
    from .collision_analysis import collision_termination_rate_percent
except ImportError:
    from collision_analysis import collision_termination_rate_percent

try:
    import seaborn as sns
except ImportError:
    sns = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TrainingAnalyzer")


class UnifiedTrainingAnalyzer:
    """Load DDPG/DQN CSV logs and generate unified comparison outputs."""

    ALGO_STYLE_MAP = {
        "fixed_apf": {"color": "#6C757D", "linestyle": "-"},
        "random_apf": {"color": "#8D99AE", "linestyle": "--"},
        "ddpg_apf": {"color": "#F4A261", "linestyle": "-"},
        "pure_dqn": {"color": "#2A9D8F", "linestyle": "--"},
        "hrl_dqn_apf": {"color": "#3A86FF", "linestyle": "-."},
        "unknown": {"color": "#7A7A7A", "linestyle": ":"},
    }

    ALGO_NAME_MAP = {
        "fixed_apf": "固定 APF 基线",
        "random_apf": "随机 APF 基线",
        "hrl_dqn_apf": "双层融合训练 (HRL+APF)",
        "pure_dqn": "纯 DQN 移动控制",
        "ddpg_apf": "DDPG 权重自适应 (APF)",
        "unknown": "未标记算法(历史数据)",
    }

    METRIC_NAME_MAP = {
        "reward": "累计奖励",
        "scan_efficiency": "扫描效率（格/步）",
        "collision_rate": "碰撞终止占比(%)",
        "collision_count": "碰撞次数",
        "scan_ratio": "扫描完成度(%)",
        "global_avg_entropy": "全局平均熵值",
        "episode": "训练轮次",
        "elapsed_time": "运行时间（秒）",
        "window_episode": "窗口内训练轮次",
        "window_elapsed_time": "窗口内运行时间（秒）",
    }

    COMPARABILITY_MAP = {
        "平均奖励": (
            "弱可比",
            "奖励函数、终止条件和 shaping 不同，更适合看各自训练趋势，不宜直接横向定优劣。",
        ),
        "最高奖励": (
            "弱可比",
            "极值对单次幸运回合敏感，且受奖励设计影响较大。",
        ),
        "训练轮次": (
            "弱可比",
            "不同算法的 episode 设计与平均长度不同，只能作为训练规模参考。",
        ),
        "总耗时(s)": (
            "弱可比",
            "受实现效率、控制链路与仿真节奏影响，不是纯策略质量指标。",
        ),
        "平均碰撞终止占比(%)": (
            "中等可比",
            "可用于比较训练稳定性，但仍受终止机制与重置口径影响，不替代最终结果指标。",
        ),
        "平均碰撞次数": (
            "中等可比",
            "可用于比较每轮碰撞负担，但仍受终止机制与场景交互影响，不替代最终结果指标。",
        ),
        "最终效率": (
            "强可比",
            "统一换算为 Cell/Step 后，可直接比较单位决策步的扫描产出。",
        ),
        "最终扫描率(%)": (
            "强可比",
            "直接描述任务覆盖率，物理语义一致。",
        ),
        "最低熵值": (
            "强可比",
            "直接反映不确定性消减程度，任务语义一致。",
        ),
    }

    METRIC_DIMENSION_MAP = {
        "平均奖励": ("过程对比", "用于观察训练过程中的学习趋势和稳定性。"),
        "最高奖励": ("过程对比", "用于观察训练过程中出现过的最佳回合表现。"),
        "训练轮次": ("过程对比", "用于描述训练规模和训练推进程度。"),
        "总耗时(s)": ("过程对比", "用于描述训练和仿真成本。"),
        "平均碰撞终止占比(%)": ("过程对比", "用于比较训练过程中因碰撞终止的频繁程度和稳定性趋势。"),
        "平均碰撞次数": ("过程对比", "用于比较训练过程中每轮碰撞事件的数量变化趋势。"),
        "最终效率": ("结果对比", "用于比较最终单位决策步的扫描产出。"),
        "最终扫描率(%)": ("结果对比", "用于比较最终任务覆盖效果。"),
        "最低熵值": ("结果对比", "用于比较最终不确定性消减效果。"),
    }

    def __init__(
        self,
        output_dir: str = "multirotor/DQN_Movement/logs/analysis_results",
        registry_path: str | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runs = []
        self.registry = None
        try:
            self.registry = load_benchmark_registry(registry_path) if registry_path else load_benchmark_registry()
        except Exception as exc:
            logger.warning("无法加载 benchmark registry，分析将不回填 family 元数据: %s", exc)
        self._setup_plotting_style()

    def _setup_plotting_style(self) -> None:
        if sns is not None:
            sns.set_theme(style="whitegrid")
        else:
            plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
        plt.rcParams["axes.unicode_minus"] = False

    def load_data(self, log_dirs: List[str]) -> None:
        """Load all CSV files from the given directories."""
        for directory in log_dirs:
            path = Path(directory)
            if not path.exists():
                logger.warning("目录不存在: %s", directory)
                continue

            for csv_file in path.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    if df.empty:
                        continue

                    algo = df["algorithm_type"].iloc[0] if "algorithm_type" in df.columns else None
                    if not algo or pd.isna(algo):
                        algo = self._infer_algorithm_from_path(csv_file)

                    env = df["env_type"].iloc[0] if "env_type" in df.columns else "unknown"
                    data_type = "training" if "training" in csv_file.name else "scan"
                    normalized = self._normalize_metrics(df, data_type)
                    if normalized.empty:
                        continue
                    resolved_meta = self._resolve_registry_meta(algo, env)

                    self.runs.append(
                        {
                            "file": csv_file,
                            "name": csv_file.stem,
                            "algorithm": algo,
                            "env": env,
                            "data": normalized,
                            "type": data_type,
                            **resolved_meta,
                            **self._extract_stage_meta(normalized, csv_file),
                        }
                    )
                    logger.info("已加载 %s (算法: %s, 类型: %s)", csv_file.name, algo, data_type)
                except Exception as exc:
                    logger.error("加载失败 %s: %s", csv_file.name, exc)

    def _extract_stage_meta(self, df: pd.DataFrame, csv_file: Path) -> dict:
        def _first_value(column: str, default):
            if column not in df.columns or df.empty:
                return default
            value = df[column].iloc[0]
            if pd.isna(value):
                return default
            return value

        experiment_id = str(_first_value("experiment_id", "") or "").strip()
        stage_name = str(_first_value("stage_name", "") or "").strip()
        stage_index_raw = _first_value("stage_index", 1)
        is_resume_raw = _first_value("is_resume", 0)
        source_model = str(_first_value("source_model", "") or "").strip()

        try:
            stage_index = max(int(stage_index_raw), 1)
        except (TypeError, ValueError):
            stage_index = 1
        try:
            is_resume = bool(int(is_resume_raw))
        except (TypeError, ValueError):
            is_resume = bool(is_resume_raw)

        if not experiment_id:
            match = re.match(
                r"(?:scan_data|dqn_training|ddpg_training|training_data)_(.+?)_stage(\d+)_\d{8}_\d{6}$",
                csv_file.stem,
            )
            if match:
                experiment_id = match.group(1)
                try:
                    stage_index = max(int(match.group(2)), 1)
                except ValueError:
                    stage_index = 1

        if not stage_name:
            stage_name = f"stage{stage_index:02d}"

        return {
            "experiment_id": experiment_id,
            "stage_name": stage_name,
            "stage_index": stage_index,
            "is_resume": is_resume,
            "source_model": source_model,
        }

    def _resolve_registry_meta(self, algorithm_type: str, env_type: str) -> dict:
        if self.registry is None:
            return {
                "primary_family": "",
                "family_memberships": [],
                "comparison_profiles": [],
                "is_trainable": False,
                "registry_version": "",
            }
        control_mode = "dqn" if str(env_type).strip().lower() == "movement" or "dqn" in str(algorithm_type).lower() else "apf"
        resolved = resolve_algorithm_registration(
            algorithm_type,
            self.registry,
            control_mode=control_mode,
            apf_weight_mode="learned" if algorithm_type == "ddpg_apf" else "fixed",
            is_trainable=algorithm_type in {"ddpg_apf", "pure_dqn", "hrl_dqn_apf"},
        )
        return {
            "primary_family": resolved.primary_family,
            "family_memberships": list(resolved.family_memberships),
            "comparison_profiles": list(resolved.comparison_profiles),
            "is_trainable": bool(resolved.is_trainable),
            "registry_version": resolved.registry_version,
        }

    def _combine_stage_runs(self, runs: List[dict], data_type: str) -> dict:
        sorted_runs = sorted(
            runs,
            key=lambda run: (int(run.get("stage_index", 1)), run["file"].name),
        )
        combined_frames = []
        episode_offset = 0.0
        elapsed_offset = 0.0
        timestep_offset = 0.0

        for run in sorted_runs:
            df = run["data"].copy()
            df["analysis_stage_index"] = int(run.get("stage_index", 1))
            df["analysis_stage_name"] = run.get("stage_name", "")

            if "episode" in df.columns:
                episode_series = pd.to_numeric(df["episode"], errors="coerce")
                valid_episode = episode_series.dropna()
                if not valid_episode.empty:
                    df["episode"] = episode_series + episode_offset
                    episode_offset += float(valid_episode.max())

            if "elapsed_time" in df.columns:
                elapsed_series = pd.to_numeric(
                    df["elapsed_time"].astype(str).str.replace("%", "", regex=False),
                    errors="coerce",
                )
                valid_elapsed = elapsed_series.dropna()
                if not valid_elapsed.empty:
                    df["elapsed_time"] = elapsed_series + elapsed_offset
                    elapsed_offset += float(valid_elapsed.max())

            if data_type == "training" and "timestep" in df.columns:
                timestep_series = pd.to_numeric(df["timestep"], errors="coerce")
                valid_timestep = timestep_series.dropna()
                if not valid_timestep.empty:
                    df["timestep"] = timestep_series + timestep_offset
                    timestep_offset += float(valid_timestep.max())

            combined_frames.append(df)

        combined_df = pd.concat(combined_frames, ignore_index=True, sort=False)
        latest_run = sorted_runs[-1]
        experiment_id = latest_run.get("experiment_id", "") or latest_run["name"]
        return {
            "file": latest_run["file"],
            "name": f"{experiment_id}_combined",
            "algorithm": latest_run["algorithm"],
            "env": latest_run["env"],
            "data": combined_df,
            "type": data_type,
            "experiment_id": latest_run.get("experiment_id", ""),
            "stage_name": latest_run.get("stage_name", ""),
            "stage_index": latest_run.get("stage_index", 1),
            "is_resume": latest_run.get("is_resume", False),
            "source_model": latest_run.get("source_model", ""),
            "stage_count": len(sorted_runs),
            "source_runs": sorted_runs,
            "latest_key": latest_run["file"].name,
        }

    def _get_target_runs(self, data_type: str, latest_only: bool = False):
        raw_runs = [run for run in self.runs if run["type"] == data_type]
        grouped = {}
        for run in raw_runs:
            experiment_id = run.get("experiment_id", "") or run["name"]
            grouped.setdefault((run["algorithm"], experiment_id), []).append(run)

        runs = []
        for grouped_runs in grouped.values():
            if len(grouped_runs) > 1 and grouped_runs[0].get("experiment_id"):
                runs.append(self._combine_stage_runs(grouped_runs, data_type))
            else:
                run = dict(grouped_runs[0])
                run["stage_count"] = 1
                run["source_runs"] = grouped_runs
                run["latest_key"] = run["file"].name
                runs.append(run)

        if not latest_only:
            return runs

        latest_by_algo = {}
        for run in runs:
            key = run["algorithm"]
            previous = latest_by_algo.get(key)
            if previous is None or run["latest_key"] > previous["latest_key"]:
                latest_by_algo[key] = run
        return [latest_by_algo[key] for key in sorted(latest_by_algo.keys())]

    def _infer_algorithm_from_path(self, csv_file: Path) -> str:
        path_str = str(csv_file).lower()
        if "fixed_apf" in path_str:
            return "fixed_apf"
        if "random_apf" in path_str:
            return "random_apf"
        if "hrl" in path_str or "hierarchical" in path_str:
            return "hrl_dqn_apf"
        if "dqn" in path_str:
            return "pure_dqn"
        if "ddpg" in path_str:
            return "ddpg_apf"
        return "unknown"

    def _normalize_metrics(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Normalize historical metrics so cross-algorithm plots use a consistent meaning.

        In particular, scan_efficiency is normalized to Cell/Step.
        """
        normalized = df.copy()
        if data_type != "training":
            return normalized

        if "episode_complete" in normalized.columns:
            episode_complete = pd.to_numeric(normalized["episode_complete"], errors="coerce").fillna(1)
            normalized = normalized[episode_complete != 0].copy()
            if normalized.empty:
                return normalized

        if "length" in normalized.columns:
            lengths = pd.to_numeric(normalized["length"], errors="coerce").replace(0, np.nan)
            efficiency_source = None
            if "global_scanned_cells" in normalized.columns:
                global_cells = pd.to_numeric(normalized["global_scanned_cells"], errors="coerce")
                if not global_cells.isna().all():
                    efficiency_source = global_cells

            if efficiency_source is None and "scanned_cells" in normalized.columns:
                scanned_cells = pd.to_numeric(normalized["scanned_cells"], errors="coerce")
                if not scanned_cells.isna().all():
                    efficiency_source = scanned_cells

            if efficiency_source is not None and not lengths.isna().all():
                normalized["scan_efficiency"] = (efficiency_source / lengths).fillna(0.0)

        normalized["collision_rate"] = collision_termination_rate_percent(normalized)
        collision_count_series = None
        for column in ("collision_count_final", "collision_count"):
            if column in normalized.columns:
                collision_count_series = pd.to_numeric(normalized[column], errors="coerce")
                break
        if collision_count_series is not None:
            normalized["collision_count"] = collision_count_series.fillna(0.0)
        return normalized

    def _find_matching_scan_run(self, training_run: dict):
        effective_scans = self._get_target_runs("scan", latest_only=False)
        experiment_id = training_run.get("experiment_id", "")
        if experiment_id:
            same_experiment = [
                run
                for run in effective_scans
                if run["algorithm"] == training_run["algorithm"]
                and run.get("experiment_id", "") == experiment_id
            ]
            if same_experiment:
                return sorted(same_experiment, key=lambda run: run["latest_key"])[-1]

        training_dir = training_run["file"].parent
        training_name = training_run["file"].name
        timestamp = training_name.split("_training_")[-1].replace(".csv", "")
        candidate = training_dir / f"scan_data_{timestamp}.csv"
        if candidate.exists():
            for run in effective_scans:
                if run["file"] == candidate:
                    return run

        same_algo_scans = sorted(
            [run for run in effective_scans if run["algorithm"] == training_run["algorithm"]],
            key=lambda run: run["latest_key"],
        )
        return same_algo_scans[-1] if same_algo_scans else None

    def _get_recent_window_runs(
        self,
        tail_episodes: int = 50,
        min_training_episodes: int = 20,
    ):
        """
        Build a fairer "recent window" comparison set.

        For each algorithm:
        - choose the latest training run with at least min_training_episodes rows,
          otherwise fall back to the latest run;
        - keep only the trailing tail_episodes;
        - pair the matching scan run and slice it to the same episode window.
        """
        prepared_runs = []
        training_runs = self._get_target_runs("training", latest_only=False)
        algorithms = sorted(set(run["algorithm"] for run in training_runs))

        for algo in algorithms:
            algo_training_runs = sorted(
                [run for run in training_runs if run["algorithm"] == algo],
                key=lambda run: run["file"].name,
            )
            if not algo_training_runs:
                continue

            selected_training = None
            for run in reversed(algo_training_runs):
                if len(run["data"]) >= min_training_episodes:
                    selected_training = run
                    break
            if selected_training is None:
                selected_training = algo_training_runs[-1]

            training_df = selected_training["data"].copy()
            if len(training_df) > tail_episodes:
                training_df = training_df.tail(tail_episodes).copy()
            training_df["window_episode"] = range(1, len(training_df) + 1)

            selected_scan = self._find_matching_scan_run(selected_training)
            if selected_scan is not None:
                scan_df = selected_scan["data"].copy()
                if "episode" in scan_df.columns and "episode" in training_df.columns:
                    min_episode = pd.to_numeric(training_df["episode"], errors="coerce").min()
                    max_episode = pd.to_numeric(training_df["episode"], errors="coerce").max()
                    if pd.notna(min_episode) and pd.notna(max_episode):
                        episode_series = pd.to_numeric(scan_df["episode"], errors="coerce")
                        scan_df = scan_df[(episode_series >= min_episode) & (episode_series <= max_episode)].copy()
                if "elapsed_time" in scan_df.columns:
                    elapsed = pd.to_numeric(scan_df["elapsed_time"], errors="coerce")
                    if not elapsed.isna().all():
                        scan_df["window_elapsed_time"] = elapsed - float(elapsed.min())
            else:
                scan_df = None

            prepared_runs.append(
                {
                    "algorithm": algo,
                    "training_run": selected_training,
                    "training_data": training_df,
                    "scan_run": selected_scan,
                    "scan_data": scan_df,
                }
            )

        return prepared_runs

    def _build_metric_metadata_df(self, metrics: List[str]) -> pd.DataFrame:
        rows = []
        for metric in metrics:
            comparability, comparability_note = self.COMPARABILITY_MAP.get(
                metric,
                ("弱可比", "未显式标注的指标默认视为弱可比，请结合任务语义谨慎解读。"),
            )
            dimension, dimension_note = self.METRIC_DIMENSION_MAP.get(
                metric,
                ("未分类", "该指标尚未归入过程对比或结果对比，请结合任务背景解释。"),
            )
            rows.append(
                {
                    "指标": metric,
                    "对比维度": dimension,
                    "维度说明": dimension_note,
                    "可比性": comparability,
                    "可比性说明": comparability_note,
                }
            )
        return pd.DataFrame(rows)

    def _safe_to_markdown(self, table, **kwargs) -> str:
        """Render table to markdown and gracefully degrade when tabulate is unavailable."""
        try:
            return table.to_markdown(**kwargs)
        except ImportError as exc:
            logger.warning("to_markdown 不可用，回退为纯文本表格: %s", exc)
            if isinstance(table, pd.DataFrame):
                if kwargs.get("index", True):
                    plain_text = table.to_string()
                else:
                    plain_text = table.to_string(index=False)
            elif isinstance(table, pd.Series):
                plain_text = table.to_string()
            else:
                plain_text = str(table)
            return f"```text\n{plain_text}\n```"

    def _get_algo_style(self, algo_id: str) -> dict:
        style = dict(self.ALGO_STYLE_MAP.get("unknown", {}))
        style.update(self.ALGO_STYLE_MAP.get(algo_id, {}))
        return style

    @staticmethod
    def _normalize_plot_series(df: pd.DataFrame, x_axis: str, metric: str) -> pd.DataFrame:
        if x_axis not in df.columns or metric not in df.columns:
            return pd.DataFrame(columns=[x_axis, metric])

        x_series = pd.to_numeric(
            df[x_axis].astype(str).str.replace("%", "", regex=False),
            errors="coerce",
        )
        y_series = pd.to_numeric(
            df[metric].astype(str).str.replace("%", "", regex=False),
            errors="coerce",
        )
        numeric_data = pd.DataFrame({x_axis: x_series, metric: y_series}).dropna()
        if numeric_data.empty:
            return numeric_data

        if x_axis in {"episode", "window_episode"}:
            numeric_data[x_axis] = numeric_data[x_axis].round().astype(int)
        elif x_axis in {"elapsed_time", "window_elapsed_time"}:
            numeric_data[x_axis] = numeric_data[x_axis].round(1)

        return numeric_data.sort_values(x_axis)

    def _build_curve_with_band(
        self,
        frames: List[pd.DataFrame],
        x_axis: str,
        metric: str,
    ) -> pd.DataFrame:
        normalized_frames = []
        for frame in frames:
            normalized = self._normalize_plot_series(frame, x_axis, metric)
            if not normalized.empty:
                normalized_frames.append(normalized)

        if not normalized_frames:
            return pd.DataFrame(columns=[x_axis, "center", "band"])

        combined = pd.concat(normalized_frames, ignore_index=True)
        grouped = (
            combined.groupby(x_axis)[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values(x_axis)
        )
        if grouped.empty:
            return pd.DataFrame(columns=[x_axis, "center", "band"])

        smooth_window = min(15, max(3, len(grouped) // 12))
        grouped["center"] = (
            grouped["mean"].rolling(window=smooth_window, min_periods=1).mean()
        )
        grouped["rolling_std"] = (
            grouped["mean"].rolling(window=smooth_window, min_periods=2).std().fillna(0.0)
        )
        grouped["band"] = grouped["std"].fillna(0.0)

        single_run = len(normalized_frames) == 1 or float(grouped["band"].max()) <= 1e-9
        if single_run:
            grouped["band"] = grouped["rolling_std"]
        else:
            grouped["band"] = grouped["band"].where(
                grouped["band"] > 1e-9, grouped["rolling_std"]
            )

        return grouped[[x_axis, "center", "band"]]

    def _plot_curve_with_band(
        self,
        ax,
        curve_df: pd.DataFrame,
        x_axis: str,
        *,
        label: str,
        color: str,
        linestyle: str,
        linewidth: float = 2.3,
        band_alpha: float = 0.18,
    ) -> None:
        if curve_df.empty:
            return

        x = curve_df[x_axis].to_numpy(dtype=float)
        center = curve_df["center"].to_numpy(dtype=float)
        band = curve_df["band"].fillna(0.0).to_numpy(dtype=float)

        ax.plot(
            x,
            center,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
        ax.fill_between(
            x,
            center - band,
            center + band,
            color=color,
            alpha=band_alpha,
            linewidth=0,
        )
    def plot_comparison(
        self,
        metric: str,
        data_type: str = "training",
        x_axis: str = "episode",
        latest_only: bool = False,
        file_prefix: str = "comparison",
    ) -> None:
        """Plot a comparison curve for one metric across algorithms."""
        target_runs = self._get_target_runs(data_type, latest_only=latest_only)
        if not target_runs:
            logger.warning("没有找到类型为 %s 的数据", data_type)
            return

        plt.figure(figsize=(14, 8))
        metric_label = self.METRIC_NAME_MAP.get(metric, metric)
        x_axis_label = self.METRIC_NAME_MAP.get(x_axis, x_axis)
        unique_algos = sorted(set(run["algorithm"] for run in target_runs))
        ax = plt.gca()

        for algo_id in unique_algos:
            algo_frames = [run["data"] for run in target_runs if run["algorithm"] == algo_id]
            display_name = self.ALGO_NAME_MAP.get(algo_id, algo_id)
            curve_df = self._build_curve_with_band(algo_frames, x_axis, metric)
            if curve_df.empty:
                continue
            style = self._get_algo_style(algo_id)
            self._plot_curve_with_band(
                ax,
                curve_df,
                x_axis,
                label=display_name,
                color=style["color"],
                linestyle=style["linestyle"],
            )

        title_prefix = "多算法最新一轮对比分析" if latest_only else "多算法对比分析"
        ax.set_title(f"{title_prefix}: {metric_label} 随 {x_axis_label} 变化趋势", fontsize=16, pad=20)
        ax.set_xlabel(x_axis_label, fontsize=12)
        ax.set_ylabel(metric_label, fontsize=12)
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

        filename = self.output_dir / f"{file_prefix}_{data_type}_{metric}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        logger.info("对比图表已保存: %s (标签: %s)", filename, unique_algos)
        plt.close()

    def generate_summary_report(
        self,
        latest_only: bool = False,
        report_prefix: str = "algorithm_comparison",
    ) -> None:
        """Generate CSV and Markdown comparison reports."""
        summary_rows = []
        target_runs = self._get_target_runs("training", latest_only=latest_only) + self._get_target_runs(
            "scan", latest_only=latest_only
        )
        for run in target_runs:
            df = run["data"]
            algo_display = self.ALGO_NAME_MAP.get(run["algorithm"], run["algorithm"])

            if run["type"] == "training":
                summary_rows.append(
                    {
                        "算法名称": algo_display,
                        "运行记录": run["name"],
                        "平均奖励": df["reward"].mean() if "reward" in df.columns else 0,
                        "最高奖励": df["reward"].max() if "reward" in df.columns else 0,
                        "训练轮次": len(df),
                        "平均碰撞终止占比(%)": (
                            pd.to_numeric(df["collision_rate"], errors="coerce").mean()
                            if "collision_rate" in df.columns
                            else 0
                        ),
                        "平均碰撞次数": (
                            pd.to_numeric(df["collision_count"], errors="coerce").mean()
                            if "collision_count" in df.columns
                            else 0
                        ),
                        "最终效率": df["scan_efficiency"].iloc[-1] if "scan_efficiency" in df.columns else 0,
                    }
                )
            elif run["type"] == "scan":
                summary_rows.append(
                    {
                        "算法名称": algo_display,
                        "运行记录": run["name"],
                        "最终扫描率(%)": (
                            pd.to_numeric(
                                df["scan_ratio"].astype(str).str.replace("%", "", regex=False),
                                errors="coerce",
                            ).iloc[-1]
                            if "scan_ratio" in df.columns
                            else 0
                        ),
                        "最低熵值": (
                            pd.to_numeric(df["global_avg_entropy"], errors="coerce").min()
                            if "global_avg_entropy" in df.columns
                            else 0
                        ),
                        "总耗时(s)": (
                            pd.to_numeric(df["elapsed_time"], errors="coerce").iloc[-1]
                            if "elapsed_time" in df.columns
                            else 0
                        ),
                    }
                )

        summary_df = pd.DataFrame(summary_rows)
        if summary_df.empty:
            logger.warning("没有可用于生成报告的数据")
            return

        algo_comparison = summary_df.groupby("算法名称").mean(numeric_only=True)
        logger.info("%s", "=" * 70)
        logger.info("多算法平均性能量化对比报告 (Averaged Performance Report)")
        logger.info("%s", "=" * 70)
        logger.info("\n%s", algo_comparison.to_string())
        logger.info("%s", "=" * 70)

        report_file = self.output_dir / f"{report_prefix}_report.csv"
        algo_comparison.to_csv(report_file, encoding="utf-8-sig")
        logger.info("对比报告已导出: %s", report_file)

        comparability_df = self._build_metric_metadata_df(list(algo_comparison.columns))
        comparability_file = self.output_dir / f"{report_prefix}_metric_comparability.csv"
        comparability_df.to_csv(comparability_file, index=False, encoding="utf-8-sig")
        logger.info("指标分类说明已导出: %s", comparability_file)

        process_report = comparability_df[comparability_df["对比维度"] == "过程对比"][
            ["指标", "维度说明", "可比性", "可比性说明"]
        ]
        outcome_report = comparability_df[comparability_df["对比维度"] == "结果对比"][
            ["指标", "维度说明", "可比性", "可比性说明"]
        ]
        process_file = self.output_dir / f"{report_prefix}_process_metrics.csv"
        outcome_file = self.output_dir / f"{report_prefix}_outcome_metrics.csv"
        process_report.to_csv(process_file, index=False, encoding="utf-8-sig")
        outcome_report.to_csv(outcome_file, index=False, encoding="utf-8-sig")
        logger.info("过程对比指标说明已导出: %s", process_file)
        logger.info("结果对比指标说明已导出: %s", outcome_file)

        markdown_file = self.output_dir / f"{report_prefix}_report.md"
        markdown_lines = [
            "# 多算法训练对比报告" if not latest_only else "# 多算法最新一轮训练对比报告",
            "",
            "## 1. 汇总结果",
            "",
            self._safe_to_markdown(algo_comparison),
            "",
            "## 2. 过程对比指标",
            "",
            self._safe_to_markdown(process_report, index=False),
            "",
            "## 3. 结果对比指标",
            "",
            self._safe_to_markdown(outcome_report, index=False),
            "",
            "## 4. 全部指标分类",
            "",
            self._safe_to_markdown(comparability_df, index=False),
            "",
            "## 5. 解读建议",
            "",
            "- 过程对比用于判断谁学得更快、更稳，适合看 reward、训练轮次、总耗时等训练维度指标。",
            "- 结果对比用于判断最终模型谁更强，优先看 `最终效率`、`最终扫描率(%)`、`最低熵值`。",
            "- 如果过程指标和结果指标出现冲突，应优先参考结果对比，再结合轨迹图、重置原因和覆盖率共同判断。",
        ]
        markdown_file.write_text("\n".join(markdown_lines), encoding="utf-8")
        logger.info("Markdown 对比报告已导出: %s", markdown_file)

    def generate_recent_window_report(
        self,
        tail_episodes: int = 50,
        min_training_episodes: int = 20,
        report_prefix: str = "recent_window_algorithm_comparison",
    ) -> None:
        """Generate report for the most recent substantial window of each algorithm."""
        prepared_runs = self._get_recent_window_runs(
            tail_episodes=tail_episodes,
            min_training_episodes=min_training_episodes,
        )
        if not prepared_runs:
            logger.warning("没有可用于生成最近窗口报告的数据")
            return

        summary_rows = []
        selection_rows = []
        for item in prepared_runs:
            algo_name = self.ALGO_NAME_MAP.get(item["algorithm"], item["algorithm"])
            training_df = item["training_data"]
            scan_df = item["scan_data"]

            episode_start = int(pd.to_numeric(training_df["episode"], errors="coerce").min())
            episode_end = int(pd.to_numeric(training_df["episode"], errors="coerce").max())
            selection_rows.append(
                {
                    "算法名称": algo_name,
                    "训练文件": item["training_run"]["file"].name,
                    "扫描文件": item["scan_run"]["file"].name if item["scan_run"] else "",
                    "窗口回合范围": f"{episode_start}-{episode_end}",
                    "窗口回合数": len(training_df),
                }
            )

            row = {
                "算法名称": algo_name,
                "平均奖励": pd.to_numeric(training_df.get("reward"), errors="coerce").mean(),
                "最高奖励": pd.to_numeric(training_df.get("reward"), errors="coerce").max(),
                "训练轮次": len(training_df),
                "平均碰撞终止占比(%)": (
                    pd.to_numeric(training_df["collision_rate"], errors="coerce").mean()
                    if "collision_rate" in training_df.columns
                    else 0
                ),
                "平均碰撞次数": (
                    pd.to_numeric(training_df["collision_count"], errors="coerce").mean()
                    if "collision_count" in training_df.columns
                    else 0
                ),
                "最终效率": pd.to_numeric(training_df.get("scan_efficiency"), errors="coerce").iloc[-1],
            }
            if scan_df is not None and not scan_df.empty:
                row["最终扫描率(%)"] = pd.to_numeric(
                    scan_df["scan_ratio"].astype(str).str.replace("%", "", regex=False),
                    errors="coerce",
                ).iloc[-1]
                row["最低熵值"] = pd.to_numeric(scan_df["global_avg_entropy"], errors="coerce").min()
                elapsed = pd.to_numeric(scan_df["window_elapsed_time"], errors="coerce")
                row["总耗时(s)"] = float(elapsed.iloc[-1]) if not elapsed.empty else 0.0
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows).set_index("算法名称")
        report_file = self.output_dir / f"{report_prefix}_report.csv"
        summary_df.to_csv(report_file, encoding="utf-8-sig")
        logger.info("最近窗口对比报告已导出: %s", report_file)

        selection_df = pd.DataFrame(selection_rows)
        selection_file = self.output_dir / f"{report_prefix}_selection.csv"
        selection_df.to_csv(selection_file, index=False, encoding="utf-8-sig")
        logger.info("最近窗口样本选择说明已导出: %s", selection_file)

        comparability_df = self._build_metric_metadata_df(list(summary_df.columns))
        comparability_file = self.output_dir / f"{report_prefix}_metric_comparability.csv"
        comparability_df.to_csv(comparability_file, index=False, encoding="utf-8-sig")

        process_report = comparability_df[comparability_df["对比维度"] == "过程对比"][
            ["指标", "维度说明", "可比性", "可比性说明"]
        ]
        outcome_report = comparability_df[comparability_df["对比维度"] == "结果对比"][
            ["指标", "维度说明", "可比性", "可比性说明"]
        ]
        process_file = self.output_dir / f"{report_prefix}_process_metrics.csv"
        outcome_file = self.output_dir / f"{report_prefix}_outcome_metrics.csv"
        process_report.to_csv(process_file, index=False, encoding="utf-8-sig")
        outcome_report.to_csv(outcome_file, index=False, encoding="utf-8-sig")

        markdown_file = self.output_dir / f"{report_prefix}_report.md"
        markdown_lines = [
            "# 多算法最近窗口对比报告",
            "",
            "## 1. 样本选择",
            "",
            self._safe_to_markdown(selection_df, index=False),
            "",
            "## 2. 汇总结果",
            "",
            self._safe_to_markdown(summary_df),
            "",
            "## 3. 过程对比指标",
            "",
            self._safe_to_markdown(process_report, index=False),
            "",
            "## 4. 结果对比指标",
            "",
            self._safe_to_markdown(outcome_report, index=False),
            "",
            "## 5. 全部指标分类",
            "",
            self._safe_to_markdown(comparability_df, index=False),
        ]
        markdown_file.write_text("\n".join(markdown_lines), encoding="utf-8")
        logger.info("最近窗口 Markdown 报告已导出: %s", markdown_file)

    def plot_recent_window_comparison(
        self,
        metric: str,
        data_type: str = "training",
        tail_episodes: int = 50,
        min_training_episodes: int = 20,
        file_prefix: str = "recent_window_comparison",
    ) -> None:
        """Plot the same core comparisons over the most recent substantial window."""
        prepared_runs = self._get_recent_window_runs(
            tail_episodes=tail_episodes,
            min_training_episodes=min_training_episodes,
        )
        if not prepared_runs:
            logger.warning("没有可用于最近窗口对比的数据")
            return

        plt.figure(figsize=(14, 8))
        x_axis = "window_episode" if data_type == "training" else "window_elapsed_time"
        metric_label = self.METRIC_NAME_MAP.get(metric, metric)
        x_axis_label = self.METRIC_NAME_MAP.get(x_axis, x_axis)
        ax = plt.gca()

        labels = []
        for item in prepared_runs:
            df = item["training_data"] if data_type == "training" else item["scan_data"]
            if df is None or df.empty:
                continue

            display_name = self.ALGO_NAME_MAP.get(item["algorithm"], item["algorithm"])
            curve_df = self._build_curve_with_band([df], x_axis, metric)
            if curve_df.empty:
                continue
            labels.append(display_name)
            style = self._get_algo_style(item["algorithm"])
            self._plot_curve_with_band(
                ax,
                curve_df,
                x_axis,
                label=display_name,
                color=style["color"],
                linestyle=style["linestyle"],
            )

        ax.set_title(f"多算法最近窗口对比: {metric_label} 随 {x_axis_label} 变化趋势", fontsize=16, pad=20)
        ax.set_xlabel(x_axis_label, fontsize=12)
        ax.set_ylabel(metric_label, fontsize=12)
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

        filename = self.output_dir / f"{file_prefix}_{data_type}_{metric}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        logger.info("最近窗口对比图表已保存: %s (标签: %s)", filename, labels)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多算法对比分析工具")
    parser.add_argument(
        "--dirs",
        nargs="+",
        help="数据目录列表",
        default=[
            "multirotor/DDPG_Weight/airsim_training_logs",
            "multirotor/DQN_Movement/logs/dqn_scan_data",
        ],
    )
    parser.add_argument("--out", default="multirotor/DQN_Movement/logs/analysis_results", help="结果保存目录")
    args = parser.parse_args()

    analyzer = UnifiedTrainingAnalyzer(output_dir=args.out)
    analyzer.load_data(args.dirs)
    analyzer.plot_comparison(metric="reward", data_type="training", x_axis="episode")
    analyzer.plot_comparison(metric="scan_efficiency", data_type="training", x_axis="episode")
    analyzer.plot_comparison(metric="scan_ratio", data_type="scan", x_axis="elapsed_time")
    analyzer.plot_comparison(metric="global_avg_entropy", data_type="scan", x_axis="elapsed_time")
    analyzer.generate_summary_report()
