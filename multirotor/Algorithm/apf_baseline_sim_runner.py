from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from four_group_benchmark_runner import _run_apf_algorithm


def _env_int(name: str) -> int | None:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return None
    try:
        return int(raw_value)
    except ValueError:
        return None


def _load_apf_baseline_default_episodes(system_config_path: Path) -> int:
    try:
        payload = json.loads(system_config_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return 100
    paper_cfg = payload.get("paper_benchmark", {}) if isinstance(payload, dict) else {}
    raw_value = paper_cfg.get("apf_baseline_episodes", 100)
    try:
        return max(int(raw_value), 1)
    except (TypeError, ValueError):
        return 100


def write_apf_baseline_outputs(
    *,
    output_root: str | Path,
    grouped_rows: Dict[str, list[dict]],
    experiment_id: str,
    stage_name: str,
    stage_index: int,
    file_token: str = "",
) -> Dict[str, Dict[str, Path]]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Dict[str, Path]] = {}
    resolved_file_token = str(file_token or time.strftime("%Y%m%d_%H%M%S")).strip()

    for algorithm_type, rows in grouped_rows.items():
        algo_root = output_root / algorithm_type
        algo_root.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(rows)
        if frame.empty:
            continue

        frame["env_type"] = "apf"
        frame["experiment_id"] = experiment_id
        frame["stage_name"] = stage_name
        frame["stage_index"] = stage_index
        frame["is_resume"] = 0
        frame["source_model"] = algorithm_type

        training_csv = algo_root / f"{algorithm_type}_training_{experiment_id}_{resolved_file_token}.csv"
        training_frame = frame[
            [
                "algorithm_type",
                "env_type",
                "experiment_id",
                "stage_name",
                "stage_index",
                "is_resume",
                "source_model",
                "episode",
                "reward",
                "length",
                "episode_elapsed_time",
                "global_scanned_cells",
                "scan_efficiency",
                "collision_count",
                "reset_reason",
                "success_flag",
            ]
        ].copy()
        training_frame.rename(columns={"episode_elapsed_time": "elapsed_time"}, inplace=True)
        training_frame.to_csv(training_csv, index=False, encoding="utf-8-sig")

        scan_csv = algo_root / f"{algorithm_type}_scan_{experiment_id}_{resolved_file_token}.csv"
        scan_frame = frame[
            [
                "algorithm_type",
                "env_type",
                "experiment_id",
                "stage_name",
                "stage_index",
                "episode",
                "episode_elapsed_time",
                "final_global_scan_ratio",
                "final_global_avg_entropy",
                "reset_reason",
            ]
        ].copy()
        scan_frame.rename(
            columns={
                "episode_elapsed_time": "elapsed_time",
                "final_global_scan_ratio": "scan_ratio",
                "final_global_avg_entropy": "global_avg_entropy",
            },
            inplace=True,
        )
        scan_frame["elapsed_time"] = pd.to_numeric(scan_frame["elapsed_time"], errors="coerce").fillna(0.0).cumsum()
        scan_frame.to_csv(scan_csv, index=False, encoding="utf-8-sig")

        outputs[algorithm_type] = {"training_csv": training_csv, "scan_csv": scan_csv}

    return outputs


def run_apf_baseline_simulation(
    *,
    output_root: str | Path,
    system_config_path: str | Path | None = None,
    episodes: int | None = None,
    seeds: Iterable[int] | None = None,
    experiment_id: str = "",
    stage_name: str = "stage00_apf_baseline",
    stage_index: int = 0,
    raw_log_dir: str | Path | None = None,
) -> Dict[str, Dict[str, Path]]:
    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(output_root)
    system_config_path = Path(system_config_path) if system_config_path else project_root / "system_config.json"
    experiment_id = experiment_id or f"apf_baseline_sim_{time.strftime('%Y%m%d_%H%M%S')}"
    default_seed = int(os.environ.get("TRAIN_SEED", "20260413"))
    resolved_seeds = [int(seed) for seed in (seeds or [default_seed])]
    env_episode_override = _env_int("AIRSIM_QUICK_APF_BASELINE_EPISODES")
    resolved_episodes = int(
        episodes
        if episodes is not None
        else env_episode_override
        if env_episode_override is not None
        else _load_apf_baseline_default_episodes(system_config_path)
    )
    raw_log_root = Path(raw_log_dir) if raw_log_dir else None
    quick_drone_count = _env_int("AIRSIM_QUICK_DRONES")
    if quick_drone_count:
        print(f"[apf-baseline] 使用 {quick_drone_count} 台无人机进行基线仿真")

    grouped_rows: Dict[str, list[dict]] = {"fixed_apf": [], "random_apf": []}
    file_token = time.strftime("%Y%m%d_%H%M%S")

    def flush_outputs() -> Dict[str, Dict[str, Path]]:
        return write_apf_baseline_outputs(
            output_root=output_root,
            grouped_rows=grouped_rows,
            experiment_id=experiment_id,
            stage_name=stage_name,
            stage_index=stage_index,
            file_token=file_token,
        )

    def record_episode(algorithm_type: str, row: dict) -> None:
        grouped_rows[algorithm_type].append(dict(row))
        flush_outputs()

    if raw_log_root:
        raw_log_root.mkdir(parents=True, exist_ok=True)

    for seed in resolved_seeds:
        _run_apf_algorithm(
            algorithm_type="fixed_apf",
            seed=seed,
            eval_episodes=resolved_episodes,
            system_config_path=system_config_path,
            ddpg_model_path=None,
            output_dir=output_root,
            data_log_dir=raw_log_root,
            training_prefix="apf",
            on_episode_complete=lambda row: record_episode("fixed_apf", row),
        )
        _run_apf_algorithm(
            algorithm_type="random_apf",
            seed=seed,
            eval_episodes=resolved_episodes,
            system_config_path=system_config_path,
            ddpg_model_path=None,
            output_dir=output_root,
            data_log_dir=raw_log_root,
            training_prefix="apf",
            on_episode_complete=lambda row: record_episode("random_apf", row),
        )

    return flush_outputs()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run APF baseline multi-episode simulations.")
    parser.add_argument("--out", default="analysis_results/apf_baseline_sim")
    parser.add_argument("--system-config", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, action="append", dest="seeds", default=None)
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--stage-name", default="stage00_apf_baseline")
    parser.add_argument("--stage-index", type=int, default=0)
    parser.add_argument("--raw-log-dir", default=None)
    args = parser.parse_args()

    run_apf_baseline_simulation(
        output_root=args.out,
        system_config_path=args.system_config,
        episodes=args.episodes,
        seeds=args.seeds,
        experiment_id=args.experiment_id,
        stage_name=args.stage_name,
        stage_index=args.stage_index,
        raw_log_dir=args.raw_log_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
