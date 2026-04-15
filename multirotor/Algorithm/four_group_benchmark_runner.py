from __future__ import annotations

import argparse
import json
import contextlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MULTIROTOR_ROOT = os.path.join(_REPO_ROOT, "multirotor")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _MULTIROTOR_ROOT not in sys.path:
    sys.path.insert(0, _MULTIROTOR_ROOT)

from apf_weight_mode import sample_random_episode_weights
from algorithm_specific_analysis import generate_algorithm_specific_reports
from benchmark_registry import load_benchmark_registry, resolve_algorithm_registration
from family_analysis import generate_family_reports
from four_group_benchmark_analyzer import generate_four_group_benchmark_report


def choose_first_existing_model(candidates: Iterable[str | Path]) -> Optional[Path]:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return None


def candidate_project_roots(project_root: Path) -> List[Path]:
    roots = [project_root]
    parts = list(project_root.parts)
    if ".worktrees" in parts:
        idx = parts.index(".worktrees")
        primary_root = Path(*parts[:idx]) / "multirotor"
        if primary_root not in roots:
            roots.append(primary_root)
    return roots


def build_apf_action_vector(coefficients: Dict[str, float]) -> np.ndarray:
    return np.array(
        [
            float(coefficients["repulsionCoefficient"]),
            float(coefficients["entropyCoefficient"]),
            float(coefficients["distanceCoefficient"]),
            float(coefficients["leaderRangeCoefficient"]),
            float(coefficients["directionRetentionCoefficient"]),
            float(coefficients.get("obstacleRepulsionDistance", 15.0)),
            float(coefficients.get("obstacleRepulsionCoefficient", 5.0)),
        ],
        dtype=np.float32,
    )


def summarize_episode_metrics(
    *,
    algorithm_type: str,
    seed: int,
    episode: int,
    total_reward: float,
    episode_elapsed_time: float,
    final_global_scan_ratio: float,
    final_global_avg_entropy: float,
    global_scanned_count: int,
    collision_count: int,
    out_of_range_count: int,
    reset_reason: str,
    terminal_battery_voltage: float,
    target_scan_ratio: float = 0.25,
) -> Dict[str, float | int | str]:
    elapsed = max(float(episode_elapsed_time), 1e-6)
    voltage_drop = max(4.2 - float(terminal_battery_voltage), 1e-6)
    normalized_reason = str(reset_reason or "").strip()
    collision_termination_flag = int(
        "collision" in normalized_reason.lower() or "碰撞" in normalized_reason
    )
    return {
        "algorithm_type": algorithm_type,
        "seed": int(seed),
        "episode": int(episode),
        "total_reward": float(total_reward),
        "success_flag": int((float(final_global_scan_ratio) / 100.0) >= float(target_scan_ratio)),
        "final_global_scan_ratio": float(final_global_scan_ratio),
        "final_global_avg_entropy": float(final_global_avg_entropy),
        "scan_efficiency": float(global_scanned_count) / elapsed,
        "avg_scan_cells_per_second": float(global_scanned_count) / elapsed,
        "avg_scan_cells_per_volt_drop": float(global_scanned_count) / voltage_drop,
        "collision_count": int(collision_count),
        "collision_termination_flag": collision_termination_flag,
        "out_of_range_count": int(out_of_range_count),
        "reset_reason": normalized_reason,
        "terminal_battery_voltage": float(terminal_battery_voltage),
    }


def _read_paper_benchmark_config(system_config_path: Path) -> dict:
    payload = json.loads(system_config_path.read_text(encoding="utf-8"))
    return payload.get("paper_benchmark", {})


def _collect_grid_metrics(server) -> Dict[str, float]:
    with server.grid_lock:
        cells = list(getattr(server.grid_data, "cells", []))
    total_cells = len(cells)
    if total_cells <= 0:
        return {
            "global_scanned_count": 0,
            "global_total_count": 0,
            "final_global_scan_ratio": 0.0,
            "final_global_avg_entropy": 100.0,
        }

    entropies = [float(getattr(cell, "entropy", 100.0)) for cell in cells]
    scanned_count = sum(1 for value in entropies if value < 30.0)
    return {
        "global_scanned_count": scanned_count,
        "global_total_count": total_cells,
        "final_global_scan_ratio": (float(scanned_count) / float(total_cells)) * 100.0,
        "final_global_avg_entropy": float(sum(entropies) / len(entropies)),
    }


def _collect_terminal_battery_voltage(server) -> float:
    values = []
    for drone_name in getattr(server, "drone_names", []):
        try:
            values.append(float(server.get_battery_voltage(drone_name)))
        except Exception:
            continue
    if not values:
        return 4.2
    return float(sum(values) / len(values))


def _make_server_kwargs(
    *,
    seed: int,
    run_kind: str,
    experiment_id: str,
    algorithm_type: str,
) -> Dict[str, object]:
    return {
        "seed": seed,
        "run_kind": run_kind,
        "experiment_id": experiment_id,
        "stage_name": "benchmark",
        "stage_index": 1,
        "is_resume": False,
        "source_model": algorithm_type,
        "enable_visualization": False,
    }


def _run_apf_algorithm(
    *,
    algorithm_type: str,
    seed: int,
    eval_episodes: int,
    system_config_path: Path,
    ddpg_model_path: Optional[Path],
    output_dir: Path,
) -> List[Dict[str, object]]:
    from stable_baselines3 import DDPG

    from multirotor.AlgorithmServer import MultiDroneAlgorithmServer
    from multirotor.DDPG_Weight.envs.simple_weight_env import SimpleWeightEnv

    experiment_id = f"{algorithm_type}_seed_{seed}"
    server = MultiDroneAlgorithmServer(
        config_file=str(system_config_path),
        control_mode="apf",
        apf_weight_mode="fixed",
        **_make_server_kwargs(
            seed=seed,
            run_kind="frozen_eval",
            experiment_id=experiment_id,
            algorithm_type=algorithm_type,
        ),
    )
    if not server.start():
        raise RuntimeError(f"Failed to start APF benchmark server for {algorithm_type}")
    if not server.start_mission():
        raise RuntimeError(f"Failed to start APF benchmark mission for {algorithm_type}")
    server.set_experiment_meta(algorithm_type=algorithm_type, env_type="weight", control_mode="apf")
    server.data_collector.set_external_data("run_kind", "frozen_eval")
    server.data_collector.set_external_data(
        "apf_weight_mode",
        "learned" if algorithm_type == "ddpg_apf" else "random_episode" if algorithm_type == "random_apf" else "fixed",
    )

    env = SimpleWeightEnv(server=server, drone_name=server.drone_names[0], reset_unity=True, reset_grid_entropy=True)
    model = None
    if algorithm_type == "ddpg_apf":
        if ddpg_model_path is None:
            raise FileNotFoundError("No DDPG benchmark model found for ddpg_apf evaluation")
        model = DDPG.load(str(ddpg_model_path), env=env)
        if hasattr(model, "set_random_seed"):
            model.set_random_seed(seed)

    results: List[Dict[str, object]] = []
    base_coefficients = server.algorithms[server.drone_names[0]].get_current_coefficients()
    random_cfg = getattr(server.config_data, "paper_benchmark", {}).get("random_apf", {})
    weight_min = float(random_cfg.get("weight_min", 0.5))
    weight_max = float(random_cfg.get("weight_max", 5.0))

    try:
        for episode_index in range(eval_episodes):
            episode_seed = int(seed) + int(episode_index)
            reset_result = env.reset(seed=episode_seed)
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            total_reward = 0.0
            done = False
            info: Dict[str, object] = {}
            sampled_weights = ""

            if algorithm_type == "fixed_apf":
                episode_action = build_apf_action_vector(base_coefficients)
            elif algorithm_type == "random_apf":
                sampled = sample_random_episode_weights(
                    seed=seed,
                    episode_index=episode_index,
                    weight_min=weight_min,
                    weight_max=weight_max,
                )
                sampled.update(
                    {
                        "obstacleRepulsionDistance": float(
                            base_coefficients.get("obstacleRepulsionDistance", 15.0)
                        ),
                        "obstacleRepulsionCoefficient": float(
                            base_coefficients.get("obstacleRepulsionCoefficient", 5.0)
                        ),
                    }
                )
                episode_action = build_apf_action_vector(sampled)
                sampled_weights = json.dumps(sampled, ensure_ascii=False)
            else:
                episode_action = None

            while not done:
                if algorithm_type == "ddpg_apf":
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = episode_action
                obs, reward, done, info = env.step(action)
                total_reward += float(reward)

            grid_metrics = _collect_grid_metrics(server)
            terminal_battery_voltage = _collect_terminal_battery_voltage(server)
            row = summarize_episode_metrics(
                algorithm_type=algorithm_type,
                seed=seed,
                episode=episode_index + 1,
                total_reward=total_reward,
                episode_elapsed_time=float(env.step_count * env.step_duration),
                final_global_scan_ratio=float(grid_metrics["final_global_scan_ratio"]),
                final_global_avg_entropy=float(grid_metrics["final_global_avg_entropy"]),
                global_scanned_count=int(grid_metrics["global_scanned_count"]),
                collision_count=int(info.get("collision_count", 0) or 0),
                out_of_range_count=int(info.get("out_of_range_count", 0) or 0),
                reset_reason=str(info.get("reset_reason", "") or ""),
                terminal_battery_voltage=terminal_battery_voltage,
                target_scan_ratio=float(env.term_cfg.get("target_scan_ratio", 0.25)),
            )
            row["sampled_apf_weights"] = sampled_weights
            results.append(row)
    finally:
        try:
            server.stop()
        except Exception:
            pass

    return results


def _run_dqn_algorithm(
    *,
    seed: int,
    eval_episodes: int,
    system_config_path: Path,
    dqn_model_path: Path,
) -> List[Dict[str, object]]:
    from stable_baselines3 import DQN

    from multirotor.Algorithm.drones_config import DronesConfig
    from multirotor.AlgorithmServer import MultiDroneAlgorithmServer
    from multirotor.DQN_Movement.envs.movement_env import MovementEnv, MultiDroneMovementEnv

    drones_config = DronesConfig()
    drone_names = drones_config.get_training_drones("dqn")
    experiment_id = f"pure_dqn_seed_{seed}"
    server = MultiDroneAlgorithmServer(
        config_file=str(system_config_path),
        drone_names=drone_names,
        control_mode="dqn",
        **_make_server_kwargs(
            seed=seed,
            run_kind="frozen_eval",
            experiment_id=experiment_id,
            algorithm_type="pure_dqn",
        ),
    )
    if not server.start():
        raise RuntimeError("Failed to start DQN benchmark server")
    if not server.start_mission():
        raise RuntimeError("Failed to start DQN benchmark mission")
    server.set_experiment_meta(algorithm_type="pure_dqn", env_type="movement", control_mode="dqn")
    server.data_collector.set_external_data("run_kind", "frozen_eval")

    dqn_config_path = system_config_path.parent / "DQN_Movement" / "configs" / "movement_dqn_config.json"
    if len(drone_names) == 1:
        env = MovementEnv(server=server, drone_name=drone_names[0], config_path=str(dqn_config_path))
    else:
        env = MultiDroneMovementEnv(server=server, drone_names=drone_names, config_path=str(dqn_config_path))
    model = DQN.load(str(dqn_model_path), env=env)
    if hasattr(model, "set_random_seed"):
        model.set_random_seed(seed)

    results: List[Dict[str, object]] = []
    try:
        for episode_index in range(eval_episodes):
            obs, _ = env.reset(seed=seed + episode_index)
            total_reward = 0.0
            terminated = False
            truncated = False
            info: Dict[str, object] = {}
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)

            grid_metrics = _collect_grid_metrics(server)
            terminal_battery_voltage = _collect_terminal_battery_voltage(server)
            reset_reason = str(info.get("last_done_reason", "") or "")
            row = summarize_episode_metrics(
                algorithm_type="pure_dqn",
                seed=seed,
                episode=episode_index + 1,
                total_reward=total_reward,
                episode_elapsed_time=float(time_safe(env, "episode_start_time")),
                final_global_scan_ratio=float(grid_metrics["final_global_scan_ratio"]),
                final_global_avg_entropy=float(grid_metrics["final_global_avg_entropy"]),
                global_scanned_count=int(grid_metrics["global_scanned_count"]),
                collision_count=int(info.get("collision_count", 0) or 0),
                out_of_range_count=int(info.get("out_of_range_count", 0) or 0),
                reset_reason=reset_reason,
                terminal_battery_voltage=terminal_battery_voltage,
                target_scan_ratio=float(getattr(env, "term_cfg", {}).get("target_scan_ratio", 0.25)),
            )
            results.append(row)
    finally:
        try:
            server.stop()
        except Exception:
            pass

    return results


def time_safe(env, attr_name: str) -> float:
    import time

    started_at = float(getattr(env, attr_name, time.time()) or time.time())
    return max(time.time() - started_at, 1e-6)


def _enrich_with_registry(rows: List[Dict[str, object]], registry) -> List[Dict[str, object]]:
    enriched = []
    for row in rows:
        resolved = resolve_algorithm_registration(
            str(row["algorithm_type"]),
            registry,
            control_mode="dqn" if row["algorithm_type"] == "pure_dqn" else "apf",
            apf_weight_mode="learned" if row["algorithm_type"] == "ddpg_apf" else "fixed",
            is_trainable=row["algorithm_type"] in {"ddpg_apf", "pure_dqn"},
        )
        row = dict(row)
        row["primary_family"] = resolved.primary_family
        row["family_memberships"] = ";".join(resolved.family_memberships)
        row["comparison_profiles"] = ";".join(resolved.comparison_profiles)
        row["is_trainable"] = int(bool(resolved.is_trainable))
        row["registry_version"] = resolved.registry_version
        enriched.append(row)
    return enriched


def run_four_group_benchmark(
    *,
    output_root: str | Path,
    system_config_path: Optional[str | Path] = None,
    registry_path: Optional[str | Path] = None,
    seeds: Optional[Iterable[int]] = None,
    eval_episodes_per_seed: Optional[int] = None,
    ddpg_model_path: Optional[str | Path] = None,
    dqn_model_path: Optional[str | Path] = None,
) -> Dict[str, Path]:
    project_root = Path(__file__).resolve().parent.parent
    project_roots = candidate_project_roots(project_root)
    system_config_path = Path(system_config_path) if system_config_path else project_root / "system_config.json"
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    paper_cfg = _read_paper_benchmark_config(system_config_path)
    resolved_seeds = [int(seed) for seed in (seeds or paper_cfg.get("seeds", []))]
    if not resolved_seeds:
        raise ValueError("No seeds configured for four-group benchmark")
    resolved_eval_episodes = int(
        eval_episodes_per_seed
        if eval_episodes_per_seed is not None
        else paper_cfg.get("eval_episodes_per_seed", 10)
    )

    ddpg_candidates: List[Optional[Path]] = [Path(ddpg_model_path) if ddpg_model_path else None]
    dqn_candidates: List[Optional[Path]] = [Path(dqn_model_path) if dqn_model_path else None]
    for root in project_roots:
        ddpg_candidates.extend(
            [
                root / "DDPG_Weight" / "models" / "best_weight_predictor_airsim.zip",
                root / "DDPG_Weight" / "models" / "best_model.zip",
                root / "DDPG_Weight" / "models" / "weight_predictor_airsim.zip",
                root / "DDPG_Weight" / "models" / "weight_predictor_simple.zip",
            ]
        )
        dqn_candidates.extend(
            [
                root / "DQN_Movement" / "models" / "movement_dqn_airsim_final.zip",
                root / "DQN_Movement" / "models" / "movement_dqn_final.zip",
                root / "DQN_Movement" / "scripts" / "models" / "movement_dqn_airsim_final.zip",
                root / "DQN_Movement" / "scripts" / "models" / "movement_dqn_final.zip",
            ]
        )
    ddpg_model = choose_first_existing_model([path for path in ddpg_candidates if path])
    dqn_model = choose_first_existing_model([path for path in dqn_candidates if path])
    if dqn_model is None:
        raise FileNotFoundError("No DQN benchmark model found")

    registry = load_benchmark_registry(registry_path) if registry_path else load_benchmark_registry()

    rows: List[Dict[str, object]] = []
    for seed in resolved_seeds:
        rows.extend(
            _run_apf_algorithm(
                algorithm_type="fixed_apf",
                seed=seed,
                eval_episodes=resolved_eval_episodes,
                system_config_path=system_config_path,
                ddpg_model_path=ddpg_model,
                output_dir=output_root,
            )
        )
        rows.extend(
            _run_apf_algorithm(
                algorithm_type="random_apf",
                seed=seed,
                eval_episodes=resolved_eval_episodes,
                system_config_path=system_config_path,
                ddpg_model_path=ddpg_model,
                output_dir=output_root,
            )
        )
        rows.extend(
            _run_apf_algorithm(
                algorithm_type="ddpg_apf",
                seed=seed,
                eval_episodes=resolved_eval_episodes,
                system_config_path=system_config_path,
                ddpg_model_path=ddpg_model,
                output_dir=output_root,
            )
        )
        rows.extend(
            _run_dqn_algorithm(
                seed=seed,
                eval_episodes=resolved_eval_episodes,
                system_config_path=system_config_path,
                dqn_model_path=dqn_model,
            )
        )

    enriched_rows = _enrich_with_registry(rows, registry)
    eval_csv_path = output_root / "four_group_eval_episodes.csv"
    pd.DataFrame(enriched_rows).to_csv(eval_csv_path, index=False, encoding="utf-8-sig")

    benchmark_outputs = generate_four_group_benchmark_report(
        eval_csv_path=eval_csv_path,
        output_dir=output_root,
    )
    family_outputs = generate_family_reports(
        eval_csv_paths=[eval_csv_path],
        registry=registry,
        output_root=output_root.parent / "family_comparisons",
    )
    algorithm_specific_outputs = generate_algorithm_specific_reports(
        eval_csv_paths=[eval_csv_path],
        output_root=output_root.parent / "algorithm_specific",
    )
    result = dict(benchmark_outputs)
    result["eval_csv"] = eval_csv_path
    result["family_output_root"] = output_root.parent / "family_comparisons"
    result["family_outputs"] = family_outputs
    result["algorithm_specific_output_root"] = output_root.parent / "algorithm_specific"
    result["algorithm_specific_outputs"] = algorithm_specific_outputs
    return result


def _parse_seed_list(raw_value: str) -> List[int]:
    return [int(item.strip()) for item in str(raw_value).split(",") if item.strip()]


@contextlib.contextmanager
def temporary_env_var(name: str, value: Optional[str]):
    previous = os.environ.get(name)
    try:
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the four-group frozen benchmark and generate reports.")
    parser.add_argument("--out", type=str, default=None, help="Output directory. Defaults to analysis_results/four_group_benchmark.")
    parser.add_argument("--system-config", type=str, default=None, help="Optional system_config.json path.")
    parser.add_argument("--registry", type=str, default=None, help="Optional benchmark_registry.json path.")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seed list. Defaults to paper_benchmark.seeds.")
    parser.add_argument("--episodes", type=int, default=None, help="Episodes per seed. Defaults to paper_benchmark.eval_episodes_per_seed.")
    parser.add_argument("--ddpg-model", type=str, default=None, help="Optional DDPG model path.")
    parser.add_argument("--dqn-model", type=str, default=None, help="Optional DQN model path.")
    parser.add_argument(
        "--unity-timeout",
        type=float,
        default=None,
        help="Optional Unity connect timeout in seconds. Useful for fast smoke checks.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(args.out) if args.out else project_root.parent / "analysis_results" / "four_group_benchmark"
    with temporary_env_var(
        "UNITY_CONNECT_TIMEOUT_SEC",
        None if args.unity_timeout is None else str(args.unity_timeout),
    ):
        run_four_group_benchmark(
            output_root=output_root,
            system_config_path=args.system_config,
            registry_path=args.registry,
            seeds=_parse_seed_list(args.seeds) if args.seeds else None,
            eval_episodes_per_seed=args.episodes,
            ddpg_model_path=args.ddpg_model,
            dqn_model_path=args.dqn_model,
        )


if __name__ == "__main__":
    main()
