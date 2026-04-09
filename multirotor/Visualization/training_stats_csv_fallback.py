from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

from multirotor.training_stats_schema import normalize_training_stats


def load_latest_ddpg_training_stats(log_dir: Path | str) -> Dict:
    snapshot = load_latest_ddpg_visualization_snapshot(log_dir)
    return snapshot.get("training_stats", {})


def _parse_timestamp(value: str) -> float | None:
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d %H:%M:%S").timestamp()
    except Exception:
        return None


def _extract_drone_positions(row: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    drone_positions = {}
    for key, value in row.items():
        if not key.endswith("_x"):
            continue
        drone_name = key[:-2]
        try:
            x = float(value)
            y = float(row.get(f"{drone_name}_y", 0.0) or 0.0)
            z = float(row.get(f"{drone_name}_z", 0.0) or 0.0)
        except ValueError:
            continue
        drone_positions[drone_name] = {"x": x, "y": y, "z": z}
    return drone_positions


def _estimate_leader_position(
    drone_positions: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    if not drone_positions:
        return {}
    count = len(drone_positions)
    return {
        "x": sum(pos["x"] for pos in drone_positions.values()) / count,
        "y": sum(pos["y"] for pos in drone_positions.values()) / count,
        "z": sum(pos["z"] for pos in drone_positions.values()) / count,
    }


def load_latest_ddpg_visualization_snapshot(
    log_dir: Path | str, now_ts: float | None = None
) -> Dict:
    log_path = Path(log_dir)
    if not log_path.exists():
        return {}

    candidates = sorted(
        log_path.glob("scan_data_*.csv"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return {}

    latest = candidates[0]
    try:
        with latest.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception:
        return {}

    if not rows:
        return {}

    last_row = rows[-1]
    raw_stats = {
        "episode": last_row.get("episode", 0),
        "step": last_row.get("episode_step", last_row.get("step", 0)),
        "reward": last_row.get("step_reward", 0.0),
        "total_reward": last_row.get("episode_reward", 0.0),
        "episode_elapsed_time": last_row.get("episode_elapsed_time", 0.0),
    }
    training_stats = normalize_training_stats(raw_stats)

    battery_data = {}
    drone_positions = _extract_drone_positions(last_row)
    current_weights = {}
    for key, value in last_row.items():
        if key.endswith("_battery_voltage") and value not in (None, ""):
            drone_name = key[: -len("_battery_voltage")]
            try:
                voltage = float(value)
                percentage = max(
                    0.0, min(100.0, round((voltage - 3.2) / (4.2 - 3.2) * 100.0))
                )
                if voltage >= 4.0:
                    status = "normal"
                elif voltage >= 3.7:
                    status = "warning"
                elif voltage >= 3.5:
                    status = "low"
                elif voltage >= 3.2:
                    status = "critical"
                else:
                    status = "empty"
                battery_data[drone_name] = {
                    "voltage": voltage,
                    "remaining_percentage": percentage,
                    "status": status,
                }
            except ValueError:
                continue
    weight_key_map = {
        "repulsion_coefficient": "repulsionCoefficient",
        "entropy_coefficient": "entropyCoefficient",
        "distance_coefficient": "distanceCoefficient",
        "leader_range_coefficient": "leaderRangeCoefficient",
        "direction_retention_coefficient": "directionRetentionCoefficient",
    }
    for csv_key, canonical_key in weight_key_map.items():
        raw_value = last_row.get(csv_key, None)
        if raw_value in (None, ""):
            continue
        try:
            current_weights[canonical_key] = float(raw_value)
        except ValueError:
            continue

    global_scanned_count = int(float(last_row.get("global_scanned_count", 0) or 0))
    global_total_count = int(float(last_row.get("global_total_count", 0) or 0))
    global_scan_ratio_text = str(last_row.get("global_scan_ratio", "0") or "0").strip()
    global_scan_ratio = 0.0
    try:
        global_scan_ratio = float(global_scan_ratio_text.rstrip("%"))
    except ValueError:
        global_scan_ratio = 0.0

    return {
        "training_stats": training_stats,
        "global_scanned_count": global_scanned_count,
        "global_total_count": global_total_count,
        "global_scan_ratio": global_scan_ratio,
        "battery_data": battery_data,
        "drone_positions": drone_positions,
        "leader_position": _estimate_leader_position(drone_positions),
        "current_weights": current_weights,
    }
