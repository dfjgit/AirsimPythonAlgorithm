from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Optional


_CANONICAL_DEFAULTS: Dict[str, Any] = {
    "episode_count": 0,
    "total_steps": 0,
    "current_episode_steps": 0,
    "current_step_reward": 0.0,
    "current_episode_reward": 0.0,
    "steps_per_sec": 0.0,
    "current_episode_time": 0.0,
    "episode_elapsed_time": 0.0,
    "last_episode_duration": 0.0,
    "total_training_time": 0.0,
    "reward_history": [],
    "episode_reward_history": [],
    "avg_reward": 0.0,
    "max_reward": 0.0,
    "min_reward": 0.0,
}


def build_default_training_stats() -> Dict[str, Any]:
    return deepcopy(_CANONICAL_DEFAULTS)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _as_history(value: Any) -> list:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return []


def _pick_first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _pick_first_from_sources(
    primary: Mapping[str, Any],
    secondary: Mapping[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    value = _pick_first(primary, *keys, default=None)
    if value is not None:
        return value
    return _pick_first(secondary, *keys, default=default)


def normalize_training_stats(
    stats: Optional[Mapping[str, Any]] = None,
    fallback: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    primary = dict(stats or {})
    secondary = dict(fallback or {})
    merged: Dict[str, Any] = {}
    merged.update(secondary)
    merged.update(primary)

    normalized = build_default_training_stats()

    normalized["episode_count"] = _as_int(
        _pick_first_from_sources(primary, secondary, "episode_count", "episode", default=0)
    )

    total_steps = _pick_first_from_sources(
        primary, secondary, "total_steps", "timestep", "step", default=0
    )
    normalized["total_steps"] = _as_int(total_steps)

    current_episode_steps = _pick_first_from_sources(
        primary,
        secondary,
        "current_episode_steps",
        "step",
        "total_steps",
        "timestep",
        default=0,
    )
    normalized["current_episode_steps"] = _as_int(current_episode_steps)

    normalized["current_step_reward"] = _as_float(
        _pick_first_from_sources(
            primary,
            secondary,
            "current_step_reward",
            "step_reward",
            "reward",
            default=0.0,
        )
    )
    normalized["current_episode_reward"] = _as_float(
        _pick_first_from_sources(
            primary,
            secondary,
            "current_episode_reward",
            "total_reward",
            "episode_reward",
            default=0.0,
        )
    )
    normalized["steps_per_sec"] = _as_float(
        _pick_first_from_sources(primary, secondary, "steps_per_sec", default=0.0)
    )

    current_episode_time = _as_float(
        _pick_first_from_sources(
            primary,
            secondary,
            "current_episode_time",
            "episode_elapsed_time",
            default=0.0,
        )
    )
    normalized["current_episode_time"] = current_episode_time
    normalized["episode_elapsed_time"] = current_episode_time
    normalized["last_episode_duration"] = _as_float(
        _pick_first_from_sources(
            primary, secondary, "last_episode_duration", default=0.0
        )
    )
    normalized["total_training_time"] = _as_float(
        _pick_first_from_sources(
            primary, secondary, "total_training_time", default=0.0
        )
    )

    reward_history = _as_history(
        _pick_first_from_sources(primary, secondary, "reward_history", default=[])
    )
    episode_reward_history = _as_history(
        _pick_first_from_sources(
            primary, secondary, "episode_reward_history", default=[]
        )
    )
    if not episode_reward_history and reward_history:
        episode_reward_history = list(reward_history)

    normalized["reward_history"] = reward_history
    normalized["episode_reward_history"] = episode_reward_history

    reward_stats_source = episode_reward_history if episode_reward_history else reward_history
    if reward_stats_source:
        normalized["avg_reward"] = _as_float(
            _pick_first_from_sources(
                primary,
                secondary,
                "avg_reward",
                default=sum(reward_stats_source) / len(reward_stats_source),
            )
        )
        normalized["max_reward"] = _as_float(
            _pick_first_from_sources(
                primary, secondary, "max_reward", default=max(reward_stats_source)
            )
        )
        normalized["min_reward"] = _as_float(
            _pick_first_from_sources(
                primary, secondary, "min_reward", default=min(reward_stats_source)
            )
        )
    else:
        normalized["avg_reward"] = _as_float(
            _pick_first_from_sources(primary, secondary, "avg_reward", default=0.0)
        )
        normalized["max_reward"] = _as_float(
            _pick_first_from_sources(primary, secondary, "max_reward", default=0.0)
        )
        normalized["min_reward"] = _as_float(
            _pick_first_from_sources(primary, secondary, "min_reward", default=0.0)
        )

    for key, value in merged.items():
        if key not in normalized:
            normalized[key] = value

    return normalized


def merge_training_stats(
    base: Optional[Mapping[str, Any]],
    patch: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    return normalize_training_stats(stats=patch, fallback=base)
