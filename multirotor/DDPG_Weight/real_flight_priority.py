from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Set

import numpy as np


@dataclass(frozen=True)
class RealFlightWeightingConfig:
    update_timing: str = "episode_end"
    enable_real_weighting: bool = True
    real_update_multiplier: int = 4
    real_batch_ratio: float = 1.0
    min_real_samples_before_update: int = 32
    max_real_updates_per_episode: int = 8
    real_buffer_capacity: int = 5000
    rollback_on_bad_update: bool = True


_TRUE_VALUES = {"true", "1", "yes", "on"}
_FALSE_VALUES = {"false", "0", "no", "off"}


def _coerce_bool(value, default: bool) -> bool:
    if value is None:
        return default

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
        raise ValueError(f"Cannot parse boolean value: {value!r}")

    return bool(value)


def normalize_real_flight_weighting_config(raw: Dict) -> RealFlightWeightingConfig:
    raw = dict(raw or {})
    return RealFlightWeightingConfig(
        update_timing=str(raw.get("update_timing", "episode_end")),
        enable_real_weighting=_coerce_bool(raw.get("enable_real_weighting"), True),
        real_update_multiplier=int(raw.get("real_update_multiplier", 4)),
        real_batch_ratio=float(raw.get("real_batch_ratio", 1.0)),
        min_real_samples_before_update=int(raw.get("min_real_samples_before_update", 32)),
        max_real_updates_per_episode=int(raw.get("max_real_updates_per_episode", 8)),
        real_buffer_capacity=int(raw.get("real_buffer_capacity", 5000)),
        rollback_on_bad_update=_coerce_bool(raw.get("rollback_on_bad_update"), True),
    )


@dataclass(frozen=True)
class RealFlightTransition:
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    done: bool
    source: str
    episode_index: int
    step_index: int
    timestamp: float


class RealFlightTransitionStore:
    def __init__(self, capacity: int):
        self.capacity = max(1, int(capacity))
        self.transitions: Deque[RealFlightTransition] = deque()
        self._truncated_episodes: Set[int] = set()
        self._episode_counts: Dict[int, int] = {}

    @property
    def size(self) -> int:
        return len(self.transitions)

    def add(self, transition: RealFlightTransition) -> None:
        snapshot = self._snapshot_transition(transition)

        if self.size >= self.capacity:
            evicted = self.transitions.popleft()
            self._decrement_episode(evicted.episode_index)
            self._mark_truncated(evicted.episode_index)

        self.transitions.append(snapshot)
        self._increment_episode(snapshot.episode_index)

    def get_episode(self, episode_index: int) -> List[RealFlightTransition]:
        if episode_index in self._truncated_episodes:
            raise ValueError(f"Episode {episode_index} has been truncated.")
        return [
            self._snapshot_transition(item)
            for item in self.transitions
            if item.episode_index == episode_index
        ]

    @staticmethod
    def _snapshot_transition(transition: RealFlightTransition) -> RealFlightTransition:
        return RealFlightTransition(
            observation=np.array(transition.observation, copy=True),
            action=np.array(transition.action, copy=True),
            reward=transition.reward,
            next_observation=np.array(transition.next_observation, copy=True),
            done=transition.done,
            source=transition.source,
            episode_index=transition.episode_index,
            step_index=transition.step_index,
            timestamp=transition.timestamp,
        )

    def _increment_episode(self, episode_index: int) -> None:
        self._episode_counts[episode_index] = self._episode_counts.get(episode_index, 0) + 1

    def _decrement_episode(self, episode_index: int) -> None:
        count = self._episode_counts.get(episode_index)
        if count is None:
            return

        if count <= 1:
            self._episode_counts.pop(episode_index, None)
            self._truncated_episodes.discard(episode_index)
        else:
            self._episode_counts[episode_index] = count - 1

    def _mark_truncated(self, episode_index: int) -> None:
        if self._episode_counts.get(episode_index, 0) > 0:
            self._truncated_episodes.add(episode_index)


@dataclass(frozen=True)
class WeightedUpdateResult:
    status: str
    episode_real_samples: int
    extra_gradient_steps: int
    rollback_triggered: bool
    policy_param_delta_norm: float


class RealFlightPriorityTrainer:
    def __init__(self, config: RealFlightWeightingConfig):
        self.config = config
        self.store = RealFlightTransitionStore(capacity=config.real_buffer_capacity)

    def record_transition(self, transition: RealFlightTransition) -> None:
        self.store.add(transition)

    def apply_post_episode_update(self, model, episode_index: int) -> WeightedUpdateResult:
        episode_transitions = self.store.get_episode(episode_index)
        episode_real_samples = len(episode_transitions)

        if not self.config.enable_real_weighting:
            return WeightedUpdateResult("disabled", episode_real_samples, 0, False, 0.0)

        if episode_real_samples < self.config.min_real_samples_before_update:
            return WeightedUpdateResult(
                "skipped_min_samples",
                episode_real_samples,
                0,
                False,
                0.0,
            )

        snapshot = copy.deepcopy(model.policy.state_dict())
        gradient_steps = min(
            self.config.max_real_updates_per_episode,
            max(
                1,
                math.ceil(episode_real_samples / max(1, int(model.batch_size)))
                * self.config.real_update_multiplier,
            ),
        )

        model.train(gradient_steps=gradient_steps, batch_size=model.batch_size)

        probe_states = [
            item.observation for item in episode_transitions[: min(4, episode_real_samples)]
        ]
        if not self._sanity_check(model, probe_states):
            if self.config.rollback_on_bad_update:
                model.policy.load_state_dict(snapshot)
                return WeightedUpdateResult(
                    "rolled_back",
                    episode_real_samples,
                    gradient_steps,
                    True,
                    0.0,
                )

        delta = abs(float(model.policy.state_dict()["weight"]) - float(snapshot["weight"]))
        return WeightedUpdateResult(
            "applied",
            episode_real_samples,
            gradient_steps,
            False,
            delta,
        )

    def _sanity_check(self, model, probe_states) -> bool:
        for observation in probe_states:
            action, _ = model.predict(observation, deterministic=True)
            if not np.isfinite(action).all():
                return False
            if np.any(action < 0.5) or np.any(action > 30.0):
                return False
        return True
