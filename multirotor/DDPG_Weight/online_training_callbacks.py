from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from multirotor.DDPG_Weight.real_flight_priority import RealFlightTransition

try:
    from stable_baselines3.common.callbacks import BaseCallback
except Exception:  # pragma: no cover - fallback for minimal test environments
    class BaseCallback:  # type: ignore[override]
        def __init__(self, verbose: int = 0):
            self.verbose = verbose
            self.locals = {}
            self.num_timesteps = 0
            self.model = None


def transition_from_info_payload(payload: Dict[str, Any]) -> RealFlightTransition:
    return RealFlightTransition(
        observation=np.array(payload["observation"], dtype=np.float32),
        action=np.array(payload["action"], dtype=np.float32),
        reward=float(payload["reward"]),
        next_observation=np.array(payload["next_observation"], dtype=np.float32),
        done=bool(payload["done"]),
        source=str(payload["source"]),
        episode_index=int(payload["episode_index"]),
        step_index=int(payload["step_index"]),
        timestamp=float(payload["timestamp"]),
    )


def _first_item(value: Any) -> Any:
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return None
        return value[0]
    return value


class EpisodeAwareTrainingCallback(BaseCallback):
    def __init__(
        self,
        total_timesteps: int,
        print_interval_steps: int = 50,
        print_interval_sec: int = 15,
        training_visualizer=None,
        data_logger=None,
        priority_trainer=None,
    ):
        super().__init__()
        self.total_timesteps = max(int(total_timesteps), 0)
        self.print_interval_steps = max(int(print_interval_steps), 1)
        self.print_interval_sec = max(int(print_interval_sec), 1)
        self.training_visualizer = training_visualizer
        self.data_logger = data_logger
        self.priority_trainer = priority_trainer
        self.episode_finished = False
        self.last_episode_index: Optional[int] = None

    def _on_step(self) -> bool:
        return self._handle_episode_boundary(
            self.locals.get("dones"), self.locals.get("infos")
        )

    def _on_training_start(self) -> None:
        self.episode_finished = False
        self.last_episode_index = None

    def _handle_episode_boundary_for_test(self) -> bool:
        return self._handle_episode_boundary(
            self.locals.get("dones"), self.locals.get("infos")
        )

    def _handle_episode_boundary(self, dones, infos) -> bool:
        done_flag = bool(_first_item(dones)) if dones is not None else False
        info = _first_item(infos) if infos is not None else None
        payload = None
        if isinstance(info, dict):
            payload = info.get("transition_payload")

        if payload is not None and self.priority_trainer is not None:
            transition = transition_from_info_payload(payload)
            self.priority_trainer.record_transition(transition)

        if done_flag:
            self.episode_finished = True
            if payload and "episode_index" in payload:
                self.last_episode_index = int(payload["episode_index"])
            else:
                self.last_episode_index = None
            return False

        if self.episode_finished or self.last_episode_index is not None:
            self.episode_finished = False
            self.last_episode_index = None

        return True
