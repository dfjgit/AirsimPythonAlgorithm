from __future__ import annotations

from typing import Callable, List


def _has_remaining_training_budget(total_timesteps: int, current_timesteps: int) -> bool:
    return int(total_timesteps) > 0 and int(current_timesteps) < int(total_timesteps)


def run_weighted_online_training(
    model,
    total_timesteps: int,
    reset_num_timesteps: bool,
    callback_factory: Callable[[], object],
    on_episode_end: Callable[[int], object],
) -> List[object]:
    first_call = True
    update_results: List[object] = []

    while True:
        if not first_call and not _has_remaining_training_budget(
            total_timesteps=total_timesteps,
            current_timesteps=getattr(model, "num_timesteps", 0),
        ):
            break

        callback = callback_factory()

        model.learn(
            total_timesteps=1,
            reset_num_timesteps=reset_num_timesteps if first_call else False,
            callback=callback,
        )
        first_call = False

        if getattr(callback, "episode_finished", False):
            episode_index = getattr(callback, "last_episode_index", None)
            if episode_index is not None:
                update_results.append(on_episode_end(int(episode_index)))

        if int(model.num_timesteps) >= int(total_timesteps):
            break

    return update_results
