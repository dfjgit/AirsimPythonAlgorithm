from __future__ import annotations

from typing import Callable, List


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
        callback = callback_factory()
        learn_total_timesteps = total_timesteps
        if not first_call or not reset_num_timesteps:
            remaining_steps = int(total_timesteps) - int(model.num_timesteps)
            if remaining_steps <= 0:
                break
            learn_total_timesteps = remaining_steps

        model.learn(
            total_timesteps=learn_total_timesteps,
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
