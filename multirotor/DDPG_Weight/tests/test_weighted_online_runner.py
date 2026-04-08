import unittest

from multirotor.DDPG_Weight.weighted_online_runner import run_weighted_online_training


class FakeCallback:
    def __init__(self):
        self.episode_finished = False
        self.last_episode_index = None


class FakeModel:
    def __init__(self, scripted_results):
        self.num_timesteps = 0
        self._scripted_results = list(scripted_results)
        self.learn_calls = []

    def learn(self, total_timesteps, reset_num_timesteps, callback):
        steps, finished, episode_index = self._scripted_results.pop(0)
        self.learn_calls.append((total_timesteps, reset_num_timesteps))
        self.num_timesteps += steps
        callback.episode_finished = finished
        callback.last_episode_index = episode_index
        return self


class WeightedOnlineRunnerTests(unittest.TestCase):
    def test_reused_model_still_runs_first_call_when_resetting_timesteps(self):
        model = FakeModel(scripted_results=[(4, True, 1)])
        model.num_timesteps = 100
        seen_updates = []

        run_weighted_online_training(
            model=model,
            total_timesteps=10,
            reset_num_timesteps=True,
            callback_factory=FakeCallback,
            on_episode_end=seen_updates.append,
        )

        self.assertEqual(model.learn_calls, [(1, True)])
        self.assertEqual(seen_updates, [1])

    def test_runner_skips_post_update_when_episode_index_is_missing(self):
        model = FakeModel(scripted_results=[(5, True, None)])
        seen_updates = []

        run_weighted_online_training(
            model=model,
            total_timesteps=5,
            reset_num_timesteps=True,
            callback_factory=FakeCallback,
            on_episode_end=seen_updates.append,
        )

        self.assertEqual(model.learn_calls, [(1, True)])
        self.assertEqual(seen_updates, [])

    def test_warm_start_first_call_uses_remaining_steps(self):
        model = FakeModel(scripted_results=[(5, True, 1)])
        model.num_timesteps = 5
        seen_updates = []

        run_weighted_online_training(
            model=model,
            total_timesteps=10,
            reset_num_timesteps=False,
            callback_factory=FakeCallback,
            on_episode_end=seen_updates.append,
        )

        self.assertEqual(model.learn_calls, [(1, False)])
        self.assertEqual(seen_updates, [1])

    def test_runner_triggers_post_update_after_each_finished_episode(self):
        model = FakeModel(scripted_results=[(12, True, 1), (12, True, 2), (6, False, None)])
        callback_instances = []
        seen_episode_updates = []

        def callback_factory():
            callback = FakeCallback()
            callback_instances.append(callback)
            return callback

        def on_episode_end(episode_index):
            seen_episode_updates.append(episode_index)

        run_weighted_online_training(
            model=model,
            total_timesteps=24,
            reset_num_timesteps=True,
            callback_factory=callback_factory,
            on_episode_end=on_episode_end,
        )

        self.assertEqual(seen_episode_updates, [1, 2])
        self.assertEqual(len(model.learn_calls), 2)
        self.assertEqual(model.learn_calls, [(1, True), (1, False)])

    def test_subsequent_calls_keep_single_step_segment_requests(self):
        model = FakeModel(scripted_results=[(5, False, None), (5, True, 1)])
        seen_updates = []

        run_weighted_online_training(
            model=model,
            total_timesteps=10,
            reset_num_timesteps=True,
            callback_factory=FakeCallback,
            on_episode_end=seen_updates.append,
        )

        self.assertEqual(model.learn_calls, [(1, True), (1, False)])
        self.assertEqual(seen_updates, [1])

    def test_runner_continues_if_episode_not_finished(self):
        model = FakeModel(scripted_results=[(3, False, None), (3, False, None), (4, True, 1)])
        seen_updates = []

        run_weighted_online_training(
            model=model,
            total_timesteps=10,
            reset_num_timesteps=True,
            callback_factory=FakeCallback,
            on_episode_end=seen_updates.append,
        )

        self.assertEqual(len(model.learn_calls), 3)
        self.assertEqual(model.learn_calls, [(1, True), (1, False), (1, False)])
        self.assertEqual(seen_updates, [1])


if __name__ == "__main__":
    unittest.main()
