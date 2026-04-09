import logging
import unittest
from unittest import mock

from multirotor.DDPG_Weight.train_with_crazyflie_online import (
    ProgressTimingState,
    TrainingProgressCallback,
    _build_weighted_callback,
    _compute_weighted_training_target,
    _load_real_weighting_config,
    _should_use_weighted_training,
)


class TrainWithCrazyflieOnlineWeightingTests(unittest.TestCase):
    def test_load_real_weighting_config_reads_nested_online_block(self):
        config = {
            "real_weighting": {
                "update_timing": "episode_end",
                "enable_real_weighting": True,
                "real_update_multiplier": 6,
                "real_batch_ratio": 1.0,
                "min_real_samples_before_update": 24,
                "max_real_updates_per_episode": 10,
                "real_buffer_capacity": 2048,
                "rollback_on_bad_update": True,
            }
        }

        weighting = _load_real_weighting_config(config)

        self.assertEqual(weighting.real_update_multiplier, 6)
        self.assertEqual(weighting.min_real_samples_before_update, 24)
        self.assertEqual(weighting.real_buffer_capacity, 2048)

    def test_load_real_weighting_config_missing_block_returns_none(self):
        config = {"total_timesteps": 100}

        weighting = _load_real_weighting_config(config)

        self.assertIsNone(weighting)

    def test_weighted_callback_wrapper_exposes_episode_state(self):
        class DummyCallback:
            def __init__(self):
                self.called = 0

            def _on_step(self):
                self.called += 1
                return True

        class DummyEpisodeCallback(DummyCallback):
            def __init__(self):
                super().__init__()
                self.episode_finished = True
                self.last_episode_index = 7

        progress_cb = DummyCallback()
        episode_cb = DummyEpisodeCallback()

        wrapper = _build_weighted_callback(progress_cb, episode_cb)

        self.assertTrue(wrapper.episode_finished)
        self.assertEqual(wrapper.last_episode_index, 7)
        wrapper._on_step()
        self.assertEqual(progress_cb.called, 1)
        self.assertEqual(episode_cb.called, 1)

    def test_should_use_weighted_training_respects_disabled_flag(self):
        config = {"real_weighting": {"enable_real_weighting": False}}
        weighting = _load_real_weighting_config(config)

        self.assertFalse(_should_use_weighted_training(weighting))

    def test_should_use_weighted_training_true_when_enabled(self):
        config = {"real_weighting": {"enable_real_weighting": True}}
        weighting = _load_real_weighting_config(config)

        self.assertTrue(_should_use_weighted_training(weighting))

    def test_compute_weighted_training_target_adds_requested_steps(self):
        target = _compute_weighted_training_target(total_timesteps=200, current_steps=150)

        self.assertEqual(target, 350)

    def test_load_real_weighting_config_rejects_invalid_update_timing(self):
        config = {"real_weighting": {"update_timing": "step"}}

        with self.assertRaises(ValueError):
            _load_real_weighting_config(config)

    def test_progress_timing_state_preserves_start_time_across_segments(self):
        timing_state = ProgressTimingState()
        logger = logging.getLogger("train_with_crazyflie_online_weighting_test")

        with mock.patch(
            "multirotor.DDPG_Weight.train_with_crazyflie_online.time.time",
            side_effect=[100.0, 100.0, 200.0, 200.0],
        ):
            callback_one = TrainingProgressCallback(
                total_timesteps=100,
                logger=logger,
                timing_state=timing_state,
            )
            callback_one.num_timesteps = 0
            callback_one._on_training_start()

            callback_two = TrainingProgressCallback(
                total_timesteps=100,
                logger=logger,
                timing_state=timing_state,
            )
            callback_two.num_timesteps = 10
            callback_two._on_training_start()

        self.assertEqual(timing_state.start_time, 100.0)
        self.assertEqual(timing_state.last_print_step, 10)
