import os
import sys
import unittest

import numpy as np

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from multirotor.DDPG_Weight.online_training_callbacks import (
    EpisodeAwareTrainingCallback,
    transition_from_info_payload,
)


class TransitionPayloadTests(unittest.TestCase):
    def test_transition_from_info_payload_builds_real_flight_transition(self):
        payload = {
            "observation": np.full(18, 1.0, dtype=np.float32),
            "action": np.full(7, 0.5, dtype=np.float32),
            "reward": 2.5,
            "next_observation": np.full(18, 2.0, dtype=np.float32),
            "done": False,
            "source": "real",
            "episode_index": 5,
            "step_index": 9,
            "timestamp": 123.0,
        }

        transition = transition_from_info_payload(payload)

        self.assertEqual(transition.episode_index, 5)
        self.assertEqual(transition.step_index, 9)
        self.assertFalse(transition.done)
        np.testing.assert_allclose(transition.action, np.full(7, 0.5, dtype=np.float32))


class EpisodeAwareTrainingCallbackTests(unittest.TestCase):
    def test_callback_records_transition_when_payload_present(self):
        class DummyTrainer:
            def __init__(self):
                self.transitions = []

            def record_transition(self, transition):
                self.transitions.append(transition)

        trainer = DummyTrainer()
        callback = EpisodeAwareTrainingCallback(
            total_timesteps=10,
            print_interval_steps=1,
            print_interval_sec=60,
            training_visualizer=None,
            data_logger=None,
            priority_trainer=trainer,
        )
        callback.locals = {
            "dones": np.array([False]),
            "infos": [
                {
                    "transition_payload": {
                        "observation": np.zeros(18, dtype=np.float32),
                        "action": np.zeros(7, dtype=np.float32),
                        "reward": 1.0,
                        "next_observation": np.ones(18, dtype=np.float32),
                        "done": False,
                        "source": "real",
                        "episode_index": 4,
                        "step_index": 5,
                        "timestamp": 500.0,
                    }
                }
            ],
        }

        should_continue = callback._handle_episode_boundary_for_test()

        self.assertTrue(should_continue)
        self.assertEqual(len(trainer.transitions), 1)
        self.assertEqual(trainer.transitions[0].episode_index, 4)

    def test_callback_keeps_latched_episode_state_until_training_restarts(self):
        callback = EpisodeAwareTrainingCallback(
            total_timesteps=10,
            print_interval_steps=1,
            print_interval_sec=60,
            training_visualizer=None,
            data_logger=None,
            priority_trainer=None,
        )
        callback.episode_finished = True
        callback.last_episode_index = 7
        callback.locals = {
            "dones": np.array([False]),
            "infos": [{}],
        }

        should_continue = callback._handle_episode_boundary_for_test()

        self.assertTrue(should_continue)
        self.assertTrue(callback.episode_finished)
        self.assertEqual(callback.last_episode_index, 7)

    def test_callback_marks_episode_finished_without_stopping_on_done(self):
        callback = EpisodeAwareTrainingCallback(
            total_timesteps=10,
            print_interval_steps=1,
            print_interval_sec=60,
            training_visualizer=None,
            data_logger=None,
            priority_trainer=None,
        )
        callback.locals = {
            "dones": np.array([True]),
            "infos": [
                {
                    "transition_payload": {
                        "observation": np.zeros(18, dtype=np.float32),
                        "action": np.zeros(7, dtype=np.float32),
                        "reward": 1.0,
                        "next_observation": np.ones(18, dtype=np.float32),
                        "done": True,
                        "source": "real",
                        "episode_index": 2,
                        "step_index": 3,
                        "timestamp": 321.0,
                    }
                }
            ],
        }

        should_continue = callback._handle_episode_boundary_for_test()

        self.assertTrue(should_continue)
        self.assertTrue(callback.episode_finished)
        self.assertEqual(callback.last_episode_index, 2)

    def test_training_start_clears_latched_episode_state(self):
        callback = EpisodeAwareTrainingCallback(
            total_timesteps=10,
            print_interval_steps=1,
            print_interval_sec=60,
            training_visualizer=None,
            data_logger=None,
            priority_trainer=None,
        )
        callback.episode_finished = True
        callback.last_episode_index = 9

        callback._on_training_start()

        self.assertFalse(callback.episode_finished)
        self.assertIsNone(callback.last_episode_index)


if __name__ == "__main__":
    unittest.main()
