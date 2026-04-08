import unittest
import numpy as np

from multirotor.DDPG_Weight.real_flight_priority import (
    RealFlightTransition,
    RealFlightTransitionStore,
    normalize_real_flight_weighting_config,
)


class RealFlightWeightingConfigTests(unittest.TestCase):
    def test_normalize_real_weighting_defaults(self):
        config = normalize_real_flight_weighting_config({})

        self.assertEqual(config.update_timing, "episode_end")
        self.assertTrue(config.enable_real_weighting)
        self.assertEqual(config.real_update_multiplier, 4)
        self.assertEqual(config.real_batch_ratio, 1.0)
        self.assertEqual(config.min_real_samples_before_update, 32)
        self.assertEqual(config.max_real_updates_per_episode, 8)
        self.assertEqual(config.real_buffer_capacity, 5000)
        self.assertTrue(config.rollback_on_bad_update)


    def test_normalize_real_weighting_parses_string_booleans(self):
        config = normalize_real_flight_weighting_config(
            {
                "enable_real_weighting": "false",
                "rollback_on_bad_update": "true",
            }
        )

        self.assertFalse(config.enable_real_weighting)
        self.assertTrue(config.rollback_on_bad_update)

    def test_normalize_real_weighting_rejects_invalid_bool_strings(self):
        with self.assertRaises(ValueError):
            normalize_real_flight_weighting_config({"enable_real_weighting": "maybe"})

class RealFlightTransitionStoreTests(unittest.TestCase):
    def test_store_evicts_oldest_transition_when_capacity_is_exceeded(self):
        store = RealFlightTransitionStore(capacity=2)

        for step in range(3):
            store.add(
                RealFlightTransition(
                    observation=np.full(18, step, dtype=np.float32),
                    action=np.full(7, step, dtype=np.float32),
                    reward=float(step),
                    next_observation=np.full(18, step + 1, dtype=np.float32),
                    done=False,
                    source="real",
                    episode_index=1,
                    step_index=step,
                    timestamp=100.0 + step,
                )
            )

        self.assertEqual(store.size, 2)
        self.assertEqual(store.transitions[0].step_index, 1)
        self.assertEqual(store.transitions[1].step_index, 2)


    def test_get_episode_returns_snapshot_copies(self):
        store = RealFlightTransitionStore(capacity=1)
        base_transition = RealFlightTransition(
            observation=np.zeros(3, dtype=np.float32),
            action=np.zeros(2, dtype=np.float32),
            reward=1.0,
            next_observation=np.ones(3, dtype=np.float32),
            done=True,
            source="real",
            episode_index=1,
            step_index=0,
            timestamp=0.0,
        )

        store.add(base_transition)
        read_sequence = store.get_episode(1)
        read_sequence[0].observation[:] = 9.0
        read_sequence[0].next_observation[:] = 7.0

        read_again = store.get_episode(1)
        self.assertTrue(np.allclose(read_again[0].observation, np.zeros(3, dtype=np.float32)))
        self.assertTrue(np.allclose(read_again[0].next_observation, np.ones(3, dtype=np.float32)))

    def test_add_stores_snapshot_copies(self):
        store = RealFlightTransitionStore(capacity=1)
        observation = np.zeros(3, dtype=np.float32)
        action = np.zeros(2, dtype=np.float32)
        transition = RealFlightTransition(
            observation=observation,
            action=action,
            reward=1.0,
            next_observation=np.zeros(3, dtype=np.float32),
            done=False,
            source="real",
            episode_index=1,
            step_index=0,
            timestamp=0.0,
        )

        store.add(transition)
        observation[:] = 5.0
        action[:] = 7.0

        stored = store.transitions[0]
        self.assertTrue(np.allclose(stored.observation, np.zeros(3, dtype=np.float32)))
        self.assertTrue(np.allclose(stored.action, np.zeros(2, dtype=np.float32)))

    def test_get_episode_raises_when_truncated(self):
        store = RealFlightTransitionStore(capacity=2)

        first = RealFlightTransition(
            observation=np.zeros(1, dtype=np.float32),
            action=np.zeros(1, dtype=np.float32),
            reward=0.0,
            next_observation=np.ones(1, dtype=np.float32),
            done=True,
            source="real",
            episode_index=1,
            step_index=0,
            timestamp=0.0,
        )

        second = RealFlightTransition(
            observation=np.ones(1, dtype=np.float32),
            action=np.ones(1, dtype=np.float32),
            reward=1.0,
            next_observation=np.ones(1, dtype=np.float32) * 2,
            done=True,
            source="real",
            episode_index=1,
            step_index=1,
            timestamp=1.0,
        )

        store.add(first)
        store.add(second)
        # Adding a third transition causes the oldest (first) to be evicted.
        store.add(
            RealFlightTransition(
                observation=np.zeros(1, dtype=np.float32),
                action=np.zeros(1, dtype=np.float32),
                reward=2.0,
                next_observation=np.zeros(1, dtype=np.float32),
                done=False,
                source="real",
                episode_index=2,
                step_index=0,
                timestamp=2.0,
            )
        )

        with self.assertRaises(ValueError):
            store.get_episode(1)

        store.add(
            RealFlightTransition(
                observation=np.zeros(1, dtype=np.float32),
                action=np.zeros(1, dtype=np.float32),
                reward=3.0,
                next_observation=np.zeros(1, dtype=np.float32),
                done=True,
                source="real",
                episode_index=2,
                step_index=1,
                timestamp=3.0,
            )
        )

        self.assertEqual(store.get_episode(1), [])


if __name__ == "__main__":
    unittest.main()
