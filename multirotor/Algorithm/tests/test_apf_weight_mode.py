import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from apf_weight_mode import (
    APF_WEIGHT_KEYS,
    resolve_apf_weight_mode,
    sample_random_episode_weights,
)


class ApfWeightModeTests(unittest.TestCase):
    def test_use_learned_weights_maps_to_learned_mode_for_apf_control(self):
        self.assertEqual(
            resolve_apf_weight_mode(
                control_mode="apf",
                use_learned_weights=True,
                explicit_mode=None,
            ),
            "learned",
        )

    def test_explicit_random_episode_mode_is_preserved_for_apf_control(self):
        self.assertEqual(
            resolve_apf_weight_mode(
                control_mode="apf",
                use_learned_weights=False,
                explicit_mode="random_episode",
            ),
            "random_episode",
        )

    def test_dqn_control_mode_forces_fixed_apf_mode(self):
        self.assertEqual(
            resolve_apf_weight_mode(
                control_mode="dqn",
                use_learned_weights=True,
                explicit_mode="random_episode",
            ),
            "fixed",
        )

    def test_random_episode_sampling_is_repeatable_and_bounded(self):
        weights_a = sample_random_episode_weights(
            seed=20260413,
            episode_index=3,
            weight_min=0.5,
            weight_max=5.0,
        )
        weights_b = sample_random_episode_weights(
            seed=20260413,
            episode_index=3,
            weight_min=0.5,
            weight_max=5.0,
        )
        weights_c = sample_random_episode_weights(
            seed=20260413,
            episode_index=4,
            weight_min=0.5,
            weight_max=5.0,
        )

        self.assertEqual(sorted(weights_a.keys()), sorted(APF_WEIGHT_KEYS))
        self.assertEqual(weights_a, weights_b)
        self.assertNotEqual(weights_a, weights_c)
        self.assertTrue(all(0.5 <= value <= 5.0 for value in weights_a.values()))


if __name__ == "__main__":
    unittest.main()
