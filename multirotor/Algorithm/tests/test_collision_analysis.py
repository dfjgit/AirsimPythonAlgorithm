import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collision_analysis import (
    collision_termination_flags,
    collision_termination_rate_percent,
    is_collision_reset_reason,
)


class CollisionAnalysisTests(unittest.TestCase):
    def test_is_collision_reset_reason_accepts_english_and_chinese_variants(self):
        self.assertTrue(is_collision_reset_reason("collision"))
        self.assertTrue(is_collision_reset_reason("发生碰撞"))
        self.assertTrue(is_collision_reset_reason("collision_with_obstacle"))
        self.assertFalse(is_collision_reset_reason("达到时长上限"))

    def test_collision_termination_flags_falls_back_to_collision_count_when_reason_is_blank(self):
        df = pd.DataFrame(
            {
                "reset_reason": ["collision", "达到时长上限", ""],
                "collision_count_final": [0, 0, 2],
            }
        )

        flags = collision_termination_flags(df)
        percents = collision_termination_rate_percent(df)

        self.assertEqual(flags.tolist(), [1.0, 0.0, 1.0])
        self.assertEqual(percents.tolist(), [100.0, 0.0, 100.0])

    def test_collision_termination_flags_does_not_fallback_when_reason_is_explicitly_non_collision(self):
        df = pd.DataFrame(
            {
                "reset_reason": ["达到时长上限"],
                "collision_count_final": [3],
            }
        )

        flags = collision_termination_flags(df)
        self.assertEqual(flags.tolist(), [0.0])

    def test_collision_termination_flags_uses_collision_count_as_secondary_fallback(self):
        df = pd.DataFrame(
            {
                "reset_reason": [""],
                "collision_count": [1],
            }
        )

        flags = collision_termination_flags(df)
        percents = collision_termination_rate_percent(df)

        self.assertEqual(flags.tolist(), [1.0])
        self.assertEqual(percents.tolist(), [100.0])


if __name__ == "__main__":
    unittest.main()
