import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmark_registry_helper import build_algorithm_template


class BenchmarkRegistryHelperTests(unittest.TestCase):
    def test_build_algorithm_template_uses_recommended_primary_family(self):
        template = build_algorithm_template(
            algorithm_type="ppo_scan",
            control_mode="dqn",
            is_trainable=True,
        )

        self.assertEqual(template["primary_family"], "learning_family")
        self.assertEqual(template["family_memberships"], ["learning_family"])
        self.assertEqual(template["comparison_profiles"], ["global_benchmark"])


if __name__ == "__main__":
    unittest.main()
