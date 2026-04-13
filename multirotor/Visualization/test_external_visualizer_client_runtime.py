import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from multirotor.Visualization.external_visualizer_client import build_visualizer


class ExternalVisualizerClientRuntimeTests(unittest.TestCase):
    def test_runtime_mode_builds_visualizer_without_env_kwarg(self):
        proxy = object()
        vis = build_visualizer("runtime", proxy)
        self.assertIsNotNone(vis)


if __name__ == "__main__":
    unittest.main()
