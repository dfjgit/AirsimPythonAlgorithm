import os
import importlib.util
import unittest
from pathlib import Path


def _load_visualization_ipc_module():
    module_path = Path(__file__).resolve().parents[1] / "Visualization" / "visualization_ipc.py"
    spec = importlib.util.spec_from_file_location("test_visualization_ipc_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


_visualization_ipc = _load_visualization_ipc_module()
_should_print_ipc_debug = _visualization_ipc._should_print_ipc_debug


class VisualizationIPCLoggingTests(unittest.TestCase):
    def setUp(self):
        self._prev_mode = os.environ.get("AIRSIM_RUNTIME_LOG_MODE")

    def tearDown(self):
        if self._prev_mode is None:
            os.environ.pop("AIRSIM_RUNTIME_LOG_MODE", None)
        else:
            os.environ["AIRSIM_RUNTIME_LOG_MODE"] = self._prev_mode

    def test_user_mode_suppresses_ipc_snapshot_debug_prints(self):
        os.environ["AIRSIM_RUNTIME_LOG_MODE"] = "user"
        self.assertFalse(_should_print_ipc_debug())

    def test_detail_mode_allows_ipc_snapshot_debug_prints(self):
        os.environ["AIRSIM_RUNTIME_LOG_MODE"] = "detail"
        self.assertTrue(_should_print_ipc_debug())


if __name__ == "__main__":
    unittest.main()
