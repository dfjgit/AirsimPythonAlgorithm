import os
import shutil
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MULTIROTOR_ROOT = os.path.join(PROJECT_ROOT, "multirotor")
DDPG_WEIGHT_ROOT = os.path.join(MULTIROTOR_ROOT, "DDPG_Weight")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if MULTIROTOR_ROOT not in sys.path:
    sys.path.insert(0, MULTIROTOR_ROOT)
if DDPG_WEIGHT_ROOT not in sys.path:
    sys.path.insert(0, DDPG_WEIGHT_ROOT)

from multirotor.Visualization.base_visualizer import BaseVisualizer
from multirotor.Visualization.ddpg_training_visualizer import DDPGTrainingVisualizer
from multirotor.Visualization.dqn_movement_visualizer import (
    DQNMovementTrainingVisualizer,
)
from multirotor.Algorithm.Vector3 import Vector3


class DummyVisualizer(BaseVisualizer):
    def setup_panels(self):
        return None

    def get_visualization_data(self):
        return {}


class BaseVisualizerLayoutTests(unittest.TestCase):
    def test_visualizer_scales_layout_to_fit_desktop_override(self):
        with patch.dict(os.environ, {"VIS_DESKTOP_SIZE": "1366x768"}, clear=False):
            visualizer = DummyVisualizer()

        self.assertLess(visualizer.SCREEN_WIDTH, 1920)
        self.assertLess(visualizer.SCREEN_HEIGHT, 1080)
        self.assertEqual(visualizer.left_panel_width, visualizer.right_panel_width)
        self.assertGreater(visualizer.view_width, visualizer.left_panel_width)
        self.assertEqual(
            visualizer.view_width,
            visualizer.SCREEN_WIDTH
            - visualizer.left_panel_width
            - visualizer.right_panel_width,
        )

    def test_ddpg_dashboard_panels_fit_within_short_screen(self):
        with patch.dict(os.environ, {"VIS_DESKTOP_SIZE": "1366x768"}, clear=False):
            visualizer = DDPGTrainingVisualizer(server=None, env=None)
            visualizer.setup_panels()

        panel_bottom = max(
            panel.y + panel.height for panel in visualizer.panel_manager.panels.values()
        )
        self.assertLessEqual(panel_bottom, visualizer.SCREEN_HEIGHT - 10)


class DDPGTrainingVisualizerDataTests(unittest.TestCase):
    def test_episode_elapsed_time_is_exposed_for_training_panel(self):
        server = SimpleNamespace(
            current_training_stats={
                "episode_count": 3,
                "total_steps": 42,
                "current_episode_steps": 6,
                "current_episode_reward": 9.5,
                "episode_elapsed_time": 12.5,
            },
            drone_names=[],
            algorithms={},
        )
        visualizer = DDPGTrainingVisualizer(server=server, env=None)

        data = visualizer.get_visualization_data()

        self.assertEqual(data["current_episode_time"], 12.5)

    @patch(
        "multirotor.Visualization.ddpg_training_visualizer.load_latest_ddpg_visualization_snapshot"
    )
    def test_ddpg_visualizer_uses_csv_fallback_when_live_stats_are_zero(
        self, load_csv_snapshot_mock
    ):
        load_csv_snapshot_mock.return_value = {
            "training_stats": {
                "episode_count": 1,
                "total_steps": 9,
                "current_episode_steps": 9,
                "current_step_reward": 48.9,
                "current_episode_reward": 1147.0,
                "current_episode_time": 45.0,
            },
            "global_scanned_count": 89,
            "global_total_count": 1613,
            "battery_data": {"UAV1": {"voltage": 4.1}},
            "current_weights": {
                "repulsionCoefficient": 1.9,
                "entropyCoefficient": 1.1,
            },
        }
        server = SimpleNamespace(
            current_training_stats={
                "episode_count": 0,
                "total_steps": 0,
                "current_episode_steps": 0,
                "current_episode_reward": 0.0,
            },
            training_stats={},
            drone_names=[],
            algorithms={},
        )
        visualizer = DDPGTrainingVisualizer(server=server, env=None)

        data = visualizer.get_visualization_data()

        self.assertEqual(data["stats_source"], "csv_fallback")
        self.assertEqual(data["total_steps"], 9)
        self.assertEqual(data["csv_global_scanned_count"], 89)
        self.assertEqual(data["csv_global_total_count"], 1613)
        self.assertEqual(data["battery_data"]["UAV1"]["voltage"], 4.1)
        self.assertEqual(data["weights"]["repulsionCoefficient"], 1.9)

    @patch(
        "multirotor.Visualization.ddpg_training_visualizer.load_latest_ddpg_visualization_snapshot"
    )
    def test_ddpg_visualizer_applies_csv_positions_to_runtime_data(
        self, load_csv_snapshot_mock
    ):
        load_csv_snapshot_mock.return_value = {
            "training_stats": {
                "episode_count": 1,
                "total_steps": 9,
                "current_episode_steps": 9,
                "current_episode_reward": 1147.0,
            },
            "drone_positions": {
                "UAV1": {"x": 1.0, "y": 2.0, "z": 3.0},
                "UAV2": {"x": 4.0, "y": 5.0, "z": 6.0},
            },
            "leader_position": {"x": 2.5, "y": 3.5, "z": 4.5},
        }
        runtime = {
            "UAV1": {
                "position": Vector3(0.0, 0.0, 0.0),
                "finalMoveDir": None,
                "leaderPosition": None,
                "leaderScanRadius": 0.0,
            },
            "UAV2": {
                "position": Vector3(0.0, 0.0, 0.0),
                "finalMoveDir": None,
                "leaderPosition": None,
                "leaderScanRadius": 0.0,
            },
        }
        server = SimpleNamespace(
            current_training_stats={"total_steps": 0},
            training_stats={"total_steps": 9, "current_episode_reward": 1147.0},
            drone_names=[],
            algorithms={},
        )
        visualizer = DDPGTrainingVisualizer(server=server, env=None)
        with patch.object(
            BaseVisualizer,
            "update_data",
            return_value=(None, runtime),
        ):
            _, runtime_data = visualizer.update_data()

        self.assertEqual(runtime_data["UAV1"]["position"].x, 1.0)
        self.assertEqual(runtime_data["UAV2"]["position"].z, 6.0)
        self.assertEqual(runtime_data["UAV1"]["leaderPosition"].x, 2.5)
        self.assertEqual(runtime_data["UAV1"]["leaderPosition"].z, 4.5)


class TrainingStatsSchemaTests(unittest.TestCase):
    def test_normalize_training_stats_maps_legacy_fields_and_preserves_extras(self):
        from multirotor.training_stats_schema import normalize_training_stats

        normalized = normalize_training_stats(
            {
                "episode_count": 4,
                "step": 11,
                "total_reward": 6.5,
                "reward_history": [1.0, 3.0],
                "episode_elapsed_time": 7.25,
                "action_counts": {"0": 2},
            }
        )

        self.assertEqual(normalized["total_steps"], 11)
        self.assertEqual(normalized["current_episode_steps"], 11)
        self.assertEqual(normalized["current_episode_reward"], 6.5)
        self.assertEqual(normalized["current_episode_time"], 7.25)
        self.assertEqual(normalized["action_counts"], {"0": 2})

    def test_dqn_visualizer_reads_normalized_training_stats(self):
        server = SimpleNamespace(
            current_training_stats={
                "episode_count": 5,
                "step": 9,
                "total_reward": 4.0,
                "episode_elapsed_time": 3.5,
                "reward_history": [2.0, 4.0],
                "action_counts": {"0": 1, "1": 2},
                "last_action": 1,
            },
            get_all_battery_data=lambda: {},
        )
        visualizer = DQNMovementTrainingVisualizer(env=None, server=server)

        data = visualizer.get_visualization_data()

        self.assertEqual(data["total_steps"], 9)
        self.assertEqual(data["current_episode_steps"], 9)
        self.assertEqual(data["current_episode_reward"], 4.0)
        self.assertEqual(data["current_episode_time"], 3.5)
        self.assertEqual(data["action_counts"][0], 1)
        self.assertEqual(data["action_counts"][1], 2)

    def test_ddpg_visualizer_falls_back_to_proxy_training_stats_when_current_stats_are_empty(self):
        server = SimpleNamespace(
            current_training_stats={
                "episode_count": 0,
                "total_steps": 0,
                "current_episode_steps": 0,
                "current_episode_reward": 0.0,
            },
            training_stats={
                "episode": 3,
                "step": 14,
                "reward": 1.25,
                "total_reward": 9.75,
                "episode_elapsed_time": 4.5,
            },
            drone_names=[],
            algorithms={},
        )
        visualizer = DDPGTrainingVisualizer(server=server, env=None)

        data = visualizer.get_visualization_data()

        self.assertEqual(data["episode_count"], 3)
        self.assertEqual(data["total_steps"], 14)
        self.assertEqual(data["current_episode_steps"], 14)
        self.assertEqual(data["current_step_reward"], 1.25)
        self.assertEqual(data["current_episode_reward"], 9.75)
        self.assertEqual(data["current_episode_time"], 4.5)


class DDPGCsvFallbackTests(unittest.TestCase):
    def test_loads_latest_ddpg_scan_csv_row_as_training_stats(self):
        from multirotor.Visualization.training_stats_csv_fallback import (
            load_latest_ddpg_visualization_snapshot,
        )

        log_dir = Path(PROJECT_ROOT) / ".codex_tmp" / f"csv_fallback_test_{time.time_ns()}"
        log_dir.mkdir(parents=True, exist_ok=True)
        scan_csv = log_dir / "scan_data_example.csv"
        scan_csv.write_text(
            "episode,timestamp,elapsed_time,episode_elapsed_time,episode_step,step_reward,episode_reward,global_scanned_count,global_total_count,global_scan_ratio,repulsion_coefficient,entropy_coefficient,distance_coefficient,leader_range_coefficient,direction_retention_coefficient,UAV1_x,UAV1_y,UAV1_z,UAV1_battery_voltage\n"
            "1,2026-04-08 12:00:30,13.03,5.00,1,297.5452,297.5452,33,1613,2.05%,2.07,0.89,2.45,2.58,2.80,-2.338,1.602,-15.923,4.185\n"
            "1,2026-04-08 12:00:35,18.04,10.01,2,108.6675,406.2128,48,1613,2.98%,1.90,1.06,2.63,2.76,2.90,-0.548,1.499,-10.167,4.170\n",
            encoding="utf-8",
        )

        snapshot = load_latest_ddpg_visualization_snapshot(log_dir)
        stats = snapshot["training_stats"]

        self.assertEqual(stats["episode_count"], 1)
        self.assertEqual(stats["total_steps"], 2)
        self.assertEqual(stats["current_episode_steps"], 2)
        self.assertAlmostEqual(stats["current_step_reward"], 108.6675)
        self.assertAlmostEqual(stats["current_episode_reward"], 406.2128)
        self.assertAlmostEqual(stats["current_episode_time"], 10.01)
        self.assertEqual(snapshot["global_scanned_count"], 48)
        self.assertEqual(snapshot["global_total_count"], 1613)
        self.assertAlmostEqual(snapshot["global_scan_ratio"], 2.98)
        self.assertEqual(snapshot["drone_positions"]["UAV1"]["x"], -0.548)
        self.assertEqual(snapshot["battery_data"]["UAV1"]["voltage"], 4.17)
        self.assertEqual(snapshot["battery_data"]["UAV1"]["status"], "normal")
        self.assertEqual(snapshot["battery_data"]["UAV1"]["remaining_percentage"], 97.0)
        self.assertAlmostEqual(
            snapshot["current_weights"]["repulsionCoefficient"], 1.90
        )
        self.assertAlmostEqual(
            snapshot["current_weights"]["leaderRangeCoefficient"], 2.76
        )
        self.assertAlmostEqual(snapshot["leader_position"]["x"], -0.548)
        self.assertAlmostEqual(snapshot["leader_position"]["y"], 1.499)
        self.assertAlmostEqual(snapshot["leader_position"]["z"], -10.167)


class VisualizationIpcEncodingTests(unittest.TestCase):
    def test_encode_snapshot_accepts_numpy_scalars(self):
        from multirotor.Visualization.visualization_ipc import (
            decode_snapshot,
            encode_snapshot,
        )

        payload = encode_snapshot(
            {
                "weights": {"repulsionCoefficient": np.float32(1.5)},
                "step": np.int64(7),
            }
        )
        decoded = decode_snapshot(payload)

        self.assertEqual(decoded["weights"]["repulsionCoefficient"], 1.5)
        self.assertEqual(decoded["step"], 7)

    @patch("multirotor.Visualization.visualization_ipc.socket.socket")
    @patch("multirotor.Visualization.visualization_ipc.threading.Thread")
    def test_ipc_server_uses_blocking_client_socket_after_accept(
        self, thread_mock, socket_cls
    ):
        from multirotor.Visualization.visualization_ipc import VisualizationIPCServer

        listen_socket = MagicMock()
        accepted_socket = MagicMock()
        socket_cls.return_value = listen_socket
        listen_socket.accept.return_value = (accepted_socket, ("127.0.0.1", 12345))
        thread_mock.return_value = MagicMock()

        server = VisualizationIPCServer(snapshot_provider=lambda: server.__dict__.update(_running=False) or {})
        server.start()
        server._run()

        accepted_socket.settimeout.assert_called_with(None)


class VisualizationRefreshSettingsTests(unittest.TestCase):
    def test_resolve_visualization_refresh_settings_uses_defaults(self):
        from multirotor.Visualization.visualization_refresh_settings import (
            DEFAULT_IPC_HZ,
            DEFAULT_RENDER_FPS,
            resolve_visualization_refresh_settings,
        )

        ipc_hz, render_fps = resolve_visualization_refresh_settings({})

        self.assertEqual(ipc_hz, DEFAULT_IPC_HZ)
        self.assertEqual(render_fps, DEFAULT_RENDER_FPS)

    def test_resolve_visualization_refresh_settings_clamps_values(self):
        from multirotor.Visualization.visualization_refresh_settings import (
            resolve_visualization_refresh_settings,
        )

        ipc_hz, render_fps = resolve_visualization_refresh_settings(
            {"visualization_ipc_hz": 120, "visualization_render_fps": 999}
        )

        self.assertEqual(ipc_hz, 60.0)
        self.assertEqual(render_fps, 120)


class ExternalVisualizerClientLoggingTests(unittest.TestCase):
    @patch("multirotor.Visualization.external_visualizer_client.print")
    @patch("multirotor.Visualization.external_visualizer_client.time.time")
    def test_apply_snapshot_throttles_verbose_logs(self, time_mock, print_mock):
        from multirotor.Visualization.external_visualizer_client import (
            SnapshotServerProxy,
            _apply_snapshot,
        )

        proxy = SnapshotServerProxy()
        proxy.visualizer_mode = "ddpg"
        snap = {
            "drone_names": ["UAV1"],
            "control_mode": "apf",
            "grid_data": {"cells": [{"x": 0.0, "y": 0.0, "z": 0.0, "entropy": 10.0}]},
            "unity_runtime_data": {
                "UAV1": {
                    "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                    "forward": None,
                    "finalMoveDir": None,
                    "leader_position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "leader_scan_radius": 1.0,
                }
            },
            "current_weights": {"repulsionCoefficient": 1.5},
            "obstacles": [],
            "current_training_stats": {},
            "training_stats": {},
        }

        time_mock.return_value = 100.0
        _apply_snapshot(proxy, snap)
        first_print_count = print_mock.call_count

        time_mock.return_value = 100.1
        _apply_snapshot(proxy, snap)
        second_print_count = print_mock.call_count

        self.assertGreater(first_print_count, 0)
        self.assertEqual(first_print_count, second_print_count)


class ExternalVisualizerClientStateTests(unittest.TestCase):
    def test_apply_snapshot_preserves_reward_history_when_reset_frame_has_zero_stats(self):
        from multirotor.Visualization.external_visualizer_client import (
            SnapshotServerProxy,
            _apply_snapshot,
        )

        proxy = SnapshotServerProxy()
        proxy.visualizer_mode = "ddpg"
        proxy.current_training_stats = {
            "episode_count": 12,
            "total_steps": 60,
            "current_episode_steps": 60,
            "current_episode_reward": 3359.69,
            "reward_history": [100.0, 200.0],
            "episode_reward_history": [100.0, 200.0],
        }
        reset_snapshot = {
            "drone_names": ["UAV1"],
            "control_mode": "apf",
            "grid_data": {"cells": []},
            "unity_runtime_data": {},
            "current_training_stats": {
                "episode_count": 13,
                "total_steps": 0,
                "current_episode_steps": 0,
                "current_episode_reward": 0.0,
                "reward_history": [],
                "episode_reward_history": [],
            },
            "training_stats": {},
            "obstacles": [],
            "current_weights": {},
        }

        _apply_snapshot(proxy, reset_snapshot)

        self.assertEqual(proxy.current_training_stats["episode_count"], 13)
        self.assertEqual(proxy.current_training_stats["reward_history"], [100.0, 200.0])
        self.assertEqual(
            proxy.current_training_stats["episode_reward_history"], [100.0, 200.0]
        )


class ExternalRuntimeVisualizerManagerTests(unittest.TestCase):
    @patch("multirotor.Visualization.external_runtime_visualizer.open", new_callable=mock_open)
    @patch("multirotor.Visualization.external_runtime_visualizer.os.makedirs")
    @patch("multirotor.Visualization.external_runtime_visualizer.subprocess.Popen")
    @patch("multirotor.Visualization.external_runtime_visualizer.VisualizationIPCServer")
    def test_start_visualization_launches_external_runtime_client(
        self,
        ipc_server_cls,
        popen_cls,
        makedirs_mock,
        open_mock,
    ):
        from multirotor.Visualization.external_runtime_visualizer import (
            ExternalRuntimeVisualizerManager,
        )

        ipc_server = ipc_server_cls.return_value
        ipc_server.bound_port = 43210
        process = popen_cls.return_value
        process.poll.return_value = None

        server = SimpleNamespace(get_visualization_snapshot=lambda: {"timestamp": 1.0})
        manager = ExternalRuntimeVisualizerManager(server)

        started = manager.start_visualization()

        self.assertTrue(started)
        ipc_server_cls.assert_called_once()
        ipc_server.start.assert_called_once_with()
        popen_cls.assert_called_once()
        command = popen_cls.call_args.args[0]
        self.assertIn("--mode", command)
        self.assertIn("runtime", command)
        self.assertIn("--port", command)
        self.assertIn("43210", command)
        makedirs_mock.assert_called()
        open_mock.assert_called()


class AlgorithmServerVisualizationInitTests(unittest.TestCase):
    def test_init_visualization_uses_external_runtime_manager(self):
        import AlgorithmServer as algorithm_server_module

        server = algorithm_server_module.MultiDroneAlgorithmServer.__new__(
            algorithm_server_module.MultiDroneAlgorithmServer
        )
        server.visualizer = None

        fake_manager = MagicMock()

        with patch.object(algorithm_server_module, "HAS_VISUALIZATION", True), patch.object(
            algorithm_server_module,
            "ExternalRuntimeVisualizerManager",
            create=True,
            return_value=fake_manager,
        ) as manager_cls:
            algorithm_server_module.MultiDroneAlgorithmServer._init_visualization(server)

        manager_cls.assert_called_once_with(server)
        self.assertIs(server.visualizer, fake_manager)


class AlgorithmServerTrainingStatsTests(unittest.TestCase):
    def test_set_training_stats_defaults_episode_steps_to_step_value(self):
        import AlgorithmServer as algorithm_server_module

        server = algorithm_server_module.MultiDroneAlgorithmServer.__new__(
            algorithm_server_module.MultiDroneAlgorithmServer
        )
        server._training_stats_lock = threading.Lock()
        server._episode_start_time = time.time() - 1.0
        server.current_training_stats = {"current_episode_steps": 0}
        server.data_collector = None

        algorithm_server_module.MultiDroneAlgorithmServer.set_training_stats(
            server,
            episode=2,
            step=7,
            reward=1.5,
            total_reward=8.0,
        )

        self.assertEqual(server.current_training_stats["current_episode_steps"], 7)

    def test_visualization_snapshot_uses_get_training_data_for_training_stats(self):
        import AlgorithmServer as algorithm_server_module

        server = algorithm_server_module.MultiDroneAlgorithmServer.__new__(
            algorithm_server_module.MultiDroneAlgorithmServer
        )
        server._training_stats_lock = threading.Lock()
        server.current_training_stats = {
            "episode_count": 2,
            "total_steps": 11,
            "current_episode_steps": 11,
            "current_step_reward": 1.25,
            "current_episode_reward": 8.5,
            "episode_elapsed_time": 4.0,
        }
        server.data_collector = SimpleNamespace(
            external_data={},
            external_data_lock=threading.Lock(),
        )
        server._vis_snapshot_cache = None
        server._vis_snapshot_cache_time = 0.0
        server._last_reset_time = 0.0
        server.drone_names = []
        server.control_mode = "apf"
        server.config_data = SimpleNamespace(
            scanRadius=1.0, moveSpeed=1.0, updateInterval=0.05
        )
        server.grid_lock = threading.Lock()
        server.grid_data = SimpleNamespace(cells=[])
        server.data_lock = threading.Lock()
        server.unity_runtime_data = {}
        server._last_reset_reason = ""
        server._last_collision_object_name = ""
        server._last_collision_penetration_depth = 0.0
        server._reset_history = []
        server.unity_socket = SimpleNamespace(received_obstacles=[])
        server.algorithms = {}
        server.get_all_battery_data = lambda: {}

        snapshot = algorithm_server_module.MultiDroneAlgorithmServer.get_visualization_snapshot(
            server
        )

        self.assertEqual(snapshot["training_stats"]["episode_count"], 2)
        self.assertEqual(snapshot["training_stats"]["total_steps"], 11)
        self.assertEqual(snapshot["training_stats"]["current_episode_reward"], 8.5)

    def test_visualization_snapshot_prefers_external_training_stats_when_current_stats_are_zero(self):
        import AlgorithmServer as algorithm_server_module

        server = algorithm_server_module.MultiDroneAlgorithmServer.__new__(
            algorithm_server_module.MultiDroneAlgorithmServer
        )
        server._training_stats_lock = threading.Lock()
        server.current_training_stats = {
            "episode_count": 0,
            "total_steps": 0,
            "current_episode_steps": 0,
            "current_step_reward": 0.0,
            "current_episode_reward": 0.0,
            "episode_elapsed_time": 0.0,
        }
        server.data_collector = SimpleNamespace(
            external_data={
                "episode": 3,
                "step": 14,
                "reward": 1.25,
                "step_reward": 1.25,
                "total_reward": 9.75,
                "episode_reward": 9.75,
                "episode_elapsed_time": 4.5,
            },
            external_data_lock=threading.Lock(),
        )
        server._vis_snapshot_cache = None
        server._vis_snapshot_cache_time = 0.0
        server._last_reset_time = 0.0
        server.drone_names = []
        server.control_mode = "apf"
        server.config_data = SimpleNamespace(
            scanRadius=1.0, moveSpeed=1.0, updateInterval=0.05
        )
        server.grid_lock = threading.Lock()
        server.grid_data = SimpleNamespace(cells=[])
        server.data_lock = threading.Lock()
        server.unity_runtime_data = {}
        server._last_reset_reason = ""
        server._last_collision_object_name = ""
        server._last_collision_penetration_depth = 0.0
        server._reset_history = []
        server.unity_socket = SimpleNamespace(received_obstacles=[])
        server.algorithms = {}
        server.get_all_battery_data = lambda: {}

        snapshot = algorithm_server_module.MultiDroneAlgorithmServer.get_visualization_snapshot(
            server
        )

        self.assertEqual(snapshot["training_stats"]["episode_count"], 3)
        self.assertEqual(snapshot["training_stats"]["total_steps"], 14)
        self.assertEqual(snapshot["current_training_stats"]["episode_count"], 3)
        self.assertEqual(snapshot["current_training_stats"]["total_steps"], 14)

    def test_visualization_snapshot_uses_prebuilt_runtime_and_grid_snapshots(self):
        import AlgorithmServer as algorithm_server_module

        server = algorithm_server_module.MultiDroneAlgorithmServer.__new__(
            algorithm_server_module.MultiDroneAlgorithmServer
        )
        server._training_stats_lock = threading.Lock()
        server.current_training_stats = {
            "episode_count": 1,
            "total_steps": 2,
            "current_episode_steps": 2,
            "current_step_reward": 1.0,
            "current_episode_reward": 3.0,
            "episode_elapsed_time": 2.0,
        }
        server.data_collector = SimpleNamespace(
            external_data={},
            external_data_lock=threading.Lock(),
        )
        server._vis_snapshot_cache = None
        server._vis_snapshot_cache_time = 0.0
        server._last_reset_time = 0.0
        server.drone_names = ["UAV1"]
        server.control_mode = "apf"
        server.config_data = SimpleNamespace(
            scanRadius=1.0, moveSpeed=1.0, updateInterval=0.05
        )
        server.grid_lock = threading.Lock()
        server.data_lock = threading.Lock()
        server.grid_data = SimpleNamespace(cells=[])
        server.unity_runtime_data = {}
        server._vis_grid_snapshot = {"cells": [{"x": 1.0, "y": 2.0, "z": 3.0, "entropy": 42.0}]}
        server._vis_runtime_snapshot = {
            "UAV1": {
                "position": {"x": 9.0, "y": 8.0, "z": 7.0},
                "forward": None,
                "finalMoveDir": None,
                "leader_position": {"x": 1.0, "y": 1.0, "z": 1.0},
                "leader_scan_radius": 2.0,
            }
        }
        server._last_reset_reason = ""
        server._last_collision_object_name = ""
        server._last_collision_penetration_depth = 0.0
        server._reset_history = []
        server.unity_socket = SimpleNamespace(received_obstacles=[])
        server.algorithms = {}
        server.get_all_battery_data = lambda: {}

        snapshot = algorithm_server_module.MultiDroneAlgorithmServer.get_visualization_snapshot(
            server
        )

        self.assertEqual(snapshot["grid_data"]["cells"][0]["entropy"], 42.0)
        self.assertEqual(snapshot["unity_runtime_data"]["UAV1"]["position"]["x"], 9.0)
        self.assertEqual(snapshot["unity_runtime_data"]["UAV1"]["leader_position"]["z"], 1.0)

    @patch("AlgorithmServer._time.time")
    def test_visualization_snapshot_refreshes_between_10hz_ticks(self, time_mock):
        import AlgorithmServer as algorithm_server_module

        server = algorithm_server_module.MultiDroneAlgorithmServer.__new__(
            algorithm_server_module.MultiDroneAlgorithmServer
        )
        server._training_stats_lock = threading.Lock()
        server.current_training_stats = {
            "episode_count": 1,
            "total_steps": 1,
            "current_episode_steps": 1,
            "current_step_reward": 1.0,
            "current_episode_reward": 1.0,
            "episode_elapsed_time": 1.0,
        }
        server.data_collector = SimpleNamespace(
            external_data={},
            external_data_lock=threading.Lock(),
        )
        server._vis_snapshot_cache = None
        server._vis_snapshot_cache_time = 0.0
        server._last_reset_time = 0.0
        server.drone_names = []
        server.control_mode = "apf"
        server.config_data = SimpleNamespace(
            scanRadius=1.0, moveSpeed=1.0, updateInterval=0.05
        )
        server.grid_lock = threading.Lock()
        server.grid_data = SimpleNamespace(cells=[])
        server.data_lock = threading.Lock()
        server.unity_runtime_data = {}
        server._last_reset_reason = ""
        server._last_collision_object_name = ""
        server._last_collision_penetration_depth = 0.0
        server._reset_history = []
        server.unity_socket = SimpleNamespace(received_obstacles=[])
        server.algorithms = {}
        server.get_all_battery_data = lambda: {}

        time_mock.return_value = 100.0
        first = algorithm_server_module.MultiDroneAlgorithmServer.get_visualization_snapshot(
            server
        )

        server.current_training_stats = {
            "episode_count": 1,
            "total_steps": 2,
            "current_episode_steps": 2,
            "current_step_reward": 2.0,
            "current_episode_reward": 3.0,
            "episode_elapsed_time": 2.0,
        }
        time_mock.return_value = 100.099
        second = algorithm_server_module.MultiDroneAlgorithmServer.get_visualization_snapshot(
            server
        )

        self.assertEqual(first["current_training_stats"]["total_steps"], 1)
        self.assertEqual(second["current_training_stats"]["total_steps"], 2)


if __name__ == "__main__":
    unittest.main()
