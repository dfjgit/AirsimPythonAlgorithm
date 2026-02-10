import argparse
import sys
import os
import time
import subprocess
from typing import Any, Dict, List

# Ensure project root in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from multirotor.Visualization.visualization_ipc import decode_snapshot, recv_frame
from multirotor.Algorithm.Vector3 import Vector3

class CellProxy:
    def __init__(self, data: Dict[str, Any]):
        self.center = Vector3(data['x'], data['y'], data['z'])
        self.entropy = data['entropy']

class GridProxy:
    def __init__(self, data: Dict[str, Any]):
        self.revision = int(data.get('revision', 0))
        self.cells = [CellProxy(c) for c in data.get('cells', [])]

class RuntimeProxy:
    def __init__(self, data: Dict[str, Any]):
        self.position = Vector3(data['position']['x'], data['position']['y'], data['position']['z']) if data.get('position') else None
        self.forward = Vector3(data['forward']['x'], data['forward']['y'], data['forward']['z']) if data.get('forward') else None
        self.finalMoveDir = Vector3(data['finalMoveDir']['x'], data['finalMoveDir']['y'], data['finalMoveDir']['z']) if data.get('finalMoveDir') else None
        self.leader_position = Vector3(data['leader_position']['x'], data['leader_position']['y'], data['leader_position']['z']) if data.get('leader_position') else None
        self.leader_scan_radius = data.get('leader_scan_radius', 0.0)

class ConfigProxy:
    def __init__(self, data: Dict[str, Any]):
        self.scanRadius = data.get('scanRadius', 1.0)
        self.moveSpeed = data.get('moveSpeed', 1.0)
        self.updateInterval = data.get('updateInterval', 0.05)

class SnapshotServerProxy:
    def __init__(self, visualizer=None):
        self.visualizer = visualizer
        self.grid_data = None
        self.unity_runtime_data = {}
        self.config_data = ConfigProxy({})
        self.algorithms = {}
        self.drone_names = []
        self.control_mode = 'dqn'
        self.use_learned_weights = False
        self.battery_data = {}
        self.current_training_stats = {}
        self._last_applied_reset_time = 0.0

    def get_all_battery_data(self) -> Dict[str, Dict[str, float]]:
        return self.battery_data

def _apply_snapshot(proxy: SnapshotServerProxy, snap: Dict[str, Any]) -> None:
    # 检查重置时间戳，如果发生新重置，则清空可视化缓存
    server_reset_time = float(snap.get('last_reset_time', 0.0))
    if server_reset_time > proxy._last_applied_reset_time:
        if proxy.visualizer and hasattr(proxy.visualizer, 'clear_cache'):
            proxy.visualizer.clear_cache()
        proxy._last_applied_reset_time = server_reset_time

    proxy.drone_names = snap.get('drone_names', proxy.drone_names)
    proxy.control_mode = snap.get('control_mode', proxy.control_mode)
    
    if 'config_data' in snap:
        proxy.config_data = ConfigProxy(snap['config_data'])

    # grid data reconstruction
    if 'grid_data' in snap:
        proxy.grid_data = GridProxy(snap['grid_data'])

    # runtime data reconstruction
    if 'unity_runtime_data' in snap:
        runtimes = {}
        for name, data in snap['unity_runtime_data'].items():
            runtimes[name] = RuntimeProxy(data)
        proxy.unity_runtime_data = runtimes

    # battery data mapping (for BatteryPanel)
    if 'battery_data' in snap:
        proxy.battery_data = snap.get('battery_data') or {}

    # training stats mapping
    if 'training_stats' in snap:
        proxy.training_stats = snap['training_stats']

    # DQN extra training stats mapping (for action panels)
    if 'current_training_stats' in snap:
        proxy.current_training_stats = snap.get('current_training_stats') or {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--mode', choices=['runtime', 'dqn', 'hrl', 'ddpg'], required=True)
    args = parser.parse_args()

    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((args.host, args.port))
    except Exception as e:
        print(f"Failed to connect to IPC server at {args.host}:{args.port}: {e}")
        sys.exit(1)

    # 将可视化器实例传入 proxy，便于在 reset 时触发 clear_cache
    proxy = SnapshotServerProxy(visualizer=vis)

    # Import visualizer based on mode
    if args.mode == 'runtime':
        from multirotor.Visualization.runtime_visualizer import RuntimeVisualizer as _Vis
    elif args.mode == 'dqn':
        from multirotor.Visualization.dqn_movement_visualizer import DQNMovementTrainingVisualizer as _Vis
    elif args.mode == 'hrl':
        from multirotor.Visualization.hierarchical_training_visualizer import HierarchicalTrainingVisualizer as _Vis
    else:
        from multirotor.Visualization.ddpg_training_visualizer import DDPGTrainingVisualizer as _Vis

    # Initialize visualizer with proxy
    vis = _Vis(server=proxy, env=None)
    
    vis.pygame_initialized = False

    import threading

    stop_event = threading.Event()

    def recv_loop():
        try:
            while not stop_event.is_set():
                payload = recv_frame(s)
                snap = decode_snapshot(payload)
                _apply_snapshot(proxy, snap)
        except Exception:
            stop_event.set()

    t = threading.Thread(target=recv_loop, daemon=True)
    t.start()

    try:
        # pygame 必须在本进程主线程中运行（Windows 上尤为重要）
        # 注意：BaseVisualizer.start_visualization() 会另起线程运行 run()，
        # 在独立进程里我们直接调用 run()，确保 pygame/窗口循环在主线程。
        vis.run()
    finally:
        stop_event.set()
        try:
            s.close()
        except Exception:
            pass

if __name__ == '__main__':
    main()
