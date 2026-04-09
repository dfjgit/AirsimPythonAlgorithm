import argparse
import sys
import os
import time
import subprocess
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from multirotor.Visualization.visualization_ipc import decode_snapshot, recv_frame
from multirotor.Algorithm.Vector3 import Vector3
from multirotor.Visualization.training_stats_csv_fallback import (
    load_latest_ddpg_visualization_snapshot,
    load_latest_ddpg_training_stats,
)
from multirotor.training_stats_schema import normalize_training_stats

VIS_BUILD_TAG = "2026-04-09T09:00"


class CellProxy:
    def __init__(self, data: Dict[str, Any]):
        self.center = Vector3(data["x"], data["y"], data["z"])
        self.entropy = data["entropy"]


class GridProxy:
    def __init__(self, data: Dict[str, Any]):
        self.revision = int(data.get("revision", 0))
        self.cells = [CellProxy(c) for c in data.get("cells", [])]


class RuntimeProxy:
    def __init__(self, data: Dict[str, Any]):
        self.position = (
            Vector3(data["position"]["x"], data["position"]["y"], data["position"]["z"])
            if data.get("position")
            else None
        )
        self.forward = (
            Vector3(data["forward"]["x"], data["forward"]["y"], data["forward"]["z"])
            if data.get("forward")
            else None
        )
        self.finalMoveDir = (
            Vector3(
                data["finalMoveDir"]["x"],
                data["finalMoveDir"]["y"],
                data["finalMoveDir"]["z"],
            )
            if data.get("finalMoveDir")
            else None
        )
        self.leader_position = (
            Vector3(
                data["leader_position"]["x"],
                data["leader_position"]["y"],
                data["leader_position"]["z"],
            )
            if data.get("leader_position")
            else None
        )
        self.leader_scan_radius = data.get("leader_scan_radius", 0.0)


class ConfigProxy:
    def __init__(self, data: Dict[str, Any]):
        self.scanRadius = data.get("scanRadius", 1.0)
        self.moveSpeed = data.get("moveSpeed", 1.0)
        self.updateInterval = data.get("updateInterval", 0.05)


class AlgorithmProxy:
    """算法代理类，用于外部可视化进程访问算法数据"""

    def __init__(self):
        self.current_weights = {}

    def get_current_coefficients(self) -> Dict[str, float]:
        """获取当前权重系数"""
        return self.current_weights if self.current_weights else {}


class SnapshotServerProxy:
    def __init__(self, visualizer=None):
        self.visualizer = visualizer
        self.grid_data = None
        self.unity_runtime_data = {}
        self.config_data = ConfigProxy({})
        self.algorithms = {}
        self.drone_names = []
        self.control_mode = "dqn"
        self.use_learned_weights = False
        self.battery_data = {}
        self.training_stats = {}
        self.current_training_stats = {}
        self.obstacles = []  # 障碍物数据
        self._last_applied_reset_time = 0.0
        self.current_weights = {}  # 当前权重数据
        self.last_reset_reason = ""  # 最后重置原因
        self.last_reset_time = 0.0  # 最后重置时间
        self.last_collision_object_name = ""
        self.last_collision_penetration_depth = 0.0
        self.reset_history = []  # 重置历史

        # 创建算法代理（用于DDPGTrainingVisualizer访问权重）
        self.algorithm_proxy = AlgorithmProxy()

    def get_all_battery_data(self) -> Dict[str, Dict[str, float]]:
        return self.battery_data


def _apply_snapshot(proxy: SnapshotServerProxy, snap: Dict[str, Any]) -> None:
    if not hasattr(proxy, "_last_verbose_snapshot_log_time"):
        proxy._last_verbose_snapshot_log_time = 0.0
    now = time.time()
    verbose_log = now - proxy._last_verbose_snapshot_log_time >= 5.0
    if verbose_log:
        proxy._last_verbose_snapshot_log_time = now

    # 检查重置时间戳，如果发生新重置，则清空可视化缓存
    server_reset_time = float(snap.get("last_reset_time", 0.0))
    if server_reset_time > proxy._last_applied_reset_time:
        if proxy.visualizer and hasattr(proxy.visualizer, "clear_cache"):
            proxy.visualizer.clear_cache()
        proxy._last_applied_reset_time = server_reset_time

    # 调试：输出快照的关键字段
    snap_keys = list(snap.keys())
    if verbose_log:
        print(f"[IPC客户端] 🔍 收到snapshot，字段数: {len(snap_keys)}")
        print(f"[IPC客户端] 🔍 snapshot字段列表: {snap_keys}")

    # 更新基础数据
    proxy.drone_names = snap.get("drone_names", proxy.drone_names)
    proxy.control_mode = snap.get("control_mode", proxy.control_mode)

    # 更新基础数据
    if "config_data" in snap:
        proxy.config_data = ConfigProxy(snap["config_data"])

    # grid data reconstruction
    if "grid_data" in snap:
        cells_count = len(snap["grid_data"].get("cells", []))
        if verbose_log:
            print(f"[IPC客户端] 🔍 grid_data存在，cells数: {cells_count}")
        # 只有当cells不为空时才更新，避免重置期间清空热力图
        if cells_count > 0:
            proxy.grid_data = GridProxy(snap["grid_data"])
        elif verbose_log:
            print(f"[IPC客户端] ⚠️ grid_data为空，保留旧数据避免热力图消失")
    elif verbose_log:
        print(f"[IPC客户端] 🔍 snapshot中没有grid_data字段，保留旧数据")

    # runtime data reconstruction
    if "unity_runtime_data" in snap:
        if verbose_log:
            print(
                f"[IPC客户端] 🔍 unity_runtime_data存在，drone数: {len(snap['unity_runtime_data'])}"
            )
        runtimes = {}
        for name, data in snap["unity_runtime_data"].items():
            runtimes[name] = RuntimeProxy(data)
        proxy.unity_runtime_data = runtimes

    # battery data mapping (for BatteryPanel)
    if "battery_data" in snap:
        proxy.battery_data = snap.get("battery_data") or {}

    # training stats mapping
    if "training_stats" in snap:
        proxy.training_stats = normalize_training_stats(snap["training_stats"])

    # DQN extra training stats mapping (for action panels)
    if "current_training_stats" in snap:
        previous_stats = dict(proxy.current_training_stats or {})
        proxy.current_training_stats = normalize_training_stats(
            snap.get("current_training_stats") or {}
        )
        if (
            proxy.current_training_stats.get("total_steps", 0) == 0
            and previous_stats.get("reward_history")
            and not proxy.current_training_stats.get("reward_history")
        ):
            proxy.current_training_stats["reward_history"] = list(
                previous_stats.get("reward_history", [])
            )
            proxy.current_training_stats["episode_reward_history"] = list(
                previous_stats.get("episode_reward_history", [])
            )

    current_stats = proxy.current_training_stats or {}
    fallback_stats = proxy.training_stats or {}
    proxy.csv_fallback_active = False
    if (
        getattr(proxy, "visualizer_mode", "") == "ddpg"
        and current_stats.get("total_steps", 0) == 0
    ):
        csv_snapshot = load_latest_ddpg_visualization_snapshot(
            Path(__file__).resolve().parent.parent / "DDPG_Weight" / "airsim_training_logs",
            now_ts=time.time(),
        )
        csv_stats = csv_snapshot.get("training_stats", {})
        if csv_stats.get("total_steps", 0) > 0:
            proxy.csv_fallback_active = True
            proxy.training_stats = csv_stats
            if csv_snapshot.get("battery_data"):
                proxy.battery_data = csv_snapshot["battery_data"]
            if csv_snapshot.get("current_weights"):
                proxy.current_weights = csv_snapshot["current_weights"]
                proxy.algorithm_proxy.current_weights = proxy.current_weights
            if csv_snapshot.get("drone_positions") and proxy.unity_runtime_data:
                leader_position = csv_snapshot.get("leader_position") or {}
                for drone_name, position in csv_snapshot["drone_positions"].items():
                    runtime = proxy.unity_runtime_data.get(drone_name)
                    if runtime is not None:
                        runtime.position = Vector3(
                            position["x"], position["y"], position["z"]
                        )
                        if leader_position:
                            runtime.leader_position = Vector3(
                                leader_position["x"],
                                leader_position["y"],
                                leader_position["z"],
                            )

    if not hasattr(proxy, "_last_training_stats_log_time"):
        proxy._last_training_stats_log_time = 0.0
    if now - proxy._last_training_stats_log_time >= 5.0:
        proxy._last_training_stats_log_time = now
        cts = proxy.current_training_stats or {}
        tts = getattr(proxy, "training_stats", {}) or {}
        print(
            "[IPC客户端] 📊 training stats "
            + f"current(ep={cts.get('episode_count', 0)}, step={cts.get('total_steps', 0)}, reward={cts.get('current_episode_reward', 0.0)}) "
            + f"fallback(ep={tts.get('episode_count', 0)}, step={tts.get('total_steps', 0)}, reward={tts.get('current_episode_reward', 0.0)}) "
            + f"csv_active={getattr(proxy, 'csv_fallback_active', False)}"
        )

    # obstacles data mapping (for visualization)
    if "obstacles" in snap:
        proxy.obstacles = snap.get("obstacles") or []
        if verbose_log:
            print(f"[IPC客户端] 🔍 收到障碍物数据: {len(proxy.obstacles)} 个")
    else:
        # 只在第一次输出警告
        if not hasattr(proxy, "_obstacles_warned"):
            proxy._obstacles_warned = True
            print(f"[IPC客户端] ⚠️ snapshot中没有'obstacles'字段！")

    # current weights mapping (for DDPG training visualization)
    if "current_weights" in snap:
        proxy.current_weights = snap.get("current_weights") or {}
        if verbose_log:
            print(f"[IPC客户端] 🔍 收到权重数据: {len(proxy.current_weights)} 个")
        # 同步更新算法代理的权重
        proxy.algorithm_proxy.current_weights = proxy.current_weights
        # 确保algorithms字典中有第一个无人机的算法代理
        if proxy.drone_names and len(proxy.drone_names) > 0:
            first_drone = proxy.drone_names[0]
            if (
                first_drone not in proxy.algorithms
                or proxy.algorithms[first_drone] != proxy.algorithm_proxy
            ):
                proxy.algorithms[first_drone] = proxy.algorithm_proxy
                if verbose_log:
                    print(f"[IPC客户端] ✅ 算法代理已设置: first_drone={first_drone}")

    # reset info mapping (for training reset visualization)
    proxy.last_reset_reason = snap.get("last_reset_reason", "")
    proxy.last_reset_time = snap.get("last_reset_time", 0)
    proxy.last_collision_object_name = snap.get("last_collision_object_name", "")
    proxy.last_collision_penetration_depth = float(
        snap.get("last_collision_penetration_depth", 0.0) or 0.0
    )
    proxy.reset_history = snap.get("reset_history", [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--render-fps", type=int, default=30)
    parser.add_argument(
        "--mode", choices=["runtime", "dqn", "hrl", "ddpg"], required=True
    )
    args = parser.parse_args()

    print(
        "[IPC客户端] 启动信息: "
        + f"build={VIS_BUILD_TAG}, file={os.path.abspath(__file__)}, cwd={os.getcwd()}, mode={args.mode}, render_fps={args.render_fps}"
    )

    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((args.host, args.port))
    except Exception as e:
        print(f"Failed to connect to IPC server at {args.host}:{args.port}: {e}")
        sys.exit(1)

    # Import visualizer based on mode
    if args.mode == "runtime":
        from multirotor.Visualization.runtime_visualizer import (
            RuntimeVisualizer as _Vis,
        )
    elif args.mode == "dqn":
        from multirotor.Visualization.dqn_movement_visualizer import (
            DQNMovementTrainingVisualizer as _Vis,
        )
    elif args.mode == "hrl":
        from multirotor.Visualization.hierarchical_training_visualizer import (
            HierarchicalTrainingVisualizer as _Vis,
        )
    else:
        from multirotor.Visualization.ddpg_training_visualizer import (
            DDPGTrainingVisualizer as _Vis,
        )

    # Create proxy first (without visualizer reference)
    proxy = SnapshotServerProxy(visualizer=None)
    proxy.visualizer_mode = args.mode

    # Initialize visualizer with proxy
    vis = _Vis(server=proxy, env=None)
    vis.render_fps = max(1, int(args.render_fps))

    # Now update proxy with the visualizer instance for clear_cache callback
    proxy.visualizer = vis

    vis.pygame_initialized = False

    import threading

    stop_event = threading.Event()

    def recv_loop():
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 10

        while not stop_event.is_set():
            try:
                payload = recv_frame(s)
                snap = decode_snapshot(payload)
                consecutive_errors = 0  # 重置错误计数
                _apply_snapshot(proxy, snap)
            except (ConnectionError, ConnectionResetError, BrokenPipeError) as e:
                print(f"[IPC客户端] ❌ 连接错误: {e}，停止接收")
                stop_event.set()
                break
            except Exception as e:
                consecutive_errors += 1
                print(
                    f"[IPC客户端] ⚠️ 接收快照异常 ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}): {e}"
                )
                import traceback

                traceback.print_exc()

                # 如果连续错误过多，退出循环
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    print(f"[IPC客户端] ❌ 连续错误过多，停止接收")
                    stop_event.set()
                    break

                # 短暂等待后重试
                time.sleep(0.1)

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


if __name__ == "__main__":
    main()
