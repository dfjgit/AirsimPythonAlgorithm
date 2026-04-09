import os
import subprocess
import sys
import time
from typing import Optional

from multirotor.Visualization.visualization_ipc import VisualizationIPCServer


class ExternalRuntimeVisualizerManager:
    """通过独立进程启动运行时可视化，避免在 Windows 后台线程运行 pygame。"""

    def __init__(
        self,
        server,
        host: str = "127.0.0.1",
        port: int = 0,
        hz: float = 10.0,
        compress_level: int = 1,
        log_dir: Optional[str] = None,
    ):
        self.server = server
        self.host = host
        self.port = port
        self.hz = hz
        self.compress_level = compress_level
        self.log_dir = log_dir or os.path.join(
            os.path.dirname(__file__), "logs", "runtime"
        )
        self.ipc_server: Optional[VisualizationIPCServer] = None
        self.process: Optional[subprocess.Popen] = None
        self._log_handle = None

    def start_visualization(self) -> bool:
        if self.process and self.process.poll() is None:
            return False

        snapshot_provider = getattr(self.server, "get_visualization_snapshot", None)
        if not callable(snapshot_provider):
            raise RuntimeError("server does not provide get_visualization_snapshot()")

        self.ipc_server = VisualizationIPCServer(
            snapshot_provider=snapshot_provider,
            host=self.host,
            port=self.port,
            hz=self.hz,
            compress_level=self.compress_level,
        )
        self.ipc_server.start()

        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, "external_vis.log")
        self._log_handle = open(log_path, "w", encoding="utf-8")

        client_entry = os.path.join(
            os.path.dirname(__file__), "external_visualizer_client.py"
        )
        command = [
            sys.executable,
            client_entry,
            "--mode",
            "runtime",
            "--host",
            self.host,
            "--port",
            str(self.ipc_server.bound_port),
        ]

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        creationflags = 0
        if os.name == "nt" and os.environ.get("VIS_NEW_CONSOLE", "0") == "1":
            creationflags = subprocess.CREATE_NEW_CONSOLE

        try:
            self.process = subprocess.Popen(
                command,
                stdout=self._log_handle,
                stderr=self._log_handle,
                creationflags=creationflags,
                env=env,
            )
            time.sleep(0.5)
            if self.process.poll() is not None:
                self.stop_visualization()
                return False
            return True
        except Exception:
            self.stop_visualization()
            raise

    def stop_visualization(self):
        if self.process:
            try:
                if self.process.poll() is None:
                    self.process.terminate()
                    self.process.wait(timeout=2.0)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            finally:
                self.process = None

        if self.ipc_server:
            try:
                self.ipc_server.stop()
            finally:
                self.ipc_server = None

        if self._log_handle:
            try:
                self._log_handle.close()
            finally:
                self._log_handle = None
