import json
import socket
import struct
import threading
import time as _time  # ⚠️ 修改：重命名time模块避免冲突
import zlib
from typing import Any, Dict, Optional, Callable
from enum import Enum
from pathlib import Path

import numpy as np


def _send_frame(conn: socket.socket, payload: bytes) -> None:
    header = struct.pack("!I", len(payload))
    conn.sendall(header + payload)


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return buf


def recv_frame(conn: socket.socket) -> bytes:
    header = _recv_exact(conn, 4)
    (length,) = struct.unpack("!I", header)
    return _recv_exact(conn, length)


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def encode_snapshot(snapshot: Dict[str, Any], compress_level: int = 1) -> bytes:
    raw = json.dumps(
        snapshot,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return zlib.compress(raw, level=compress_level)


def decode_snapshot(payload: bytes) -> Dict[str, Any]:
    raw = zlib.decompress(payload)
    return json.loads(raw.decode("utf-8"))


class VisualizationIPCServer:
    def __init__(
        self,
        snapshot_provider: Callable[[], Dict[str, Any]],
        host: str = "127.0.0.1",
        port: int = 0,
        hz: float = 10.0,
        compress_level: int = 1,
    ):
        self.snapshot_provider = snapshot_provider
        self.host = host
        self.port = port
        self.hz = hz
        self.compress_level = compress_level

        self._sock: Optional[socket.socket] = None
        self._client: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    @property
    def bound_port(self) -> int:
        if not self._sock:
            return 0
        return int(self._sock.getsockname()[1])

    def start(self) -> None:
        if self._running:
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(1)
        self._sock.settimeout(0.5)
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        try:
            if self._client:
                try:
                    self._client.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                self._client.close()
        finally:
            self._client = None

        try:
            if self._sock:
                self._sock.close()
        finally:
            self._sock = None

    def _run(self) -> None:
        next_send = _time.time()
        period = 1.0 / max(self.hz, 1e-6)

        while self._running:
            if not self._client:
                try:
                    conn, _addr = self._sock.accept()  # type: ignore[union-attr]
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    # 连接建立后只负责持续发送快照，不应因为瞬时发送阻塞就主动断开连接。
                    # Windows 上大帧 payload + 调度抖动会让 0.5s timeout 过于敏感，表现为
                    # “几秒后实时链路断开，可视化退回 CSV 回放”。
                    conn.settimeout(None)
                    self._client = conn
                except socket.timeout:
                    continue
                except Exception:
                    continue

            now = _time.time()
            if now < next_send:
                _time.sleep(min(0.01, next_send - now))
                continue

            next_send = now + period
            try:
                snapshot = self.snapshot_provider()
                # 调试日志：每5秒输出一次，显示发送的快照字段
                current_time = _time.time()
                if not hasattr(self, '_last_snapshot_log_time'):
                    self._last_snapshot_log_time = 0
                if current_time - self._last_snapshot_log_time > 5.0:
                    self._last_snapshot_log_time = current_time
                    snap_keys = list(snapshot.keys())
                    print(f"[IPC服务端] 📤 发送snapshot，字段数: {len(snap_keys)}，字段列表: {snap_keys}")
                payload = encode_snapshot(snapshot, compress_level=self.compress_level)
                _send_frame(self._client, payload)
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
                # 客户端断开连接是正常情况（关闭可视化窗口时），不输出警告
                # 清除客户端连接，等待重新连接
                if self._client:
                    try:
                        self._client.close()
                    except Exception:
                        pass
                    self._client = None
            except Exception as e:
                # 非 socket 异常通常来自快照构造/序列化，不应直接断开可视化连接。
                print(f"[IPC服务端] ⚠️ 快照发送异常: {e}")
                _time.sleep(min(0.05, period))
