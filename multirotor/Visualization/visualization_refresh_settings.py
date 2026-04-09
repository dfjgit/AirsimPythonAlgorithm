from __future__ import annotations

from typing import Any, Dict, Tuple


DEFAULT_IPC_HZ = 10.0
DEFAULT_RENDER_FPS = 30
MIN_IPC_HZ = 1.0
MAX_IPC_HZ = 60.0
MIN_RENDER_FPS = 5
MAX_RENDER_FPS = 120


def _coerce_float(value: Any, default: float, lower: float, upper: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    return max(lower, min(parsed, upper))


def _coerce_int(value: Any, default: int, lower: int, upper: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(lower, min(parsed, upper))


def resolve_visualization_refresh_settings(config: Dict[str, Any]) -> Tuple[float, int]:
    ipc_hz = _coerce_float(
        config.get("visualization_ipc_hz", DEFAULT_IPC_HZ),
        DEFAULT_IPC_HZ,
        MIN_IPC_HZ,
        MAX_IPC_HZ,
    )
    render_fps = _coerce_int(
        config.get("visualization_render_fps", DEFAULT_RENDER_FPS),
        DEFAULT_RENDER_FPS,
        MIN_RENDER_FPS,
        MAX_RENDER_FPS,
    )
    return ipc_hz, render_fps
