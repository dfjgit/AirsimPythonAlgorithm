from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


_CONSOLE_INFO_RULES = {
    "AlgorithmServer": (
        "启动 Unity Socket 服务",
        "等待 Unity 连接中",
        "Unity 连接成功",
        "连接到AirSim模拟器",
        "AirSim连接成功",
        "准备开始任务",
        "无人机起飞完成",
        "所有虚拟无人机已确认起飞",
        "服务初始化成功",
        "[重置]",
        "等待系统完全稳定",
    ),
    "UnitySocketServer": (
        "服务器启动成功",
        "Unity已连接",
        "连接已断开",
        "服务器已停止",
    ),
}

_CONSOLE_WARNING_SUPPRESS_RULES = {
    "AlgorithmServer": (
        "[网格更新]",
    ),
    "DroneController": (
        "发生碰撞: 对象=",
        "穿透深度=",
    ),
}


def is_user_runtime_log_mode() -> bool:
    mode = os.environ.get("AIRSIM_RUNTIME_LOG_MODE", "detail").strip().lower() or "detail"
    return mode == "user"


def should_emit_detail_feedback() -> bool:
    return not is_user_runtime_log_mode()


class _UserModeConsoleFilter(logging.Filter):
    def __init__(self, mode: str):
        super().__init__()
        self.mode = str(mode or "detail").strip().lower()

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()

        if record.levelno >= logging.ERROR:
            return True
        if self.mode == "user" and record.levelno == logging.WARNING:
            for snippet in _CONSOLE_WARNING_SUPPRESS_RULES.get(record.name, ()):
                if snippet in message:
                    return False
            return True
        if self.mode != "user":
            return True

        for prefix in _CONSOLE_INFO_RULES.get(record.name, ()):
            if prefix in message:
                return True
        return False


def _resolve_runtime_log_dir() -> Path:
    raw_dir = os.environ.get("AIRSIM_RUNTIME_LOG_DIR", "").strip()
    if raw_dir:
        return Path(raw_dir)
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "analysis_results" / "runtime_logs"


def configure_runtime_logging(
    *,
    force_reconfigure: bool = False,
    console_stream: Any = None,
) -> dict[str, str]:
    root_logger = logging.getLogger()
    if force_reconfigure:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    if getattr(root_logger, "_airsim_runtime_logging_configured", False) and not force_reconfigure:
        return {
            "mode": getattr(root_logger, "_airsim_runtime_log_mode", "detail"),
            "log_file": getattr(root_logger, "_airsim_runtime_log_file", ""),
        }

    mode = os.environ.get("AIRSIM_RUNTIME_LOG_MODE", "detail").strip().lower() or "detail"
    log_dir = _resolve_runtime_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"runtime_session_{timestamp}_{os.getpid()}.log"

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    setattr(file_handler, "_airsim_runtime_handler", True)

    console_handler = logging.StreamHandler(console_stream or sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(_UserModeConsoleFilter(mode))
    setattr(console_handler, "_airsim_runtime_handler", True)

    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.propagate = False
    root_logger._airsim_runtime_logging_configured = True  # type: ignore[attr-defined]
    root_logger._airsim_runtime_log_mode = mode  # type: ignore[attr-defined]
    root_logger._airsim_runtime_log_file = str(log_file)  # type: ignore[attr-defined]
    return {"mode": mode, "log_file": str(log_file)}
