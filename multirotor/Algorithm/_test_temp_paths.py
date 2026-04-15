from __future__ import annotations

import tempfile
import uuid
from pathlib import Path


def _normalize_prefix(prefix: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(prefix).strip())
    cleaned = cleaned.strip("_") or "tmp"
    return cleaned[:24]


def test_temp_root() -> Path:
    root = Path(tempfile.gettempdir()) / "apa_t"
    root.mkdir(parents=True, exist_ok=True)
    return root


def make_temp_dir(prefix: str) -> Path:
    path = test_temp_root() / _normalize_prefix(prefix) / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


def suite_temp_root(prefix: str) -> Path:
    path = test_temp_root() / _normalize_prefix(prefix)
    path.mkdir(parents=True, exist_ok=True)
    return path
