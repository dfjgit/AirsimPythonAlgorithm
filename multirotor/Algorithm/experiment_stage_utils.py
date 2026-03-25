"""Utilities for naming, tracking, and persisting multi-stage training runs."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _sanitize_token(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return text.strip("._-") or "unknown"


def stage_meta_sidecar_path(model_path: str) -> str:
    """Return the sidecar JSON path used to persist training lineage metadata."""
    if not model_path:
        return ""
    base = model_path[:-4] if model_path.endswith(".zip") else model_path
    return f"{base}.stage_meta.json"


def load_stage_meta_for_model(model_path: Optional[str]) -> Dict[str, Any]:
    """Load stage metadata from a model sidecar if it exists."""
    if not model_path:
        return {}

    sidecar = stage_meta_sidecar_path(model_path)
    if not sidecar or not os.path.exists(sidecar):
        return {}

    try:
        with open(sidecar, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def build_stage_meta(
    algorithm_tag: str,
    is_resume: bool = False,
    source_model_path: Optional[str] = None,
    experiment_id: Optional[str] = None,
    stage_name: Optional[str] = None,
    stage_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Create normalized stage metadata for from-scratch or resumed training."""
    previous_meta = load_stage_meta_for_model(source_model_path) if is_resume else {}
    source_model_name = ""
    if source_model_path:
        source_model_name = os.path.basename(source_model_path)
        if source_model_name.endswith(".zip"):
            source_model_name = source_model_name[:-4]

    previous_experiment_id = str(previous_meta.get("experiment_id", "")).strip()
    previous_stage_index = int(previous_meta.get("stage_index", 0) or 0)

    if experiment_id:
        normalized_experiment_id = _sanitize_token(experiment_id)
    elif previous_experiment_id:
        normalized_experiment_id = _sanitize_token(previous_experiment_id)
    elif is_resume and source_model_name:
        normalized_experiment_id = _sanitize_token(source_model_name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        normalized_experiment_id = _sanitize_token(f"{algorithm_tag}_{timestamp}")

    if stage_index is not None:
        normalized_stage_index = max(int(stage_index), 1)
    elif is_resume:
        normalized_stage_index = previous_stage_index + 1 if previous_stage_index > 0 else 2
    else:
        normalized_stage_index = 1

    if stage_name:
        normalized_stage_name = _sanitize_token(stage_name)
    else:
        suffix = "finetune" if is_resume else "from_scratch"
        normalized_stage_name = f"stage{normalized_stage_index:02d}_{suffix}"

    return {
        "experiment_id": normalized_experiment_id,
        "stage_name": normalized_stage_name,
        "stage_index": normalized_stage_index,
        "is_resume": bool(is_resume),
        "source_model": source_model_name,
    }


def build_stage_file_token(stage_meta: Dict[str, Any]) -> str:
    """Create a stable filename token for a training stage."""
    experiment_id = _sanitize_token(stage_meta.get("experiment_id", "exp"))
    stage_index = max(int(stage_meta.get("stage_index", 1) or 1), 1)
    return f"{experiment_id}_stage{stage_index:02d}"


def write_stage_meta_for_model(model_path: Optional[str], stage_meta: Dict[str, Any]) -> Optional[str]:
    """Persist training lineage metadata next to a saved model."""
    if not model_path:
        return None

    sidecar = stage_meta_sidecar_path(model_path)
    if not sidecar:
        return None

    payload = dict(stage_meta or {})
    payload["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return sidecar
