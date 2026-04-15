from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path


_TOKEN_RE = re.compile(r"[^\w\-.]+")
_DEFAULT_WORKFLOW_DIR = "workflow"
_RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *[f"com{i}" for i in range(1, 10)],
    *[f"lpt{i}" for i in range(1, 10)],
}


def _sanitize_segment(value: str) -> str:
    cleaned = _TOKEN_RE.sub("_", value or "").strip("_.")
    if not cleaned or cleaned in {".", ".."}:
        return ""
    if cleaned.lower() in _RESERVED_NAMES:
        return ""
    return cleaned


def _sanitize_timestamp_token(value: str) -> str:
    sanitized = _sanitize_segment(value)
    if not sanitized:
        return ""
    return sanitized


def _parse_updated_at(value) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def create_experiment_root(*, base_root: Path, workflow_type: str, alias: str = "", now_token: str | None = None) -> Path:
    if now_token:
        timestamp = _sanitize_timestamp_token(now_token) or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    workflow_dir = _sanitize_segment(workflow_type) or _DEFAULT_WORKFLOW_DIR
    workflow_root = base_root / workflow_dir
    workflow_root.mkdir(parents=True, exist_ok=True)
    sanitized_alias = _sanitize_segment(alias) or workflow_dir
    base_name = f"{timestamp}_{sanitized_alias}"
    exp_root = workflow_root / base_name
    counter = 1
    while exp_root.exists():
        exp_root = workflow_root / f"{base_name}_{counter}"
        counter += 1
    exp_root.mkdir(parents=True, exist_ok=True)
    (exp_root / "artifacts").mkdir(parents=True, exist_ok=True)
    return exp_root


def initialize_workflow_state(exp_root: Path, *, workflow_type: str, alias: str = "") -> dict:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state = {
        "workflow_type": workflow_type,
        "experiment_id": exp_root.name,
        "alias": alias,
        "status": "pending",
        "current_phase": "",
        "steps": {},
        "artifacts": {},
        "recommendations": {},
        "checkpoint_manifest": {},
        "created_at": now,
    }
    return save_workflow_state(exp_root, state, updated_at=now)


def save_workflow_state(exp_root: Path, state: dict, *, updated_at: str | None = None) -> dict:
    payload = dict(state)
    payload["updated_at"] = updated_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    (exp_root / "workflow_state.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    state.update(payload)
    return state


def load_workflow_state(exp_root: Path) -> dict:
    return json.loads((exp_root / "workflow_state.json").read_text(encoding="utf-8"))


def list_resumable_experiments(base_root: Path, *, workflow_type: str) -> list[dict]:
    workflow_dir = _sanitize_segment(workflow_type) or _DEFAULT_WORKFLOW_DIR
    workflow_root = base_root / workflow_dir
    if not workflow_root.exists():
        return []
    result = []
    for state_path in workflow_root.glob("*/workflow_state.json"):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(state, dict):
            continue
        if state.get("status") == "completed":
            continue
        result.append({"experiment_root": state_path.parent, "state": state})
    def _updated_at_key(item: dict) -> tuple[int, datetime]:
        dt = _parse_updated_at(item["state"].get("updated_at"))
        if dt is None:
            return (0, datetime.min)
        return (1, dt)

    result.sort(key=_updated_at_key, reverse=True)
    return result
