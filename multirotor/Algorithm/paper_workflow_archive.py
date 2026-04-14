from __future__ import annotations

import json
import re
import shutil
from pathlib import Path


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def archive_directory_tree(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copytree(src, dst, dirs_exist_ok=True)


def _sorted_candidates(path: Path, pattern: str) -> list[Path]:
    if not path.exists():
        return []
    return sorted(path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)


def _candidate_project_roots(project_root: Path) -> list[Path]:
    roots = [Path(project_root)]
    parts = list(project_root.parts)
    if ".worktrees" in parts:
        primary_root = Path(*parts[: parts.index(".worktrees")])
        if primary_root not in roots:
            roots.append(primary_root)
    return roots


def _extract_stage_token(log_path: Path, stage_name: str) -> str | None:
    stem = log_path.stem
    index = stem.find(stage_name)
    if index == -1:
        return None
    token = stem[index + len(stage_name) :]
    if token.startswith("_"):
        token = token[1:]
    return token or None


def _choose_by_token(candidates: list[Path], token: str | None) -> Path | None:
    if not candidates or not token:
        return None
    for candidate in candidates:
        if token in candidate.name:
            return candidate
    return None


def _determine_stage_token(logs: list[Path], stage_name: str, reference_files: list[Path]) -> str | None:
    for log_path in logs:
        token = _extract_stage_token(log_path, stage_name)
        if token and any(token in reference.name for reference in reference_files):
            return token
    if logs:
        return _extract_stage_token(logs[0], stage_name)
    return None


def collect_ddpg_stage_outputs(project_root: Path, *, stage_name: str) -> dict:
    models_dir = project_root / "multirotor" / "DDPG_Weight" / "models"
    logs_dir = project_root / "multirotor" / "DDPG_Weight" / "airsim_training_logs"
    finals = _sorted_candidates(models_dir, "weight_predictor_airsim*.zip")
    bests = _sorted_candidates(models_dir, "best*.zip")
    metas = _sorted_candidates(models_dir, "*.stage_meta.json")
    training_logs = _sorted_candidates(logs_dir, f"*{stage_name}*.csv")
    stage_token = _determine_stage_token(training_logs, stage_name, finals + metas + bests)
    if stage_token:
        filtered_logs = [
            log for log in training_logs if stage_token in log.name
        ]
        training_logs = filtered_logs or training_logs
    return {
        "final_model": _choose_by_token(finals, stage_token),
        "best_model": _choose_by_token(bests, stage_token),
        "stage_meta": _choose_by_token(metas, stage_token),
        "training_logs": training_logs,
    }


def _extract_stage_index(stage_name: str) -> int | None:
    match = re.search(r"stage(\d+)", str(stage_name or ""), re.IGNORECASE)
    if not match:
        return None
    return max(int(match.group(1)), 1)


def _load_stage_meta(meta_path: Path | None) -> dict:
    if meta_path is None or not meta_path.exists():
        return {}
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _dqn_model_candidates(project_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for root in _candidate_project_roots(project_root):
        candidates.extend(
            [
                root / "multirotor" / "DQN_Movement" / "models" / "movement_dqn_airsim_final.zip",
                root / "multirotor" / "DQN_Movement" / "models" / "movement_dqn_final.zip",
                root / "multirotor" / "DQN_Movement" / "scripts" / "models" / "movement_dqn_airsim_final.zip",
                root / "multirotor" / "DQN_Movement" / "scripts" / "models" / "movement_dqn_final.zip",
            ]
        )
    return candidates


def _build_stage_file_token(stage_meta: dict) -> str | None:
    experiment_id = str(stage_meta.get("experiment_id", "") or "").strip()
    stage_index = stage_meta.get("stage_index")
    if not experiment_id:
        return None
    try:
        normalized_stage_index = max(int(stage_index or 1), 1)
    except (TypeError, ValueError):
        normalized_stage_index = 1
    return f"{experiment_id}_stage{normalized_stage_index:02d}"


def _stage_meta_matches(stage_meta: dict, *, stage_name: str) -> bool:
    if not stage_meta:
        return False
    requested_index = _extract_stage_index(stage_name)
    meta_stage_name = str(stage_meta.get("stage_name", "") or "")
    if meta_stage_name.startswith(stage_name):
        return True
    if requested_index is None:
        return False
    try:
        return int(stage_meta.get("stage_index", 0) or 0) == requested_index
    except (TypeError, ValueError):
        return False


def collect_dqn_stage_outputs(project_root: Path, *, stage_name: str) -> dict:
    final_model = next((candidate for candidate in _dqn_model_candidates(project_root) if candidate.exists()), None)
    stage_meta = None
    stage_meta_payload: dict = {}
    if final_model is not None:
        candidate_meta = final_model.with_suffix("").with_suffix(".stage_meta.json")
        if candidate_meta.exists():
            payload = _load_stage_meta(candidate_meta)
            if payload and not _stage_meta_matches(payload, stage_name=stage_name):
                final_model = None
            else:
                stage_meta = candidate_meta
                stage_meta_payload = payload

    training_logs: list[Path] = []
    for root in _candidate_project_roots(project_root):
        logs_dir = root / "multirotor" / "DQN_Movement" / "logs" / "dqn_scan_data"
        training_logs.extend(_sorted_candidates(logs_dir, f"*{stage_name}*.csv"))

    stage_token = _build_stage_file_token(stage_meta_payload)
    if stage_token:
        filtered_logs = [log_path for log_path in training_logs if stage_token in log_path.name]
        training_logs = filtered_logs or training_logs

    deduped_logs: list[Path] = []
    seen_paths: set[str] = set()
    for log_path in training_logs:
        resolved = str(log_path.resolve())
        if resolved in seen_paths:
            continue
        deduped_logs.append(log_path)
        seen_paths.add(resolved)

    return {
        "final_model": final_model,
        "best_model": None,
        "stage_meta": stage_meta,
        "training_logs": deduped_logs,
    }


def archive_comparison_stage_outputs(project_root: Path, exp_root: Path, *, algorithm: str, stage_bucket: str) -> dict:
    target_root = exp_root / "artifacts" / stage_bucket / algorithm
    (target_root / "models").mkdir(parents=True, exist_ok=True)
    (target_root / "logs").mkdir(parents=True, exist_ok=True)
    if algorithm == "ddpg_apf":
        outputs = collect_ddpg_stage_outputs(project_root, stage_name=stage_bucket)
    elif algorithm == "pure_dqn":
        outputs = collect_dqn_stage_outputs(project_root, stage_name=stage_bucket)
    else:
        raise ValueError(f"Unsupported comparison workflow algorithm: {algorithm}")
    if outputs["final_model"]:
        _copy_if_exists(outputs["final_model"], target_root / "models" / outputs["final_model"].name)
    if outputs["best_model"]:
        _copy_if_exists(outputs["best_model"], target_root / "models" / outputs["best_model"].name)
    if outputs["stage_meta"]:
        _copy_if_exists(outputs["stage_meta"], target_root / "models" / outputs["stage_meta"].name)
    for log_path in outputs["training_logs"]:
        _copy_if_exists(log_path, target_root / "logs" / log_path.name)
    return outputs
