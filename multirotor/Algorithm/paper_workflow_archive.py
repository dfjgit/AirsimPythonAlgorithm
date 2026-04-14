from __future__ import annotations

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


def archive_comparison_stage_outputs(project_root: Path, exp_root: Path, *, algorithm: str, stage_bucket: str) -> dict:
    target_root = exp_root / "artifacts" / stage_bucket / algorithm
    (target_root / "models").mkdir(parents=True, exist_ok=True)
    (target_root / "logs").mkdir(parents=True, exist_ok=True)
    outputs = collect_ddpg_stage_outputs(project_root, stage_name=stage_bucket)
    if outputs["final_model"]:
        _copy_if_exists(outputs["final_model"], target_root / "models" / outputs["final_model"].name)
    if outputs["best_model"]:
        _copy_if_exists(outputs["best_model"], target_root / "models" / outputs["best_model"].name)
    if outputs["stage_meta"]:
        _copy_if_exists(outputs["stage_meta"], target_root / "models" / outputs["stage_meta"].name)
    for log_path in outputs["training_logs"]:
        _copy_if_exists(log_path, target_root / "logs" / log_path.name)
    return outputs
