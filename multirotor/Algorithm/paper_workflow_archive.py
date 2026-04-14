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


def collect_ddpg_stage_outputs(project_root: Path, *, stage_name: str) -> dict:
    models_dir = project_root / "multirotor" / "DDPG_Weight" / "models"
    logs_dir = project_root / "multirotor" / "DDPG_Weight" / "airsim_training_logs"
    finals = sorted(models_dir.glob("weight_predictor_airsim*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    bests = sorted(models_dir.glob("best*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    metas = sorted(models_dir.glob("*.stage_meta.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    training_logs = sorted(logs_dir.glob(f"*{stage_name}*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {
        "final_model": finals[0] if finals else None,
        "best_model": bests[0] if bests else None,
        "stage_meta": metas[0] if metas else None,
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
