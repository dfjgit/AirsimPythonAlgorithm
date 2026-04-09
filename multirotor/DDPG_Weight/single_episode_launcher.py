from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DefaultModelSelection:
    status: str
    model_path: Optional[str]


def normalize_model_path_input(raw_path: str) -> str:
    normalized = str(raw_path or "").strip().strip('"').strip("'")
    if normalized.lower().endswith(".zip"):
        normalized = normalized[:-4]
    return normalized


def resolve_default_model(models_dir: str | Path) -> DefaultModelSelection:
    models_path = Path(models_dir)
    online_candidates = sorted(
        models_path.glob("weight_predictor_crazyflie_online_*.zip"),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    if online_candidates:
        return DefaultModelSelection(
            status="online",
            model_path=str(online_candidates[0].with_suffix("")),
        )

    airsim_model = models_path / "weight_predictor_airsim.zip"
    if airsim_model.exists():
        return DefaultModelSelection(
            status="airsim",
            model_path=str(airsim_model.with_suffix("")),
        )

    return DefaultModelSelection(status="missing", model_path=None)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resolve default single-episode models")
    parser.add_argument("--models-dir", type=str, default=None)
    parser.add_argument("--emit-env", action="store_true")
    parser.add_argument("--normalize-model-path", type=str, default=None)
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    if args.normalize_model_path is not None:
        print(normalize_model_path_input(args.normalize_model_path))
        return 0

    if args.emit_env:
        if not args.models_dir:
            raise ValueError("--models-dir is required with --emit-env")
        result = resolve_default_model(args.models_dir)
        print(f"MODEL_STATUS={result.status}")
        print(f"MODEL_PATH={result.model_path or ''}")
        return 0

    raise ValueError("No operation specified")


if __name__ == "__main__":
    raise SystemExit(main())
