from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from benchmark_registry import BenchmarkRegistry, resolve_algorithm_registration


def _localized_text(zh_text: str, en_text: str) -> str:
    return zh_text if os.environ.get("AIRSIM_UI_LANG", "").lower() == "zh" else en_text


def configure_plot_fonts():
    fonts = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]
    plt.rcParams["font.sans-serif"] = fonts
    plt.rcParams["axes.unicode_minus"] = False
    return fonts


def _parse_multi_value(raw_value) -> List[str]:
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    if raw_value is None:
        return []
    text = str(raw_value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in text.split(";") if item.strip()]


def _join_multi_value(values: Iterable[str]) -> str:
    seen = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.append(text)
    return ";".join(seen)


def _normalize_eval_frame(frame: pd.DataFrame, registry: BenchmarkRegistry) -> pd.DataFrame:
    normalized = frame.copy()
    if "algorithm_type" not in normalized.columns:
        raise ValueError("evaluation frame must include algorithm_type")

    primary_families: List[str] = []
    family_memberships: List[str] = []
    comparison_profiles: List[str] = []
    is_trainable_values: List[bool] = []

    for _, row in normalized.iterrows():
        resolved = resolve_algorithm_registration(
            str(row.get("algorithm_type", "")).strip(),
            registry,
            control_mode=str(row.get("control_mode", "")).strip().lower(),
            apf_weight_mode=str(row.get("apf_weight_mode", "")).strip().lower(),
            is_trainable=bool(row.get("is_trainable", False)),
        )
        current_memberships = _parse_multi_value(row.get("family_memberships", ""))
        current_profiles = _parse_multi_value(row.get("comparison_profiles", ""))
        primary_families.append(row.get("primary_family") or resolved.primary_family)
        family_memberships.append(
            _join_multi_value(current_memberships or resolved.family_memberships)
        )
        comparison_profiles.append(
            _join_multi_value(current_profiles or resolved.comparison_profiles)
        )
        is_trainable_values.append(bool(row.get("is_trainable", resolved.is_trainable)))

    normalized["primary_family"] = primary_families
    normalized["family_memberships"] = family_memberships
    normalized["comparison_profiles"] = comparison_profiles
    normalized["is_trainable"] = is_trainable_values
    return normalized


def _ensure_registry_memberships_exist(registry: BenchmarkRegistry, frame: pd.DataFrame) -> None:
    known_families = set(registry.families.keys())
    unknown_families = set()
    for raw_value in frame.get("family_memberships", []):
        for family_id in _parse_multi_value(raw_value):
            if family_id not in known_families:
                unknown_families.add(family_id)
    if unknown_families:
        raise ValueError(
            "Unknown family memberships referenced in evaluation data: "
            + ", ".join(sorted(unknown_families))
        )


def _write_bar_plot(summary: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.bar(summary["algorithm_type"], summary[metric], color="#457b9d")
    plt.title(title)
    plt.ylabel(_localized_text("指标值", metric))
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def generate_family_reports(
    *,
    eval_csv_paths: Iterable[str | Path],
    registry: BenchmarkRegistry,
    output_root: str | Path,
) -> Dict[str, Dict[str, Path]]:
    frames = []
    for csv_path in eval_csv_paths:
        frame = pd.read_csv(csv_path, encoding="utf-8-sig")
        if frame.empty:
            continue
        frames.append(_normalize_eval_frame(frame, registry))

    if not frames:
        raise ValueError("No evaluation CSV data available for family analysis")

    merged = pd.concat(frames, ignore_index=True, sort=False)
    _ensure_registry_memberships_exist(registry, merged)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    configure_plot_fonts()
    generated: Dict[str, Dict[str, Path]] = {}

    for family_id, family in registry.families.items():
        if not family.enabled:
            continue

        family_rows = merged[
            merged["family_memberships"].map(
                lambda raw_value: family_id in _parse_multi_value(raw_value)
            )
        ].copy()
        if family_rows.empty:
            continue

        family_dir = output_root / family_id
        family_dir.mkdir(parents=True, exist_ok=True)
        summary = (
            family_rows.groupby("algorithm_type", dropna=False)
            .agg(
                episodes=("episode", "count"),
                success_rate=("success_flag", "mean"),
                mean_final_global_scan_ratio=("final_global_scan_ratio", "mean"),
                mean_final_global_avg_entropy=("final_global_avg_entropy", "mean"),
                mean_scan_efficiency=("scan_efficiency", "mean"),
                mean_collision_count=("collision_count", "mean"),
            )
            .reset_index()
        )

        summary_csv = family_dir / "family_summary.csv"
        summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

        scan_ratio_plot = family_dir / "scan_ratio_bar.png"
        _write_bar_plot(
            summary,
            "mean_final_global_scan_ratio",
            scan_ratio_plot,
            _localized_text(f"{family.display_name} 扫描率", f"{family.display_name} Scan Ratio"),
        )

        success_rate_plot = family_dir / "success_rate_bar.png"
        _write_bar_plot(
            summary,
            "success_rate",
            success_rate_plot,
            _localized_text(f"{family.display_name} 成功率", f"{family.display_name} Success Rate"),
        )

        generated[family_id] = {
            "summary_csv": summary_csv,
            "scan_ratio_plot": scan_ratio_plot,
            "success_rate_plot": success_rate_plot,
        }

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description=_localized_text("根据评测 CSV 生成 family 对比分析。", "Generate family comparison reports from evaluation CSV files.")
    )
    parser.add_argument(
        "--eval-csv",
        action="append",
        required=True,
        help=_localized_text("评测 CSV 路径，可重复指定。", "Path to an evaluation CSV. Repeatable."),
    )
    parser.add_argument(
        "--registry",
        type=str,
        default=None,
        help=_localized_text(
            "可选 benchmark_registry.json 路径，默认使用 multirotor/benchmark_registry.json。",
            "Optional benchmark registry path. Defaults to multirotor/benchmark_registry.json.",
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help=_localized_text("family 对比分析输出目录。", "Output directory for family reports."),
    )
    args = parser.parse_args()

    from benchmark_registry import load_benchmark_registry

    registry = load_benchmark_registry(args.registry)
    generate_family_reports(
        eval_csv_paths=args.eval_csv,
        registry=registry,
        output_root=args.out,
    )


if __name__ == "__main__":
    main()
