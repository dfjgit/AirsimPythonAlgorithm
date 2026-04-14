from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

try:
    from .collision_analysis import collision_termination_rate_percent
except ImportError:
    from collision_analysis import collision_termination_rate_percent


def _manual_review(reasons: Iterable[str] | str) -> dict:
    reason_list = list(reasons) if isinstance(reasons, Iterable) and not isinstance(reasons, str) else [reasons]
    return {"decision": "可选续训", "reasons": reason_list}


def recommend_comparison_stage02(
    training_csv,
    benchmark_csv,
    *,
    algorithm_type: str,
    recent_window: int = 50,
    min_recent_window: int = 20,
) -> dict:
    training_df = pd.read_csv(training_csv, encoding="utf-8-sig")
    benchmark_df = pd.read_csv(benchmark_csv, encoding="utf-8-sig")
    recent_df = training_df.tail(recent_window)
    if len(recent_df) < min_recent_window:
        return _manual_review("最近窗口样本不足，建议人工确认")

    if "success_flag" not in recent_df.columns or "scan_efficiency" not in recent_df.columns:
        return _manual_review("训练数据缺少 success_flag 或 scan_efficiency，无法判断")

    success_series = pd.to_numeric(recent_df["success_flag"], errors="coerce").dropna()
    eff_series = pd.to_numeric(recent_df["scan_efficiency"], errors="coerce").dropna()
    if success_series.empty or eff_series.empty:
        return _manual_review("训练数据 success_flag/scan_efficiency 全部缺失")

    recent_success = float(success_series.mean())
    recent_eff = float(eff_series.mean())

    collision_columns = {"collision_rate", "reset_reason", "collision_count_final", "collision_count"}
    if not collision_columns & set(training_df.columns):
        return _manual_review(
            "训练数据缺少 collision_rate/reset_reason/collision_count，无法评估安全性"
        )

    if "collision_rate" in recent_df.columns and recent_df["collision_rate"].dropna().any():
        collision_series = pd.to_numeric(recent_df["collision_rate"], errors="coerce").dropna()
    else:
        collision_series = collision_termination_rate_percent(recent_df)
    recent_collision = float(collision_series.mean()) / 100.0 if not collision_series.empty else 0.0

    previous_df = training_df.iloc[
        max(0, len(training_df) - 2 * len(recent_df)) : len(training_df) - len(recent_df)
    ]
    previous_eff = (
        float(pd.to_numeric(previous_df["scan_efficiency"], errors="coerce").mean())
        if not previous_df.empty
        else recent_eff
    )
    eff_gain = 0.0 if previous_eff == 0 else (recent_eff - previous_eff) / abs(previous_eff)

    algo_benchmark = benchmark_df[benchmark_df["algorithm_type"] == algorithm_type]
    fixed_benchmark = benchmark_df[benchmark_df["algorithm_type"] == "fixed_apf"]
    benchmark_ready = not algo_benchmark.empty and not fixed_benchmark.empty
    benchmark_scan_gap = None
    if benchmark_ready:
        algo_scan = pd.to_numeric(algo_benchmark["final_global_scan_ratio"], errors="coerce").mean()
        fixed_scan = pd.to_numeric(fixed_benchmark["final_global_scan_ratio"], errors="coerce").mean()
        if pd.isna(algo_scan) or pd.isna(fixed_scan):
            benchmark_ready = False
        else:
            benchmark_scan_gap = float(fixed_scan - algo_scan)

    reasons: list[str] = []
    if recent_success < 0.8:
        reasons.append(f"recent success={recent_success:.2f} below 0.80")
    if recent_collision > 0.2:
        reasons.append(
            f"recent collision termination ratio={recent_collision:.2f} above 0.20"
        )
    if eff_gain > 0.05:
        reasons.append(f"scan efficiency still improving by {eff_gain:.2%}")
    if benchmark_scan_gap is not None and benchmark_scan_gap > 0.0:
        reasons.append(
            f"frozen benchmark scan ratio trails fixed_apf by {benchmark_scan_gap:.2f}"
        )

    if reasons:
        return {"decision": "建议续训", "reasons": reasons}
    if not benchmark_ready:
        return _manual_review("封存 benchmark 数据不全，无法确认是否完成 stage01")

    return {
        "decision": "当前可结束 stage01",
        "reasons": [
            "recent window is stable and benchmark no longer trails fixed_apf"
        ],
    }
