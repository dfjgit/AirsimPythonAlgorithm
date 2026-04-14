from __future__ import annotations

import pandas as pd


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
        return {"decision": "可选续训", "reasons": ["最近窗口样本不足，建议人工确认"]}

    recent_success = float(pd.to_numeric(recent_df["success_flag"], errors="coerce").mean())
    recent_collision = float(
        pd.to_numeric(recent_df["collision_rate"], errors="coerce").mean()
    ) / 100.0
    recent_eff = float(pd.to_numeric(recent_df["scan_efficiency"], errors="coerce").mean())
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
    benchmark_scan_gap = 0.0
    if not algo_benchmark.empty and not fixed_benchmark.empty:
        benchmark_scan_gap = float(
            pd.to_numeric(fixed_benchmark["final_global_scan_ratio"], errors="coerce").mean()
        ) - float(
            pd.to_numeric(algo_benchmark["final_global_scan_ratio"], errors="coerce").mean()
        )

    reasons = []
    if recent_success < 0.8:
        reasons.append(f"recent success={recent_success:.2f} below 0.80")
    if recent_collision > 0.2:
        reasons.append(
            f"recent collision termination ratio={recent_collision:.2f} above 0.20"
        )
    if eff_gain > 0.05:
        reasons.append(f"scan efficiency still improving by {eff_gain:.2%}")
    if benchmark_scan_gap > 0.0:
        reasons.append(
            f"frozen benchmark scan ratio trails fixed_apf by {benchmark_scan_gap:.2f}"
        )

    if reasons:
        return {"decision": "建议续训", "reasons": reasons}
    return {
        "decision": "当前可结束 stage01",
        "reasons": [
            "recent window is stable and benchmark no longer trails fixed_apf"
        ],
    }
