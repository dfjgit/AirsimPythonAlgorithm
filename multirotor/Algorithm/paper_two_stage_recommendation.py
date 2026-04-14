from __future__ import annotations

from pathlib import Path

import pandas as pd

CONTINUE_DECISION = "寤鸿缁х画瀹為淇"
CAUTION_DECISION = "寤鸿缁х画瀵嗗垏瑙傚療"
STOP_DECISION = "褰撳墠鍙粨鏉熷弻闃舵瀹為獙"

REGRESSION_THRESHOLD = -0.01
REGRESSION_REASON = "real-weighted refinement regression observed"
HIGH_SUCCESS_REASON = "real-weighted refinement success is already high"
PLATEAU_REASON = "real-weighted refinement has reached a small-gain plateau"


def recommend_real_weighted_continue(summary_csv: str | Path) -> dict[str, list[str] | str]:  # type: ignore[type-arg]
    df = pd.read_csv(summary_csv, encoding="utf-8-sig")
    sim_row = df[df["phase"] == "sim_pretrain"].iloc[0]
    refine_row = df[df["phase"] == "real_weighted_refine"].iloc[0]
    baseline_efficiency = float(sim_row["avg_scan_efficiency"])
    refine_efficiency = float(refine_row["avg_scan_efficiency"])
    efficiency_gain = (
        (refine_efficiency - baseline_efficiency)
        / max(baseline_efficiency, 1e-6)
    )
    success_gain = float(refine_row["success_rate"]) - float(sim_row["success_rate"])
    refine_success_rate = float(refine_row["success_rate"])

    reasons = [
        f"efficiency gain={efficiency_gain:.2%}",
        f"success gain={success_gain:.2%}",
    ]

    if efficiency_gain < REGRESSION_THRESHOLD or success_gain < REGRESSION_THRESHOLD:
        return {
            "decision": CAUTION_DECISION,
            "reasons": reasons + [REGRESSION_REASON],
        }
    if efficiency_gain > 0.10 or success_gain > 0.10:
        return {"decision": CONTINUE_DECISION, "reasons": reasons}
    if refine_success_rate >= 0.9 and efficiency_gain <= 0.03:
        return {
            "decision": STOP_DECISION,
            "reasons": reasons + [HIGH_SUCCESS_REASON],
        }
    if efficiency_gain > 0.03 or success_gain > 0.03:
        return {"decision": CAUTION_DECISION, "reasons": reasons}

    return {
        "decision": STOP_DECISION,
        "reasons": reasons + [PLATEAU_REASON],
    }
