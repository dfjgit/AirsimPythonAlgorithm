from __future__ import annotations

import pandas as pd

COLLISION_RESET_REASON_KEYWORDS = ("collision", "collid", "碰撞")


def is_collision_reset_reason(reason: str) -> bool:
    text = str(reason or "").strip().lower()
    if not text:
        return False
    return any(keyword in text for keyword in COLLISION_RESET_REASON_KEYWORDS)


def collision_termination_flags(
    frame: pd.DataFrame,
    *,
    reason_column: str = "reset_reason",
    fallback_columns: tuple[str, ...] = ("collision_count_final", "collision_count"),
) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)

    reset_reason = (
        frame[reason_column]
        if reason_column in frame.columns
        else pd.Series("", index=frame.index, dtype=object)
    )
    reset_reason = reset_reason.fillna("").astype(str).str.strip()
    collision_from_reason = reset_reason.map(is_collision_reset_reason)

    fallback_numeric = None
    for column in fallback_columns:
        if column in frame.columns:
            fallback_numeric = pd.to_numeric(frame[column], errors="coerce").fillna(0)
            break

    if fallback_numeric is None:
        fallback_numeric = pd.Series(0, index=frame.index, dtype=float)

    collision_from_fallback = reset_reason.eq("") & fallback_numeric.gt(0)
    return (collision_from_reason | collision_from_fallback).astype(float)


def collision_termination_rate_percent(frame: pd.DataFrame, **kwargs) -> pd.Series:
    return collision_termination_flags(frame, **kwargs) * 100.0
