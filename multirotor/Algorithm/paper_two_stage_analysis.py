from __future__ import annotations

from pathlib import Path

import pandas as pd


def _safe_mean(series: pd.Series) -> float:
    return pd.to_numeric(series, errors="coerce").mean()


def build_two_stage_summary(
    sim_training_csv: Path, refine_training_csv: Path, output_root: Path
) -> dict[str, Path]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    sim_df = pd.read_csv(sim_training_csv, encoding="utf-8-sig")
    refine_df = pd.read_csv(refine_training_csv, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            {
                "phase": "sim_pretrain",
                "episodes": len(sim_df),
                "avg_scan_efficiency": _safe_mean(sim_df["scan_efficiency"]),
                "success_rate": _safe_mean(sim_df["success_flag"]),
            },
            {
                "phase": "real_weighted_refine",
                "episodes": len(refine_df),
                "avg_scan_efficiency": _safe_mean(refine_df["scan_efficiency"]),
                "success_rate": _safe_mean(refine_df["success_flag"]),
            },
        ],
    )

    summary_csv = output_root / "two_stage_summary.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    summary_md = output_root / "two_stage_summary.md"
    summary_md.write_text(
        "# Two-Stage Summary\n\n" + summary.to_markdown(index=False),
        encoding="utf-8",
    )

    return {"summary_csv": summary_csv, "summary_md": summary_md}
