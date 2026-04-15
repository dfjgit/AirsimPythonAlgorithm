from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


def generate_algorithm_specific_reports(
    *,
    eval_csv_paths: Iterable[str | Path],
    output_root: str | Path,
) -> Dict[str, Dict[str, Path]]:
    frames = []
    for csv_path in eval_csv_paths:
        frame = pd.read_csv(csv_path, encoding="utf-8-sig")
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise ValueError("No evaluation CSV data available for algorithm-specific analysis")

    merged = pd.concat(frames, ignore_index=True, sort=False)
    if "algorithm_type" not in merged.columns:
        raise ValueError("evaluation CSV must include algorithm_type")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    generated: Dict[str, Dict[str, Path]] = {}

    for algorithm_type, frame in merged.groupby("algorithm_type", dropna=False):
        algo_dir = output_root / str(algorithm_type)
        algo_dir.mkdir(parents=True, exist_ok=True)

        episodes_csv = algo_dir / "eval_episodes.csv"
        frame.to_csv(episodes_csv, index=False, encoding="utf-8-sig")

        numeric_frame = frame.select_dtypes(include=["number"]).copy()
        if numeric_frame.empty:
            summary = pd.DataFrame({"metric": [], "count": [], "mean": [], "median": []})
        else:
            summary = (
                numeric_frame.agg(["count", "mean", "median"])
                .transpose()
                .reset_index()
                .rename(columns={"index": "metric"})
            )
        summary_csv = algo_dir / "summary.csv"
        summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

        generated[str(algorithm_type)] = {
            "episodes_csv": episodes_csv,
            "summary_csv": summary_csv,
        }

    return generated
