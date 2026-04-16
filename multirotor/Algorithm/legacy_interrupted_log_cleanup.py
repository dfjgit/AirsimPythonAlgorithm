from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TRAINING_PREFIXES = (
    "ddpg_training_",
    "dqn_training_",
    "training_data_",
    "apf_training_",
)


def _scan_name_candidates(training_csv: Path) -> list[str]:
    stem = training_csv.stem
    for prefix in TRAINING_PREFIXES:
        if stem.startswith(prefix):
            suffix = stem[len(prefix) :]
            return [f"scan_data_{suffix}.csv"]
    return [f"scan_data_{stem}.csv"]


def resolve_scan_csv(training_csv: Path, *, scan_csv: Path | None = None, scan_dir: Path | None = None) -> Path | None:
    if scan_csv is not None:
        return Path(scan_csv)

    candidate_names = _scan_name_candidates(training_csv)
    search_roots = [training_csv.parent]
    if scan_dir is not None:
        search_roots.append(Path(scan_dir))

    for root in search_roots:
        for candidate_name in candidate_names:
            candidate = root / candidate_name
            if candidate.exists():
                return candidate
    return None


def _load_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not Path(path).exists() or Path(path).stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def _has_terminal_scan_row(scan_df: pd.DataFrame, episode: int) -> bool:
    if scan_df.empty or "episode" not in scan_df.columns:
        return False
    working = scan_df.copy()
    working["episode"] = pd.to_numeric(working["episode"], errors="coerce")
    working = working.dropna(subset=["episode"])
    if working.empty:
        return False
    working["episode"] = working["episode"].astype(int)
    if "reset_reason" not in working.columns:
        return False
    reason = working["reset_reason"].fillna("").astype(str).str.strip()
    return bool(((working["episode"] == int(episode)) & (reason != "")).any())


def analyze_training_csv(training_csv: Path, *, scan_csv: Path | None = None, scan_dir: Path | None = None) -> dict:
    training_csv = Path(training_csv)
    resolved_scan_csv = resolve_scan_csv(training_csv, scan_csv=scan_csv, scan_dir=scan_dir)
    training_df = _load_csv(training_csv)
    if training_df.empty:
        return {
            "training_csv": training_csv,
            "scan_csv": resolved_scan_csv,
            "status": "empty",
            "candidate_episode": None,
            "removed_rows": 0,
            "retained_rows": 0,
        }

    if "episode_complete" in training_df.columns:
        return {
            "training_csv": training_csv,
            "scan_csv": resolved_scan_csv,
            "status": "already_migrated",
            "candidate_episode": None,
            "removed_rows": 0,
            "retained_rows": len(training_df),
        }

    last_row = training_df.iloc[-1]
    last_episode = pd.to_numeric(pd.Series([last_row.get("episode")]), errors="coerce").iloc[0]
    if pd.isna(last_episode):
        return {
            "training_csv": training_csv,
            "scan_csv": resolved_scan_csv,
            "status": "invalid_episode",
            "candidate_episode": None,
            "removed_rows": 0,
            "retained_rows": len(training_df),
        }
    last_episode = int(last_episode)

    if resolved_scan_csv is None:
        return {
            "training_csv": training_csv,
            "scan_csv": None,
            "status": "missing_scan",
            "candidate_episode": None,
            "removed_rows": 0,
            "retained_rows": len(training_df),
        }

    scan_df = _load_csv(resolved_scan_csv)
    if _has_terminal_scan_row(scan_df, last_episode):
        return {
            "training_csv": training_csv,
            "scan_csv": resolved_scan_csv,
            "status": "complete",
            "candidate_episode": None,
            "removed_rows": 0,
            "retained_rows": len(training_df),
        }

    return {
        "training_csv": training_csv,
        "scan_csv": resolved_scan_csv,
        "status": "interrupted_last_episode",
        "candidate_episode": last_episode,
        "removed_rows": 1,
        "retained_rows": max(len(training_df) - 1, 0),
    }


def apply_cleanup(training_csv: Path, *, scan_csv: Path | None = None, scan_dir: Path | None = None) -> dict:
    summary = analyze_training_csv(training_csv, scan_csv=scan_csv, scan_dir=scan_dir)
    training_csv = Path(training_csv)
    if summary["status"] != "interrupted_last_episode":
        summary["applied"] = False
        return summary

    training_df = _load_csv(training_csv)
    retained = training_df.iloc[:-1].copy()
    removed = training_df.iloc[[-1]].copy()
    retained["episode_complete"] = 1
    removed["episode_complete"] = 0
    if "reset_reason" in removed.columns:
        reset_reason = removed["reset_reason"].fillna("").astype(str).str.strip()
        removed.loc[reset_reason.eq(""), "reset_reason"] = "interrupted_migrated"

    backup_csv = training_csv.with_suffix(training_csv.suffix + ".bak")
    training_csv.replace(backup_csv)
    retained.to_csv(training_csv, index=False, encoding="utf-8-sig")

    interrupted_dir = training_csv.parent / "interrupted_runs"
    interrupted_dir.mkdir(parents=True, exist_ok=True)
    quarantined_csv = interrupted_dir / training_csv.name
    removed.to_csv(quarantined_csv, index=False, encoding="utf-8-sig")

    summary.update(
        {
            "applied": True,
            "backup_csv": backup_csv,
            "quarantined_csv": quarantined_csv,
        }
    )
    return summary


def _collect_training_csvs(*, csv_path: Path | None = None, directory: Path | None = None) -> list[Path]:
    if csv_path is not None:
        return [Path(csv_path)]
    if directory is None:
        return []
    files = []
    for path in sorted(Path(directory).glob("*.csv")):
        if path.name.startswith("scan_data_"):
            continue
        if any(path.name.startswith(prefix) for prefix in TRAINING_PREFIXES):
            files.append(path)
    return files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Conservatively clean legacy interrupted training logs.")
    parser.add_argument("--csv", type=Path, help="Single training CSV to inspect.")
    parser.add_argument("--dir", type=Path, help="Directory containing legacy training CSV files.")
    parser.add_argument("--scan-csv", type=Path, help="Explicit paired scan_data CSV.")
    parser.add_argument("--scan-dir", type=Path, help="Directory to search for paired scan_data CSV files.")
    parser.add_argument("--apply", action="store_true", help="Rewrite training CSV in place and quarantine the interrupted last row.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    training_csvs = _collect_training_csvs(csv_path=args.csv, directory=args.dir)
    if not training_csvs:
        parser.error("Provide --csv or --dir with at least one training CSV.")

    results = []
    for training_csv in training_csvs:
        if args.apply:
            result = apply_cleanup(training_csv, scan_csv=args.scan_csv, scan_dir=args.scan_dir)
        else:
            result = analyze_training_csv(training_csv, scan_csv=args.scan_csv, scan_dir=args.scan_dir)
        printable = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in result.items()
        }
        results.append(printable)

    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
