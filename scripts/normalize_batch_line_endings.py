from __future__ import annotations

import argparse
import sys
from pathlib import Path


TEXT_SUFFIXES = {".bat", ".cmd"}


def iter_batch_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            files.append(path)
            continue
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in TEXT_SUFFIXES:
                    files.append(child)
    return files


def normalize_crlf(data: bytes) -> bytes:
    unified = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return unified.replace(b"\n", b"\r\n")


def needs_normalization(data: bytes) -> bool:
    return data != normalize_crlf(data)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check or normalize CRLF line endings in Windows batch files."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Report files that still need CRLF normalization.")
    mode.add_argument("--write", action="store_true", help="Rewrite files in-place with CRLF line endings.")
    parser.add_argument("paths", nargs="+", help="Batch files or directories to inspect.")
    args = parser.parse_args()

    input_paths = [Path(path).resolve() for path in args.paths]
    batch_files = iter_batch_files(input_paths)
    if not batch_files:
        print("No batch files found.")
        return 1

    dirty: list[Path] = []
    for batch_file in batch_files:
        data = batch_file.read_bytes()
        if not needs_normalization(data):
            continue
        dirty.append(batch_file)
        if args.write:
            batch_file.write_bytes(normalize_crlf(data))
            print(f"normalized {batch_file}")
        else:
            print(f"needs CRLF normalization: {batch_file}")

    if dirty and not args.write:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
