from __future__ import annotations

import argparse
import json

from benchmark_registry import (
    GLOBAL_BENCHMARK_PROFILE,
    load_benchmark_registry,
    recommend_family_memberships,
)


def build_algorithm_template(
    *,
    algorithm_type: str,
    control_mode: str,
    is_trainable: bool,
) -> dict:
    recommended = recommend_family_memberships(
        control_mode=control_mode,
        is_trainable=is_trainable,
    )
    return {
        "algorithm_type": algorithm_type,
        "display_name": algorithm_type,
        "primary_family": recommended[0] if recommended else "",
        "family_memberships": recommended,
        "comparison_profiles": [GLOBAL_BENCHMARK_PROFILE],
        "is_trainable": bool(is_trainable),
        "control_mode": control_mode,
        "enabled": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate or scaffold benchmark registry entries.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate the registry file.")
    validate_parser.add_argument("--registry", type=str, default=None)

    recommend_parser = subparsers.add_parser("recommend", help="Recommend family memberships.")
    recommend_parser.add_argument("--algorithm-type", type=str, required=True)
    recommend_parser.add_argument("--control-mode", type=str, required=True)
    recommend_parser.add_argument("--trainable", action="store_true")

    scaffold_parser = subparsers.add_parser("scaffold", help="Print a registry entry template.")
    scaffold_parser.add_argument("--algorithm-type", type=str, required=True)
    scaffold_parser.add_argument("--control-mode", type=str, required=True)
    scaffold_parser.add_argument("--trainable", action="store_true")

    args = parser.parse_args()

    if args.command == "validate":
        registry = load_benchmark_registry(args.registry)
        print(
            f"OK: registry_version={registry.registry_version}, "
            f"families={len(registry.families)}, algorithms={len(registry.algorithms)}"
        )
        return

    if args.command == "recommend":
        payload = {
            "algorithm_type": args.algorithm_type,
            "recommended_family_memberships": recommend_family_memberships(
                control_mode=args.control_mode,
                is_trainable=bool(args.trainable),
            ),
            "comparison_profiles": [GLOBAL_BENCHMARK_PROFILE],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "scaffold":
        print(
            json.dumps(
                build_algorithm_template(
                    algorithm_type=args.algorithm_type,
                    control_mode=args.control_mode,
                    is_trainable=bool(args.trainable),
                ),
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
