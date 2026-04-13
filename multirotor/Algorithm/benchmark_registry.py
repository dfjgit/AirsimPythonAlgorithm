from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


GLOBAL_BENCHMARK_PROFILE = "global_benchmark"


@dataclass(frozen=True)
class FamilyDefinition:
    family_id: str
    display_name: str
    analysis_template: str
    description: str
    enabled: bool = True


@dataclass(frozen=True)
class AlgorithmRegistration:
    algorithm_type: str
    display_name: str
    primary_family: str
    family_memberships: List[str]
    comparison_profiles: List[str]
    is_trainable: bool
    control_mode: str
    enabled: bool = True


@dataclass(frozen=True)
class ResolvedAlgorithmRegistration:
    algorithm_type: str
    display_name: str
    primary_family: str
    family_memberships: List[str]
    comparison_profiles: List[str]
    is_trainable: bool
    control_mode: str
    enabled: bool
    registry_version: int
    recommended_family_memberships: List[str] = field(default_factory=list)
    is_fallback: bool = False


@dataclass(frozen=True)
class BenchmarkRegistry:
    registry_version: int
    families: Dict[str, FamilyDefinition]
    algorithms: Dict[str, AlgorithmRegistration]
    source_path: Path


def default_registry_path() -> Path:
    return Path(__file__).resolve().parent.parent / "benchmark_registry.json"


def _dedupe_keep_order(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def recommend_family_memberships(
    *,
    control_mode: str,
    apf_weight_mode: Optional[str] = None,
    is_trainable: Optional[bool] = None,
) -> List[str]:
    recommended: List[str] = []
    normalized_control_mode = str(control_mode or "").strip().lower()
    normalized_apf_mode = str(apf_weight_mode or "").strip().lower()

    if normalized_control_mode == "apf":
        recommended.append("apf_family")
    elif not normalized_control_mode and normalized_apf_mode in {
        "fixed",
        "random_episode",
        "learned",
    }:
        recommended.append("apf_family")
    if bool(is_trainable):
        recommended.append("learning_family")
    return _dedupe_keep_order(recommended)


def _parse_family(raw_family: dict) -> FamilyDefinition:
    return FamilyDefinition(
        family_id=str(raw_family["family_id"]).strip(),
        display_name=str(raw_family.get("display_name", raw_family["family_id"])).strip(),
        analysis_template=str(
            raw_family.get("analysis_template", "benchmark_common")
        ).strip(),
        description=str(raw_family.get("description", "")).strip(),
        enabled=bool(raw_family.get("enabled", True)),
    )


def _parse_algorithm(raw_algorithm: dict) -> AlgorithmRegistration:
    return AlgorithmRegistration(
        algorithm_type=str(raw_algorithm["algorithm_type"]).strip(),
        display_name=str(
            raw_algorithm.get("display_name", raw_algorithm["algorithm_type"])
        ).strip(),
        primary_family=str(raw_algorithm.get("primary_family", "")).strip(),
        family_memberships=_dedupe_keep_order(
            raw_algorithm.get("family_memberships", [])
        ),
        comparison_profiles=_dedupe_keep_order(
            raw_algorithm.get("comparison_profiles", [])
        ),
        is_trainable=bool(raw_algorithm.get("is_trainable", False)),
        control_mode=str(raw_algorithm.get("control_mode", "")).strip().lower(),
        enabled=bool(raw_algorithm.get("enabled", True)),
    )


def _validate_registry(registry: BenchmarkRegistry) -> None:
    for family_id, family in registry.families.items():
        if not family_id:
            raise ValueError("family_id cannot be empty")
        if not family.analysis_template:
            raise ValueError(f"family {family_id!r} must define analysis_template")

    for algorithm_type, algorithm in registry.algorithms.items():
        if not algorithm_type:
            raise ValueError("algorithm_type cannot be empty")
        if GLOBAL_BENCHMARK_PROFILE not in algorithm.comparison_profiles:
            raise ValueError(
                f"algorithm {algorithm_type!r} must include {GLOBAL_BENCHMARK_PROFILE!r}"
            )
        if algorithm.primary_family and algorithm.primary_family not in registry.families:
            raise ValueError(
                f"algorithm {algorithm_type!r} references unknown primary_family "
                f"{algorithm.primary_family!r}"
            )
        unknown_families = [
            family_id
            for family_id in algorithm.family_memberships
            if family_id not in registry.families
        ]
        if unknown_families:
            raise ValueError(
                f"algorithm {algorithm_type!r} references unknown families: "
                f"{', '.join(unknown_families)}"
            )
        if algorithm.primary_family and algorithm.primary_family not in algorithm.family_memberships:
            raise ValueError(
                f"algorithm {algorithm_type!r} primary_family must also appear in "
                "family_memberships"
            )


def load_benchmark_registry(path: Optional[Path | str] = None) -> BenchmarkRegistry:
    registry_path = Path(path) if path else default_registry_path()
    with registry_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    families = {
        family.family_id: family
        for family in (_parse_family(item) for item in payload.get("families", []))
    }
    algorithms = {
        algorithm.algorithm_type: algorithm
        for algorithm in (
            _parse_algorithm(item) for item in payload.get("algorithms", [])
        )
    }

    registry = BenchmarkRegistry(
        registry_version=int(payload.get("registry_version", 1) or 1),
        families=families,
        algorithms=algorithms,
        source_path=registry_path,
    )
    _validate_registry(registry)
    return registry


def resolve_algorithm_registration(
    algorithm_type: str,
    registry: BenchmarkRegistry,
    *,
    control_mode: str,
    apf_weight_mode: Optional[str] = None,
    is_trainable: Optional[bool] = None,
) -> ResolvedAlgorithmRegistration:
    normalized_algorithm_type = str(algorithm_type or "unknown").strip() or "unknown"
    registered = registry.algorithms.get(normalized_algorithm_type)
    if registered is not None:
        return ResolvedAlgorithmRegistration(
            algorithm_type=registered.algorithm_type,
            display_name=registered.display_name,
            primary_family=registered.primary_family,
            family_memberships=list(registered.family_memberships),
            comparison_profiles=list(registered.comparison_profiles),
            is_trainable=registered.is_trainable,
            control_mode=registered.control_mode,
            enabled=registered.enabled,
            registry_version=registry.registry_version,
            recommended_family_memberships=list(registered.family_memberships),
            is_fallback=False,
        )

    recommended_families = recommend_family_memberships(
        control_mode=control_mode,
        apf_weight_mode=apf_weight_mode,
        is_trainable=is_trainable,
    )
    return ResolvedAlgorithmRegistration(
        algorithm_type=normalized_algorithm_type,
        display_name=normalized_algorithm_type,
        primary_family="",
        family_memberships=[],
        comparison_profiles=[GLOBAL_BENCHMARK_PROFILE],
        is_trainable=bool(is_trainable),
        control_mode=str(control_mode or "").strip().lower(),
        enabled=True,
        registry_version=registry.registry_version,
        recommended_family_memberships=recommended_families,
        is_fallback=True,
    )
