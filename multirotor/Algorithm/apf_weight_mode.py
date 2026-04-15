from __future__ import annotations

import random
from typing import Dict, Optional


APF_WEIGHT_KEYS = (
    "repulsionCoefficient",
    "entropyCoefficient",
    "distanceCoefficient",
    "leaderRangeCoefficient",
    "directionRetentionCoefficient",
)

VALID_APF_WEIGHT_MODES = {"fixed", "random_episode", "learned"}


def resolve_apf_weight_mode(
    control_mode: str,
    use_learned_weights: bool = False,
    explicit_mode: Optional[str] = None,
) -> str:
    """Resolve the APF weight mode while keeping DQN execution unchanged."""

    normalized_control_mode = str(control_mode or "apf").strip().lower()
    if normalized_control_mode != "apf":
        return "fixed"

    if explicit_mode:
        normalized_mode = str(explicit_mode).strip().lower()
        if normalized_mode not in VALID_APF_WEIGHT_MODES:
            raise ValueError(
                f"Unsupported apf_weight_mode={explicit_mode!r}; "
                f"expected one of {sorted(VALID_APF_WEIGHT_MODES)}"
            )
        return normalized_mode

    return "learned" if bool(use_learned_weights) else "fixed"


def sample_random_episode_weights(
    seed: Optional[int],
    episode_index: int,
    weight_min: float,
    weight_max: float,
) -> Dict[str, float]:
    """Sample a stable APF weight profile for one episode."""

    if weight_max < weight_min:
        raise ValueError("weight_max must be >= weight_min")

    rng = random.Random(int(seed or 0) + int(episode_index))
    return {
        key: rng.uniform(float(weight_min), float(weight_max))
        for key in APF_WEIGHT_KEYS
    }
