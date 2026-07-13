"""Measured-only, raw-token cost/quality frontier utilities.

Each model is treated as an independent arm.  Missing outcomes remain missing:
there is no assumption that a model with a higher tier number passes, costs the
same number of tokens, or dominates another model.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any


def wilson_lower_bound(successes: float, total: int, z: float = 1.96) -> float | None:
    """Return a conservative 95% Wilson lower bound for a success rate."""
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = proportion + z * z / (2.0 * total)
    margin = z * math.sqrt(
        (proportion * (1.0 - proportion) + z * z / (4.0 * total)) / total
    )
    return max(0.0, (centre - margin) / denominator)


def measured_arm_statistics(
    records: Sequence[dict], tier: str, *, category: str | None = None
) -> dict[str, Any]:
    """Summarize an arm only where it was actually observed."""
    selected = [
        record
        for record in records
        if category is None or record.get("category") == category
    ]
    observations: list[dict] = []
    for record in selected:
        outcome = record.get("_measured_tier_outcomes", {}).get(tier)
        if isinstance(outcome, dict):
            observations.append(outcome)
    measured = len(observations)
    total = len(selected)
    successes = sum(float(outcome["success_rate"]) for outcome in observations)
    total_tokens = sum(float(outcome["mean_total_tokens"]) for outcome in observations)
    return {
        "tier": tier,
        "category": category,
        "measured_prompt_groups": measured,
        "total_prompt_groups": total,
        "coverage": measured / max(total, 1),
        "empirical_accuracy": successes / measured if measured else None,
        "accuracy_wilson_95_lower": wilson_lower_bound(successes, measured),
        "mean_raw_tokens": total_tokens / measured if measured else None,
        "observed_calls": sum(
            int(outcome.get("observations", 1)) for outcome in observations
        ),
    }


def select_category_policy(
    records: Sequence[dict],
    tiers: Sequence[str],
    *,
    quality_target: float,
) -> dict[str, Any]:
    """Pick the lowest-token fully measured arm per category.

    Feasibility uses empirical accuracy; the Wilson lower bound is reported so
    reviewers can see when the sample is too small to establish the target
    confidently.  An arm with incomplete category coverage is never selected.
    """
    if not 0.0 <= quality_target <= 1.0:
        raise ValueError("quality_target must be between 0 and 1")
    categories = sorted({str(record.get("category", "")) for record in records})
    policy: dict[str, str] = {}
    category_details: dict[str, Any] = {}
    for category in categories:
        candidates = [
            measured_arm_statistics(records, tier, category=category)
            for tier in tiers
        ]
        eligible = [
            candidate
            for candidate in candidates
            if candidate["coverage"] == 1.0
            and candidate["empirical_accuracy"] is not None
            and candidate["empirical_accuracy"] >= quality_target
        ]
        selected = min(
            eligible,
            key=lambda candidate: (candidate["mean_raw_tokens"], candidate["tier"]),
            default=None,
        )
        if selected is not None:
            policy[category] = selected["tier"]
        category_details[category] = {
            "selected_tier": selected["tier"] if selected else None,
            "selected_meets_wilson_95_target": bool(
                selected
                and selected["accuracy_wilson_95_lower"] is not None
                and selected["accuracy_wilson_95_lower"] >= quality_target
            ),
            "arms": candidates,
        }
    return {
        "quality_target": quality_target,
        "policy": policy,
        "complete": len(policy) == len(categories),
        "unresolved_categories": sorted(set(categories) - set(policy)),
        "category_details": category_details,
        "selection_and_evaluation_use_same_dataset": True,
    }


def evaluate_category_policy(
    records: Sequence[dict],
    policy: dict[str, str],
    simulate: Callable[..., dict],
) -> dict[str, Any]:
    """Evaluate a policy through the measured-only replay implementation."""
    missing = sorted(
        {
            str(record.get("category", ""))
            for record in records
            if str(record.get("category", "")) not in policy
        }
    )
    if missing:
        return {
            "evaluable": False,
            "reason": "no fully measured arm met the quality target",
            "unresolved_categories": missing,
        }
    metrics = simulate(records, lambda record: policy[str(record.get("category", ""))])
    return {"evaluable": True, **metrics}


def build_frontier_analysis(
    records: Sequence[dict],
    tiers: Sequence[str],
    simulate: Callable[..., dict],
    targets: Sequence[float] = (0.98, 0.99, 1.0),
) -> dict[str, Any]:
    """Build target policies and the full per-category independent-arm table."""
    policies: dict[str, Any] = {}
    for target in targets:
        selection = select_category_policy(records, tiers, quality_target=target)
        selection["evaluation"] = evaluate_category_policy(
            records, selection["policy"], simulate
        )
        policies[f"{target:.2f}"] = selection
    return {
        "objective": "minimize mean raw total_tokens subject to measured success",
        "independent_model_arms": True,
        "missing_outcomes_are_unknown": True,
        "confidence_bound": "Wilson 95% lower bound (reported, not imputed)",
        "overall_arm_statistics": {
            tier: measured_arm_statistics(records, tier) for tier in tiers
        },
        "target_policies": policies,
    }

