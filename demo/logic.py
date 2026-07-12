"""Pure decision logic used by the Streamlit binary-router demo."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from router.labels import EASY_CATEGORIES
from router.route_binary import binary_thresholds_from_env, tier_from_probability


@dataclass(frozen=True)
class BinaryDecision:
    tier: str
    probability: float | None
    threshold: float | None
    reason: str


def decide_binary(
    prompt: str,
    category: str | None,
    *,
    predict_proba: Callable[[str, str | None], float] | None = None,
    tau: float | None = None,
    ner_tau: float | None = None,
) -> BinaryDecision:
    """Apply the exact binary category/threshold policy used by the agent."""
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    if category in EASY_CATEGORIES:
        return BinaryDecision(
            tier="tier0",
            probability=None,
            threshold=None,
            reason=f"{category} is a cheap-default category rule.",
        )

    if tau is None:
        tau, env_ner_tau = binary_thresholds_from_env()
        if ner_tau is None:
            ner_tau = env_ner_tau
    elif ner_tau is None:
        ner_tau = tau

    if predict_proba is None:
        from router.infer_binary_router import (
            checkpoint_available,
            predict_cheap_ok_proba,
        )

        if not checkpoint_available():
            raise RuntimeError(
                "No binary checkpoint found. Run: "
                "python -m router.train_binary_router"
            )
        predict_proba = predict_cheap_ok_proba

    probability = float(predict_proba(prompt, category))
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Invalid cheap_ok probability: {probability}")

    threshold = ner_tau if category == "ner" else tau
    tier = tier_from_probability(probability, category, tau, ner_tau)
    comparison = ">=" if tier == "tier0" else "<"
    return BinaryDecision(
        tier=tier,
        probability=probability,
        threshold=threshold,
        reason=(
            f"P(cheap_ok)={probability:.3f} {comparison} "
            f"threshold={threshold:.3f}."
        ),
    )
