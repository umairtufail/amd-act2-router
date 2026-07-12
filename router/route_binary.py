"""Shared category-aware policy for binary tier0/tier3 routing."""

from __future__ import annotations

import os
from collections.abc import Callable

from router.labels import EASY_CATEGORIES

DEFAULT_BINARY_ROUTER_TAU = 0.8


def binary_thresholds_from_env() -> tuple[float, float]:
    """Return (general tau, NER tau), with NER inheriting the general value."""
    tau = float(os.environ.get("BINARY_ROUTER_TAU", str(DEFAULT_BINARY_ROUTER_TAU)))
    ner_tau = float(os.environ.get("NER_BINARY_TAU", str(tau)))
    return tau, ner_tau


def tier_from_probability(
    probability: float,
    category: str | None,
    tau: float = DEFAULT_BINARY_ROUTER_TAU,
    ner_tau: float | None = None,
) -> str:
    """Apply category rules and a cheap_ok threshold to a known probability."""
    if category in EASY_CATEGORIES:
        return "tier0"
    threshold = (tau if ner_tau is None else ner_tau) if category == "ner" else tau
    return "tier0" if probability >= threshold else "tier3"


def choose_binary_tier(
    prompt: str,
    category: str | None = None,
    tau: float | None = None,
    ner_tau: float | None = None,
    predict_proba: Callable[[str, str | None], float] | None = None,
) -> str:
    """Choose tier0/tier3 with the exact policy shared by agent and eval.

    Supplying an explicit ``tau`` without ``ner_tau`` applies that threshold
    to NER as well, keeping evaluation sweeps independent of environment
    overrides.
    """
    if category in EASY_CATEGORIES:
        return "tier0"

    if tau is None:
        env_tau, env_ner_tau = binary_thresholds_from_env()
        tau = env_tau
        if ner_tau is None:
            ner_tau = env_ner_tau
    elif ner_tau is None:
        ner_tau = tau

    if predict_proba is None:
        from router.infer_binary_router import predict_cheap_ok_proba

        predict_proba = predict_cheap_ok_proba
    probability = predict_proba(prompt, category)
    return tier_from_probability(probability, category, tau, ner_tau)
