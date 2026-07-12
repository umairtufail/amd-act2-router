"""Central config loading so no module duplicates YAML/env-var logic.

All Fireworks model IDs are resolved from environment variables (per the
hackathon rules, model IDs must be swappable to ALLOWED_MODELS without code
changes). This module is the single place that reads config/models.yaml.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

_CONFIG_PATH = Path(__file__).parent / "models.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_fireworks_base_url() -> str:
    """Env var wins over the yaml default."""
    return os.environ.get("FIREWORKS_BASE_URL", load_config()["fireworks_base_url"])


def get_tier_names() -> list[str]:
    """Tier names in cheapest-first order, e.g. ['tier0', 'tier1', 'tier2', 'tier3']."""
    return [t["name"] for t in load_config()["model_tiers"]]


def get_allowed_model_ids() -> set[str] | None:
    """Parse the optional judging-harness ALLOWED_MODELS value.

    JSON lists and comma/whitespace-separated strings are accepted. Explicit
    MODEL_TIER* variables remain the source of tier ordering; this list is a
    compliance guard, not a guess at cheapest-to-strongest ordering.
    """
    raw = os.environ.get("ALLOWED_MODELS", "").strip()
    if not raw:
        return None

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        parsed = parsed.get("models", parsed.get("allowed_models"))
        if isinstance(parsed, dict):
            parsed = list(parsed.values())
    if isinstance(parsed, list):
        models = [str(model).strip() for model in parsed]
    elif isinstance(parsed, str):
        models = [parsed.strip()]
    elif parsed is None:
        models = [part for part in re.split(r"[\s,]+", raw) if part]
    else:
        raise RuntimeError(
            "ALLOWED_MODELS must be a JSON list or a comma-separated list."
        )

    allowed = {model for model in models if model}
    if not allowed:
        raise RuntimeError("ALLOWED_MODELS is set but contains no model IDs.")
    return allowed


def get_model_id_for_tier(tier_name: str) -> str:
    """Resolve a tier name to an actual Fireworks model ID via its env var."""
    for tier in load_config()["model_tiers"]:
        if tier["name"] == tier_name:
            model_id = os.environ.get(tier["id_env"], "").strip()
            if not model_id:
                raise RuntimeError(
                    f"Env var {tier['id_env']} is not set (needed for {tier_name}). "
                    "See .env.example."
                )
            allowed = get_allowed_model_ids()
            if allowed is not None and model_id not in allowed:
                raise RuntimeError(
                    f"{model_id!r} configured for {tier_name} is not present in "
                    "ALLOWED_MODELS."
                )
            return model_id
    raise ValueError(f"Unknown tier: {tier_name!r}. Known: {get_tier_names()}")


def get_judge_model_id() -> str:
    model_id = os.environ.get("MODEL_JUDGE", "").strip()
    if not model_id:
        raise RuntimeError("Env var MODEL_JUDGE is not set. See .env.example.")
    return model_id
