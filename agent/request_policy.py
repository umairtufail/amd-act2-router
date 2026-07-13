"""Pure, shared construction of answer-generation requests.

The labeling, evaluation, demo, and submission paths must ask a model the
same question in the same way.  Keeping that policy here prevents a router
from being trained on responses produced with a different temperature,
system prompt, or reasoning setting than the deployed agent.

This module performs no I/O and makes no model calls.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

POLICY_VERSION = "answer-request-v2"

ANSWER_MAX_TOKENS = int(os.environ.get("ANSWER_MAX_TOKENS", "700"))
ANSWER_TEMPERATURE = float(os.environ.get("ANSWER_TEMPERATURE", "0.0"))

ANSWER_SYSTEM_PROMPT = os.environ.get(
    "ANSWER_SYSTEM_PROMPT",
    "Answer directly, completely, and concisely. Follow every requested format "
    "and constraint exactly. Include all requested items without unnecessary "
    "headings, tables, preambles, or repetition.",
)

CATEGORY_ALIASES = {
    "mathematical_reasoning": "math_reasoning",
    "sentiment_classification": "sentiment",
    "text_summarization": "summarization",
    "named_entity_recognition": "ner",
}

MEDIUM_REASONING_CATEGORIES = {
    "math_reasoning",
    "logic_puzzles",
    "code_debugging",
    "code_generation",
}

FACTUAL_SYSTEM_SUFFIX = (
    " For factual comparisons, cover definitions, relationship, mechanism, "
    "use, persistence, and speed or performance when relevant. State "
    "comparisons explicitly (for example, one is faster than the other), "
    "rather than describing only one side. Stay under 140 words."
)

SENTIMENT_SYSTEM_SUFFIX = (
    " If both positive and negative evidence are material, classify the "
    "sentiment as Neutral unless Mixed is an allowed label; do not classify "
    "such a mixed review as Negative. In the single reason, explicitly "
    "mention every material positive and negative detail from the input. "
    "Preserve concrete facts such as timing, missing items, damage, and "
    "performance instead of replacing them with vague paraphrases."
)

NER_SYSTEM_SUFFIX = (
    " Follow the requested NER schema exactly. Keep each comma-qualified location "
    "whole as one entry. For organizations, use separate entries for a full name, "
    "its parenthesized acronym, and any later unambiguous shortened mention."
)

FACTUAL_PROMPT_SUFFIX = (
    "\n\nResponse requirements: Stay under 140 words. If comparing concepts, "
    "explicitly contrast both on every relevant dimension. When their speed "
    "or performance differs, state directly which one is faster or slower."
)


def normalize_category(category: str | None) -> str | None:
    """Map evaluator category names to the router's training vocabulary."""
    if category is None:
        return None
    normalized = str(category).strip().lower()
    return CATEGORY_ALIASES.get(normalized, normalized)


def answer_reasoning_effort(category: str | None) -> str:
    """Use measured category defaults while allowing an explicit override."""
    override = os.environ.get("ANSWER_REASONING_EFFORT", "").strip().lower()
    if override:
        return override
    return (
        "medium"
        if normalize_category(category) in MEDIUM_REASONING_CATEGORIES
        else "low"
    )


def answer_system_prompt(category: str | None) -> str:
    """Return concise, category-aware instructions used by every answer path."""
    normalized = normalize_category(category)
    if normalized == "factual_knowledge":
        return ANSWER_SYSTEM_PROMPT + FACTUAL_SYSTEM_SUFFIX
    if normalized == "sentiment":
        return ANSWER_SYSTEM_PROMPT + SENTIMENT_SYSTEM_SUFFIX
    if normalized == "ner":
        return ANSWER_SYSTEM_PROMPT + NER_SYSTEM_SUFFIX
    return ANSWER_SYSTEM_PROMPT


def prepare_answer_prompt(prompt: str, category: str | None) -> str:
    """Attach only requirements that are part of the deployed request policy."""
    if normalize_category(category) == "factual_knowledge":
        return prompt + FACTUAL_PROMPT_SUFFIX
    return prompt


def build_answer_request(
    prompt: str,
    category: str | None,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Build kwargs accepted by :func:`agent.llm_backend.answer_chat`.

    The returned object is deliberately plain data so callers can record or
    hash it without importing a model runtime.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    return {
        "prompt": prepare_answer_prompt(prompt, category),
        "max_tokens": ANSWER_MAX_TOKENS if max_tokens is None else int(max_tokens),
        "temperature": (
            ANSWER_TEMPERATURE if temperature is None else float(temperature)
        ),
        "system_prompt": answer_system_prompt(category),
        "reasoning_effort": answer_reasoning_effort(category),
    }


def _policy_descriptor() -> dict[str, Any]:
    """Stable description of all values that can alter generated responses."""
    return {
        "version": POLICY_VERSION,
        "max_tokens": ANSWER_MAX_TOKENS,
        "temperature": ANSWER_TEMPERATURE,
        "base_system_prompt": ANSWER_SYSTEM_PROMPT,
        "category_aliases": CATEGORY_ALIASES,
        "medium_reasoning_categories": sorted(MEDIUM_REASONING_CATEGORIES),
        "reasoning_effort_override": os.environ.get(
            "ANSWER_REASONING_EFFORT", ""
        ).strip().lower(),
        "factual_system_suffix": FACTUAL_SYSTEM_SUFFIX,
        "sentiment_system_suffix": SENTIMENT_SYSTEM_SUFFIX,
        "ner_system_suffix": NER_SYSTEM_SUFFIX,
        "factual_prompt_suffix": FACTUAL_PROMPT_SUFFIX,
    }


def request_policy_hash() -> str:
    """Return a reproducible SHA-256 for evaluation freshness checks."""
    encoded = json.dumps(
        _policy_descriptor(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
