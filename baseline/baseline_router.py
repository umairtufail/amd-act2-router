"""Prompt-based baseline router: pays Fireworks tokens for every routing decision.

This is the approach most routing guides describe — ask an LLM how hard the
query is before answering it. It exists here as the comparison point that the
local fine-tuned router must beat on token efficiency.
"""

from __future__ import annotations

import logging
import re

from agent.fireworks_client import chat
from config import get_model_id_for_tier, get_tier_names

logger = logging.getLogger(__name__)

CLASSIFY_PROMPT = """You are a router deciding which model should answer a query.
Tiers, from cheapest to strongest:
- 0: trivial queries (simple facts, basic arithmetic, obvious classification)
- 1: moderate queries (multi-step but routine reasoning or standard code)
- 2: hard queries (dense multi-step reasoning, subtle bugs, tricky specs)
- 3: hardest queries (adversarial, compound, or highly intricate problems)

Reply with exactly one digit: 0, 1, 2, or 3.

Query: {prompt}"""

# Reasoning models spend tokens thinking before answering. Too small a budget
# and they never emit the digit, silently defaulting the parse. Keep this
# generous and always inspect the raw text when debugging.
CLASSIFY_MAX_TOKENS = 200

_DIGIT_RE = re.compile(r"[0-3]")


def classify_tier(prompt: str) -> tuple[str, int]:
    """Classify a prompt into a tier using the cheapest model.

    Returns (tier_name, classification_tokens). The token count is the price
    this baseline pays per decision — the whole point of comparison.
    """
    tier_names = get_tier_names()
    result = chat(
        get_model_id_for_tier(tier_names[0]),
        CLASSIFY_PROMPT.format(prompt=prompt),
        max_tokens=CLASSIFY_MAX_TOKENS,
        temperature=0.0,
    )
    match = _DIGIT_RE.search(result["text"].strip())
    if match:
        tier = tier_names[int(match.group(0))]
    else:
        # No parseable digit: safest cheap default, but log it loudly.
        logger.warning("Baseline classifier emitted no digit, defaulting to %s. Raw: %r",
                       tier_names[0], result["text"][:200])
        tier = tier_names[0]
    return tier, result["total_tokens"]
