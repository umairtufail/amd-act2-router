"""Hackathon Track 1 agent entrypoint.

Contract: read /input/tasks.json, write /output/results.json, exit 0.
Routing is decided locally (zero Fireworks tokens in the default mode);
answers are always generated via Fireworks with ALLOWED_MODELS.

ROUTER_MODE env var selects the routing strategy without code changes:
  multitier       local fine-tuned classifier (default, submission mode)
  prompt_baseline LLM-based tier classification (costs tokens; for A/B tests)
  always_tier0    fixed cheapest tier
  always_tier3    fixed strongest tier

INPUT_PATH / OUTPUT_PATH env vars override the container paths for local runs.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

from agent.fireworks_client import chat_safe
from config import get_model_id_for_tier, get_tier_names

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("agent")

ROUTER_MODE = os.environ.get("ROUTER_MODE", "multitier")
INPUT_PATH = Path(os.environ.get("INPUT_PATH", "/input/tasks.json"))
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", "/output/results.json"))
ANSWER_MAX_TOKENS = int(os.environ.get("ANSWER_MAX_TOKENS", "700"))


def route(prompt: str, category: str | None = None) -> tuple[str, str, int]:
    """Pick a model for the prompt. Returns (tier_name, model_id, routing_tokens)."""
    tiers = get_tier_names()

    if ROUTER_MODE == "always_tier0":
        tier, tokens = tiers[0], 0
    elif ROUTER_MODE == "always_tier3":
        tier, tokens = tiers[-1], 0
    elif ROUTER_MODE == "prompt_baseline":
        from baseline.baseline_router import classify_tier
        tier, tokens = classify_tier(prompt)
    else:  # multitier — the local classifier, zero tokens
        from router.infer_multitier_router import checkpoint_available, predict_tier
        if checkpoint_available():
            tier, tokens = predict_tier(prompt, category), 0
        else:
            # No trained checkpoint in the image: degrade gracefully instead
            # of crashing the whole run.
            logger.warning("No router checkpoint found — falling back to %s.", tiers[0])
            tier, tokens = tiers[0], 0

    return tier, get_model_id_for_tier(tier), tokens


def main() -> int:
    start = time.time()
    tasks = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    logger.info("Router mode: %s | Processing %d tasks...", ROUTER_MODE, len(tasks))

    results = []
    tier_usage: Counter = Counter()
    totals = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0,
              "routing_tokens": 0}

    for task in tasks:
        tier, model_id, routing_tokens = route(task["prompt"], task.get("category"))
        tier_usage[tier] += 1
        totals["routing_tokens"] += routing_tokens

        answer = chat_safe(model_id, task["prompt"],
                           max_tokens=ANSWER_MAX_TOKENS, temperature=0.2)
        for key in ("total_tokens", "prompt_tokens", "completion_tokens"):
            totals[key] += answer[key]

        results.append({"task_id": task["task_id"], "answer": answer["text"]})
        logger.info("task %s -> %s (%s), %d tokens",
                    task["task_id"], tier, model_id, answer["total_tokens"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    logger.info("Tier distribution: %s", dict(tier_usage))
    logger.info("Total Fireworks tokens: %d (prompt: %d, completion: %d, routing: %d)",
                totals["total_tokens"] + totals["routing_tokens"],
                totals["prompt_tokens"], totals["completion_tokens"],
                totals["routing_tokens"])
    logger.info("Wrote %d results to %s in %.1fs",
                len(results), OUTPUT_PATH, time.time() - start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
