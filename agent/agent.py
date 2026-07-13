"""Hackathon Track 1 agent entrypoint.

Contract: read /input/tasks.json, write /output/results.json, exit 0.
Routing is decided locally (zero Fireworks tokens in the default mode);
answers are always generated via Fireworks with ALLOWED_MODELS.

ROUTER_MODE env var selects the routing strategy without code changes:
  verified_tier0  tier0 first, deterministic verification, Fireworks fallback
  binary          legacy local tier0/tier3 classifier (analysis mode)
  multitier       legacy local 4-class classifier (A/B mode)
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agent.llm_backend import answer_chat_safe as chat_safe
from agent.quality_gate import assess_answer
from agent.request_policy import (
    ANSWER_MAX_TOKENS,
    ANSWER_SYSTEM_PROMPT,
    ANSWER_TEMPERATURE,
    CATEGORY_ALIASES,
    MEDIUM_REASONING_CATEGORIES,
    answer_reasoning_effort,
    answer_system_prompt,
    build_answer_request,
    normalize_category,
    prepare_answer_prompt,
)
from config import get_model_id_for_tier, get_tier_names

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("agent")

ROUTER_MODE = os.environ.get("ROUTER_MODE", "verified_tier0")
INPUT_PATH = Path(os.environ.get("INPUT_PATH", "/input/tasks.json"))
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", "/output/results.json"))
ANSWER_WORKERS = max(1, int(os.environ.get("ANSWER_WORKERS", "4")))
ANSWER_FALLBACKS = max(0, int(os.environ.get("ANSWER_FALLBACKS", "1")))


def route(prompt: str, category: str | None = None) -> tuple[str, str, int]:
    """Pick a model for the prompt. Returns (tier_name, model_id, routing_tokens)."""
    category = normalize_category(category)
    tiers = get_tier_names()

    if ROUTER_MODE in {"verified_tier0", "always_tier0"}:
        tier, tokens = tiers[0], 0
    elif ROUTER_MODE == "always_tier3":
        tier, tokens = tiers[-1], 0
    elif ROUTER_MODE == "prompt_baseline":
        from baseline.baseline_router import classify_tier
        tier, tokens = classify_tier(prompt)
    elif ROUTER_MODE == "binary":
        from router.labels import EASY_CATEGORIES

        if category in EASY_CATEGORIES:
            # This is the same policy as choose_binary_tier(), but avoids
            # importing torch/transformers for categories that bypass inference.
            tier, tokens = tiers[0], 0
        else:
            from router.infer_binary_router import checkpoint_available

            if checkpoint_available():
                from router.route_binary import choose_binary_tier

                tier, tokens = choose_binary_tier(prompt, category), 0
            else:
                logger.warning(
                    "No binary router checkpoint found - falling back to %s.",
                    tiers[0],
                )
                tier, tokens = tiers[0], 0
    elif ROUTER_MODE == "multitier":
        from router.infer_multitier_router import checkpoint_available, predict_tier
        if checkpoint_available():
            tier, tokens = predict_tier(prompt, category), 0
        else:
            # No trained checkpoint in the image: degrade gracefully instead
            # of crashing the whole run.
            logger.warning("No router checkpoint found — falling back to %s.", tiers[0])
            tier, tokens = tiers[0], 0
    else:
        raise ValueError(
            f"Unknown ROUTER_MODE {ROUTER_MODE!r}; expected verified_tier0, "
            "binary, multitier, prompt_baseline, always_tier0, or always_tier3"
        )

    return tier, get_model_id_for_tier(tier), tokens


def route_safe(prompt: str, category: str | None = None) -> tuple[str, str, int]:
    """Keep one broken local routing decision from crashing the whole episode."""
    try:
        return route(prompt, category)
    except Exception as exc:  # noqa: BLE001 - submission boundary
        fallback_tier = get_tier_names()[0]
        logger.exception(
            "Local routing failed; falling back to %s: %s", fallback_tier, exc
        )
        return fallback_tier, get_model_id_for_tier(fallback_tier), 0


def load_tasks(path: Path = INPUT_PATH) -> list[dict]:
    """Load and strictly validate the fixed container input contract."""
    try:
        tasks = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"input file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"input file is not valid JSON: {exc}") from exc

    if not isinstance(tasks, list):
        raise ValueError("input root must be a JSON array")

    seen_ids: set[tuple[type, object]] = set()
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"task at index {index} must be an object")
        missing = {"task_id", "prompt"} - task.keys()
        if missing:
            raise ValueError(
                f"task at index {index} is missing {', '.join(sorted(missing))}"
            )
        task_id = task["task_id"]
        if isinstance(task_id, bool) or not isinstance(task_id, (str, int, float)):
            raise ValueError(
                f"task_id at index {index} must be a string or number"
            )
        if isinstance(task_id, str) and not task_id.strip():
            raise ValueError(f"task_id at index {index} must not be blank")
        typed_id = (type(task_id), task_id)
        if typed_id in seen_ids:
            raise ValueError(f"duplicate task_id at index {index}: {task_id!r}")
        seen_ids.add(typed_id)

        prompt = task["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"prompt at index {index} must be non-empty text")
        category = task.get("category")
        if category is not None and not isinstance(category, str):
            raise ValueError(f"category at index {index} must be text when present")
    return tasks


def write_results_atomic(results: list[dict], path: Path = OUTPUT_PATH) -> None:
    """Replace results.json only after a complete, valid document is written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _fallback_tiers(primary_tier: str) -> list[str]:
    """Return distinct, configured fallback arms without assuming tier monotonicity."""
    tiers = get_tier_names()
    raw = os.environ.get("ANSWER_FALLBACK_TIERS", "").strip()
    preferred = (
        [part.strip() for part in raw.split(",") if part.strip()]
        if raw
        else [tiers[-1], tiers[0], *reversed(tiers[1:-1])]
    )
    result: list[str] = []
    for tier in preferred:
        if tier in tiers and tier != primary_tier and tier not in result:
            result.append(tier)
    return result


def answer_with_fallback(
    prompt: str,
    category: str | None,
    primary_tier: str,
    primary_model_id: str,
    request: dict,
    max_fallbacks: int | None = None,
) -> dict:
    """Generate once, retrying a different approved Fireworks arm only on failure.

    The local gate checks structural requirements only.  It never supplies an
    answer, and normal usable completions therefore retain the one-call cost.
    Token counts include every attempted generation so evaluation accounting
    remains honest when a fallback is needed.
    """
    attempts: list[dict] = []
    candidates: list[tuple[str, str]] = [(primary_tier, primary_model_id)]
    fallback_limit = ANSWER_FALLBACKS if max_fallbacks is None else max_fallbacks
    for tier in _fallback_tiers(primary_tier)[:max(0, fallback_limit)]:
        try:
            model_id = get_model_id_for_tier(tier)
        except (RuntimeError, ValueError) as exc:
            logger.warning("Skipping unavailable fallback %s: %s", tier, exc)
            continue
        if model_id != primary_model_id:
            candidates.append((tier, model_id))

    chosen: dict | None = None
    chosen_assessment = None
    for index, (tier, model_id) in enumerate(candidates):
        response = chat_safe(model_id, **request)
        assessment = assess_answer(
            prompt,
            category,
            response.get("text"),
            response.get("finish_reason"),
        )
        response = dict(response)
        response["text"] = assessment.text
        response["attempted_tier"] = tier
        response["quality_gate_reason"] = assessment.reason
        attempts.append(response)
        chosen = response
        chosen_assessment = assessment
        if assessment.usable:
            break
        if index + 1 < len(candidates):
            logger.warning(
                "Answer from %s failed structural gate (%s); trying %s.",
                tier,
                assessment.reason,
                candidates[index + 1][0],
            )

    assert chosen is not None and chosen_assessment is not None
    # If every call failed structurally, prefer a non-error response over an
    # empty/error marker, while retaining the conservative failure metadata.
    if not chosen_assessment.usable:
        non_error = [
            item
            for item in attempts
            if item.get("text") and not item.get("error")
        ]
        if non_error:
            chosen = non_error[-1]

    final = dict(chosen)
    for key in ("total_tokens", "prompt_tokens", "completion_tokens"):
        final[key] = sum(int(item.get(key, 0) or 0) for item in attempts)
    if not final.get("error"):
        final.pop("error", None)
    final["fallback_used"] = len(attempts) > 1
    final["attempted_models"] = [item.get("model_id") for item in attempts]
    final["attempted_tiers"] = [item.get("attempted_tier") for item in attempts]
    return final


def main() -> int:
    start = time.time()
    tasks = load_tasks(INPUT_PATH)
    logger.info("Router mode: %s | Processing %d tasks...", ROUTER_MODE, len(tasks))

    results = []
    tier_usage: Counter = Counter()
    model_attempts: Counter = Counter()
    totals = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0,
              "routing_tokens": 0}

    jobs = []
    with ThreadPoolExecutor(max_workers=ANSWER_WORKERS) as executor:
        for task in tasks:
            category = normalize_category(task.get("category"))
            tier, model_id, routing_tokens = route_safe(task["prompt"], category)
            totals["routing_tokens"] += routing_tokens

            request = build_answer_request(task["prompt"], category)
            future = executor.submit(
                answer_with_fallback,
                task["prompt"],
                category,
                tier,
                model_id,
                request,
                0 if ROUTER_MODE.startswith("always_") else None,
            )
            jobs.append((task, tier, model_id, future))

        # Resolve in input order so results.json always preserves task ordering.
        for task, tier, model_id, future in jobs:
            try:
                answer = future.result()
            except Exception as exc:  # noqa: BLE001 - preserve container contract
                logger.exception("Unhandled answer worker failure for %s", task["task_id"])
                answer = {
                    "text": f"ERROR: model call failed ({exc})",
                    "total_tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "finish_reason": "error",
                    "model_id": model_id,
                    "error": str(exc),
                }
            for key in ("total_tokens", "prompt_tokens", "completion_tokens"):
                totals[key] += answer[key]

            final_tier = answer.get("attempted_tier", tier)
            final_model_id = answer.get("model_id", model_id)
            tier_usage[final_tier] += 1
            model_attempts.update(answer.get("attempted_tiers", [tier]))
            results.append({"task_id": task["task_id"], "answer": answer["text"]})
            logger.info(
                "task %s -> %s (%s), %d tokens%s",
                task["task_id"],
                final_tier,
                final_model_id,
                answer["total_tokens"],
                " with fallback" if answer.get("fallback_used") else "",
            )

    write_results_atomic(results, OUTPUT_PATH)

    logger.info("Tier distribution: %s", dict(tier_usage))
    logger.info("Model attempts: %s", dict(model_attempts))
    logger.info("Total Fireworks tokens: %d (prompt: %d, completion: %d, routing: %d)",
                totals["total_tokens"] + totals["routing_tokens"],
                totals["prompt_tokens"], totals["completion_tokens"],
                totals["routing_tokens"])
    logger.info("Wrote %d results to %s in %.1fs",
                len(results), OUTPUT_PATH, time.time() - start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
