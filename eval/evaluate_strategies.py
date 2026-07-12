"""Compare routing strategies on the labeled dataset: accuracy vs tokens.

Strategies:
  always_tier0      always the cheapest model
  always_tier3      always the strongest model
  multitier_router  the local fine-tuned classifier (needs a trained checkpoint)
  binary_router     category rules + local cheap_ok probability threshold
  binary router tau sweep over 0.6, 0.7, 0.8, and 0.9
  prompt_baseline   LLM-based classification — OPT-IN via --include-prompt-baseline
                    because it makes a real Fireworks call per example

Answer outcomes are replayed from labeling data (tier_results), so no new
answer-generation calls happen here. If labeling ran in early-stop mode, tiers
above the labeled tier were never attempted; for those we assume stronger
tiers also pass and reuse the labeled tier's token count as the estimate.
Run labeling with --all-tiers for exact numbers (the report shows how many
outcomes were assumed vs measured).

Human runs:  python -m eval.evaluate_strategies
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from config import get_tier_names

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data" / "labeled_multitier.jsonl"
OUT_PATH = Path(__file__).parent / "results.json"
BINARY_TAU_SWEEP = (0.6, 0.7, 0.8, 0.9)


def load_records() -> list[dict]:
    if not DATA_PATH.exists():
        raise SystemExit(f"{DATA_PATH} not found — run labeling first.")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def outcome_for_tier(record: dict, tier: str) -> tuple[bool, int, bool]:
    """(passed, answer_tokens, was_assumed) for routing this record to `tier`."""
    results = record["tier_results"]
    if tier in results:
        return results[tier]["passed"], results[tier]["total_tokens"], False
    # Tier never attempted (early-stop labeling). It is at least as strong as
    # the labeled passing tier, so assume it passes; estimate tokens from the
    # closest attempted tier.
    attempted = [results[t]["total_tokens"] for t in results]
    est_tokens = attempted[-1] if attempted else 0
    assumed_pass = record["tier_label"] != "none"
    return assumed_pass, est_tokens, True


def simulate(records: list[dict], choose_tier, routing_tokens_fn=None) -> dict:
    """Run one strategy over all records and aggregate metrics."""
    stats = {
        "correct": 0, "total": len(records),
        "answer_tokens": 0, "routing_tokens": 0, "assumed_outcomes": 0,
        "per_category": defaultdict(lambda: [0, 0]),   # correct, total
        "per_difficulty": defaultdict(lambda: [0, 0]),
        "tier_usage": defaultdict(int),
    }
    for rec in records:
        tier = choose_tier(rec)
        stats["tier_usage"][tier] += 1
        passed, tokens, assumed = outcome_for_tier(rec, tier)
        stats["answer_tokens"] += tokens
        stats["assumed_outcomes"] += assumed
        if routing_tokens_fn:
            stats["routing_tokens"] += routing_tokens_fn(rec)
        stats["correct"] += passed
        for key, group in (("per_category", rec["category"]),
                           ("per_difficulty", rec["difficulty_pool"])):
            stats[key][group][0] += passed
            stats[key][group][1] += 1

    total_tokens = stats["answer_tokens"] + stats["routing_tokens"]
    return {
        "accuracy": stats["correct"] / max(stats["total"], 1),
        "correct": stats["correct"],
        "total": stats["total"],
        "answer_tokens": stats["answer_tokens"],
        "routing_tokens": stats["routing_tokens"],
        "total_tokens": total_tokens,
        "tokens_per_example": total_tokens / max(stats["total"], 1),
        "assumed_outcomes": stats["assumed_outcomes"],
        "tier_usage": dict(stats["tier_usage"]),
        "per_category": {k: v[0] / v[1] for k, v in stats["per_category"].items()},
        "per_difficulty": {k: v[0] / v[1] for k, v in stats["per_difficulty"].items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate routing strategies.")
    parser.add_argument("--include-prompt-baseline", action="store_true",
                        help="also run the LLM prompt baseline (REAL Fireworks "
                             "calls, one per example)")
    args = parser.parse_args()

    records = load_records()
    tiers = get_tier_names()
    cheapest, strongest = tiers[0], tiers[-1]
    logger.info("Loaded %d labeled examples.\n", len(records))

    results: dict[str, dict] = {}
    results["always_tier0"] = simulate(records, lambda r: cheapest)
    results["always_tier3"] = simulate(records, lambda r: strongest)

    # Local fine-tuned multitier router (zero routing tokens).
    from router.infer_multitier_router import (
        checkpoint_available as multitier_checkpoint_available,
        predict_tier as predict_multitier,
    )
    if multitier_checkpoint_available():
        results["multitier_router"] = simulate(
            records,
            lambda r: predict_multitier(r["prompt"], r.get("category")),
        )
    else:
        logger.warning("Skipping multitier_router: no checkpoint. "
                       "Train first: python -m router.train_multitier_router\n")

    # Binary router: evaluate the same category-aware policy used by the agent.
    # Cache probabilities by prompt/category so the default strategy and all
    # four thresholds reuse a single DistilBERT inference per unique routed
    # input. EASY_CATEGORIES short-circuit inside choose_binary_tier.
    from router.infer_binary_router import (
        checkpoint_available as binary_checkpoint_available,
        predict_cheap_ok_proba,
    )
    from router.route_binary import choose_binary_tier

    binary_sweep: dict[str, dict] = {}
    if binary_checkpoint_available():
        probability_cache: dict[tuple[str, str | None], float] = {}

        def cached_probability(prompt: str, category: str | None) -> float:
            key = (prompt, category)
            if key not in probability_cache:
                probability_cache[key] = predict_cheap_ok_proba(prompt, category)
            return probability_cache[key]

        def binary_tier(rec: dict, tau: float | None = None) -> str:
            return choose_binary_tier(
                rec["prompt"],
                rec.get("category"),
                tau=tau,
                predict_proba=cached_probability,
            )

        results["binary_router"] = simulate(records, binary_tier)
        for tau in BINARY_TAU_SWEEP:
            binary_sweep[f"{tau:.1f}"] = simulate(
                records,
                lambda r, threshold=tau: binary_tier(r, threshold),
            )
    else:
        logger.warning("Skipping binary_router and tau sweep: no checkpoint. "
                       "Train first: python -m router.train_binary_router\n")

    # Prompt baseline (costs real tokens per example) — opt-in.
    if args.include_prompt_baseline:
        from baseline.baseline_router import classify_tier
        baseline_cache: dict[str, tuple[str, int]] = {}

        def baseline_tier(rec):
            if rec["id"] not in baseline_cache:
                baseline_cache[rec["id"]] = classify_tier(rec["prompt"])
            return baseline_cache[rec["id"]][0]

        results["prompt_baseline"] = simulate(
            records, baseline_tier,
            routing_tokens_fn=lambda r: baseline_cache[r["id"]][1])
    else:
        logger.info("prompt_baseline skipped (pass --include-prompt-baseline to "
                    "run it; it makes one Fireworks call per example).\n")

    # Human-readable table.
    header = f"{'Strategy':<20} {'Accuracy':>9} {'Answer tok':>11} {'Routing tok':>12} {'Total tok':>10} {'Tok/ex':>8}"
    logger.info(header)
    logger.info("-" * len(header))
    for name, r in results.items():
        logger.info(f"{name:<20} {r['accuracy']:>8.1%} {r['answer_tokens']:>11,} "
                    f"{r['routing_tokens']:>12,} {r['total_tokens']:>10,} "
                    f"{r['tokens_per_example']:>8.1f}")
    logger.info("")
    for name, r in results.items():
        if r["assumed_outcomes"]:
            logger.info("note: %s used %d assumed (not measured) outcomes — "
                        "run labeling with --all-tiers for exact numbers.",
                        name, r["assumed_outcomes"])
        if "tier_usage" in r:
            logger.info("%s tier usage: %s", name, r["tier_usage"])

    if binary_sweep:
        logger.info("\nBinary router tau sweep:")
        sweep_header = (
            f"{'τ':>4} {'Accuracy':>9} {'Total tok':>10} {'Tok/ex':>8} "
            f"{'tier0%':>8} {'tier3%':>8}"
        )
        logger.info(sweep_header)
        logger.info("-" * len(sweep_header))
        for tau, r in binary_sweep.items():
            total = max(r["total"], 1)
            tier0_pct = r["tier_usage"].get(cheapest, 0) / total
            tier3_pct = r["tier_usage"].get(strongest, 0) / total
            logger.info(
                f"{tau:>4} {r['accuracy']:>8.1%} {r['total_tokens']:>10,} "
                f"{r['tokens_per_example']:>8.1f} {tier0_pct:>8.1%} "
                f"{tier3_pct:>8.1%}"
            )
        for tau, r in binary_sweep.items():
            if r["assumed_outcomes"]:
                logger.info(
                    "note: binary_router tau=%s used %d assumed (not measured) "
                    "outcomes.",
                    tau,
                    r["assumed_outcomes"],
                )

        # Keep the nested sweep out of the uniform main-strategy table above.
        results["binary_router_tau_sweep"] = binary_sweep

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("\nFull metrics (incl. per-category/per-difficulty) -> %s", OUT_PATH)


if __name__ == "__main__":
    main()
