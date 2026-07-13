"""Measured-only replay of routing strategies: accuracy versus raw tokens.

The evaluator never invents a result for an untried model.  Exact duplicate
prompts are one evaluation group, and repeated calls become repeated
observations for that group.  A missing arm is reported as unknown, making a
strategy non-comparable until the necessary measurements are collected.

Human runs: ``python -m eval.evaluate_strategies``.  This makes no answer calls
unless the explicitly opt-in ``--include-prompt-baseline`` flag is supplied.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from agent.request_policy import request_policy_hash
from config import get_tier_names
from data.integrity import (
    collapse_prompt_groups,
    dataset_hash,
    stable_json_hash,
)
from eval.offline_frontier import build_frontier_analysis

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data" / "labeled_multitier.jsonl"
OUT_PATH = Path(__file__).parent / "results.json"
BINARY_TAU_SWEEP = (0.6, 0.7, 0.8, 0.9)


def load_records() -> list[dict]:
    if not DATA_PATH.exists():
        raise SystemExit(f"{DATA_PATH} not found - run labeling first.")
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def collapse_replay_records(
    records: list[dict], *, required_request_policy_hash: str | None = None
) -> list[dict]:
    """Collapse duplicates into measured per-arm observations.

    When ``required_request_policy_hash`` is set, stale/legacy calls remain
    unknown rather than being mixed with outcomes from the deployed policy.
    """

    def resolve(group: list[dict]) -> dict[str, Any]:
        outcomes: dict[str, dict[str, Any]] = {}
        all_tiers = sorted(
            {
                tier
                for record in group
                for tier in record.get("tier_results", {}).keys()
            }
        )
        for tier in all_tiers:
            observations = [
                record["tier_results"][tier]
                for record in group
                if isinstance(record.get("tier_results", {}).get(tier), dict)
                and "passed" in record["tier_results"][tier]
                and "total_tokens" in record["tier_results"][tier]
                and (
                    required_request_policy_hash is None
                    or record["tier_results"][tier].get("request_policy_hash")
                    == required_request_policy_hash
                )
            ]
            if not observations:
                continue
            tokens = [float(observation["total_tokens"]) for observation in observations]
            outcomes[tier] = {
                "success_rate": sum(
                    float(bool(observation["passed"])) for observation in observations
                )
                / len(observations),
                "mean_total_tokens": sum(tokens) / len(tokens),
                "observations": len(observations),
                "model_ids": sorted(
                    {
                        str(observation.get("model_id", "unknown"))
                        for observation in observations
                    }
                ),
                "request_policy_hashes": dict(
                    Counter(
                        observation.get("request_policy_hash") or "legacy-or-missing"
                        for observation in observations
                    )
                ),
            }
        return {"_measured_tier_outcomes": outcomes}

    return collapse_prompt_groups(records, resolve)


def outcome_for_tier(
    record: dict, tier: str
) -> tuple[float | None, float | None, int]:
    """Return measured (success rate, mean raw tokens, call count).

    ``(None, None, 0)`` means this independent model arm was never measured for
    the prompt group.  It is intentionally not inferred from another arm.
    """
    outcome = record.get("_measured_tier_outcomes", {}).get(tier)
    if not isinstance(outcome, dict):
        return None, None, 0
    return (
        float(outcome["success_rate"]),
        float(outcome["mean_total_tokens"]),
        int(outcome.get("observations", 1)),
    )


def simulate(records: list[dict], choose_tier, routing_tokens_fn=None) -> dict:
    """Replay a strategy without treating unknown arm outcomes as failures."""
    stats: dict[str, Any] = {
        "total": len(records),
        "measured_outcomes": 0,
        "expected_correct_measured": 0.0,
        "measured_answer_tokens": 0.0,
        "routing_tokens": 0,
        "tier_usage": Counter(),
        "per_category": defaultdict(
            lambda: {"total": 0, "measured": 0, "correct": 0.0, "tokens": 0.0}
        ),
        "per_difficulty": defaultdict(
            lambda: {"total": 0, "measured": 0, "correct": 0.0, "tokens": 0.0}
        ),
        "unknown_by_tier": Counter(),
    }
    for record in records:
        tier = choose_tier(record)
        stats["tier_usage"][tier] += 1
        if routing_tokens_fn:
            stats["routing_tokens"] += int(routing_tokens_fn(record))

        success_rate, tokens, _ = outcome_for_tier(record, tier)
        measured = success_rate is not None and tokens is not None
        if not measured:
            stats["unknown_by_tier"][tier] += 1
        else:
            stats["measured_outcomes"] += 1
            stats["expected_correct_measured"] += success_rate
            stats["measured_answer_tokens"] += tokens

        for dimension, value in (
            ("per_category", record.get("category", "")),
            ("per_difficulty", record.get("difficulty_pool", "")),
        ):
            bucket = stats[dimension][value]
            bucket["total"] += 1
            if measured:
                bucket["measured"] += 1
                bucket["correct"] += success_rate
                bucket["tokens"] += tokens

    measured = stats["measured_outcomes"]
    total = stats["total"]
    fully_measured = measured == total
    measured_answer_tokens = stats["measured_answer_tokens"]

    def dimensions(name: str) -> dict[str, dict[str, Any]]:
        output: dict[str, dict[str, Any]] = {}
        for key, bucket in sorted(stats[name].items()):
            bucket_measured = bucket["measured"]
            bucket_total = bucket["total"]
            output[key] = {
                "accuracy": (
                    bucket["correct"] / bucket_total
                    if bucket_measured == bucket_total
                    else None
                ),
                "accuracy_on_measured": (
                    bucket["correct"] / bucket_measured if bucket_measured else None
                ),
                "measurement_coverage": bucket_measured / max(bucket_total, 1),
                "measured_prompt_groups": bucket_measured,
                "total_prompt_groups": bucket_total,
                "mean_raw_tokens_on_measured": (
                    bucket["tokens"] / bucket_measured if bucket_measured else None
                ),
            }
        return output

    return {
        "accuracy": (
            stats["expected_correct_measured"] / total if fully_measured else None
        ),
        "accuracy_on_measured": (
            stats["expected_correct_measured"] / measured if measured else None
        ),
        "expected_correct_measured": stats["expected_correct_measured"],
        "total": total,
        "measured_outcomes": measured,
        "unknown_outcomes": total - measured,
        "measurement_coverage": measured / max(total, 1),
        "fully_measured": fully_measured,
        "answer_tokens": measured_answer_tokens if fully_measured else None,
        "measured_answer_tokens": measured_answer_tokens,
        "routing_tokens": stats["routing_tokens"],
        "total_tokens": (
            measured_answer_tokens + stats["routing_tokens"]
            if fully_measured
            else None
        ),
        "tokens_per_example": (
            (measured_answer_tokens + stats["routing_tokens"]) / max(total, 1)
            if fully_measured
            else None
        ),
        "tier_usage": dict(stats["tier_usage"]),
        "unknown_by_tier": dict(stats["unknown_by_tier"]),
        "assumed_outcomes": 0,
        "per_category": dimensions("per_category"),
        "per_difficulty": dimensions("per_difficulty"),
    }


def _accuracy_text(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.1%}"


def _tokens_text(value: float | None) -> str:
    return "unknown" if value is None else f"{value:,.0f}"


def build_shadow_router_analysis(
    source_records: list[dict], *, val_frac: float = 0.2, seed: int = 7
) -> dict[str, Any]:
    """Compare tiny offline baselines on the same leakage-safe prompt groups.

    These results diagnose whether prompt/category features contain any hard
    class signal.  They are never imported by the submission runtime and are
    not deployment evidence because today's labels use a legacy request policy.
    """
    from eval.shadow_routers import (
        ShadowSplit,
        binary_metrics,
        run_shadow_baselines,
    )
    from router.labels import CHEAP_OK_LABEL
    from router.train_binary_router import (
        binary_label,
        collapse_binary_records,
        stratified_split,
    )

    grouped = collapse_binary_records(source_records)
    train_records, validation_records = stratified_split(grouped, val_frac, seed)

    def split(records: list[dict]) -> ShadowSplit:
        return ShadowSplit(
            prompts=[record["prompt"] for record in records],
            categories=[record.get("category") for record in records],
            labels=[
                0 if binary_label(record) == CHEAP_OK_LABEL else 1
                for record in records
            ],
            groups=[record["_prompt_group_key"] for record in records],
        )

    train = split(train_records)
    validation = split(validation_records)
    runs = run_shadow_baselines(
        train,
        validation,
        n_features=512,
        knn_k=3,
        logistic_epochs=400,
    )
    always_cheap = binary_metrics(validation.labels, [0] * len(validation.labels))
    return {
        "analysis_only": True,
        "eligible_for_submission_runtime": False,
        "reason": "shadow comparison uses legacy-policy labels and sparse hard groups",
        "seed": seed,
        "train_prompt_groups": len(train_records),
        "validation_prompt_groups": len(validation_records),
        "train_hard_groups": sum(train.labels),
        "validation_hard_groups": sum(validation.labels),
        "constant_tier0": always_cheap.as_dict(),
        "routers": {name: run.as_dict() for name, run in runs.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate routing strategies.")
    parser.add_argument(
        "--include-prompt-baseline",
        action="store_true",
        help="also run the LLM prompt baseline (REAL Fireworks calls)",
    )
    args = parser.parse_args()

    source_records = load_records()
    current_policy_hash = request_policy_hash()
    records = collapse_replay_records(source_records)
    current_policy_records = collapse_replay_records(
        source_records, required_request_policy_hash=current_policy_hash
    )
    tiers = get_tier_names()
    cheapest, strongest = tiers[0], tiers[-1]
    logger.info(
        "Loaded %d rows / %d unique prompt groups (%d duplicates removed).\n",
        len(source_records),
        len(records),
        len(source_records) - len(records),
    )

    results: dict[str, dict] = {
        "always_tier0": simulate(records, lambda _record: cheapest),
        "always_tier3": simulate(records, lambda _record: strongest),
    }

    from router.infer_multitier_router import (
        checkpoint_available as multitier_checkpoint_available,
        predict_tier as predict_multitier,
    )

    if multitier_checkpoint_available():
        results["multitier_router"] = simulate(
            records,
            lambda record: predict_multitier(
                record["prompt"], record.get("category")
            ),
        )
    else:
        logger.warning("Skipping multitier_router: no checkpoint.\n")

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

        def binary_tier(record: dict, tau: float | None = None) -> str:
            return choose_binary_tier(
                record["prompt"],
                record.get("category"),
                tau=tau,
                predict_proba=cached_probability,
            )

        results["binary_router"] = simulate(records, binary_tier)
        for tau in BINARY_TAU_SWEEP:
            binary_sweep[f"{tau:.1f}"] = simulate(
                records,
                lambda record, threshold=tau: binary_tier(record, threshold),
            )
    else:
        logger.warning("Skipping binary_router and tau sweep: no checkpoint.\n")

    if args.include_prompt_baseline:
        from baseline.baseline_router import classify_tier

        baseline_cache: dict[str, tuple[str, int]] = {}

        def baseline_tier(record: dict) -> str:
            group_id = record["_prompt_group_key"]
            if group_id not in baseline_cache:
                baseline_cache[group_id] = classify_tier(record["prompt"])
            return baseline_cache[group_id][0]

        results["prompt_baseline"] = simulate(
            records,
            baseline_tier,
            routing_tokens_fn=lambda record: baseline_cache[
                record["_prompt_group_key"]
            ][1],
        )
    else:
        logger.info(
            "prompt_baseline skipped (opt in with --include-prompt-baseline; "
            "it makes real calls).\n"
        )

    header = (
        f"{'Strategy':<20} {'Accuracy':>10} {'Coverage':>9} "
        f"{'Total tok':>11} {'Unknown':>8}"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for name, result in results.items():
        logger.info(
            f"{name:<20} {_accuracy_text(result['accuracy']):>10} "
            f"{result['measurement_coverage']:>8.1%} "
            f"{_tokens_text(result['total_tokens']):>11} "
            f"{result['unknown_outcomes']:>8}"
        )
    for name, result in results.items():
        if not result["fully_measured"]:
            logger.info(
                "note: %s is non-comparable: %d chosen arm outcomes are "
                "unmeasured (measured-subset accuracy %s).",
                name,
                result["unknown_outcomes"],
                _accuracy_text(result["accuracy_on_measured"]),
            )

    label_policy_hashes = Counter(
        record.get("request_policy_hash") or "legacy-or-mixed"
        for record in source_records
    )
    try:
        shadow_analysis = build_shadow_router_analysis(source_records)
    except (RuntimeError, ValueError) as exc:
        shadow_analysis = {
            "analysis_only": True,
            "eligible_for_submission_runtime": False,
            "error": str(exc),
        }
    output: dict[str, Any] = {
        "metadata": {
            "source_rows": len(source_records),
            "unique_prompt_groups": len(records),
            "duplicates_removed": len(source_records) - len(records),
            "source_dataset_sha256": dataset_hash(source_records),
            "evaluation_dataset_sha256": dataset_hash(records),
            "current_policy_evaluation_dataset_sha256": dataset_hash(
                current_policy_records
            ),
            "request_policy_sha256": current_policy_hash,
            "label_request_policy_hashes": dict(label_policy_hashes),
            "labels_match_current_request_policy": (
                set(label_policy_hashes) == {current_policy_hash}
            ),
            "unknown_outcomes_are_not_imputed": True,
            "default_strategy_replay_scope": "all historical measured outcomes",
            "deployment_evidence_scope": "current_request_policy_frontier",
        },
        "strategies": results,
        "binary_router_tau_sweep": binary_sweep,
        "offline_frontier": build_frontier_analysis(records, tiers, simulate),
        "current_request_policy_frontier": build_frontier_analysis(
            current_policy_records, tiers, simulate
        ),
        "shadow_routers": shadow_analysis,
    }
    output["results_sha256"] = stable_json_hash(output)
    OUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("\nMeasured-only metrics and frontier -> %s", OUT_PATH)


if __name__ == "__main__":
    main()
