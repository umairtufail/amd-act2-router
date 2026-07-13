"""Compare routing strategies on holdout prompts not used for training.

Answers use a configured local OpenAI-compatible backend by default. Pass
``--fireworks`` to opt into real Fireworks answer calls. Results include both
per-strategy aggregates and per-task routing/grading details.

Usage::

    python -m eval.holdout_generalization
    python -m eval.holdout_generalization --strategy binary
    python -m eval.holdout_generalization --strategy all --fireworks
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from pathlib import Path

from agent import local_llm_client
from agent.llm_backend import answer_chat
from agent.quality_gate import assess_answer, verification_policy_hash
from agent.request_policy import build_answer_request, request_policy_hash
from config import get_model_id_for_tier
from data.integrity import dataset_hash, stable_json_hash
from data.judge import grade_answer
from data.schema import CATEGORIES
from router.infer_binary_router import (
    checkpoint_available as binary_checkpoint_available,
)
from router.infer_multitier_router import (
    checkpoint_available as multitier_checkpoint_available,
    predict_tier as predict_multitier_tier,
)
from router.route_binary import choose_binary_tier

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

HOLDOUT_PATH = Path(__file__).parent.parent / "data" / "holdout_tasks.json"
RESULTS_PATH = Path(__file__).parent / "holdout_results.json"
STRATEGIES = (
    "verified_tier0",
    "binary",
    "multitier",
    "always_tier0",
    "always_tier3",
)
AnswerCache = dict[tuple[str, str, str], dict]
GradeCache = dict[tuple[str, str, str, str], bool]


def load_tasks() -> list[dict]:
    if not HOLDOUT_PATH.exists():
        raise SystemExit(f"{HOLDOUT_PATH} not found.")
    tasks = json.loads(HOLDOUT_PATH.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise SystemExit(f"{HOLDOUT_PATH} must contain a JSON list of tasks.")
    required = {"id", "category", "prompt", "ground_truth"}
    seen_ids: set[str] = set()
    category_counts: Counter[str] = Counter()
    for index, task in enumerate(tasks):
        missing = required - task.keys() if isinstance(task, dict) else required
        if not isinstance(task, dict) or missing:
            missing_text = ", ".join(sorted(missing))
            raise SystemExit(
                f"Invalid holdout task at index {index}; missing: {missing_text}"
            )
        task_id = task["id"]
        if not isinstance(task_id, str) or not task_id.strip():
            raise SystemExit(f"Invalid holdout task at index {index}; id must be text.")
        if task_id in seen_ids:
            raise SystemExit(f"Duplicate holdout task id: {task_id!r}.")
        seen_ids.add(task_id)

        category = task["category"]
        if category not in CATEGORIES:
            raise SystemExit(
                f"Invalid category {category!r} for holdout task {task_id!r}; "
                f"expected one of {CATEGORIES}."
            )
        category_counts[category] += 1

    missing_categories = [
        category for category in CATEGORIES if category_counts[category] == 0
    ]
    if missing_categories:
        raise SystemExit(
            "Holdout set has no coverage for categories: "
            + ", ".join(missing_categories)
        )
    return tasks


def selected_strategies(requested: str) -> list[str]:
    return list(STRATEGIES) if requested == "all" else [requested]


def require_checkpoints(
    strategies: list[str], *, skip_unavailable: bool = False
) -> list[str]:
    """Validate learned strategies, optionally skipping missing ones for ``all``."""
    missing: list[str] = []
    available: list[str] = []
    requirements = {
        "binary": (
            binary_checkpoint_available,
            "binary checkpoint (run: python -m router.train_binary_router)",
        ),
        "multitier": (
            multitier_checkpoint_available,
            "multitier checkpoint (run: python -m router.train_multitier_router)",
        ),
    }
    for strategy in strategies:
        requirement = requirements.get(strategy)
        if requirement is None or requirement[0]():
            available.append(strategy)
            continue
        if skip_unavailable:
            logger.warning("Skipping %s: missing %s.", strategy, requirement[1])
        else:
            missing.append(requirement[1])
    if missing:
        raise SystemExit("Missing required " + "; ".join(missing) + ".")
    return available


def configure_answer_backend(use_fireworks: bool) -> tuple[str, str]:
    """Configure answer dispatch and return (backend, actual model name)."""
    if use_fireworks:
        os.environ.pop("DEV_LOCAL_ANSWERS", None)
        return "fireworks", "per-tier"

    os.environ["DEV_LOCAL_ANSWERS"] = "1"
    missing: list[str] = []
    if not local_llm_client.is_configured():
        missing.append("LOCAL_LLM_BASE_URL")
    local_model = os.environ.get("LOCAL_LLM_MODEL", "").strip()
    if not local_model:
        missing.append("LOCAL_LLM_MODEL")
    if missing:
        raise SystemExit(
            "Local answers are the default, but the local backend is not "
            f"configured (missing {', '.join(missing)}). Set the variables "
            "in .env or pass --fireworks to opt into Fireworks calls."
        )
    return "local", local_model


def choose_tier(strategy: str, task: dict) -> str:
    prompt = task["prompt"]
    category = task.get("category")
    if strategy == "binary":
        return choose_binary_tier(prompt, category)
    if strategy == "multitier":
        return predict_multitier_tier(prompt, category)
    if strategy in {"verified_tier0", "always_tier0"}:
        return "tier0"
    if strategy == "always_tier3":
        return "tier3"
    raise ValueError(f"Unknown strategy: {strategy!r}")


def cached_answer(
    cache: AnswerCache,
    *,
    backend: str,
    backend_model: str,
    routed_model_id: str,
    prompt: str,
    category: str | None,
) -> tuple[dict, bool]:
    """Generate once for each backend/model/exact shared request combination."""
    request = build_answer_request(prompt, category)
    key = (backend, backend_model, stable_json_hash(request))
    if key in cache:
        return cache[key], True
    result = answer_chat(routed_model_id, **request)
    cache[key] = result
    return result, False


def cached_grade(
    cache: GradeCache,
    task: dict,
    answer_text: str,
) -> tuple[bool, bool]:
    """Cache verdicts for identical grading inputs, including LLM judgments."""
    key = (
        task.get("category", ""),
        task["prompt"],
        task["ground_truth"],
        answer_text,
    )
    if key in cache:
        return cache[key], True
    passed = bool(
        grade_answer(
            task.get("category", ""),
            task["prompt"],
            task["ground_truth"],
            answer_text,
        )
    )
    cache[key] = passed
    return passed, False


def regrade_failed_results(saved_results: dict, tasks: list[dict]) -> dict:
    """Regrade saved failures in place without generating any new answers."""
    expected_dataset_hash = dataset_hash(tasks)
    saved_dataset_hash = saved_results.get("dataset_sha256")
    if saved_dataset_hash != expected_dataset_hash:
        raise SystemExit(
            "Cannot regrade stale holdout results: dataset hash differs "
            f"(saved={saved_dataset_hash or 'missing'}, "
            f"current={expected_dataset_hash}). Run a fresh evaluation."
        )
    expected_policy_hash = request_policy_hash()
    saved_policy_hash = saved_results.get("request_policy_sha256")
    if saved_policy_hash != expected_policy_hash:
        raise SystemExit(
            "Cannot regrade stale holdout results: answer request policy hash "
            f"differs (saved={saved_policy_hash or 'missing'}, "
            f"current={expected_policy_hash}). Run a fresh evaluation."
        )
    expected_verifier_hash = verification_policy_hash()
    saved_verifier_hash = saved_results.get("verification_policy_sha256")
    if saved_verifier_hash != expected_verifier_hash:
        raise SystemExit(
            "Cannot regrade stale holdout results: verification policy hash "
            f"differs (saved={saved_verifier_hash or 'missing'}, "
            f"current={expected_verifier_hash}). Run a fresh evaluation."
        )
    saved_results_hash = saved_results.pop("results_sha256", None)
    actual_results_hash = stable_json_hash(saved_results)
    if saved_results_hash != actual_results_hash:
        if saved_results_hash is not None:
            saved_results["results_sha256"] = saved_results_hash
        raise SystemExit(
            "Cannot regrade holdout results: results hash is missing or does "
            "not match the file contents. Run a fresh evaluation."
        )

    tasks_by_id = {task.get("id"): task for task in tasks}
    strategies = saved_results.get("strategies")
    if not isinstance(strategies, dict):
        raise SystemExit(f"{RESULTS_PATH} has no strategy results to regrade.")

    expected_ids = set(tasks_by_id)
    for strategy, result in strategies.items():
        saved_ids = {
            task.get("task_id") for task in result.get("tasks", [])
        }
        if saved_ids != expected_ids:
            missing = sorted(expected_ids - saved_ids)
            extra = sorted(saved_ids - expected_ids, key=str)
            raise SystemExit(
                f"Cannot regrade stale {strategy} results: holdout task IDs changed "
                f"(missing={missing[:5]}, extra={extra[:5]}). Run a fresh full "
                "holdout evaluation first."
            )

    grade_cache: GradeCache = {}
    failed_entries = 0
    changed_to_pass = 0
    for strategy, result in strategies.items():
        task_results = result.get("tasks", [])
        for saved_task in task_results:
            if saved_task.get("passed"):
                continue
            task_id = saved_task.get("task_id")
            task = tasks_by_id.get(task_id)
            if task is None:
                raise SystemExit(
                    f"Cannot regrade {strategy}/{task_id}: task is missing from "
                    f"{HOLDOUT_PATH}."
                )
            if "answer" not in saved_task:
                raise SystemExit(
                    f"Cannot regrade {strategy}/{task_id}: saved answer is missing."
                )

            failed_entries += 1
            passed, cache_hit = cached_grade(
                grade_cache, task, saved_task["answer"]
            )
            saved_task["passed"] = passed
            saved_task["grade_cache_hit"] = cache_hit
            changed_to_pass += int(passed)
            logger.info(
                "[%s] regrade strategy=%s task=%s%s",
                "PASS" if passed else "FAIL",
                strategy,
                task_id,
                " cached-verdict" if cache_hit else "",
            )

        total = len(task_results)
        correct = sum(bool(task.get("passed")) for task in task_results)
        result["correct"] = correct
        result["total"] = total
        result["accuracy"] = correct / max(total, 1)
        logger.info(
            "%s after regrade: %d/%d = %.1f%%",
            strategy,
            correct,
            total,
            100 * result["accuracy"],
        )

    saved_results["last_regrade"] = {
        "failed_entries": failed_entries,
        "unique_grade_calls": len(grade_cache),
        "changed_to_pass": changed_to_pass,
        "answer_calls": 0,
    }
    saved_results["results_sha256"] = stable_json_hash(saved_results)
    return saved_results


def regrade_saved_failures() -> None:
    if not RESULTS_PATH.exists():
        raise SystemExit(f"{RESULTS_PATH} not found; run holdout evaluation first.")
    saved_results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    regrade_failed_results(saved_results, load_tasks())
    RESULTS_PATH.write_text(json.dumps(saved_results, indent=2), encoding="utf-8")
    summary = saved_results["last_regrade"]
    logger.info(
        "Regraded %d failed entries with %d unique judge calls and 0 answer "
        "calls; %d changed to PASS.\nUpdated results -> %s",
        summary["failed_entries"],
        summary["unique_grade_calls"],
        summary["changed_to_pass"],
        RESULTS_PATH,
    )


def evaluate_strategy(
    strategy: str,
    tasks: list[dict],
    *,
    answer_backend: str,
    local_model: str,
    answer_cache: AnswerCache,
    grade_cache: GradeCache,
) -> dict:
    correct = 0
    total_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0
    tier_usage: Counter[str] = Counter()
    model_attempts: Counter[str] = Counter()
    category_correct: Counter[str] = Counter()
    category_total: Counter[str] = Counter()
    task_results: list[dict] = []

    for index, task in enumerate(tasks):
        task_id = task.get("id", f"task_{index}")
        initial_tier = choose_tier(strategy, task)
        tier = initial_tier
        routed_model_id = get_model_id_for_tier(tier)
        backend_model = routed_model_id if answer_backend == "fireworks" else local_model
        answer, answer_cache_hit = cached_answer(
            answer_cache,
            backend=answer_backend,
            backend_model=backend_model,
            routed_model_id=routed_model_id,
            prompt=task["prompt"],
            category=task.get("category"),
        )
        attempts = [
            {
                "tier": tier,
                "model_id": routed_model_id,
                "backend_model": backend_model,
                "answer": answer,
                "cache_hit": answer_cache_hit,
            }
        ]
        fallback_used = False
        quality_gate_reason = None
        if strategy == "verified_tier0":
            assessment = assess_answer(
                task["prompt"],
                task.get("category"),
                answer.get("text"),
                answer.get("finish_reason"),
            )
            answer = {**answer, "text": assessment.text}
            attempts[0]["answer"] = answer
            quality_gate_reason = assessment.reason
            if not assessment.usable:
                fallback_used = True
                tier = "tier3"
                routed_model_id = get_model_id_for_tier(tier)
                backend_model = (
                    routed_model_id if answer_backend == "fireworks" else local_model
                )
                answer, fallback_cache_hit = cached_answer(
                    answer_cache,
                    backend=answer_backend,
                    backend_model=backend_model,
                    routed_model_id=routed_model_id,
                    prompt=task["prompt"],
                    category=task.get("category"),
                )
                fallback_assessment = assess_answer(
                    task["prompt"],
                    task.get("category"),
                    answer.get("text"),
                    answer.get("finish_reason"),
                )
                answer = {**answer, "text": fallback_assessment.text}
                attempts.append(
                    {
                        "tier": tier,
                        "model_id": routed_model_id,
                        "backend_model": backend_model,
                        "answer": answer,
                        "cache_hit": fallback_cache_hit,
                    }
                )

        answer_text = answer.get("text", "")
        passed, grade_cache_hit = cached_grade(grade_cache, task, answer_text)

        answer_total = sum(
            int(attempt["answer"].get("total_tokens", 0) or 0)
            for attempt in attempts
        )
        answer_prompt = sum(
            int(attempt["answer"].get("prompt_tokens", 0) or 0)
            for attempt in attempts
        )
        answer_completion = sum(
            int(attempt["answer"].get("completion_tokens", 0) or 0)
            for attempt in attempts
        )
        correct += int(passed)
        category = task.get("category", "")
        category_correct[category] += int(passed)
        category_total[category] += 1
        total_tokens += answer_total
        prompt_tokens += answer_prompt
        completion_tokens += answer_completion
        tier_usage[tier] += 1
        model_attempts.update(attempt["tier"] for attempt in attempts)

        cache_note = " cached-answer" if all(a["cache_hit"] for a in attempts) else ""
        logger.info(
            "[%s] strategy=%s task=%s tier=%s tokens=%d%s",
            "PASS" if passed else "FAIL",
            strategy,
            task_id,
            tier,
            answer_total,
            cache_note,
        )

        task_results.append(
            {
                "task_id": task_id,
                "category": task.get("category"),
                "initial_tier": initial_tier,
                "tier": tier,
                "routed_model_id": routed_model_id,
                "answer_backend": answer_backend,
                "answer_model": backend_model,
                "passed": passed,
                "total_tokens": answer_total,
                "prompt_tokens": answer_prompt,
                "completion_tokens": answer_completion,
                "answer_cache_hit": all(a["cache_hit"] for a in attempts),
                "grade_cache_hit": grade_cache_hit,
                "fallback_used": fallback_used,
                "quality_gate_reason": quality_gate_reason,
                "attempted_tiers": [attempt["tier"] for attempt in attempts],
                "answer": answer_text,
            }
        )

    total = len(tasks)
    result = {
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens_per_example": total_tokens / max(total, 1),
        "tier_usage": dict(tier_usage),
        "model_attempts": dict(model_attempts),
        "per_category": {
            category: {
                "accuracy": category_correct[category] / total_for_category,
                "correct": category_correct[category],
                "total": total_for_category,
            }
            for category, total_for_category in sorted(category_total.items())
        },
        "tasks": task_results,
    }
    logger.info(
        "%s: %d/%d = %.1f%% | total tokens=%d | tier usage=%s\n",
        strategy,
        correct,
        total,
        100 * result["accuracy"],
        total_tokens,
        dict(tier_usage),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Holdout generalization test.")
    parser.add_argument(
        "--strategy",
        choices=(*STRATEGIES, "all"),
        default="all",
        help="routing strategy to evaluate (default: all)",
    )
    parser.add_argument(
        "--fireworks",
        action="store_true",
        help="use Fireworks for answers, or for judging with --regrade",
    )
    parser.add_argument(
        "--regrade",
        action="store_true",
        help="regrade only failures saved in holdout_results.json; makes no "
             "answer-generation calls",
    )
    args = parser.parse_args()

    if args.regrade:
        if args.fireworks:
            os.environ["JUDGE_BACKEND"] = "fireworks"
        regrade_saved_failures()
        return

    tasks = load_tasks()
    strategies = require_checkpoints(
        selected_strategies(args.strategy),
        skip_unavailable=args.strategy == "all",
    )
    answer_backend, local_model = configure_answer_backend(args.fireworks)
    if answer_backend == "local":
        logger.warning(
            "Local holdout mode uses the same LOCAL_LLM_MODEL for every routed "
            "tier; use --fireworks to compare tier-specific answer quality.\n"
        )
    logger.info(
        "Holdout tasks: %d | strategies: %s | answers via: %s\n",
        len(tasks),
        ", ".join(strategies),
        answer_backend if args.fireworks else f"local ({local_model})",
    )

    answer_cache: AnswerCache = {}
    grade_cache: GradeCache = {}
    strategy_results: dict[str, dict] = {}
    for strategy in strategies:
        strategy_results[strategy] = evaluate_strategy(
            strategy,
            tasks,
            answer_backend=answer_backend,
            local_model=local_model,
            answer_cache=answer_cache,
            grade_cache=grade_cache,
        )

    results = {
        "answer_backend": answer_backend,
        "answer_model": local_model,
        "task_count": len(tasks),
        "answer_calls": len(answer_cache),
        "grade_calls": len(grade_cache),
        "dataset_sha256": dataset_hash(tasks),
        "request_policy_sha256": request_policy_hash(),
        "verification_policy_sha256": verification_policy_hash(),
        "strategies": strategy_results,
    }
    results["results_sha256"] = stable_json_hash(results)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Holdout results -> %s", RESULTS_PATH)


if __name__ == "__main__":
    main()
