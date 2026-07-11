"""Label each task with the cheapest model tier that passes grading.

For every task in data/tasks_raw.jsonl, models are tried in order
tier0 -> tier1 -> tier2 -> tier3. Each answer is graded (real test execution
for code_generation, LLM judge for everything else). The first passing tier
becomes the label.

Resumable: task IDs already present in data/labeled_multitier.jsonl are
skipped, and results are appended one line at a time, so the script can be
stopped and restarted without wasting tokens.

Modes:
  default      stop at the first tier that passes (cheapest labeling run)
  --all-tiers  grade EVERY tier for every task; costs more tokens but makes
               eval/evaluate_strategies.py exact instead of assuming that
               stronger tiers also pass.

THIS SCRIPT CONSUMES FIREWORKS TOKENS. The human runs it explicitly.
"""

import argparse
import json
import logging
from pathlib import Path

from agent.fireworks_client import chat
from config import get_model_id_for_tier, get_tier_names
from data.judge import grade_code_answer, grade_text_answer
from data.schema import CODE_CATEGORIES, LabeledExample, TaskExample

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

IN_PATH = Path(__file__).parent / "tasks_raw.jsonl"
OUT_PATH = Path(__file__).parent / "labeled_multitier.jsonl"


def _load_done_ids() -> set[str]:
    if not OUT_PATH.exists():
        return set()
    with open(OUT_PATH, "r", encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def _grade(task: TaskExample, answer_text: str) -> bool:
    if task.category in CODE_CATEGORIES:
        return grade_code_answer(task.ground_truth, answer_text)
    return grade_text_answer(task.prompt, task.ground_truth, answer_text)


def label_task(task: TaskExample, all_tiers: bool) -> LabeledExample:
    tier_results: dict = {}
    tier_label = "none"
    for tier in get_tier_names():
        model_id = get_model_id_for_tier(tier)
        result = chat(model_id, task.prompt, max_tokens=700, temperature=0.2)
        passed = _grade(task, result["text"])
        tier_results[tier] = {
            "passed": passed,
            "total_tokens": result["total_tokens"],
            "model_id": model_id,
        }
        logger.info("  %s -> %s (%d tokens)", tier, "PASS" if passed else "FAIL",
                    result["total_tokens"])
        if passed and tier_label == "none":
            tier_label = tier
            if not all_tiers:
                break
    return LabeledExample(**task.model_dump(), tier_label=tier_label,
                          tier_results=tier_results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-tier empirical labeling.")
    parser.add_argument("--all-tiers", action="store_true",
                        help="grade every tier (more tokens, exact eval)")
    parser.add_argument("--limit", type=int, default=0,
                        help="label at most N new tasks this run (0 = no limit)")
    args = parser.parse_args()

    if not IN_PATH.exists():
        raise SystemExit(f"{IN_PATH} not found — run `python -m data.generate_tasks` first.")

    with open(IN_PATH, "r", encoding="utf-8") as f:
        tasks = [TaskExample(**json.loads(line)) for line in f if line.strip()]

    done = _load_done_ids()
    todo = [t for t in tasks if t.id not in done]
    if args.limit:
        todo = todo[: args.limit]
    logger.info("Total tasks: %d | already labeled: %d | labeling now: %d",
                len(tasks), len(done), len(todo))

    counts: dict[str, int] = {}
    for i, task in enumerate(todo, 1):
        logger.info("[%d/%d] %s (%s/%s)", i, len(todo), task.id,
                    task.category, task.difficulty_pool)
        labeled = label_task(task, args.all_tiers)
        counts[labeled.tier_label] = counts.get(labeled.tier_label, 0) + 1
        # Append immediately so an interrupted run loses at most one task.
        with open(OUT_PATH, "a", encoding="utf-8") as f:
            f.write(labeled.model_dump_json() + "\n")

    logger.info("Done. New labels this run: %s", counts or "(nothing to do)")
    logger.info("Output: %s", OUT_PATH)


if __name__ == "__main__":
    # Human: run this when you're ready — it will consume Fireworks tokens:
    #   python -m data.label_multitier            (early-stop, cheapest)
    #   python -m data.label_multitier --all-tiers  (exact eval data)
    main()
