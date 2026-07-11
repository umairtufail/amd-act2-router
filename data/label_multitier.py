"""Label each task with the cheapest model tier that passes grading.

For every task in data/tasks_raw.jsonl, models are tried in order
tier0 -> tier1 -> tier2 -> tier3. Each answer is graded (local graders
where possible; LLM judge only for summarization/NER). The first passing
tier becomes the label.

Resumable: task IDs already present in data/labeled_multitier.jsonl are
skipped, and results are appended one line at a time.

To re-label a bad row, remove its line from labeled_multitier.jsonl first.

THIS SCRIPT CONSUMES FIREWORKS TOKENS. The human runs it explicitly.
"""

import argparse
import json
import logging
import time
from pathlib import Path

from agent.fireworks_client import chat
from config import get_model_id_for_tier, get_tier_names
from data.judge import grade_answer
from data.schema import LabeledExample, TaskExample

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

IN_PATH = Path(__file__).parent / "tasks_raw.jsonl"
OUT_PATH = Path(__file__).parent / "labeled_multitier.jsonl"
ANSWER_PREVIEW_LEN = 300


def _load_done_ids() -> set[str]:
    if not OUT_PATH.exists():
        return set()
    with open(OUT_PATH, "r", encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def label_task(
    task: TaskExample,
    all_tiers: bool,
    tier_sleep: float,
) -> LabeledExample:
    tier_results: dict = {}
    tier_label = "none"
    tiers = get_tier_names()
    for i, tier in enumerate(tiers):
        if i > 0 and tier_sleep > 0:
            time.sleep(tier_sleep)
        model_id = get_model_id_for_tier(tier)
        result = chat(model_id, task.prompt, max_tokens=700, temperature=0.2)
        passed = grade_answer(task.category, task.prompt, task.ground_truth, result["text"])
        tier_results[tier] = {
            "passed": passed,
            "total_tokens": result["total_tokens"],
            "model_id": model_id,
            "answer_text": result["text"][:ANSWER_PREVIEW_LEN],
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
    parser.add_argument("--sleep", type=float, default=2.0,
                        help="seconds to pause between tasks (default 2; helps avoid 429)")
    parser.add_argument("--tier-sleep", type=float, default=1.0,
                        help="seconds to pause between tier attempts within one task")
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
        if i > 1 and args.sleep > 0:
            time.sleep(args.sleep)
        logger.info("[%d/%d] %s (%s/%s)", i, len(todo), task.id,
                    task.category, task.difficulty_pool)
        labeled = label_task(task, args.all_tiers, args.tier_sleep)
        counts[labeled.tier_label] = counts.get(labeled.tier_label, 0) + 1
        with open(OUT_PATH, "a", encoding="utf-8") as f:
            f.write(labeled.model_dump_json() + "\n")

    logger.info("Done. New labels this run: %s", counts or "(nothing to do)")
    logger.info("Output: %s", OUT_PATH)


if __name__ == "__main__":
    # Human: run when ready — consumes Fireworks tokens:
    #   python -m data.label_multitier --limit 5
    #   python -m data.label_multitier --limit 5 --sleep 5   # slower, fewer 429s
    # Re-label a bad row: delete its line from labeled_multitier.jsonl, then re-run.
    main()
