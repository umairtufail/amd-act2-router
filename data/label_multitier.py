"""Label each task with the cheapest model tier that passes grading.

For every task in data/tasks_raw.jsonl, models are tried in order
tier0 -> tier1 -> tier2 -> tier3. Each answer is graded (local graders
where possible; LLM judge only for summarization/NER). The first passing
tier becomes the label.

Resumable: task IDs already present in data/labeled_multitier.jsonl are
skipped, and results are appended one line at a time.  ``--fill-missing-tiers``
updates existing rows atomically. ``--refresh-policy`` can remeasure stale
selected arms, while category/tier/group filters keep paid collection targeted.
Use ``--dry-run`` to inspect any operation without calls.

THIS SCRIPT CONSUMES FIREWORKS TOKENS. The human runs it explicitly.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

from agent.fireworks_client import chat
from agent.request_policy import build_answer_request, request_policy_hash
from config import get_model_id_for_tier, get_tier_names
from data.judge import grade_answer
from data.integrity import prompt_group_key
from data.schema import CATEGORIES, LabeledExample, TaskExample

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

IN_PATH = Path(__file__).parent / "tasks_raw.jsonl"
OUT_PATH = Path(__file__).parent / "labeled_multitier.jsonl"
ANSWER_PREVIEW_LEN = 300


def _load_rows(path: Path = OUT_PATH) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_rows_atomic(rows: list[dict], path: Path = OUT_PATH) -> None:
    """Replace the JSONL file without exposing a partially updated dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp_path.replace(path)


def _validate_dataset_boundary(
    tasks: list[TaskExample],
    rows: list[dict],
    *,
    expected_split: str | None,
    output_path: Path,
) -> None:
    """Fail closed when a stress partition could enter training labels."""
    task_splits = {task.dataset_split for task in tasks}
    row_splits = {row.get("dataset_split") for row in rows}
    if expected_split is not None:
        bad_tasks = sorted(
            task.id for task in tasks if task.dataset_split != expected_split
        )
        bad_rows = sorted(
            str(row.get("id", "<missing>"))
            for row in rows
            if row.get("dataset_split") != expected_split
        )
        if bad_tasks or bad_rows:
            preview = ", ".join((bad_tasks + bad_rows)[:5])
            raise SystemExit(
                f"dataset split mismatch: expected {expected_split!r}; "
                f"offending rows: {preview}"
            )
    if "stress_holdout" in task_splits | row_splits:
        if output_path.resolve() == OUT_PATH.resolve():
            raise SystemExit(
                "stress_holdout records cannot be written to the normal "
                "training-label output"
            )


def _validate_unique_task_ids(tasks: list[TaskExample], input_path: Path) -> None:
    task_ids = [task.id for task in tasks]
    if len(task_ids) == len(set(task_ids)):
        return
    duplicates = sorted(
        task_id for task_id in set(task_ids) if task_ids.count(task_id) > 1
    )
    raise SystemExit(f"{input_path} contains duplicate task IDs: {duplicates[:5]}")


def _select_todo(
    tasks: list[TaskExample],
    rows: list[dict],
    selected_tiers: list[str],
    *,
    fill_missing_tiers: bool,
    refresh_policy: bool,
    unique_prompts: bool,
) -> list[TaskExample]:
    """Select work without charging repeatedly for duplicate prompt groups."""
    row_index = {row["id"]: index for index, row in enumerate(rows)}
    if len(row_index) != len(rows):
        raise SystemExit("output contains duplicate task IDs; repair it first")

    if not unique_prompts:
        if fill_missing_tiers:
            return [
                task
                for task in tasks
                if task.id in row_index
                and bool(
                    _tiers_to_measure(
                        rows[row_index[task.id]].get("tier_results", {}),
                        selected_tiers,
                        fill_missing_tiers=True,
                        refresh_policy=refresh_policy,
                    )
                )
            ]
        return [task for task in tasks if task.id not in row_index]

    task_groups: dict[str, list[TaskExample]] = {}
    for task in tasks:
        task_groups.setdefault(prompt_group_key(task.prompt), []).append(task)

    todo: list[TaskExample] = []
    for group_tasks in task_groups.values():
        group_rows = [
            rows[row_index[task.id]]
            for task in group_tasks
            if task.id in row_index
        ]
        if not fill_missing_tiers:
            if not group_rows:
                todo.append(group_tasks[0])
            continue

        missing_tiers = []
        for tier in selected_tiers:
            tier_is_current = any(
                not _tiers_to_measure(
                    row.get("tier_results", {}),
                    [tier],
                    fill_missing_tiers=True,
                    refresh_policy=refresh_policy,
                )
                for row in group_rows
            )
            if not tier_is_current:
                missing_tiers.append(tier)
        if missing_tiers and group_rows:
            todo.append(next(task for task in group_tasks if task.id in row_index))
    return todo


def _tier_label_from_measured(tier_results: dict) -> str:
    """Return a definitive label only after every cheaper arm was measured."""
    for tier in get_tier_names():
        result = tier_results.get(tier)
        if not isinstance(result, dict) or not isinstance(result.get("passed"), bool):
            return "unresolved"
        if result["passed"] is True:
            return tier
    return "none"


def _uniform_request_policy_hash(tier_results: dict) -> str | None:
    hashes = {
        result.get("request_policy_hash")
        for result in tier_results.values()
        if isinstance(result, dict)
    }
    return hashes.pop() if len(hashes) == 1 and None not in hashes else None


def _tiers_to_measure(
    tier_results: dict,
    selected_tiers: list[str],
    *,
    fill_missing_tiers: bool,
    refresh_policy: bool,
) -> list[str]:
    if not fill_missing_tiers:
        return selected_tiers
    current_hash = request_policy_hash()
    return [
        tier
        for tier in selected_tiers
        if tier not in tier_results
        or (
            refresh_policy
            and tier_results.get(tier, {}).get("request_policy_hash")
            != current_hash
        )
    ]


def label_task(
    task: TaskExample,
    all_tiers: bool,
    tier_sleep: float,
    existing: LabeledExample | None = None,
    fill_missing_tiers: bool = False,
    selected_tiers: list[str] | None = None,
    refresh_policy: bool = False,
) -> LabeledExample:
    tier_results: dict = dict(existing.tier_results) if existing else {}
    tiers_to_measure = _tiers_to_measure(
        tier_results,
        selected_tiers or get_tier_names(),
        fill_missing_tiers=fill_missing_tiers,
        refresh_policy=refresh_policy,
    )
    for i, tier in enumerate(tiers_to_measure):
        if i > 0 and tier_sleep > 0:
            time.sleep(tier_sleep)
        model_id = get_model_id_for_tier(tier)
        result = chat(model_id, **build_answer_request(task.prompt, task.category))
        passed = grade_answer(task.category, task.prompt, task.ground_truth, result["text"])
        tier_results[tier] = {
            "passed": passed,
            "total_tokens": result["total_tokens"],
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0),
            "finish_reason": result.get("finish_reason"),
            "model_id": model_id,
            "answer_text": result["text"][:ANSWER_PREVIEW_LEN],
            "request_policy_hash": request_policy_hash(),
        }
        logger.info("  %s -> %s (%d tokens)", tier, "PASS" if passed else "FAIL",
                    result["total_tokens"])
        if passed and not all_tiers and not fill_missing_tiers:
            break
    tier_label = _tier_label_from_measured(tier_results)
    return LabeledExample(
        **task.model_dump(),
        tier_label=tier_label,
        tier_results=tier_results,
        request_policy_hash=_uniform_request_policy_hash(tier_results),
        measurement_status=(
            "partial" if tier_label == "unresolved" else "resolved"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-tier empirical labeling.")
    parser.add_argument(
        "--input",
        type=Path,
        default=IN_PATH,
        help="task JSONL to label (default: data/tasks_raw.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_PATH,
        help="resumable labeled JSONL (default: data/labeled_multitier.jsonl)",
    )
    parser.add_argument(
        "--expected-split",
        choices=("train", "stress_holdout"),
        help="require every input/output row to belong to this preassigned split",
    )
    parser.add_argument("--all-tiers", action="store_true",
                        help="grade every tier (more tokens, exact eval)")
    parser.add_argument(
        "--fill-missing-tiers",
        action="store_true",
        help="atomically update existing rows by measuring only missing arms",
    )
    parser.add_argument(
        "--refresh-policy",
        action="store_true",
        help="with --fill-missing-tiers, also remeasure selected arms whose "
        "saved request-policy hash is stale or missing",
    )
    parser.add_argument(
        "--category",
        action="append",
        choices=CATEGORIES,
        help="process only this category (repeatable)",
    )
    parser.add_argument(
        "--tier",
        action="append",
        choices=get_tier_names(),
        help="measure only this arm (repeatable); useful for tier0-first mining",
    )
    parser.add_argument(
        "--unique-prompts",
        action="store_true",
        help="process one representative per normalized prompt group",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show task/arm calls that would be made, then exit without API calls",
    )
    parser.add_argument("--limit", type=int, default=0,
                        help="label at most N new tasks this run (0 = no limit)")
    parser.add_argument("--until-total", type=int, default=0,
                        help="stop once the output file has this many labeled rows "
                             "(e.g. 176); overrides --limit when both set")
    parser.add_argument("--sleep", type=float, default=2.0,
                        help="seconds to pause between tasks (default 2; helps avoid 429)")
    parser.add_argument("--tier-sleep", type=float, default=1.0,
                        help="seconds to pause between tier attempts within one task")
    args = parser.parse_args()

    if args.fill_missing_tiers and args.until_total:
        parser.error("--until-total cannot be combined with --fill-missing-tiers")
    if args.refresh_policy and not args.fill_missing_tiers:
        parser.error("--refresh-policy requires --fill-missing-tiers")
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if input_path == output_path:
        parser.error("--input and --output must be different files")
    if not input_path.exists():
        raise SystemExit(f"{input_path} not found")

    with input_path.open("r", encoding="utf-8") as f:
        tasks = [TaskExample(**json.loads(line)) for line in f if line.strip()]
    _validate_unique_task_ids(tasks, input_path)

    if args.category:
        selected_categories = set(args.category)
        tasks = [task for task in tasks if task.category in selected_categories]

    rows = _load_rows(output_path)
    _validate_dataset_boundary(
        tasks,
        rows,
        expected_split=args.expected_split,
        output_path=output_path,
    )
    row_index = {row["id"]: index for index, row in enumerate(rows)}
    if len(row_index) != len(rows):
        raise SystemExit(f"{output_path} contains duplicate task IDs; repair it first.")
    done = set(row_index)
    selected_tiers = list(dict.fromkeys(args.tier or get_tier_names()))
    todo = _select_todo(
        tasks,
        rows,
        selected_tiers,
        fill_missing_tiers=args.fill_missing_tiers,
        refresh_policy=args.refresh_policy,
        unique_prompts=args.unique_prompts,
    )

    if args.until_total:
        need = max(args.until_total - len(done), 0)
        if need == 0:
            logger.info("Already at %d labeled (target %d). Nothing to do.",
                        len(done), args.until_total)
            return
        todo = todo[:need]
    elif args.limit:
        todo = todo[: args.limit]
    logger.info("Total tasks: %d | already labeled: %d | processing now: %d%s",
                len(tasks), len(done), len(todo),
                f" (target total {args.until_total})" if args.until_total else "")

    if args.dry_run:
        for task in todo:
            existing_results = (
                rows[row_index[task.id]].get("tier_results", {})
                if task.id in row_index
                else {}
            )
            tiers = _tiers_to_measure(
                existing_results,
                selected_tiers,
                fill_missing_tiers=args.fill_missing_tiers,
                refresh_policy=args.refresh_policy,
            )
            logger.info("DRY RUN %s -> %s", task.id, ", ".join(tiers))
        logger.info("Dry run complete: 0 API calls.")
        return

    counts: dict[str, int] = {}
    for i, task in enumerate(todo, 1):
        if i > 1 and args.sleep > 0:
            time.sleep(args.sleep)
        logger.info("[%d/%d] %s (%s/%s)", i, len(todo), task.id,
                    task.category, task.difficulty_pool)
        existing = (
            LabeledExample(**rows[row_index[task.id]])
            if task.id in row_index
            else None
        )
        labeled = label_task(
            task,
            args.all_tiers,
            args.tier_sleep,
            existing=existing,
            fill_missing_tiers=args.fill_missing_tiers,
            selected_tiers=selected_tiers,
            refresh_policy=args.refresh_policy,
        )
        counts[labeled.tier_label] = counts.get(labeled.tier_label, 0) + 1
        serialized = labeled.model_dump()
        if existing is not None:
            rows[row_index[task.id]] = serialized
            _write_rows_atomic(rows, output_path)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("a", encoding="utf-8") as f:
                f.write(labeled.model_dump_json() + "\n")
            row_index[task.id] = len(rows)
            rows.append(serialized)

    logger.info("Done. New labels this run: %s", counts or "(nothing to do)")
    logger.info("Output: %s", output_path)


if __name__ == "__main__":
    # Human: run when ready — consumes Fireworks tokens:
    #   python -m data.label_multitier --limit 5
    #   python -m data.label_multitier --limit 5 --sleep 5   # slower, fewer 429s
    # Inspect missing-arm work without spending tokens:
    #   python -m data.label_multitier --fill-missing-tiers --refresh-policy \
    #     --category ner --category summarization --tier tier0 --tier tier3 \
    #     --unique-prompts --dry-run
    main()
