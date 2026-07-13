from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.label_multitier import (
    OUT_PATH,
    _select_todo,
    _tier_label_from_measured,
    _validate_dataset_boundary,
    _validate_unique_task_ids,
)
from data.schema import TaskExample
from router.train_binary_router import (
    CKPT_DIR,
    _validate_candidate_dir,
    load_records,
    stratified_split,
)


def _task(task_id: str, *, split: str = "train", prompt: str = "Same prompt"):
    return TaskExample(
        id=task_id,
        category="sentiment",
        difficulty_pool="adversarial",
        prompt=prompt,
        ground_truth="neutral",
        dataset_split=split,
        prompt_family_id="sentiment.mixed.01",
        generator_version="hard-cases-v1",
        mining_round=1,
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_unique_prompt_refresh_does_not_advance_to_duplicate_rows():
    tasks = [_task("a"), _task("b")]
    rows = [
        {
            **tasks[0].model_dump(),
            "tier_label": "tier0",
            "tier_results": {
                "tier1": {"passed": True},
                "tier2": {"passed": True},
            },
        }
    ]

    todo = _select_todo(
        tasks,
        rows,
        ["tier1", "tier2"],
        fill_missing_tiers=True,
        refresh_policy=False,
        unique_prompts=True,
    )

    assert todo == []


def test_stress_rows_cannot_target_normal_training_output():
    with pytest.raises(SystemExit, match="stress_holdout"):
        _validate_dataset_boundary(
            [_task("stress", split="stress_holdout")],
            [],
            expected_split="stress_holdout",
            output_path=OUT_PATH,
        )


def test_expected_split_is_enforced(tmp_path: Path):
    with pytest.raises(SystemExit, match="dataset split mismatch"):
        _validate_dataset_boundary(
            [_task("stress", split="stress_holdout")],
            [],
            expected_split="train",
            output_path=tmp_path / "labels.jsonl",
        )


def test_duplicate_input_ids_are_rejected(tmp_path: Path):
    with pytest.raises(SystemExit, match="duplicate task IDs"):
        _validate_unique_task_ids([_task("same"), _task("same")], tmp_path / "in")


def test_partial_arm_results_are_not_given_definitive_labels():
    assert _tier_label_from_measured({"tier0": {"passed": False}}) == "unresolved"
    assert _tier_label_from_measured({"tier0": {"passed": True}}) == "tier0"
    assert (
        _tier_label_from_measured(
            {
                "tier0": {"passed": False},
                "tier1": {"passed": False},
                "tier2": {"passed": False},
                "tier3": {"passed": False},
            }
        )
        == "none"
    )


def test_binary_loader_combines_train_sources(tmp_path: Path):
    first = tmp_path / "legacy.jsonl"
    second = tmp_path / "mined.jsonl"
    _write_jsonl(
        first,
        [{"id": "first", "tier_label": "tier0", "dataset_split": "train"}],
    )
    _write_jsonl(
        second,
        [{"id": "mined", "tier_label": "tier3", "dataset_split": "train"}],
    )

    records = load_records([first, second])

    assert [record["id"] for record in records] == ["first", "mined"]


def test_binary_loader_rejects_custom_sources_without_train_provenance(tmp_path: Path):
    path = tmp_path / "ambiguous.jsonl"
    _write_jsonl(path, [{"id": "unknown", "tier_label": "tier0"}])
    with pytest.raises(SystemExit, match="without dataset_split='train'"):
        load_records([path])


def test_binary_loader_rejects_stress_rows(tmp_path: Path):
    path = tmp_path / "stress.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "never-train",
                "tier_label": "tier3",
                "dataset_split": "stress_holdout",
            }
        ],
    )

    with pytest.raises(SystemExit, match="refusing stress_holdout"):
        load_records([path])


def test_binary_loader_accepts_only_evidenced_partial_hard_rows(tmp_path: Path):
    policy_hash = "a" * 64
    path = tmp_path / "confirmed.jsonl"
    record = {
        "id": "confirmed",
        "dataset_split": "train",
        "tier_label": "unresolved",
        "measurement_status": "partial",
        "binary_target": "needs_strong",
        "hard_case_confirmed": True,
        "selection_request_policy_hash": policy_hash,
        "tier_results": {
            "tier0": {
                "passed": False,
                "total_tokens": 10,
                "finish_reason": "stop",
                "request_policy_hash": policy_hash,
            },
            "tier3": {
                "passed": True,
                "total_tokens": 20,
                "finish_reason": "stop",
                "request_policy_hash": policy_hash,
            },
        },
    }
    _write_jsonl(path, [record])
    assert load_records([path])[0]["id"] == "confirmed"

    record["tier_results"]["tier3"]["passed"] = False
    _write_jsonl(path, [record])
    with pytest.raises(SystemExit, match="partially measured"):
        load_records([path])


def test_family_aware_split_keeps_paraphrases_together():
    records = []
    for family_index in range(4):
        for example_index in range(2):
            records.append(
                {
                    "id": f"{family_index}-{example_index}",
                    "prompt": f"unique prompt {family_index} {example_index}",
                    "category": "ner",
                    "ground_truth": "truth",
                    "tier_label": "tier3" if family_index == 0 else "tier0",
                    "prompt_family_id": f"family-{family_index}",
                }
            )
    train, validation = stratified_split(records, 0.5, seed=3)
    train_families = {row["prompt_family_id"] for row in train}
    validation_families = {row["prompt_family_id"] for row in validation}
    assert not (train_families & validation_families)


def test_candidate_directory_cannot_overlap_active_checkpoint_tree():
    with pytest.raises(ValueError, match="overlaps active"):
        _validate_candidate_dir(CKPT_DIR)
