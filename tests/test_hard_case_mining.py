"""Offline tests for deterministic, leakage-safe active hard-case mining."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from data.generate_hard_candidates import (
    GENERATOR_VERSION,
    build_candidate_partitions,
    write_candidate_partitions,
)
from data.integrity import (
    assert_disjoint_mining_partitions,
    assert_no_prompt_overlap,
    prompt_family_key,
    prompt_group_key,
    stable_json_hash,
    validate_mining_partition,
)
from data.schema import TaskExample
from data.select_hard_cases import (
    build_hard_case_selection,
    confirm_recoverable_failures,
    select_current_policy_tier0_failures,
    write_hard_case_selection,
)

POLICY_HASH = "a" * 64


def _mining_record(
    record_id: str,
    prompt: str,
    split: str,
    family: str,
    *,
    passed: bool | None = None,
    policy_hash: str = POLICY_HASH,
    total_tokens: int = 10,
    finish_reason: str = "stop",
) -> dict:
    record = TaskExample(
        id=record_id,
        category="ner",
        difficulty_pool="hard",
        prompt=prompt,
        ground_truth='{"persons": [], "organizations": [], "locations": []}',
        dataset_split=split,
        prompt_family_id=family,
        generator_version=GENERATOR_VERSION,
        mining_round=1,
    ).model_dump()
    if passed is not None:
        record["tier_label"] = "tier0" if passed else "none"
        record["tier_results"] = {
            "tier0": {
                "passed": passed,
                "total_tokens": total_tokens,
                "finish_reason": finish_reason,
                "request_policy_hash": policy_hash,
                "model_id": "tier0-model",
            }
        }
    return record


class SchemaAndIntegrityTests(unittest.TestCase):
    def test_legacy_task_schema_remains_compatible(self) -> None:
        task = TaskExample(
            id="legacy",
            category="ner",
            difficulty_pool="hard",
            prompt="legacy prompt",
            ground_truth="truth",
        )
        self.assertIsNone(task.dataset_split)
        self.assertIsNone(task.prompt_family_id)
        mined = TaskExample(
            **{
                **task.model_dump(),
                "dataset_split": "train",
                "prompt_family_id": "family-a",
                "generator_version": "v1",
                "mining_round": 2,
            }
        )
        self.assertEqual(mined.model_dump()["prompt_family_id"], "family-a")

    def test_partition_validation_rejects_duplicate_ids_and_mixed_splits(self) -> None:
        one = _mining_record("one", "prompt one", "train", "family-a")
        duplicate = _mining_record("one", "prompt two", "train", "family-b")
        with self.assertRaisesRegex(ValueError, "duplicate mining record id"):
            validate_mining_partition([one, duplicate], "train")
        wrong_split = _mining_record(
            "stress", "prompt stress", "stress_holdout", "family-c"
        )
        with self.assertRaisesRegex(ValueError, "expected 'train'"):
            validate_mining_partition([wrong_split], "train")

    def test_disjointness_rejects_exact_and_family_leakage(self) -> None:
        train = [_mining_record("train", "same prompt", "train", "family-a")]
        exact_stress = [
            _mining_record(
                "stress", " SAME   prompt ", "stress_holdout", "family-b"
            )
        ]
        with self.assertRaisesRegex(ValueError, "exact-prompt leakage"):
            assert_disjoint_mining_partitions(train, exact_stress)

        family_stress = [
            _mining_record(
                "stress", "different prompt", "stress_holdout", "FAMILY-A"
            )
        ]
        with self.assertRaisesRegex(ValueError, "prompt-family leakage"):
            assert_disjoint_mining_partitions(train, family_stress)

    def test_reference_overlap_is_rejected(self) -> None:
        candidate = _mining_record("candidate", "overlap me", "train", "family-a")
        reference = {"id": "old", "prompt": "  OVERLAP   me "}
        with self.assertRaisesRegex(ValueError, "tasks_raw"):
            assert_no_prompt_overlap(
                [candidate], [reference], reference_name="tasks_raw"
            )


class CandidateGenerationTests(unittest.TestCase):
    def test_generation_is_deterministic_and_family_disjoint(self) -> None:
        first_train, first_stress, first_manifest = build_candidate_partitions(
            seed=41, raw_references=[], holdout_references=[]
        )
        second_train, second_stress, second_manifest = build_candidate_partitions(
            seed=41, raw_references=[], holdout_references=[]
        )
        self.assertEqual(first_train, second_train)
        self.assertEqual(first_stress, second_stress)
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(len(first_train), 24)
        self.assertEqual(len(first_stress), 12)
        assert_disjoint_mining_partitions(first_train, first_stress)
        self.assertFalse(
            {prompt_family_key(row) for row in first_train}
            & {prompt_family_key(row) for row in first_stress}
        )
        unsigned = dict(first_manifest)
        manifest_hash = unsigned.pop("manifest_sha256")
        self.assertEqual(manifest_hash, stable_json_hash(unsigned))

    def test_generation_has_no_overlap_with_current_training_or_holdout(self) -> None:
        train, stress, manifest = build_candidate_partitions(seed=31)
        self.assertEqual(manifest["counts"]["train_records"], len(train))
        self.assertEqual(manifest["counts"]["stress_records"], len(stress))
        all_prompt_keys = {prompt_group_key(row) for row in train + stress}
        raw = [
            json.loads(line)
            for line in (
                Path(__file__).parent.parent / "data" / "tasks_raw.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        holdout = json.loads(
            (Path(__file__).parent.parent / "data" / "holdout_tasks.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(all_prompt_keys & {prompt_group_key(row) for row in raw})
        self.assertFalse(all_prompt_keys & {prompt_group_key(row) for row in holdout})

    def test_artifact_writes_are_atomic_and_no_overwrite_by_default(self) -> None:
        train, stress, manifest = build_candidate_partitions(
            raw_references=[], holdout_references=[]
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_candidate_partitions(output, train, stress, manifest)
            self.assertTrue((output / "train_candidates.jsonl").exists())
            self.assertTrue((output / "stress_candidates.jsonl").exists())
            saved_manifest = json.loads(
                (output / "split_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_manifest, manifest)
            with self.assertRaises(FileExistsError):
                write_candidate_partitions(output, train, stress, manifest)


class HardCaseSelectionTests(unittest.TestCase):
    def test_selector_uses_only_valid_current_policy_failures(self) -> None:
        records = [
            _mining_record("hard", "hard", "train", "family-a", passed=False),
            _mining_record("pass", "pass", "train", "family-b", passed=True),
            _mining_record(
                "stale",
                "stale",
                "train",
                "family-c",
                passed=False,
                policy_hash="b" * 64,
            ),
            _mining_record("missing", "missing", "train", "family-d"),
            _mining_record(
                "error",
                "error",
                "train",
                "family-e",
                passed=False,
                total_tokens=0,
                finish_reason="error",
            ),
        ]
        selected, audit = select_current_policy_tier0_failures(
            records,
            expected_split="train",
            policy_hash=POLICY_HASH,
            require_complete=False,
        )
        self.assertEqual([row["id"] for row in selected], ["hard"])
        self.assertEqual(
            audit["counts"],
            {
                "total": 5,
                "current_valid": 2,
                "passed": 1,
                "hard_failures": 1,
                "missing_tier0": 1,
                "stale_policy": 1,
                "invalid_observation": 1,
            },
        )
        self.assertFalse(audit["complete_current_tier0_coverage"])

    def test_selector_fails_closed_on_incomplete_or_stale_labels(self) -> None:
        records = [
            _mining_record("hard", "hard", "train", "family-a", passed=False),
            _mining_record("missing", "missing", "train", "family-b"),
        ]
        with self.assertRaisesRegex(ValueError, "incomplete/stale"):
            select_current_policy_tier0_failures(
                records,
                expected_split="train",
                policy_hash=POLICY_HASH,
            )

    def test_train_and_stress_selection_preserves_preassigned_split(self) -> None:
        train = [
            _mining_record("train-hard", "train hard", "train", "train-family", passed=False),
            _mining_record("train-pass", "train pass", "train", "train-pass-family", passed=True),
        ]
        stress = [
            _mining_record("stress-hard", "stress hard", "stress_holdout", "stress-family", passed=False),
            _mining_record("stress-pass", "stress pass", "stress_holdout", "stress-pass-family", passed=True),
        ]
        train[0]["tier_results"]["tier3"] = {
            "passed": True,
            "total_tokens": 20,
            "finish_reason": "stop",
            "request_policy_hash": POLICY_HASH,
        }
        stress[0]["tier_results"]["tier3"] = {
            "passed": True,
            "total_tokens": 20,
            "finish_reason": "stop",
            "request_policy_hash": POLICY_HASH,
        }
        train_hard, stress_hard, manifest = build_hard_case_selection(
            train, stress, policy_hash=POLICY_HASH
        )
        self.assertEqual([row["id"] for row in train_hard], ["train-hard"])
        self.assertEqual([row["id"] for row in stress_hard], ["stress-hard"])
        self.assertEqual(train_hard[0]["dataset_split"], "train")
        self.assertEqual(stress_hard[0]["dataset_split"], "stress_holdout")
        self.assertTrue(manifest["split_preserved"])
        self.assertTrue(manifest["requires_confirmed_higher_arm_recovery"])

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_hard_case_selection(
                output, train_hard, stress_hard, manifest
            )
            self.assertEqual(
                json.loads(
                    (output / "selection_manifest.json").read_text(encoding="utf-8")
                ),
                manifest,
            )
            with self.assertRaises(FileExistsError):
                write_hard_case_selection(
                    output, train_hard, stress_hard, manifest
                )

    def test_recovery_confirmation_fails_closed_until_higher_arms_are_measured(self) -> None:
        failure = _mining_record(
            "needs-recovery", "needs recovery", "train", "family-a", passed=False
        )
        with self.assertRaisesRegex(ValueError, "need current-policy"):
            confirm_recoverable_failures(
                [failure], policy_hash=POLICY_HASH
            )

        failure["tier_results"]["tier2"] = {
            "passed": True,
            "total_tokens": 15,
            "finish_reason": "stop",
            "request_policy_hash": POLICY_HASH,
        }
        selected, counts = confirm_recoverable_failures(
            [failure], policy_hash=POLICY_HASH
        )
        self.assertEqual([row["id"] for row in selected], ["needs-recovery"])
        self.assertEqual(counts["confirmed_recoverable"], 1)
        self.assertEqual(selected[0]["binary_target"], "needs_strong")
        self.assertTrue(selected[0]["hard_case_confirmed"])
        self.assertEqual(selected[0]["recovery_tiers"], ["tier2"])


if __name__ == "__main__":
    unittest.main()
