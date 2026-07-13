"""Offline regression tests for leakage, unknown outcomes, and frontiers."""

from __future__ import annotations

import unittest

import numpy as np

from agent.request_policy import request_policy_hash
from data.label_multitier import _tiers_to_measure
from data.integrity import (
    assert_disjoint_prompt_groups,
    canonical_prompt,
    dataset_hash,
    prompt_group_key,
    stable_json_hash,
)
from eval.evaluate_strategies import collapse_replay_records, simulate
from eval.offline_frontier import build_frontier_analysis, wilson_lower_bound
from router.train_binary_router import (
    MIN_HARD_GROUPS_FOR_PROMOTION,
    classification_metrics,
    collapse_binary_records,
    constant_cheap_confusion,
    promotion_recommended,
    stratified_split,
)


def _row(
    row_id: str,
    prompt: str,
    *,
    tier0_passed: bool,
    tier3_passed: bool | None = None,
    category: str = "ner",
) -> dict:
    results = {
        "tier0": {
            "passed": tier0_passed,
            "total_tokens": 10,
            "model_id": "model-0",
        }
    }
    if tier3_passed is not None:
        results["tier3"] = {
            "passed": tier3_passed,
            "total_tokens": 20,
            "model_id": "model-3",
        }
    return {
        "id": row_id,
        "category": category,
        "difficulty_pool": "hard",
        "prompt": prompt,
        "ground_truth": "answer",
        "tier_label": "tier0" if tier0_passed else "tier3",
        "tier_results": results,
    }


class IntegrityTests(unittest.TestCase):
    def test_targeted_refresh_selects_only_missing_or_stale_arms(self) -> None:
        current = request_policy_hash()
        results = {
            "tier0": {"passed": True},
            "tier1": {"passed": True, "request_policy_hash": current},
        }
        self.assertEqual(
            _tiers_to_measure(
                results,
                ["tier0", "tier1", "tier3"],
                fill_missing_tiers=True,
                refresh_policy=True,
            ),
            ["tier0", "tier3"],
        )
        self.assertEqual(
            _tiers_to_measure(
                results,
                ["tier0", "tier1", "tier3"],
                fill_missing_tiers=True,
                refresh_policy=False,
            ),
            ["tier3"],
        )

    def test_prompt_identity_normalizes_case_unicode_and_whitespace(self) -> None:
        left = "  Caf\u00e9\n  TEST "
        right = "cafe\u0301 test"
        self.assertEqual(canonical_prompt(left), canonical_prompt(right))
        self.assertEqual(prompt_group_key(left), prompt_group_key(right))

    def test_stable_hash_ignores_mapping_key_order(self) -> None:
        self.assertEqual(stable_json_hash({"b": 2, "a": 1}), stable_json_hash({"a": 1, "b": 2}))
        self.assertNotEqual(dataset_hash([{"a": 1}, {"a": 2}]), dataset_hash([{"a": 2}, {"a": 1}]))

    def test_binary_duplicates_collapse_conservatively(self) -> None:
        records = [
            _row("a", "same prompt", tier0_passed=True),
            _row("b", " SAME   prompt ", tier0_passed=False),
        ]
        collapsed = collapse_binary_records(records)
        self.assertEqual(len(collapsed), 1)
        self.assertEqual(collapsed[0]["tier_label"], "tier3")
        self.assertEqual(collapsed[0]["_prompt_group_size"], 2)

    def test_group_split_has_no_prompt_leakage_and_retains_hard_validation(self) -> None:
        records = []
        for index in range(12):
            records.append(
                _row(f"easy-{index}", f"easy {index}", tier0_passed=True)
            )
        for index in range(3):
            records.extend(
                [
                    _row(f"hard-{index}-a", f"hard {index}", tier0_passed=False),
                    _row(f"hard-{index}-b", f" HARD   {index} ", tier0_passed=False),
                ]
            )
        train, validation = stratified_split(records, 0.25, seed=7)
        assert_disjoint_prompt_groups(train, validation)
        self.assertEqual(len(train) + len(validation), 15)
        self.assertTrue(any(row["tier_label"] == "tier3" for row in train))
        self.assertTrue(any(row["tier_label"] == "tier3" for row in validation))

    def test_hard_class_metrics_and_constant_baseline_are_explicit(self) -> None:
        confusion = np.array([[8, 2], [1, 3]])
        metrics = classification_metrics(confusion)
        self.assertAlmostEqual(metrics["hard_precision"], 3 / 5)
        self.assertAlmostEqual(metrics["hard_recall"], 3 / 4)
        records = [
            _row("easy", "easy", tier0_passed=True),
            _row("hard", "hard", tier0_passed=False),
        ]
        self.assertEqual(constant_cheap_confusion(records).tolist(), [[1, 0], [1, 0]])

    def test_checkpoint_promotion_requires_hard_class_value(self) -> None:
        baseline = classification_metrics(np.array([[9, 0], [1, 0]]))
        predicts_only_cheap = classification_metrics(np.array([[9, 0], [1, 0]]))
        useful_candidate = classification_metrics(np.array([[9, 0], [0, 1]]))
        self.assertFalse(
            promotion_recommended(
                predicts_only_cheap,
                baseline,
                hard_group_count=MIN_HARD_GROUPS_FOR_PROMOTION,
            )
        )
        self.assertFalse(
            promotion_recommended(useful_candidate, baseline, hard_group_count=1)
        )
        self.assertTrue(
            promotion_recommended(
                useful_candidate,
                baseline,
                hard_group_count=MIN_HARD_GROUPS_FOR_PROMOTION,
            )
        )


class MeasuredReplayTests(unittest.TestCase):
    def test_policy_filter_excludes_legacy_measurements(self) -> None:
        current = request_policy_hash()
        fresh = _row("fresh", "same", tier0_passed=True)
        fresh["tier_results"]["tier0"]["request_policy_hash"] = current
        legacy = _row("legacy", " SAME ", tier0_passed=False)
        records = collapse_replay_records(
            [fresh, legacy], required_request_policy_hash=current
        )
        result = simulate(records, lambda _record: "tier0")
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["measured_outcomes"], 1)

    def test_unmeasured_arm_is_unknown_not_assumed(self) -> None:
        records = collapse_replay_records(
            [_row("one", "one prompt", tier0_passed=True)]
        )
        result = simulate(records, lambda _record: "tier3")
        self.assertIsNone(result["accuracy"])
        self.assertIsNone(result["total_tokens"])
        self.assertEqual(result["unknown_outcomes"], 1)
        self.assertEqual(result["assumed_outcomes"], 0)

    def test_duplicate_calls_become_one_group_with_empirical_rate(self) -> None:
        records = collapse_replay_records(
            [
                _row("pass", "same", tier0_passed=True),
                _row("fail", " SAME ", tier0_passed=False),
            ]
        )
        result = simulate(records, lambda _record: "tier0")
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["total_tokens"], 10)

    def test_frontier_treats_models_as_independent_raw_token_arms(self) -> None:
        source = [
            _row("a", "category a", tier0_passed=True, tier3_passed=True, category="a"),
            _row("b", "category b", tier0_passed=False, tier3_passed=True, category="b"),
        ]
        records = collapse_replay_records(source)
        frontier = build_frontier_analysis(records, ["tier0", "tier3"], simulate)
        target = frontier["target_policies"]["0.98"]
        self.assertTrue(target["complete"])
        self.assertEqual(target["policy"], {"a": "tier0", "b": "tier3"})
        self.assertEqual(target["evaluation"]["total_tokens"], 30)
        self.assertTrue(frontier["independent_model_arms"])

    def test_wilson_bound_exposes_small_sample_uncertainty(self) -> None:
        self.assertLess(wilson_lower_bound(9, 9), 0.98)


if __name__ == "__main__":
    unittest.main()
