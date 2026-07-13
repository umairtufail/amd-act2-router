"""Select genuine current-policy tier0 failures from pre-split mining data.

Selection is offline and never calls a model.  Missing, stale, malformed, or
infrastructure-error observations are not hard examples.  The default CLI
fails closed until every candidate has one valid current-policy tier0 result.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from agent.request_policy import request_policy_hash
from data.integrity import (
    assert_disjoint_mining_partitions,
    dataset_hash,
    stable_json_hash,
    validate_mining_partition,
)

MINING_DIR = Path(__file__).parent / "mining"
DEFAULT_TRAIN_LABELED = MINING_DIR / "train_labeled.jsonl"
DEFAULT_STRESS_LABELED = MINING_DIR / "stress_labeled.jsonl"
DEFAULT_OUTPUT_DIR = MINING_DIR / "selected"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_current_policy_tier0_failures(
    records: list[dict],
    *,
    expected_split: str,
    policy_hash: str | None = None,
    require_complete: bool = True,
) -> tuple[list[dict], dict[str, Any]]:
    """Return only valid tier0 quality failures under one exact policy hash."""
    validate_mining_partition(records, expected_split)
    current_hash = policy_hash or request_policy_hash()
    selected: list[dict] = []
    counts = {
        "total": len(records),
        "current_valid": 0,
        "passed": 0,
        "hard_failures": 0,
        "missing_tier0": 0,
        "stale_policy": 0,
        "invalid_observation": 0,
    }

    incomplete_ids: list[str] = []
    for record in records:
        result = record.get("tier_results", {}).get("tier0")
        if not isinstance(result, dict):
            counts["missing_tier0"] += 1
            incomplete_ids.append(str(record["id"]))
            continue
        if result.get("request_policy_hash") != current_hash:
            counts["stale_policy"] += 1
            incomplete_ids.append(str(record["id"]))
            continue

        passed = result.get("passed")
        token_count = result.get("total_tokens")
        finish_reason = str(result.get("finish_reason") or "").strip().lower()
        valid_tokens = (
            isinstance(token_count, (int, float))
            and not isinstance(token_count, bool)
            and token_count > 0
        )
        if (
            not isinstance(passed, bool)
            or not valid_tokens
            or finish_reason == "error"
        ):
            counts["invalid_observation"] += 1
            incomplete_ids.append(str(record["id"]))
            continue

        counts["current_valid"] += 1
        if passed is True:
            counts["passed"] += 1
        else:
            counts["hard_failures"] += 1
            selected.append(dict(record))

    incomplete = (
        counts["missing_tier0"]
        + counts["stale_policy"]
        + counts["invalid_observation"]
    )
    if require_complete and incomplete:
        raise ValueError(
            f"{expected_split} mining labels are incomplete/stale: {counts}; "
            f"example IDs={incomplete_ids[:5]}"
        )

    audit = {
        "dataset_split": expected_split,
        "request_policy_sha256": current_hash,
        "counts": counts,
        "input_dataset_sha256": dataset_hash(records),
        "selected_dataset_sha256": dataset_hash(selected),
        "complete_current_tier0_coverage": incomplete == 0,
    }
    return selected, audit


def confirm_recoverable_failures(
    tier0_failures: list[dict],
    *,
    policy_hash: str,
    require_complete: bool = True,
) -> tuple[list[dict], dict[str, Any]]:
    """Keep only failures with a valid current-policy passing higher arm."""
    selected: list[dict] = []
    counts = {
        "tier0_failures": len(tier0_failures),
        "confirmed_recoverable": 0,
        "confirmed_unrecoverable": 0,
        "incomplete_recovery": 0,
    }
    incomplete_ids: list[str] = []
    for record in tier0_failures:
        valid_results: list[dict] = []
        for tier in ("tier1", "tier2", "tier3"):
            result = record.get("tier_results", {}).get(tier)
            if not isinstance(result, dict):
                continue
            token_count = result.get("total_tokens")
            if (
                result.get("request_policy_hash") == policy_hash
                and isinstance(result.get("passed"), bool)
                and isinstance(token_count, (int, float))
                and not isinstance(token_count, bool)
                and token_count > 0
                and str(result.get("finish_reason") or "").strip().lower()
                != "error"
            ):
                valid_results.append(result)

        passing_tiers = [
            tier
            for tier in ("tier1", "tier2", "tier3")
            if isinstance(record.get("tier_results", {}).get(tier), dict)
            and record["tier_results"][tier] in valid_results
            and record["tier_results"][tier]["passed"] is True
        ]
        if passing_tiers:
            counts["confirmed_recoverable"] += 1
            confirmed = dict(record)
            confirmed["binary_target"] = "needs_strong"
            confirmed["hard_case_confirmed"] = True
            confirmed["recovery_tiers"] = passing_tiers
            confirmed["selection_request_policy_hash"] = policy_hash
            selected.append(confirmed)
        elif len(valid_results) == 3:
            counts["confirmed_unrecoverable"] += 1
        else:
            counts["incomplete_recovery"] += 1
            incomplete_ids.append(str(record["id"]))

    if require_complete and counts["incomplete_recovery"]:
        raise ValueError(
            "tier0 failures need current-policy tier1/tier2/tier3 recovery "
            f"measurements: {counts}; example IDs={incomplete_ids[:5]}"
        )
    return selected, counts


def build_hard_case_selection(
    train_records: list[dict],
    stress_records: list[dict],
    *,
    policy_hash: str | None = None,
    require_complete: bool = True,
    require_recovery: bool = True,
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """Select each preassigned split independently and preserve separation."""
    assert_disjoint_mining_partitions(train_records, stress_records)
    current_hash = policy_hash or request_policy_hash()
    train_hard, train_audit = select_current_policy_tier0_failures(
        train_records,
        expected_split="train",
        policy_hash=current_hash,
        require_complete=require_complete,
    )
    stress_hard, stress_audit = select_current_policy_tier0_failures(
        stress_records,
        expected_split="stress_holdout",
        policy_hash=current_hash,
        require_complete=require_complete,
    )
    recovery_audits = None
    if require_recovery:
        train_hard, train_recovery = confirm_recoverable_failures(
            train_hard,
            policy_hash=current_hash,
            require_complete=require_complete,
        )
        stress_hard, stress_recovery = confirm_recoverable_failures(
            stress_hard,
            policy_hash=current_hash,
            require_complete=require_complete,
        )
        recovery_audits = {
            "train": train_recovery,
            "stress_holdout": stress_recovery,
        }
    # A selection can contain zero records in one split, so validate the source
    # partitions above and then check overlap directly only when both are nonempty.
    if train_hard and stress_hard:
        assert_disjoint_mining_partitions(train_hard, stress_hard)
    manifest: dict[str, Any] = {
        "manifest_version": 1,
        "selection_rule": (
            "current-policy tier0 failed and a higher arm passed"
            if require_recovery
            else "current-policy valid tier0 passed == false (screen only)"
        ),
        "request_policy_sha256": current_hash,
        "split_preserved": True,
        "requires_confirmed_higher_arm_recovery": require_recovery,
        "train": train_audit,
        "stress_holdout": stress_audit,
    }
    if recovery_audits is not None:
        manifest["recovery"] = recovery_audits
    manifest["manifest_sha256"] = stable_json_hash(manifest)
    return train_hard, stress_hard, manifest


def _atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_hard_case_selection(
    output_dir: Path,
    train_hard: list[dict],
    stress_hard: list[dict],
    manifest: dict[str, Any],
    *,
    force: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "train_hard.jsonl",
        output_dir / "stress_hard.jsonl",
        output_dir / "selection_manifest.json",
    ]
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "refusing to overwrite hard-case selection: "
            + ", ".join(str(path) for path in existing)
        )
    _atomic_write_jsonl(paths[0], train_hard)
    _atomic_write_jsonl(paths[1], stress_hard)
    temporary = paths[2].with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(paths[2])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select current-policy tier0 failures without model calls."
    )
    parser.add_argument("--train-labeled", type=Path, default=DEFAULT_TRAIN_LABELED)
    parser.add_argument("--stress-labeled", type=Path, default=DEFAULT_STRESS_LABELED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write only valid current observations while reporting exclusions",
    )
    parser.add_argument(
        "--screen-only",
        action="store_true",
        help="select tier0 failures for second-stage arm collection; final hard "
        "selection requires a current-policy higher-arm pass",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    train_hard, stress_hard, manifest = build_hard_case_selection(
        _read_jsonl(args.train_labeled),
        _read_jsonl(args.stress_labeled),
        require_complete=not args.allow_incomplete,
        require_recovery=not args.screen_only,
    )
    write_hard_case_selection(
        args.output_dir,
        train_hard,
        stress_hard,
        manifest,
        force=args.force,
    )
    print(
        f"Selected {len(train_hard)} train and {len(stress_hard)} stress hard "
        f"groups in {args.output_dir} (no model calls)."
    )


if __name__ == "__main__":
    main()
