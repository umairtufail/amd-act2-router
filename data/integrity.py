"""Deterministic dataset identities and prompt-group utilities.

Router metrics are only meaningful when repeated copies of the same prompt do
not cross the train/validation boundary.  This module deliberately stays free
of ML dependencies so labeling, training, and evaluation can share one notion
of prompt identity and result freshness.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, TypeVar

T = TypeVar("T", bound=Mapping[str, Any])
_WHITESPACE = re.compile(r"\s+")


def canonical_prompt(prompt: str) -> str:
    """Normalize harmless textual differences before grouping prompts."""
    if not isinstance(prompt, str):
        raise TypeError("prompt must be text")
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", prompt)).strip().casefold()


def prompt_group_key(record_or_prompt: Mapping[str, Any] | str) -> str:
    """Return a compact, stable key for an exact normalized prompt group."""
    prompt = (
        record_or_prompt
        if isinstance(record_or_prompt, str)
        else record_or_prompt.get("prompt")
    )
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("record prompt must be non-empty text")
    return hashlib.sha256(canonical_prompt(prompt).encode("utf-8")).hexdigest()


def stable_json_hash(value: Any) -> str:
    """Hash JSON-compatible data without depending on whitespace or key order."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dataset_hash(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash the ordered logical dataset used by an evaluation or training run."""
    return stable_json_hash(list(records))


def group_by_prompt(records: Iterable[T]) -> dict[str, list[T]]:
    """Group rows by normalized prompt, preserving first-seen group order."""
    groups: dict[str, list[T]] = defaultdict(list)
    for record in records:
        groups[prompt_group_key(record)].append(record)
    return dict(groups)


def collapse_prompt_groups(
    records: Sequence[T],
    resolve: Callable[[list[T]], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse duplicate prompt rows with a caller-provided label resolver.

    Every duplicate must describe the same category and ground truth.  A
    mismatch is more likely a data bug than a valid repeated observation, so it
    fails closed instead of silently joining unrelated training examples.
    """
    collapsed: list[dict[str, Any]] = []
    for key, group in group_by_prompt(records).items():
        identities = {
            (row.get("category"), row.get("ground_truth")) for row in group
        }
        if len(identities) != 1:
            ids = [str(row.get("id", "<missing>")) for row in group]
            raise ValueError(
                "same prompt has conflicting category/ground truth: " + ", ".join(ids)
            )
        representative = dict(group[0])
        representative.update(resolve(group))
        representative["_prompt_group_key"] = key
        representative["_prompt_group_size"] = len(group)
        collapsed.append(representative)
    return collapsed


def assert_disjoint_prompt_groups(
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
) -> None:
    """Raise when an exact normalized prompt appears in both splits."""
    train_keys = {prompt_group_key(record) for record in train_records}
    validation_keys = {prompt_group_key(record) for record in validation_records}
    overlap = train_keys & validation_keys
    if overlap:
        raise AssertionError(
            f"prompt leakage across train/validation split ({len(overlap)} groups)"
        )


def prompt_family_key(record: Mapping[str, Any]) -> str:
    """Return the normalized explicit template/paraphrase family identifier."""
    value = record.get("prompt_family_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"record {record.get('id', '<missing>')!r} needs prompt_family_id"
        )
    return unicodedata.normalize("NFKC", value).strip().casefold()


def validate_mining_partition(
    records: Sequence[Mapping[str, Any]],
    expected_split: str,
) -> None:
    """Validate IDs, exact prompts, families, and one declared mining split."""
    if expected_split not in {"train", "stress_holdout"}:
        raise ValueError("expected_split must be 'train' or 'stress_holdout'")
    if not records:
        raise ValueError(f"{expected_split} mining partition must not be empty")

    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    family_categories: dict[str, Any] = {}
    for index, record in enumerate(records):
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"mining record at index {index} needs a non-empty id")
        if record_id in seen_ids:
            raise ValueError(f"duplicate mining record id: {record_id!r}")
        seen_ids.add(record_id)

        actual_split = record.get("dataset_split")
        if actual_split != expected_split:
            raise ValueError(
                f"record {record_id!r} has split {actual_split!r}; "
                f"expected {expected_split!r}"
            )

        prompt_key = prompt_group_key(record)
        if prompt_key in seen_prompts:
            raise ValueError(
                f"duplicate normalized prompt inside {expected_split}: {record_id!r}"
            )
        seen_prompts.add(prompt_key)

        family = prompt_family_key(record)
        category = record.get("category")
        if family in family_categories and family_categories[family] != category:
            raise ValueError(
                f"prompt family {family!r} spans multiple categories"
            )
        family_categories[family] = category


def assert_disjoint_mining_partitions(
    train_records: Sequence[Mapping[str, Any]],
    stress_records: Sequence[Mapping[str, Any]],
) -> None:
    """Reject both exact-prompt and template/paraphrase-family leakage."""
    validate_mining_partition(train_records, "train")
    validate_mining_partition(stress_records, "stress_holdout")

    prompt_overlap = {
        prompt_group_key(record) for record in train_records
    }.intersection(prompt_group_key(record) for record in stress_records)
    if prompt_overlap:
        raise ValueError(
            f"train/stress exact-prompt leakage ({len(prompt_overlap)} groups)"
        )

    family_overlap = {
        prompt_family_key(record) for record in train_records
    }.intersection(prompt_family_key(record) for record in stress_records)
    if family_overlap:
        preview = sorted(family_overlap)[:5]
        raise ValueError(
            "train/stress prompt-family leakage "
            f"({len(family_overlap)} families): {preview}"
        )


def assert_no_prompt_overlap(
    candidates: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
    *,
    reference_name: str,
) -> None:
    """Reject exact normalized prompt reuse against an existing dataset."""
    reference_keys = {prompt_group_key(record) for record in references}
    overlap = {
        prompt_group_key(record) for record in candidates
    }.intersection(reference_keys)
    if overlap:
        raise ValueError(
            f"candidate prompts overlap {reference_name} ({len(overlap)} groups)"
        )
