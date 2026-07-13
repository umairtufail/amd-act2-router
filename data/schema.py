"""Shared data schemas for the dataset pipeline."""

from typing import Literal

from pydantic import BaseModel

CATEGORIES = [
    "math_reasoning",
    "logic_puzzles",
    "sentiment",
    "summarization",
    "ner",
    "code_debugging",
    "code_generation",
    "factual_knowledge",
]

DIFFICULTY_POOLS = ["trivial", "medium", "hard", "adversarial"]

# Categories whose ground_truth is a JSON spec graded by executing real tests.
CODE_CATEGORIES = {"code_generation"}


class TaskExample(BaseModel):
    id: str
    category: str
    difficulty_pool: str  # trivial | medium | hard | adversarial
    prompt: str
    ground_truth: str  # plain text, or JSON string for code_generation specs
    # Optional provenance for active hard-case mining.  Defaults keep every
    # existing JSONL row backwards compatible while allowing mined examples to
    # carry a split and a paraphrase/template-family boundary through labeling.
    dataset_split: Literal["train", "stress_holdout"] | None = None
    prompt_family_id: str | None = None
    generator_version: str | None = None
    mining_round: int | None = None


class LabeledExample(TaskExample):
    """A task after multi-tier labeling."""

    tier_label: str  # cheapest passing tier, "none", or "unresolved" while partial
    # Per-tier: passed, total_tokens, model_id, answer_text (preview for debugging)
    tier_results: dict
    # Present only when every recorded tier outcome used the exact same shared
    # answer request policy.  Legacy or mixed-policy rows remain ``None``.
    request_policy_hash: str | None = None
    measurement_status: Literal["partial", "resolved"] | None = None
