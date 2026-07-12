"""Shared data schemas for the dataset pipeline."""

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


class LabeledExample(TaskExample):
    """A task after multi-tier labeling."""

    tier_label: str  # cheapest tier that passed, or "none" if all failed
    # Per-tier: passed, total_tokens, model_id, answer_text (preview for debugging)
    tier_results: dict
