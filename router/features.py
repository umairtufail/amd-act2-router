"""Feature extraction for the multi-tier router.

The classifier sees the prompt text (encoded by DistilBERT) plus a small
vector of cheap numeric/structural features. Difficulty is deliberately NOT
a feature: it exists only in our generated dataset and won't be present in
the hackathon's real /input/tasks.json. Category is optional for the same
reason — an "unknown" bucket covers tasks that arrive without one.
"""

from __future__ import annotations

import re

from data.schema import CATEGORIES

# Fixed category vocabulary; index len(CATEGORIES) is the "unknown" bucket.
CATEGORY_TO_INDEX = {c: i for i, c in enumerate(CATEGORIES)}
UNKNOWN_CATEGORY_INDEX = len(CATEGORIES)
NUM_CATEGORIES = len(CATEGORIES) + 1

_CODE_FENCE_RE = re.compile(r"```")
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*]|\d+\.)\s", re.MULTILINE)

# Numeric features, in order. Scales keep values roughly in [0, ~few].
NUMERIC_FEATURE_NAMES = [
    "char_len", "word_count", "line_count", "question_marks",
    "digit_ratio", "code_fences", "list_markers",
]
NUM_NUMERIC_FEATURES = len(NUMERIC_FEATURE_NAMES)


def category_index(category: str | None) -> int:
    return CATEGORY_TO_INDEX.get(category or "", UNKNOWN_CATEGORY_INDEX)


def numeric_features(prompt: str) -> list[float]:
    n_chars = len(prompt)
    digits = sum(ch.isdigit() for ch in prompt)
    return [
        n_chars / 1000.0,
        len(prompt.split()) / 200.0,
        (prompt.count("\n") + 1) / 20.0,
        prompt.count("?") / 5.0,
        digits / max(n_chars, 1),
        len(_CODE_FENCE_RE.findall(prompt)) / 2.0,
        len(_LIST_MARKER_RE.findall(prompt)) / 10.0,
    ]


def extract_features(prompt: str, category: str | None = None) -> dict:
    """Everything the model needs besides tokenized text."""
    return {
        "text": prompt,
        "numeric": numeric_features(prompt),
        "category_index": category_index(category),
    }
