"""Shared labels and category rules for binary routing."""

CHEAP_OK_LABEL = "cheap_ok"
NEEDS_STRONG_LABEL = "needs_strong"


def tier_label_to_binary(tier_label: str) -> str:
    """Map the cheapest passing tier to the binary routing target."""
    return CHEAP_OK_LABEL if tier_label == "tier0" else NEEDS_STRONG_LABEL


EASY_CATEGORIES = {
    "factual_knowledge",
    "math_reasoning",
    "logic_puzzles",
    "code_debugging",
    "code_generation",
    "sentiment",
}

ROUTER_CATEGORIES = {"ner", "summarization"}
