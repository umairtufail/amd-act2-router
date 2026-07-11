"""Grading helpers used during dataset labeling.

Grading strategy by category (cheapest reliable method first):
- code_generation  → execute real tests (local, free)
- math_reasoning   → extract numeric answer, compare locally (free)
- logic_puzzles    → match ground-truth name locally (free)
- sentiment        → match classification label locally (free)
- code_debugging   → match expected program output locally (free)
- summarization, ner → LLM judge (MODEL_JUDGE; consumes tokens)

The LLM judge uses a high max_tokens budget so reasoning models can finish
thinking and still emit {"correct": true/false}.
"""

from __future__ import annotations

import json
import logging
import re

from agent.fireworks_client import chat
from config import get_judge_model_id
from data.code_exec import extract_code, run_tests

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are grading an AI model's answer against a known ground truth.

Question:
{prompt}

Ground truth (correct answer or required key facts):
{ground_truth}

Model's answer:
{answer}

Is the model's answer correct? Judge semantic correctness, not wording:
summaries/extractions must contain the ground truth facts without contradictions.

Respond with ONLY a JSON object on its own, no other text:
{{"correct": true}} or {{"correct": false}}"""

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_CORRECT_RE = re.compile(r'"correct"\s*:\s*(true|false)', re.IGNORECASE)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
# Default judge budget; override via JUDGE_MAX_TOKENS env var.
JUDGE_MAX_TOKENS = 700


def _parse_verdict(text: str) -> bool:
    """Extract correct=true/false from judge output."""
    match = _JSON_RE.search(text)
    if match:
        try:
            return bool(json.loads(match.group(0)).get("correct", False))
        except json.JSONDecodeError:
            pass
    # Fallback: reasoning models sometimes emit bare "correct": true without braces.
    m = _CORRECT_RE.search(text)
    if m:
        return m.group(1).lower() == "true"
    logger.warning("Judge returned no parseable verdict, treating as FAIL: %r", text[:200])
    return False


def _first_token(text: str) -> str:
    text = text.strip().strip("\"'.,;:")
    return text.split()[0] if text.split() else ""


def grade_math_answer(ground_truth: str, answer: str) -> bool:
    """Numeric ground truths — compare extracted number, no LLM judge."""
    try:
        expected = float(ground_truth.strip())
    except ValueError:
        logger.warning("math ground_truth not numeric: %r", ground_truth)
        return False
    nums = _NUM_RE.findall(answer.replace(",", ""))
    if not nums:
        return False
    try:
        got = float(nums[-1])  # models often reason, then give the final number
    except ValueError:
        return False
    return abs(got - expected) < 0.011


def grade_logic_answer(ground_truth: str, answer: str) -> bool:
    """Single-name answers — exact match after normalization."""
    expected = ground_truth.strip().lower()
    got = _first_token(answer).lower()
    if got == expected:
        return True
    # Allow "The answer is Kai" style responses.
    return bool(re.search(rf"\b{re.escape(expected)}\b", answer.lower()))


def grade_sentiment_answer(ground_truth: str, answer: str) -> bool:
    """positive / negative / neutral — match the label word."""
    label = ground_truth.strip().lower()
    ans = answer.strip().lower()
    first = _first_token(ans).lower()
    return first == label or ans == label


def grade_code_debugging_answer(ground_truth: str, answer: str) -> bool:
    """Expected stdout from a program — match exactly (last line if multi-line)."""
    expected = ground_truth.strip()
    text = extract_code(answer) if "```" in answer else answer
    got = text.strip()
    if got == expected:
        return True
    lines = [ln.strip() for ln in got.splitlines() if ln.strip()]
    if lines and lines[-1] == expected:
        return True
    return expected in got


def grade_text_answer(prompt: str, ground_truth: str, answer: str) -> bool:
    """LLM judge for open-ended tasks (summarization, NER). Consumes tokens."""
    import os

    max_tokens = int(os.environ.get("JUDGE_MAX_TOKENS", str(JUDGE_MAX_TOKENS)))
    judge_prompt = JUDGE_PROMPT.format(
        prompt=prompt, ground_truth=ground_truth, answer=answer
    )
    result = chat(get_judge_model_id(), judge_prompt, max_tokens=max_tokens, temperature=0.0)
    return _parse_verdict(result["text"])


def grade_code_answer(ground_truth_spec_json: str, answer_text: str) -> bool:
    """Grade code_generation by running tests. Free (local execution)."""
    spec = json.loads(ground_truth_spec_json)
    return run_tests(answer_text, spec["function_name"], spec["tests"])


def grade_answer(category: str, prompt: str, ground_truth: str, answer: str) -> bool:
    """Route to the right grader for a task category."""
    if category == "code_generation":
        return grade_code_answer(ground_truth, answer)
    if category == "math_reasoning":
        return grade_math_answer(ground_truth, answer)
    if category == "logic_puzzles":
        return grade_logic_answer(ground_truth, answer)
    if category == "sentiment":
        return grade_sentiment_answer(ground_truth, answer)
    if category == "code_debugging":
        return grade_code_debugging_answer(ground_truth, answer)
    # summarization, ner, and anything else → LLM judge
    return grade_text_answer(prompt, ground_truth, answer)
