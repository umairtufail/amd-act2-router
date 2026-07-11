"""Grading helpers used during dataset labeling.

- Text answers are graded by an independent judge model (MODEL_JUDGE) so a
  model never grades its own answers.
- Code answers are graded by executing real tests (data/code_exec.py).

Nothing here is called automatically; labeling (which the human runs) invokes
these functions.
"""

import json
import logging
import re

from agent.fireworks_client import chat
from config import get_judge_model_id
from data.code_exec import run_tests

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are grading an AI model's answer against a known ground truth.

Question:
{prompt}

Ground truth (correct answer or required key facts):
{ground_truth}

Model's answer:
{answer}

Is the model's answer correct? Judge semantic correctness, not wording:
numeric answers must match the ground truth value; classification answers must
match the label; summaries/extractions must contain the ground truth facts
without contradictions.

Respond with ONLY a JSON object, no other text:
{{"correct": true}} or {{"correct": false}}"""

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _parse_verdict(text: str) -> bool:
    """Extract {"correct": bool} from judge output; be strict on failure."""
    match = _JSON_RE.search(text)
    if not match:
        logger.warning("Judge returned no JSON, treating as FAIL: %r", text[:200])
        return False
    try:
        return bool(json.loads(match.group(0)).get("correct", False))
    except json.JSONDecodeError:
        logger.warning("Judge returned invalid JSON, treating as FAIL: %r", text[:200])
        return False


def grade_text_answer(prompt: str, ground_truth: str, answer: str) -> bool:
    """Grade a free-text answer using the judge model. Consumes Fireworks tokens."""
    judge_prompt = JUDGE_PROMPT.format(
        prompt=prompt, ground_truth=ground_truth, answer=answer
    )
    # max_tokens generous enough for reasoning models to get past their
    # thinking phase and still emit the JSON verdict.
    result = chat(get_judge_model_id(), judge_prompt, max_tokens=300, temperature=0.0)
    return _parse_verdict(result["text"])


def grade_code_answer(ground_truth_spec_json: str, answer_text: str) -> bool:
    """Grade a code answer by running its tests. Free (local execution)."""
    spec = json.loads(ground_truth_spec_json)
    return run_tests(answer_text, spec["function_name"], spec["tests"])
