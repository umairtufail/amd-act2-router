"""Route LLM calls to Fireworks (submission) or local (dev) based on env vars.

Policy:
  - Submission answers: always Fireworks (agent.py default).
  - Judge / holdout eval: local if JUDGE_BACKEND=local or LOCAL_LLM_BASE_URL set.
  - Dev answer testing: local only when DEV_LOCAL_ANSWERS=1 (never in Dockerfile).

Env vars:
  JUDGE_BACKEND=fireworks|local|auto   (default auto: local if configured)
  DEV_LOCAL_ANSWERS=1                  (agent uses local for answers — dev only)
  LOCAL_LLM_BASE_URL, LOCAL_LLM_MODEL
"""

from __future__ import annotations

import logging
import os

from agent import fireworks_client, local_llm_client
from config import get_judge_model_id

logger = logging.getLogger(__name__)

JUDGE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "judge_verdict",
        "schema": {
            "type": "object",
            "properties": {"correct": {"type": "boolean"}},
            "required": ["correct"],
            "additionalProperties": False,
        },
    },
}


def _judge_uses_local() -> bool:
    backend = os.environ.get("JUDGE_BACKEND", "auto").strip().lower()
    if backend == "local":
        return True
    if backend == "fireworks":
        return False
    return local_llm_client.is_configured()


def judge_chat(prompt: str, max_tokens: int = 700, temperature: float = 0.0) -> dict:
    """LLM judge for labeling/eval — may use local LLM in development."""
    if _judge_uses_local():
        logger.debug("Judge: using local LLM")
        return local_llm_client.chat(prompt, max_tokens=max_tokens, temperature=temperature)
    return fireworks_client.chat(
        get_judge_model_id(),
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format=JUDGE_RESPONSE_FORMAT,
    )


def answer_chat(
    model_id: str,
    prompt: str,
    max_tokens: int = 700,
    temperature: float = 0.2,
    system_prompt: str | None = None,
    reasoning_effort: str | None = None,
) -> dict:
    """Answer generation — Fireworks by default; local only with DEV_LOCAL_ANSWERS=1."""
    if os.environ.get("DEV_LOCAL_ANSWERS", "").strip() == "1":
        if not local_llm_client.is_configured():
            raise RuntimeError("DEV_LOCAL_ANSWERS=1 but LOCAL_LLM_BASE_URL is not set.")
        logger.warning(
            "DEV_LOCAL_ANSWERS=1: routing answers to LOCAL LLM (not valid for submission)."
        )
        local_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        result = local_llm_client.chat(
            local_prompt, max_tokens=max_tokens, temperature=temperature
        )
        result.setdefault("finish_reason", None)
        result.setdefault("model_id", os.environ.get("LOCAL_LLM_MODEL", "local"))
        result.setdefault("attempts", 1)
        return result
    return fireworks_client.chat(
        model_id,
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        reasoning_effort=reasoning_effort,
    )


def answer_chat_safe(model_id: str, prompt: str, **kwargs) -> dict:
    try:
        return answer_chat(model_id, prompt, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.error("Answer call failed for %s: %s", model_id, exc)
        return {
            "text": f"ERROR: model call failed ({exc})",
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "finish_reason": "error",
            "model_id": model_id,
            "attempts": fireworks_client.MAX_RETRIES,
            "error": str(exc),
        }
