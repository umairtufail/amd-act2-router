"""Thin wrapper around the Fireworks chat-completions API.

Every Fireworks call in this project goes through chat() so that:
- the API key / base URL are read from env vars in exactly one place,
- timeouts and retries are applied consistently,
- token usage is always captured for evaluation.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, TypedDict

import requests

from config import get_fireworks_base_url

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_S = float(os.environ.get("FIREWORKS_TIMEOUT_S", "60"))
MAX_RETRIES = max(1, int(os.environ.get("FIREWORKS_MAX_RETRIES", "3")))
# Status codes worth retrying: rate limits and transient server errors.
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class FireworksError(RuntimeError):
    """Raised when a call fails after all retries."""


class ChatResult(TypedDict, total=False):
    """Normalized response shape used throughout the project."""

    text: str
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None
    model_id: str
    attempts: int
    error: str


def _backoff_seconds(attempt: int, status_code: int | None, resp) -> float:
    """Longer waits for 429 rate limits; respect Retry-After when present."""
    if status_code == 429 and resp is not None:
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            try:
                return min(float(retry_after), 120.0)
            except ValueError:
                pass
        return min(15 * attempt, 60)  # 15s, 30s, 45s
    return float(2 ** attempt)  # 2s, 4s for other transient errors


def _api_key() -> str:
    key = os.environ.get("FIREWORKS_API_KEY", "").strip()
    if not key:
        raise FireworksError("FIREWORKS_API_KEY is not set. See .env.example.")
    return key


def _token_count(usage: dict[str, Any], key: str) -> int:
    value = usage.get(key, 0)
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _message_text(message: dict[str, Any]) -> str:
    """Normalize Fireworks/OpenAI-compatible message content to plain text."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        if parts:
            return "".join(parts)
    reasoning = message.get("reasoning_content")
    return reasoning if isinstance(reasoning, str) else ""


def _parse_success(data: Any, model_id: str, attempt: int) -> ChatResult:
    """Validate a successful HTTP response before exposing it to callers."""
    if not isinstance(data, dict):
        raise ValueError("response JSON must be an object")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response has no choices")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise ValueError("response choice has no message")
    usage = data.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        finish_reason = str(finish_reason)
    return {
        "text": _message_text(choice["message"]),
        "total_tokens": _token_count(usage, "total_tokens"),
        "prompt_tokens": _token_count(usage, "prompt_tokens"),
        "completion_tokens": _token_count(usage, "completion_tokens"),
        "finish_reason": finish_reason,
        "model_id": model_id,
        "attempts": attempt,
    }


def chat(
    model_id: str,
    prompt: str,
    max_tokens: int = 700,
    temperature: float = 0.2,
    response_format: dict | None = None,
    system_prompt: str | None = None,
    reasoning_effort: str | None = None,
) -> ChatResult:
    """Call Fireworks chat completion for a single prompt.

    Returns normalized text, token counts, finish reason, model ID, and attempts.
    Raises FireworksError after MAX_RETRIES failed attempts.
    """
    url = f"{get_fireworks_base_url().rstrip('/')}/chat/completions"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }

    last_error = None
    last_status = None
    resp = None
    retry_usage = {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT_S)
            if resp.status_code in RETRYABLE_STATUS:
                last_status = resp.status_code
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                logger.warning("Fireworks transient error (attempt %d/%d): %s",
                               attempt, MAX_RETRIES, last_error)
            elif resp.status_code != 200:
                raise FireworksError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            else:
                data = None
                try:
                    data = resp.json()
                    parsed = _parse_success(data, model_id, attempt)
                    for key, previous in retry_usage.items():
                        parsed[key] = int(parsed.get(key, 0) or 0) + previous
                    return parsed
                except (TypeError, ValueError, KeyError, IndexError) as exc:
                    if isinstance(data, dict) and isinstance(data.get("usage"), dict):
                        for key in retry_usage:
                            retry_usage[key] += _token_count(data["usage"], key)
                    last_status = resp.status_code
                    last_error = f"Malformed success response: {exc}"
                    logger.warning(
                        "Fireworks malformed response (attempt %d/%d): %s",
                        attempt,
                        MAX_RETRIES,
                        last_error,
                    )
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_status = None
            resp = None
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Fireworks network error (attempt %d/%d): %s",
                           attempt, MAX_RETRIES, last_error)
        if attempt < MAX_RETRIES:
            wait = _backoff_seconds(attempt, last_status, resp)
            logger.info("Retrying in %.0fs...", wait)
            time.sleep(wait)

    raise FireworksError(f"Fireworks call failed after {MAX_RETRIES} attempts: {last_error}")


def chat_safe(model_id: str, prompt: str, **kwargs) -> ChatResult:
    """Like chat() but never raises — returns an error marker dict instead.

    Used by the agent so one bad task can't kill the whole container run.
    """
    try:
        return chat(model_id, prompt, **kwargs)
    except Exception as exc:  # noqa: BLE001 — deliberate catch-all at the boundary
        logger.error("Fireworks call failed permanently for model %s: %s", model_id, exc)
        return {
            "text": f"ERROR: model call failed ({exc})",
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "finish_reason": "error",
            "model_id": model_id,
            "attempts": MAX_RETRIES,
            "error": str(exc),
        }
