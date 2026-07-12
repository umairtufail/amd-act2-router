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

import requests

from config import get_fireworks_base_url

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_S = float(os.environ.get("FIREWORKS_TIMEOUT_S", "60"))
MAX_RETRIES = 3
# Status codes worth retrying: rate limits and transient server errors.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class FireworksError(RuntimeError):
    """Raised when a call fails after all retries."""


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


def chat(
    model_id: str,
    prompt: str,
    max_tokens: int = 700,
    temperature: float = 0.2,
    response_format: dict | None = None,
) -> dict:
    """Call Fireworks chat completion for a single prompt.

    Returns a dict with keys: text, total_tokens, prompt_tokens, completion_tokens.
    Raises FireworksError after MAX_RETRIES failed attempts.
    """
    url = f"{get_fireworks_base_url().rstrip('/')}/chat/completions"
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }

    last_error = None
    last_status = None
    resp = None
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
                data = resp.json()
                usage = data.get("usage", {})
                msg = data["choices"][0]["message"]
                text = msg.get("content") or msg.get("reasoning_content") or ""
                return {
                    "text": text,
                    "total_tokens": usage.get("total_tokens", 0),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                }
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


def chat_safe(model_id: str, prompt: str, **kwargs) -> dict:
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
            "error": str(exc),
        }
