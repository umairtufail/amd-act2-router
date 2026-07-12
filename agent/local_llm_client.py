"""OpenAI-compatible local LLM client (Ollama, LM Studio, llama.cpp server, etc.).

DEV / RESEARCH ONLY — never used in the hackathon submission container for
scored answers. Set LOCAL_LLM_BASE_URL in .env for local development.

Default Ollama:  LOCAL_LLM_BASE_URL=http://localhost:11434/v1
                 LOCAL_LLM_MODEL=llama3.2
"""

from __future__ import annotations

import logging
import os

import requests

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_S = float(os.environ.get("LOCAL_LLM_TIMEOUT_S", "120"))


class LocalLLMError(RuntimeError):
    pass


def _base_url() -> str:
    url = os.environ.get("LOCAL_LLM_BASE_URL", "").strip().rstrip("/")
    if not url:
        raise LocalLLMError(
            "LOCAL_LLM_BASE_URL is not set. Example: http://localhost:11434/v1"
        )
    return url


def _model_id(override: str | None = None) -> str:
    model = (override or os.environ.get("LOCAL_LLM_MODEL", "")).strip()
    if not model:
        raise LocalLLMError("LOCAL_LLM_MODEL is not set (e.g. llama3.2, qwen2.5:7b).")
    return model


def chat(
    prompt: str,
    model_id: str | None = None,
    max_tokens: int = 700,
    temperature: float = 0.2,
) -> dict:
    """Call a local OpenAI-compatible /chat/completions endpoint."""
    url = f"{_base_url()}/chat/completions"
    payload = {
        "model": _model_id(model_id),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_S)
    if resp.status_code != 200:
        raise LocalLLMError(f"HTTP {resp.status_code}: {resp.text[:500]}")
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


def is_configured() -> bool:
    return bool(os.environ.get("LOCAL_LLM_BASE_URL", "").strip())
