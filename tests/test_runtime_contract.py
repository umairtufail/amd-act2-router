"""Offline tests for the shared request and container runtime contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_request_policy_is_shared_and_hashable():
    from agent.request_policy import (
        ANSWER_MAX_TOKENS,
        ANSWER_SYSTEM_PROMPT,
        NER_SYSTEM_SUFFIX,
        POLICY_VERSION,
        answer_system_prompt,
        build_answer_request,
        request_policy_hash,
    )

    request = build_answer_request(
        "Compare A and B.", "factual_knowledge"
    )
    assert request["temperature"] == 0.0
    assert request["reasoning_effort"] == "low"
    assert "which one is faster or slower" in request["prompt"]
    assert "under 140 words" in request["system_prompt"]
    assert request["max_tokens"] == 700 == ANSWER_MAX_TOKENS
    assert POLICY_VERSION == "answer-request-v2"
    assert len(ANSWER_SYSTEM_PROMPT) == 189
    assert len(ANSWER_SYSTEM_PROMPT + NER_SYSTEM_SUFFIX) == 414
    ner_prompt = answer_system_prompt("named_entity_recognition")
    assert "comma-qualified location" in ner_prompt
    assert "parenthesized acronym" in ner_prompt
    assert "later unambiguous shortened mention" in ner_prompt
    assert "requested NER schema" not in answer_system_prompt("summarization")
    assert len(request_policy_hash()) == 64
    int(request_policy_hash(), 16)
    assert request_policy_hash() != "c2acb787fc151252a7b536dda2ebbc1b466030c1af2519c2f8f6c46be7b58506"


def test_fireworks_result_exposes_finish_reason(monkeypatch):
    import agent.fireworks_client as client

    class FakeResponse:
        status_code = 200
        headers = {}

        @staticmethod
        def json():
            return {
                "choices": [
                    {"message": {"content": "done"}, "finish_reason": "stop"}
                ],
                "usage": {
                    "total_tokens": 9,
                    "prompt_tokens": 6,
                    "completion_tokens": 3,
                },
            }

    monkeypatch.setenv("FIREWORKS_API_KEY", "offline-test-key")
    monkeypatch.setattr(client.requests, "post", lambda *args, **kwargs: FakeResponse())
    result = client.chat("accounts/fake/models/test", "hello")
    assert result == {
        "text": "done",
        "total_tokens": 9,
        "prompt_tokens": 6,
        "completion_tokens": 3,
        "finish_reason": "stop",
        "model_id": "accounts/fake/models/test",
        "attempts": 1,
    }


def test_fireworks_malformed_success_is_retried(monkeypatch):
    import agent.fireworks_client as client

    calls = 0

    class FakeResponse:
        status_code = 200
        headers = {}

        def json(self):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {
                    "choices": [],
                    "usage": {
                        "total_tokens": 4,
                        "prompt_tokens": 3,
                        "completion_tokens": 1,
                    },
                }
            return {
                "choices": [
                    {"message": {"content": "recovered"}, "finish_reason": "stop"}
                ],
                "usage": {},
            }

    monkeypatch.setenv("FIREWORKS_API_KEY", "offline-test-key")
    monkeypatch.setattr(client.requests, "post", lambda *args, **kwargs: FakeResponse())
    monkeypatch.setattr(client.time, "sleep", lambda *_: None)
    result = client.chat("accounts/fake/models/test", "hello")
    assert result["text"] == "recovered"
    assert result["attempts"] == 2
    assert result["total_tokens"] == 4
    assert calls == 2


def test_load_tasks_rejects_duplicate_ids(tmp_path: Path):
    from agent.agent import load_tasks

    path = tmp_path / "tasks.json"
    path.write_text(
        json.dumps(
            [
                {"task_id": "same", "prompt": "one"},
                {"task_id": "same", "prompt": "two"},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate task_id"):
        load_tasks(path)


def test_atomic_results_writer_preserves_contract(tmp_path: Path):
    from agent.agent import write_results_atomic

    output = tmp_path / "nested" / "results.json"
    expected = [{"task_id": "a", "answer": "caf\u00e9"}]
    write_results_atomic(expected, output)
    assert json.loads(output.read_text(encoding="utf-8")) == expected
    assert not list(output.parent.glob(".*.tmp"))


def test_structural_failure_uses_one_fireworks_fallback(monkeypatch):
    import agent.agent as runtime

    calls: list[str] = []

    def fake_chat(model_id: str, **request):
        calls.append(model_id)
        if model_id == "primary-model":
            return {
                "text": "",
                "total_tokens": 4,
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "finish_reason": "stop",
                "model_id": model_id,
                "error": "empty completion",
            }
        return {
            "text": "94",
            "total_tokens": 5,
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "finish_reason": "stop",
            "model_id": model_id,
        }

    monkeypatch.setattr(runtime, "chat_safe", fake_chat)
    monkeypatch.setattr(runtime, "get_model_id_for_tier", lambda tier: f"{tier}-model")
    monkeypatch.setattr(runtime, "ANSWER_FALLBACKS", 1)
    result = runtime.answer_with_fallback(
        "What is 47 + 47? Answer with the number only.",
        "math_reasoning",
        "tier0",
        "primary-model",
        {"prompt": "task", "max_tokens": 700},
    )
    assert calls == ["primary-model", "tier3-model"]
    assert result["text"] == "94"
    assert result["fallback_used"] is True
    assert result["total_tokens"] == 9
    assert result["prompt_tokens"] == 6
    assert "error" not in result


def test_strict_baseline_can_disable_structural_fallback(monkeypatch):
    import agent.agent as runtime

    calls: list[str] = []

    def fake_chat(model_id: str, **request):
        calls.append(model_id)
        return {
            "text": "",
            "total_tokens": 2,
            "prompt_tokens": 2,
            "completion_tokens": 0,
            "finish_reason": "stop",
            "model_id": model_id,
        }

    monkeypatch.setattr(runtime, "chat_safe", fake_chat)
    result = runtime.answer_with_fallback(
        "task",
        None,
        "tier0",
        "primary-model",
        {"prompt": "task"},
        max_fallbacks=0,
    )
    assert calls == ["primary-model"]
    assert result["fallback_used"] is False


def test_holdout_verified_tier0_accounts_for_both_calls(monkeypatch):
    import eval.holdout_generalization as holdout

    task = {
        "id": "ner-ambiguous",
        "category": "ner",
        "prompt": (
            'Respond with ONLY a JSON object with keys "persons", '
            '"organizations", and "locations", each a list of strings.\n\n'
            "Sentence: Paris-based Acme appointed Ada in Lyon."
        ),
        "ground_truth": "unused",
    }

    def fake_answer(cache, *, routed_model_id, **kwargs):
        if routed_model_id == "model-tier0":
            return {
                "text": '{"persons":["Ada"],"organizations":["Acme"],'
                '"locations":["Lyon"]}',
                "total_tokens": 10,
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "finish_reason": "stop",
            }, False
        return {
            "text": '{"persons":["Ada"],"organizations":["Acme"],'
            '"locations":["Paris","Lyon"]}',
            "total_tokens": 6,
            "prompt_tokens": 4,
            "completion_tokens": 2,
            "finish_reason": "stop",
        }, False

    monkeypatch.setattr(holdout, "cached_answer", fake_answer)
    monkeypatch.setattr(
        holdout, "cached_grade", lambda cache, task, answer: ("Paris" in answer, False)
    )
    monkeypatch.setattr(holdout, "get_model_id_for_tier", lambda tier: f"model-{tier}")
    result = holdout.evaluate_strategy(
        "verified_tier0",
        [task],
        answer_backend="fireworks",
        local_model="unused",
        answer_cache={},
        grade_cache={},
    )
    assert result["accuracy"] == 1.0
    assert result["total_tokens"] == 16
    assert result["model_attempts"] == {"tier0": 1, "tier3": 1}
    assert result["tasks"][0]["fallback_used"] is True


def test_submission_copy_parser_handles_continuations_and_rejects_add():
    from scripts.preflight_submission import _docker_copy_sources

    dockerfile = (
        "COPY first.txt second.txt /app/\n"
        "COPY third.txt \\\n"
        "    fourth.txt /opt/contract/\n"
    )
    assert _docker_copy_sources(dockerfile) == {
        "first.txt",
        "second.txt",
        "third.txt",
        "fourth.txt",
    }
    with pytest.raises(ValueError, match="ADD instructions are forbidden"):
        _docker_copy_sources("ADD offline-dataset.jsonl /app/\n")


def test_submission_artifact_scan_is_scoped_to_runtime_tree(tmp_path: Path):
    from scripts.preflight_submission import _find_forbidden_data_artifacts

    app = tmp_path / "app"
    app.mkdir()
    (app / "agent.py").write_text("pass\n", encoding="utf-8")
    (app / "hard_cases.jsonl").write_text("{}\n", encoding="utf-8")
    (app / "answer_cache.json").write_text("{}\n", encoding="utf-8")
    contract = tmp_path / "opt" / "contract"
    contract.mkdir(parents=True)
    (contract / "tasks.json").write_text("[]\n", encoding="utf-8")

    found = {
        path.relative_to(app).as_posix()
        for path in _find_forbidden_data_artifacts(app)
    }
    assert found == {"answer_cache.json", "hard_cases.jsonl"}
