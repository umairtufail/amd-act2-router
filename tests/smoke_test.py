"""Quick sanity checks before building/pushing the container.

Run with either:
  python -m tests.smoke_test
  python -m pytest tests/smoke_test.py -v

No Fireworks calls are made; fake env vars are injected so config resolution
can be tested without real model IDs.
"""

import os
import sys
import json
from collections import Counter
from pathlib import Path

# Fake env so config/get_model_id_for_tier works without real secrets.
_FAKE_ENV = {
    "MODEL_TIER0": "accounts/fake/models/tier0-model",
    "MODEL_TIER1": "accounts/fake/models/tier1-model",
    "MODEL_TIER2": "accounts/fake/models/tier2-model",
    "MODEL_TIER3": "accounts/fake/models/tier3-model",
    "MODEL_JUDGE": "accounts/fake/models/judge-model",
    "ROUTER_MODE": "binary",
    "BINARY_ROUTER_TAU": "0.8",
    "NER_BINARY_TAU": "0.8",
}
for key, value in _FAKE_ENV.items():
    os.environ.setdefault(key, value)


def test_config_resolves_all_tiers():
    from config import get_model_id_for_tier, get_tier_names

    tiers = get_tier_names()
    assert tiers == ["tier0", "tier1", "tier2", "tier3"]
    for tier in tiers:
        assert get_model_id_for_tier(tier)  # non-empty


def test_allowed_models_guard():
    from config import get_allowed_model_ids, get_model_id_for_tier

    previous = os.environ.get("ALLOWED_MODELS")
    allowed = [_FAKE_ENV[f"MODEL_TIER{i}"] for i in range(4)]
    try:
        os.environ["ALLOWED_MODELS"] = json.dumps(allowed)
        assert get_allowed_model_ids() == set(allowed)
        assert get_model_id_for_tier("tier0") == allowed[0]

        os.environ["ALLOWED_MODELS"] = json.dumps(allowed[1:])
        try:
            get_model_id_for_tier("tier0")
        except RuntimeError as exc:
            assert "not present in ALLOWED_MODELS" in str(exc)
        else:
            raise AssertionError("disallowed tier0 model was accepted")
    finally:
        if previous is None:
            os.environ.pop("ALLOWED_MODELS", None)
        else:
            os.environ["ALLOWED_MODELS"] = previous


def test_features_extract():
    from router.features import UNKNOWN_CATEGORY_INDEX, extract_features

    feats = extract_features("What is 2 + 2?\n```python\nx = 1\n```", "math_reasoning")
    assert feats["text"]
    assert len(feats["numeric"]) == 7
    assert isinstance(feats["category_index"], int)
    assert (
        extract_features("What is the capital of Canada?", "factual_knowledge")
        ["category_index"]
        < UNKNOWN_CATEGORY_INDEX
    )
    # Unknown / missing category must not crash.
    assert extract_features("hello", None)["category_index"] >= 0
    assert extract_features("hello", "not_a_category")["category_index"] >= 0


def test_local_graders():
    from data.judge import (
        grade_factual_answer,
        grade_logic_answer,
        grade_math_answer,
        grade_sentiment_answer,
    )

    assert grade_math_answer("94", "The answer is 94.") is True
    assert grade_math_answer("94", "93") is False
    assert grade_logic_answer("Kai", "Kai") is True
    assert grade_logic_answer("Kai", "The answer is Kai.") is True
    assert grade_logic_answer("Kai", "Pia") is False
    assert grade_sentiment_answer("negative", "negative") is True
    assert grade_sentiment_answer("positive", "negative") is False
    assert grade_factual_answer("tungsten||wolfram", "Wolfram.") is True
    assert grade_factual_answer("Pablo Picasso||Picasso", "The answer is Picasso.") is True
    assert grade_factual_answer("Sao Paulo", "São Paulo") is True
    assert grade_factual_answer("Paris", "Paris or London") is False
    assert grade_factual_answer("Paris", "The answer is not Paris") is False
    assert grade_factual_answer("Mars", "Marshall") is False


def test_structured_judge_contract():
    import agent.llm_backend as backend
    from data.judge import _parse_verdict

    assert _parse_verdict('{"correct": true}') is True
    assert _parse_verdict('{"correct": false}') is False
    # A string must never become truthy via bool("false").
    assert _parse_verdict('{"correct": "false"}') is False

    captured = {}
    original_chat = backend.fireworks_client.chat
    original_backend = os.environ.get("JUDGE_BACKEND")

    def fake_chat(model_id, prompt, **kwargs):
        captured.update(kwargs)
        return {"text": '{"correct": true}', "total_tokens": 1}

    backend.fireworks_client.chat = fake_chat
    os.environ["JUDGE_BACKEND"] = "fireworks"
    try:
        result = backend.judge_chat("grade this")
    finally:
        backend.fireworks_client.chat = original_chat
        if original_backend is None:
            os.environ.pop("JUDGE_BACKEND", None)
        else:
            os.environ["JUDGE_BACKEND"] = original_backend

    assert result["text"] == '{"correct": true}'
    schema = captured["response_format"]
    assert schema["type"] == "json_schema"
    assert schema["json_schema"]["schema"]["properties"]["correct"] == {
        "type": "boolean"
    }


def test_code_exec_grades_correctly():
    from data.code_exec import extract_code, run_tests

    good = "```python\ndef double(x):\n    return x * 2\n```"
    bad = "def double(x):\n    return x + 2\n"
    tests = [{"args": [3], "expected": 6}, {"args": [0], "expected": 0}]
    assert extract_code(good).strip().startswith("def double")
    assert run_tests(good, "double", tests) is True
    assert run_tests(bad, "double", tests) is False


def test_predict_tier_returns_valid_tier():
    """Keep legacy multitier checkpoint inference covered for A/B mode."""
    from data.schema import CATEGORIES
    from router.infer_multitier_router import checkpoint_available, predict_tier

    if not checkpoint_available():
        print("  (skipped: no trained checkpoint yet)")
        return
    config_path = Path(__file__).parent.parent / "router" / "checkpoints" / "router_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config.get("categories") == CATEGORIES
    for prompt in ["What is 12 + 7?",
                   "Write a Python function levenshtein(a, b)...",
                   "Five people stand in a line..."]:
        tier = predict_tier(prompt)
        assert tier in {"tier0", "tier1", "tier2", "tier3"}, tier


def test_binary_checkpoint_predicts_tier03():
    """Binary checkpoint inference must return a probability and tier0/tier3."""
    from router.infer_binary_router import (
        checkpoint_available,
        predict_cheap_ok_proba,
        predict_tier,
    )

    if not checkpoint_available():
        print("  (skipped: no trained binary checkpoint yet)")
        return
    from data.schema import CATEGORIES
    from router.infer_binary_router import CONFIG_PATH

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config.get("categories") == CATEGORIES
    cases = [
        ("What is 12 + 7?", "math_reasoning"),
        (
            "Extract all people from: Dr. Jane Smith met Bob in Berlin.",
            "ner",
        ),
        ("Summarize a long technical report in one sentence.", "summarization"),
    ]
    for prompt, category in cases:
        probability = predict_cheap_ok_proba(prompt, category)
        assert 0.0 <= probability <= 1.0, probability
        tier = predict_tier(prompt, category)
        assert tier in {"tier0", "tier3"}, tier


def test_binary_labels_and_policy_boundaries():
    from router.labels import CHEAP_OK_LABEL, NEEDS_STRONG_LABEL, tier_label_to_binary
    from router.route_binary import choose_binary_tier

    assert tier_label_to_binary("tier0") == CHEAP_OK_LABEL
    for label in ("tier1", "tier2", "tier3", "none"):
        assert tier_label_to_binary(label) == NEEDS_STRONG_LABEL

    def must_not_run(prompt, category):
        raise AssertionError("easy categories must bypass binary inference")

    assert choose_binary_tier(
        "A clearly positive review", "sentiment", predict_proba=must_not_run
    ) == "tier0"
    assert choose_binary_tier(
        "What is the capital of Canada?",
        "factual_knowledge",
        predict_proba=must_not_run,
    ) == "tier0"
    assert choose_binary_tier(
        "summary", "summarization", tau=0.8, predict_proba=lambda *_: 0.79
    ) == "tier3"
    assert choose_binary_tier(
        "summary", "summarization", tau=0.8, predict_proba=lambda *_: 0.8
    ) == "tier0"
    assert choose_binary_tier(
        "entities", "ner", tau=0.8, ner_tau=0.9,
        predict_proba=lambda *_: 0.85,
    ) == "tier3"


def test_holdout_covers_all_categories_without_training_overlap():
    from data.schema import CATEGORIES
    from eval.holdout_generalization import load_tasks

    tasks = load_tasks()
    counts = Counter(task["category"] for task in tasks)
    assert counts == Counter({category: 9 for category in CATEGORIES})
    assert len({task["id"] for task in tasks}) == len(tasks)

    raw_path = Path(__file__).parent.parent / "data" / "tasks_raw.jsonl"
    training_prompts = {
        json.loads(line)["prompt"]
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert not training_prompts.intersection(task["prompt"] for task in tasks)


def test_holdout_regrade_reuses_saved_answers():
    import eval.holdout_generalization as holdout

    task = {
        "id": "saved_task",
        "category": "ner",
        "prompt": "prompt",
        "ground_truth": "truth",
    }
    saved_task = {"task_id": "saved_task", "passed": False, "answer": "good"}
    saved = {
        "strategies": {
            "binary": {"tasks": [saved_task.copy()]},
            "always_tier0": {"tasks": [saved_task.copy()]},
        }
    }
    original_grade = holdout.grade_answer
    holdout.grade_answer = lambda *args: args[-1] == "good"
    try:
        holdout.regrade_failed_results(saved, [task])
    finally:
        holdout.grade_answer = original_grade

    assert saved["strategies"]["binary"]["accuracy"] == 1.0
    assert saved["strategies"]["always_tier0"]["accuracy"] == 1.0
    assert saved["last_regrade"] == {
        "failed_entries": 2,
        "unique_grade_calls": 1,
        "changed_to_pass": 2,
        "answer_calls": 0,
    }


def test_route_returns_model_id():
    import agent.agent as agent_module

    old_mode = agent_module.ROUTER_MODE
    agent_module.ROUTER_MODE = "binary"
    try:
        tier, model_id, routing_tokens = agent_module.route(
            "Extract all people from: Jane Smith met Bob in Berlin.", "ner"
        )
    finally:
        agent_module.ROUTER_MODE = old_mode
    assert tier in {"tier0", "tier3"}
    assert model_id  # non-empty
    assert routing_tokens == 0  # local (or fallback) routing costs nothing


def test_fireworks_client_error_marker():
    """chat_safe must return an error marker, never raise."""
    os.environ.setdefault("FIREWORKS_API_KEY", "fake-key-for-smoke-test")
    os.environ["FIREWORKS_BASE_URL"] = "http://127.0.0.1:9"  # nothing listens here
    import agent.fireworks_client as fc

    old_retries = fc.MAX_RETRIES
    fc.MAX_RETRIES = 1  # don't sit through backoff in a smoke test
    try:
        result = fc.chat_safe("accounts/fake/models/x", "hi")
    finally:
        fc.MAX_RETRIES = old_retries
        os.environ.pop("FIREWORKS_BASE_URL", None)
    assert "error" in result
    assert result["total_tokens"] == 0


def main() -> int:
    checks = [
        test_config_resolves_all_tiers,
        test_allowed_models_guard,
        test_features_extract,
        test_local_graders,
        test_structured_judge_contract,
        test_code_exec_grades_correctly,
        test_predict_tier_returns_valid_tier,
        test_binary_checkpoint_predicts_tier03,
        test_binary_labels_and_policy_boundaries,
        test_holdout_covers_all_categories_without_training_overlap,
        test_holdout_regrade_reuses_saved_answers,
        test_route_returns_model_id,
        test_fireworks_client_error_marker,
    ]
    failed = 0
    for check in checks:
        try:
            check()
            print(f"PASS  {check.__name__}")
        except Exception as exc:  # noqa: BLE001 — report every failure
            failed += 1
            print(f"FAIL  {check.__name__}: {exc}")
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
