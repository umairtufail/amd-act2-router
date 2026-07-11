"""Quick sanity checks before building/pushing the container.

Run with either:
  python -m tests.smoke_test
  python -m pytest tests/smoke_test.py -v

No Fireworks calls are made; fake env vars are injected so config resolution
can be tested without real model IDs.
"""

import os
import sys

# Fake env so config/get_model_id_for_tier works without real secrets.
_FAKE_ENV = {
    "MODEL_TIER0": "accounts/fake/models/tier0-model",
    "MODEL_TIER1": "accounts/fake/models/tier1-model",
    "MODEL_TIER2": "accounts/fake/models/tier2-model",
    "MODEL_TIER3": "accounts/fake/models/tier3-model",
}
for key, value in _FAKE_ENV.items():
    os.environ.setdefault(key, value)


def test_config_resolves_all_tiers():
    from config import get_model_id_for_tier, get_tier_names

    tiers = get_tier_names()
    assert tiers == ["tier0", "tier1", "tier2", "tier3"]
    for tier in tiers:
        assert get_model_id_for_tier(tier)  # non-empty


def test_features_extract():
    from router.features import extract_features

    feats = extract_features("What is 2 + 2?\n```python\nx = 1\n```", "math_reasoning")
    assert feats["text"]
    assert len(feats["numeric"]) == 7
    assert isinstance(feats["category_index"], int)
    # Unknown / missing category must not crash.
    assert extract_features("hello", None)["category_index"] >= 0
    assert extract_features("hello", "not_a_category")["category_index"] >= 0


def test_code_exec_grades_correctly():
    from data.code_exec import extract_code, run_tests

    good = "```python\ndef double(x):\n    return x * 2\n```"
    bad = "def double(x):\n    return x + 2\n"
    tests = [{"args": [3], "expected": 6}, {"args": [0], "expected": 0}]
    assert extract_code(good).strip().startswith("def double")
    assert run_tests(good, "double", tests) is True
    assert run_tests(bad, "double", tests) is False


def test_predict_tier_returns_valid_tier():
    """Only meaningful once a checkpoint exists; skipped otherwise."""
    from router.infer_multitier_router import checkpoint_available, predict_tier

    if not checkpoint_available():
        print("  (skipped: no trained checkpoint yet)")
        return
    for prompt in ["What is 12 + 7?",
                   "Write a Python function levenshtein(a, b)...",
                   "Five people stand in a line..."]:
        tier = predict_tier(prompt)
        assert tier in {"tier0", "tier1", "tier2", "tier3"}, tier


def test_route_returns_model_id():
    from agent.agent import route

    tier, model_id, routing_tokens = route("What is 2 + 2?", "math_reasoning")
    assert tier in {"tier0", "tier1", "tier2", "tier3"}
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
        test_features_extract,
        test_code_exec_grades_correctly,
        test_predict_tier_returns_valid_tier,
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
