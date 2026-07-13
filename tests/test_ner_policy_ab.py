from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval import ner_policy_ab


def _artifact(policy: str, passed: bool, tokens: int) -> dict:
    artifact = {
        "artifact_version": 1,
        "kind": "ner_policy_arm_results",
        "source": "test",
        "request_policy_sha256": policy,
        "model_id": "tier0",
        "rows": [
            {
                "id": "same",
                "scope": "mining_train",
                "category": "ner",
                "prompt": "prompt",
                "ground_truth": "truth",
                "answer": "answer",
                "passed": passed,
                "total_tokens": tokens,
                "prompt_tokens": tokens // 2,
                "completion_tokens": tokens - tokens // 2,
                "finish_reason": "stop",
            }
        ],
    }
    artifact["artifact_sha256"] = ner_policy_ab._artifact_hash(artifact)
    return artifact


def test_external_set_is_fixed_and_disjoint():
    tasks = ner_policy_ab.external_tasks()
    assert len(tasks) == 10
    assert len({task["id"] for task in tasks}) == 10
    assert sum(task["scope"] == "natural_holdout" for task in tasks) == 9
    assert sum(task["scope"] == "public_fixture" for task in tasks) == 1


def test_compare_requires_identical_ids_and_reports_deltas(tmp_path: Path):
    old = _artifact("old", False, 100)
    new = _artifact("new", True, 80)
    old_path, new_path = tmp_path / "old.json", tmp_path / "new.json"
    ner_policy_ab._write_artifact(old_path, old)
    ner_policy_ab._write_artifact(new_path, new)

    result = ner_policy_ab.compare([old_path], [new_path])

    assert result["scopes"]["overall"]["accuracy_delta_points"] == 100
    assert result["scopes"]["overall"]["token_delta"] == -20
    assert result["changed_verdicts"][0]["id"] == "same"


def test_artifact_hash_detects_edits(tmp_path: Path):
    path = tmp_path / "artifact.json"
    ner_policy_ab._write_artifact(path, _artifact("p", True, 10))
    value = json.loads(path.read_text(encoding="utf-8"))
    value["rows"][0]["passed"] = False
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        ner_policy_ab._load_artifact(path)


def test_run_external_requires_fireworks_judge(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("JUDGE_BACKEND", raising=False)
    with pytest.raises(RuntimeError, match="JUDGE_BACKEND=fireworks"):
        ner_policy_ab.run_external(tmp_path / "unused.json")
