"""Reproducible request-policy A/B evaluation for tier0 NER answers.

The mining snapshot reuses already-paid tier0 outcomes.  The external runner
uses the nine natural-holdout NER tasks plus the public T05 NER task and writes
resumable artifacts after every Fireworks answer + judge pair.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from agent.fireworks_client import chat
from agent.request_policy import build_answer_request, request_policy_hash
from config import get_model_id_for_tier
from data.integrity import stable_json_hash
from data.judge import grade_answer

ROOT = Path(__file__).parent.parent
DEFAULT_MINING_PATHS = (
    ROOT / "data" / "mining" / "train_labeled.jsonl",
    ROOT / "data" / "mining" / "stress_labeled.jsonl",
)
HOLDOUT_PATH = ROOT / "data" / "holdout_tasks.json"
PUBLIC_PATH = ROOT / "tests" / "fixtures" / "public_validation" / "tasks.json"
PUBLIC_TASK_ID = "T05"
PUBLIC_GROUND_TRUTH = json.dumps(
    {
        "persons": ["Sundar Pichai"],
        "organizations": ["Google", "ETH Zurich"],
        "locations": ["Zurich"],
        "dates": ["March 15 2023"],
    },
    sort_keys=True,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _artifact_hash(artifact: dict[str, Any]) -> str:
    unsigned = dict(artifact)
    unsigned.pop("artifact_sha256", None)
    return stable_json_hash(unsigned)


def _write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(artifact)
    artifact["artifact_sha256"] = _artifact_hash(artifact)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _load_artifact(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("artifact_sha256") != _artifact_hash(artifact):
        raise ValueError(f"artifact hash mismatch: {path}")
    return artifact


def capture_mining(paths: list[Path] | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    hashes: set[str] = set()
    for path in paths or list(DEFAULT_MINING_PATHS):
        for record in _read_jsonl(path):
            if record.get("category") != "ner":
                continue
            result = record.get("tier_results", {}).get("tier0")
            if not isinstance(result, dict) or not isinstance(result.get("passed"), bool):
                raise ValueError(f"{record.get('id')} has no valid tier0 outcome")
            policy_hash = result.get("request_policy_hash")
            if not isinstance(policy_hash, str):
                raise ValueError(f"{record.get('id')} has no request-policy hash")
            hashes.add(policy_hash)
            rows.append(
                {
                    "id": record["id"],
                    "scope": f"mining_{record.get('dataset_split')}",
                    "category": "ner",
                    "prompt": record["prompt"],
                    "ground_truth": record["ground_truth"],
                    "answer": result.get("answer_text", ""),
                    "passed": result["passed"],
                    "total_tokens": int(result.get("total_tokens", 0) or 0),
                    "prompt_tokens": int(result.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(result.get("completion_tokens", 0) or 0),
                    "finish_reason": result.get("finish_reason"),
                }
            )
    if len(hashes) != 1:
        raise ValueError(f"mining snapshot mixes request policies: {sorted(hashes)}")
    return {
        "artifact_version": 1,
        "kind": "ner_policy_arm_results",
        "source": "mining",
        "request_policy_sha256": hashes.pop(),
        "model_id": get_model_id_for_tier("tier0"),
        "rows": sorted(rows, key=lambda row: row["id"]),
    }


def external_tasks() -> list[dict[str, str]]:
    holdout = json.loads(HOLDOUT_PATH.read_text(encoding="utf-8"))
    tasks = [
        {
            "id": task["id"],
            "scope": "natural_holdout",
            "category": "ner",
            "prompt": task["prompt"],
            "ground_truth": task["ground_truth"],
        }
        for task in holdout
        if task.get("category") == "ner"
    ]
    public = json.loads(PUBLIC_PATH.read_text(encoding="utf-8"))
    match = next(task for task in public if str(task.get("task_id")) == PUBLIC_TASK_ID)
    tasks.append(
        {
            "id": f"public_{PUBLIC_TASK_ID}",
            "scope": "public_fixture",
            "category": str(match.get("category") or "named_entity_recognition"),
            "prompt": match["prompt"],
            "ground_truth": PUBLIC_GROUND_TRUTH,
        }
    )
    if len(tasks) != 10 or len({task["id"] for task in tasks}) != 10:
        raise ValueError("expected nine natural NER tasks plus public T05")
    return tasks


def run_external(output: Path) -> dict[str, Any]:
    if os.environ.get("JUDGE_BACKEND", "").strip().lower() != "fireworks":
        raise RuntimeError("set JUDGE_BACKEND=fireworks for a comparable A/B run")
    policy_hash = request_policy_hash()
    model_id = get_model_id_for_tier("tier0")
    artifact: dict[str, Any]
    if output.exists():
        artifact = _load_artifact(output)
        if artifact.get("request_policy_sha256") != policy_hash:
            raise ValueError("refusing to resume an artifact from another request policy")
    else:
        artifact = {
            "artifact_version": 1,
            "kind": "ner_policy_arm_results",
            "source": "external",
            "request_policy_sha256": policy_hash,
            "model_id": model_id,
            "rows": [],
        }
    completed = {row["id"] for row in artifact["rows"]}
    for task in external_tasks():
        if task["id"] in completed:
            continue
        result = chat(model_id, **build_answer_request(task["prompt"], task["category"]))
        passed = grade_answer(
            task["category"], task["prompt"], task["ground_truth"], result["text"]
        )
        artifact["rows"].append(
            {
                **task,
                "answer": result["text"],
                "passed": passed,
                "total_tokens": int(result.get("total_tokens", 0) or 0),
                "prompt_tokens": int(result.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(result.get("completion_tokens", 0) or 0),
                "finish_reason": result.get("finish_reason"),
            }
        )
        artifact["rows"].sort(key=lambda row: row["id"])
        _write_artifact(output, artifact)
        print(
            f"[{'PASS' if passed else 'FAIL'}] {task['id']} "
            f"{result.get('total_tokens', 0)} tokens"
        )
    return _load_artifact(output)


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    passed = sum(row["passed"] is True for row in rows)
    tokens = sum(int(row["total_tokens"]) for row in rows)
    return {
        "tasks": total,
        "passed": passed,
        "accuracy": passed / total if total else None,
        "total_tokens": tokens,
        "mean_tokens": tokens / total if total else None,
    }


def compare(baseline_paths: list[Path], candidate_paths: list[Path]) -> dict[str, Any]:
    baseline_artifacts = [_load_artifact(path) for path in baseline_paths]
    candidate_artifacts = [_load_artifact(path) for path in candidate_paths]
    baseline_rows = {
        row["id"]: row for artifact in baseline_artifacts for row in artifact["rows"]
    }
    candidate_rows = {
        row["id"]: row for artifact in candidate_artifacts for row in artifact["rows"]
    }
    if set(baseline_rows) != set(candidate_rows):
        raise ValueError("baseline and candidate task IDs differ")
    for task_id in baseline_rows:
        if baseline_rows[task_id]["prompt"] != candidate_rows[task_id]["prompt"]:
            raise ValueError(f"prompt changed for {task_id}")
    scopes = sorted({row["scope"] for row in baseline_rows.values()})
    scope_results: dict[str, Any] = {}
    for scope in [*scopes, "overall"]:
        ids = [
            task_id
            for task_id, row in baseline_rows.items()
            if scope == "overall" or row["scope"] == scope
        ]
        old = _stats([baseline_rows[task_id] for task_id in ids])
        new = _stats([candidate_rows[task_id] for task_id in ids])
        scope_results[scope] = {
            "baseline": old,
            "candidate": new,
            "accuracy_delta_points": (new["accuracy"] - old["accuracy"]) * 100,
            "token_delta": new["total_tokens"] - old["total_tokens"],
            "token_delta_percent": (
                (new["total_tokens"] - old["total_tokens"])
                / old["total_tokens"]
                * 100
                if old["total_tokens"]
                else None
            ),
        }
    changed = [
        {
            "id": task_id,
            "scope": baseline_rows[task_id]["scope"],
            "baseline_passed": baseline_rows[task_id]["passed"],
            "candidate_passed": candidate_rows[task_id]["passed"],
            "baseline_tokens": baseline_rows[task_id]["total_tokens"],
            "candidate_tokens": candidate_rows[task_id]["total_tokens"],
        }
        for task_id in sorted(baseline_rows)
        if baseline_rows[task_id]["passed"] != candidate_rows[task_id]["passed"]
    ]
    return {
        "artifact_version": 1,
        "kind": "ner_policy_ab_comparison",
        "baseline_policy_sha256": baseline_artifacts[0]["request_policy_sha256"],
        "candidate_policy_sha256": candidate_artifacts[0]["request_policy_sha256"],
        "scopes": scope_results,
        "changed_verdicts": changed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tier0 NER request-policy A/B.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture_parser = subparsers.add_parser("capture-mining")
    capture_parser.add_argument("--output", type=Path, required=True)

    run_parser = subparsers.add_parser("run-external")
    run_parser.add_argument("--output", type=Path, required=True)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--baseline", type=Path, action="append", required=True)
    compare_parser.add_argument("--candidate", type=Path, action="append", required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "capture-mining":
        artifact = capture_mining()
        _write_artifact(args.output, artifact)
        print(f"Captured {len(artifact['rows'])} mining NER results -> {args.output}")
    elif args.command == "run-external":
        artifact = run_external(args.output)
        stats = _stats(artifact["rows"])
        print(f"External NER: {stats['passed']}/{stats['tasks']} | {stats['total_tokens']} tokens")
    else:
        result = compare(args.baseline, args.candidate)
        _write_artifact(args.output, result)
        print(json.dumps(result["scopes"]["overall"], indent=2))


if __name__ == "__main__":
    main()
