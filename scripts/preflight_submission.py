"""Fail-fast checks for the Track 1 source tree and submission image.

The submission uses the measured ``verified_tier0`` policy and does not need a
local ML router. Runtime checks are network-free: they import the exact agent
entrypoint and exercise the input/output contract with a mocked Fireworks call.

Examples:
    python -m scripts.preflight_submission --source
    python -m scripts.preflight_submission --image amd-act2-router:candidate
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


EXPECTED_CMD = ["python", "-m", "agent.agent"]
EXPECTED_BASE_URL = "https://api.fireworks.ai/inference/v1"
EXPECTED_ROUTER_MODE = "verified_tier0"
MAX_IMAGE_BYTES = 500_000_000
EXPECTED_RUNTIME_REQUIREMENTS = {
    "requests==2.34.2",
    "python-dotenv==1.2.2",
    "PyYAML==6.0.3",
    "certifi==2026.6.17",
    "charset-normalizer==3.4.9",
    "idna==3.18",
    "urllib3==2.7.0",
}
EXPECTED_MODEL_DEFAULTS = {
    "MODEL_TIER0": "accounts/fireworks/models/gpt-oss-120b",
    "MODEL_TIER1": "accounts/fireworks/models/kimi-k2p6",
    "MODEL_TIER2": "accounts/fireworks/models/glm-5p2",
    "MODEL_TIER3": "accounts/fireworks/models/deepseek-v4-pro",
}
EXPECTED_DOCKER_COPY_SOURCES = {
    "requirements.txt",
    "config/__init__.py",
    "config/models.yaml",
    "agent/__init__.py",
    "agent/agent.py",
    "agent/fireworks_client.py",
    "agent/llm_backend.py",
    "agent/local_llm_client.py",
    "agent/request_policy.py",
    "agent/quality_gate.py",
    "scripts/preflight_submission.py",
    "tests/fixtures/container_input/tasks.json",
}
EXPECTED_APP_PATHS = {
    "requirements.txt",
    "agent",
    "agent/__init__.py",
    "agent/agent.py",
    "agent/fireworks_client.py",
    "agent/llm_backend.py",
    "agent/local_llm_client.py",
    "agent/request_policy.py",
    "agent/quality_gate.py",
    "config",
    "config/__init__.py",
    "config/models.yaml",
    "scripts",
    "scripts/preflight_submission.py",
}
FORBIDDEN_RUNTIME_MODULES = (
    "torch",
    "transformers",
    "numpy",
    "pydantic",
    "pandas",
    "streamlit",
)
FORBIDDEN_RUNTIME_DIRS = ("router", "data", "demo", "eval", "baseline")
FORBIDDEN_MODEL_SUFFIXES = (".pt", ".pth", ".bin", ".safetensors", ".onnx")
FORBIDDEN_DATA_SUFFIXES = {
    ".jsonl",
    ".csv",
    ".tsv",
    ".parquet",
    ".arrow",
    ".feather",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".pkl",
    ".pickle",
    ".joblib",
    ".npy",
    ".npz",
}
FORBIDDEN_DATA_NAME_FRAGMENTS = (
    "cache",
    "dataset",
    "holdout",
    "labeled",
    "mining",
    "hard_case",
    "hard-case",
)
REQUIRED_DOCKERIGNORE_RULES = {
    "data/",
    "eval/",
    "router/",
    "baseline/",
    "demo/",
    "outputs/",
    "mining/",
    "datasets/",
    "artifacts/",
    "cache/",
    "caches/",
    "reports/",
    "scripts/*",
    "!scripts/preflight_submission.py",
    "**/*.jsonl",
    "**/*.csv",
    "**/*.tsv",
    "**/*.parquet",
    "**/*.sqlite",
    "**/*.db",
    "**/*.pkl",
    "**/*.joblib",
    "**/*cache*.json",
}


class Checks:
    """Collect failures so one run reports every actionable issue."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        print(f"{'PASS' if condition else 'FAIL'}  {message}")
        if not condition:
            self.failures.append(message)

    def finish(self) -> None:
        if self.failures:
            raise SystemExit(
                f"Submission preflight failed: {len(self.failures)} check(s)"
            )
        print("Submission preflight passed.")


def _read_json(path: Path, checks: Checks) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        checks.require(False, f"valid JSON at {path}: {exc}")
        return None


def _requirement_lines(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "-"))
    }


def _docker_logical_instructions(dockerfile: str) -> list[str]:
    """Join Dockerfile continuation lines without interpreting shell syntax."""
    instructions: list[str] = []
    current: list[str] = []
    for raw_line in dockerfile.splitlines():
        line = raw_line.strip()
        if not current and (not line or line.startswith("#")):
            continue
        continued = line.endswith("\\")
        if continued:
            line = line[:-1].rstrip()
        if line:
            current.append(line)
        if not continued and current:
            instructions.append(" ".join(current))
            current = []
    if current:
        raise ValueError("Dockerfile ends with an unterminated continuation")
    return instructions


def _docker_copy_sources(dockerfile: str) -> set[str]:
    """Return COPY sources and reject ADD, flags, JSON form, or malformed COPY."""
    sources: set[str] = set()
    for instruction in _docker_logical_instructions(dockerfile):
        try:
            parts = shlex.split(instruction, posix=True)
        except ValueError as exc:
            raise ValueError(f"invalid Docker instruction: {exc}") from exc
        if not parts:
            continue
        operation = parts[0].upper()
        if operation == "ADD":
            raise ValueError("ADD instructions are forbidden")
        if operation != "COPY":
            continue
        if len(parts) < 3:
            raise ValueError(f"malformed COPY instruction: {instruction}")
        if any(part.startswith("--") for part in parts[1:-1]):
            raise ValueError("COPY flags are not permitted in the submission Dockerfile")
        if instruction.lstrip().upper().startswith("COPY ["):
            raise ValueError("JSON-form COPY is not permitted")
        sources.update(parts[1:-1])
    return sources


def _runtime_relative_paths(root: Path) -> set[str]:
    """Return the complete relative /app tree, including directories."""
    return {path.relative_to(root).as_posix() for path in root.rglob("*")}


def _find_forbidden_data_artifacts(root: Path) -> list[Path]:
    """Find offline datasets/caches in /app; /opt/contract is intentionally separate."""
    artifacts: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        lower_name = path.name.lower()
        if (
            path.suffix.lower() in FORBIDDEN_DATA_SUFFIXES
            or any(fragment in lower_name for fragment in FORBIDDEN_DATA_NAME_FRAGMENTS)
        ):
            artifacts.append(path)
    return artifacts


def _check_contract_fixture(path: Path, checks: Checks) -> list[dict[str, Any]]:
    tasks = _read_json(path, checks) if path.is_file() else None
    checks.require(path.is_file(), "container contract fixture exists")
    valid = isinstance(tasks, list) and bool(tasks)
    if valid:
        ids: list[str] = []
        for task in tasks:
            if not isinstance(task, dict):
                valid = False
                break
            task_id = task.get("task_id")
            prompt = task.get("prompt")
            if not isinstance(task_id, str) or not task_id.strip():
                valid = False
                break
            if not isinstance(prompt, str) or not prompt.strip():
                valid = False
                break
            if "category" in task and not isinstance(task["category"], str):
                valid = False
                break
            ids.append(task_id)
        valid = valid and len(ids) == len(set(ids))
    checks.require(valid, "contract fixture has unique IDs and valid task fields")
    return tasks if isinstance(tasks, list) else []


def check_source(root: Path) -> None:
    checks = Checks()
    dockerfile_path = root / "Dockerfile"
    dockerignore_path = root / ".dockerignore"
    requirements_path = root / "requirements.txt"
    dev_requirements_path = root / "requirements-dev.txt"
    demo_requirements_path = root / "requirements-demo.txt"

    checks.require(dockerfile_path.is_file(), "Dockerfile exists")
    checks.require(requirements_path.is_file(), "runtime requirements exist")
    if not dockerfile_path.is_file() or not requirements_path.is_file():
        checks.finish()
        return

    dockerfile = dockerfile_path.read_text(encoding="utf-8")
    dockerignore = (
        dockerignore_path.read_text(encoding="utf-8").splitlines()
        if dockerignore_path.is_file()
        else []
    )
    runtime_requirements = _requirement_lines(requirements_path)

    checks.require(
        re.search(
            r"(?m)^FROM\s+python:3\.11\.15-slim@sha256:"
            r"e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3\s*$",
            dockerfile,
        )
        is not None,
        "Docker base pins the validated Python patch and image digest",
    )
    checks.require(
        'CMD ["python", "-m", "agent.agent"]' in dockerfile,
        "Docker CMD matches the Track 1 entrypoint",
    )
    checks.require(
        f"ENV ROUTER_MODE={EXPECTED_ROUTER_MODE}" in dockerfile,
        "Docker defaults to the verified tier-0 policy",
    )
    checks.require(
        not re.search(r"(?mi)^\s*(COPY|ADD)\s+\.\s", dockerfile),
        "Dockerfile uses selective COPY instructions",
    )
    checks.require(
        not re.search(r"(?mi)^\s*(ARG|ENV)\s+FIREWORKS_API_KEY\b", dockerfile),
        "Dockerfile never bakes the Fireworks API key",
    )
    try:
        copy_sources = _docker_copy_sources(dockerfile)
    except ValueError as exc:
        checks.require(False, f"Docker COPY instructions are valid and allowlisted: {exc}")
    else:
        checks.require(
            copy_sources == EXPECTED_DOCKER_COPY_SOURCES,
            "Docker COPY sources match the exact submission allowlist",
        )
    checks.require(
        "requirements-dev.txt" not in dockerfile
        and "requirements-demo.txt" not in dockerfile,
        "submission image installs runtime requirements only",
    )
    checks.require("&& pip check" in dockerfile, "Docker build verifies dependencies")
    checks.require("ENV PIP_NO_INDEX=1" in dockerfile, "runtime package downloads are disabled")
    checks.require(
        runtime_requirements == EXPECTED_RUNTIME_REQUIREMENTS,
        "runtime dependency set is minimal, exact, and pinned",
    )
    checks.require(
        all("==" in requirement for requirement in runtime_requirements),
        "direct runtime dependencies are exactly pinned",
    )
    checks.require(dev_requirements_path.is_file(), "development requirements exist")
    if dev_requirements_path.is_file():
        dev_requirements = _requirement_lines(dev_requirements_path)
        checks.require(
            {"torch==2.13.0", "transformers==5.13.1", "numpy==2.4.6", "pydantic==2.13.4"}
            <= dev_requirements,
            "ML and schema dependencies remain available for local development",
        )
    checks.require(demo_requirements_path.is_file(), "demo requirements exist")
    if demo_requirements_path.is_file():
        demo_requirements = _requirement_lines(demo_requirements_path)
        checks.require(
            {"pandas==3.0.3", "streamlit==1.59.1"} <= demo_requirements,
            "demo-only dependencies remain available outside the image",
        )

    checks.require(dockerignore_path.is_file(), ".dockerignore exists")
    checks.require(".env" in dockerignore, ".dockerignore excludes .env")
    checks.require(".env.*" in dockerignore, ".dockerignore excludes env variants")
    checks.require("router/" in dockerignore, ".dockerignore excludes all router artifacts")
    checks.require(
        REQUIRED_DOCKERIGNORE_RULES.issubset(set(dockerignore)),
        ".dockerignore excludes offline scripts, datasets, caches, and artifacts",
    )

    for env_name, expected in EXPECTED_MODEL_DEFAULTS.items():
        checks.require(
            f"ENV {env_name}={expected}" in dockerfile,
            f"{env_name} uses the verified allowed default",
        )

    _check_contract_fixture(
        root / "tests" / "fixtures" / "container_input" / "tasks.json", checks
    )
    checks.finish()


def _mock_contract_run(tasks: list[dict[str, Any]], checks: Checks) -> None:
    """Run agent.main while replacing the Fireworks boundary only."""
    try:
        import agent.agent as agent_module
    except Exception as exc:  # noqa: BLE001
        checks.require(False, f"agent entrypoint imports without ML packages: {exc}")
        return

    def fake_chat_safe(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        prompt = str(kwargs.get("prompt", ""))
        text = (
            '{"persons":["Ada Lovelace"],"organizations":["Royal Society"],'
            '"locations":["London"]}'
            if "persons" in prompt and "organizations" in prompt
            else "Ottawa"
        )
        return {
            "text": text,
            "total_tokens": 1,
            "prompt_tokens": 1,
            "completion_tokens": 0,
            "finish_reason": "stop",
            "model_id": EXPECTED_MODEL_DEFAULTS["MODEL_TIER0"],
            "attempts": 1,
        }

    with tempfile.TemporaryDirectory(prefix="router-preflight-") as tmp:
        input_path = Path(tmp) / "input" / "tasks.json"
        output_path = Path(tmp) / "output" / "results.json"
        input_path.parent.mkdir(parents=True)
        input_path.write_text(json.dumps(tasks), encoding="utf-8")

        previous = {
            "INPUT_PATH": agent_module.INPUT_PATH,
            "OUTPUT_PATH": agent_module.OUTPUT_PATH,
            "ROUTER_MODE": agent_module.ROUTER_MODE,
            "chat_safe": agent_module.chat_safe,
        }
        try:
            agent_module.INPUT_PATH = input_path
            agent_module.OUTPUT_PATH = output_path
            agent_module.ROUTER_MODE = EXPECTED_ROUTER_MODE
            agent_module.chat_safe = fake_chat_safe
            exit_code = agent_module.main()
        except Exception as exc:  # noqa: BLE001
            checks.require(False, f"mocked container contract executes: {exc}")
            return
        finally:
            for name, value in previous.items():
                setattr(agent_module, name, value)

        results = _read_json(output_path, checks) if output_path.is_file() else None
        expected_ids = [task["task_id"] for task in tasks]
        actual_ids = (
            [item.get("task_id") for item in results]
            if isinstance(results, list)
            and all(isinstance(item, dict) for item in results)
            else []
        )
        answers_valid = isinstance(results, list) and len(results) == len(tasks) and all(
            isinstance(item, dict)
            and isinstance(item.get("answer"), str)
            and bool(item["answer"].strip())
            for item in results
        )
        checks.require(exit_code == 0, "agent exits zero for the contract fixture")
        checks.require(actual_ids == expected_ids, "results preserve task IDs and order")
        checks.require(answers_valid, "results contain one non-empty answer per task")


def check_runtime(root: Path = Path("/app")) -> None:
    checks = Checks()
    machine = platform.machine().lower()
    checks.require(platform.system() == "Linux", "runtime OS is Linux")
    checks.require(machine in {"x86_64", "amd64"}, "runtime architecture is amd64")
    checks.require(
        os.environ.get("ROUTER_MODE") == EXPECTED_ROUTER_MODE,
        "image defaults to verified_tier0 mode",
    )
    checks.require(
        os.environ.get("FIREWORKS_BASE_URL") == EXPECTED_BASE_URL,
        "Fireworks base URL uses the inference endpoint",
    )
    checks.require(os.environ.get("PIP_NO_INDEX") == "1", "runtime downloads are disabled")

    try:
        from config import get_model_id_for_tier

        resolved = {
            f"MODEL_TIER{index}": get_model_id_for_tier(f"tier{index}")
            for index in range(4)
        }
    except Exception as exc:  # noqa: BLE001
        checks.require(False, f"model configuration resolves against ALLOWED_MODELS: {exc}")
    else:
        checks.require(
            resolved == EXPECTED_MODEL_DEFAULTS,
            "all configured answer arms are verified Fireworks model IDs",
        )

    env_files = [
        path for path in root.rglob("*") if path.is_file() and path.name.startswith(".env")
    ]
    checks.require(not env_files, "image contains no .env or secret env file")
    checks.require(
        _runtime_relative_paths(root) == EXPECTED_APP_PATHS,
        "/app contains exactly the allowlisted runtime files and directories",
    )
    checks.require(
        all(not (root / name).exists() for name in FORBIDDEN_RUNTIME_DIRS),
        "router, model, data, eval, baseline, and demo trees are absent",
    )
    model_artifacts = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_MODEL_SUFFIXES
    ]
    checks.require(not model_artifacts, "image contains no local model weights")
    checks.require(
        not _find_forbidden_data_artifacts(root),
        "image contains no offline dataset, mining, or cache artifacts",
    )
    for module in FORBIDDEN_RUNTIME_MODULES:
        checks.require(importlib.util.find_spec(module) is None, f"{module} is absent")

    runtime_requirements = _requirement_lines(root / "requirements.txt")
    checks.require(
        runtime_requirements == EXPECTED_RUNTIME_REQUIREMENTS,
        "installed image carries only the minimal direct requirements",
    )
    tasks = _check_contract_fixture(Path("/opt/contract/tasks.json"), checks)
    if tasks and not checks.failures:
        _mock_contract_run(tasks, checks)
    checks.finish()


def _image_env(config: dict[str, Any]) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in config.get("Env") or []:
        if "=" in item:
            key, value = item.split("=", 1)
            values[key] = value
    return values


def check_image(image: str) -> None:
    try:
        inspect = subprocess.run(
            ["docker", "image", "inspect", image],
            check=True,
            capture_output=True,
            text=True,
        )
        metadata = json.loads(inspect.stdout)[0]
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError, IndexError) as exc:
        raise SystemExit(f"Could not inspect image {image!r}: {exc}") from exc

    checks = Checks()
    checks.require(metadata.get("Os") == "linux", "image metadata OS is Linux")
    checks.require(metadata.get("Architecture") == "amd64", "image metadata is amd64")
    config = metadata.get("Config") or {}
    checks.require(config.get("Cmd") == EXPECTED_CMD, "image CMD is the Track 1 entrypoint")
    checks.require(config.get("WorkingDir") == "/app", "image working directory is /app")
    checks.require(
        int(metadata.get("Size") or 0) < MAX_IMAGE_BYTES,
        "image is smaller than 500 MB",
    )
    image_env = _image_env(config)
    checks.require(
        "FIREWORKS_API_KEY" not in image_env,
        "image config contains no Fireworks API key",
    )
    checks.require(
        image_env.get("ROUTER_MODE") == EXPECTED_ROUTER_MODE,
        "image config defaults to verified_tier0 mode",
    )
    checks.require(
        all(image_env.get(key) == value for key, value in EXPECTED_MODEL_DEFAULTS.items()),
        "image config contains the verified Fireworks model defaults",
    )
    checks.finish()

    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--entrypoint",
        "python",
        image,
        "-m",
        "scripts.preflight_submission",
        "--runtime",
    ]
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Offline runtime preflight failed for {image!r}: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source", action="store_true", help="check the source tree")
    mode.add_argument("--runtime", action="store_true", help="check inside the image")
    mode.add_argument("--image", metavar="REF", help="inspect and test a built image")
    args = parser.parse_args()

    if args.source:
        check_source(Path(__file__).resolve().parents[1])
    elif args.runtime:
        check_runtime()
    else:
        check_image(args.image)
    return 0


if __name__ == "__main__":
    sys.exit(main())
