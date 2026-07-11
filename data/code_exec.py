"""Grade generated code by actually executing it against test cases.

Used for code_generation tasks: the ground truth is a JSON spec
{"function_name": ..., "tests": [{"args": [...], "expected": ...}, ...]}
and an answer passes only if every test passes in a subprocess.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(answer_text: str) -> str:
    """Pull Python source out of a markdown fence; fall back to the raw text."""
    match = CODE_BLOCK_RE.search(answer_text)
    return match.group(1) if match else answer_text


def run_tests(answer_text: str, function_name: str, tests: list, timeout: int = 10) -> bool:
    """Execute the answer's code plus a test harness in a subprocess.

    The harness prints PASS as its last line only if every test case matches.
    Any exception, timeout, or wrong result counts as a failure.
    """
    code = extract_code(answer_text)
    harness = (
        code
        + "\n\n"
        + f"_tests = {tests!r}\n"
        + f"_fn = {function_name}\n"
        + "_all_ok = True\n"
        + "for _t in _tests:\n"
        + "    try:\n"
        + "        if _fn(*_t['args']) != _t['expected']:\n"
        + "            _all_ok = False\n"
        + "    except Exception:\n"
        + "        _all_ok = False\n"
        + "print('PASS' if _all_ok else 'FAIL')\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(harness)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip().endswith("PASS")
    except subprocess.TimeoutExpired:
        return False
    finally:
        Path(path).unlink(missing_ok=True)
