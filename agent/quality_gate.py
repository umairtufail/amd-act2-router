"""Conservative, zero-token checks for generated answer structure.

The quality gate deliberately does *not* attempt to judge whether an answer is
factually correct.  It only checks constraints that can be established from
the task text without an LLM (for example, valid JSON, an exact bullet count,
or a requested Python function).  Unsupported or ambiguous prompts pass
through unchanged.

``assess_answer`` never calls an API and never creates substantive answer
content.  The returned ``text`` is either the original answer with surrounding
whitespace removed, or a content-preserving formatting normalization such as
removing a Markdown fence when the prompt explicitly asks for JSON only.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass


VERIFICATION_POLICY_VERSION = "verified-tier0-v1"


def verification_policy_hash() -> str:
    """Stable identity used to reject stale cascade evaluations."""
    return hashlib.sha256(VERIFICATION_POLICY_VERSION.encode("utf-8")).hexdigest()


_CATEGORY_ALIASES = {
    "mathematical_reasoning": "math_reasoning",
    "sentiment_classification": "sentiment",
    "text_summarization": "summarization",
    "named_entity_recognition": "ner",
}

_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

_ERROR_MARKER = re.compile(
    r"^\s*(?:\[?error\]?\s*:|model\s+call\s+failed\b)", re.IGNORECASE
)
_NUMBER_ONLY = re.compile(
    r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?(?:/\d+)?"
)
_BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+(.+?)\s*$")
_WORD = re.compile(r"\b[^\W_]+(?:[-'\u2019][^\W_]+)*\b", re.UNICODE)


@dataclass(frozen=True)
class AnswerAssessment:
    """Result of deterministic answer checks.

    ``usable=False`` means a structural contract is definitely broken and the
    caller may choose to make another Fireworks call.  A caller that cannot
    retry can still return ``text``: the gate is fail-open and never discards
    the model response.
    """

    usable: bool
    reason: str
    text: str
    normalized: bool = False
    checks: tuple[str, ...] = ()


def _category_name(category: str | None) -> str:
    value = (category or "").strip().lower()
    return _CATEGORY_ALIASES.get(value, value)


def _failed(text: str, reason: str, *checks: str, normalized: bool = False):
    return AnswerAssessment(False, reason, text, normalized, tuple(checks))


def _passed(text: str, *checks: str, normalized: bool = False):
    reason = "passed deterministic format checks" if checks else "no safe format check applicable"
    return AnswerAssessment(True, reason, text, normalized, tuple(checks))


def _unwrap_fence(text: str, language: str | None = None) -> str | None:
    language_pattern = re.escape(language) if language else r"[A-Za-z0-9_+-]*"
    match = re.fullmatch(
        rf"```{language_pattern}\s*\n?(.*?)\n?```", text, re.IGNORECASE | re.DOTALL
    )
    return match.group(1).strip() if match else None


def _strip_answer_wrapper(text: str) -> str:
    """Remove only an unambiguous answer-label wrapper, not answer content."""
    match = re.fullmatch(
        r"(?:the\s+answer\s+is|answer)\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL
    )
    if not match:
        match = re.fullmatch(
            r"the\s+answer\s+is\s+(.+)", text, re.IGNORECASE | re.DOTALL
        )
    if not match:
        return text
    inner = match.group(1).strip()
    # A final full stop is presentation punctuation for an explicit "only"
    # response.  Other punctuation is retained because it may be substantive.
    if inner.endswith("."):
        inner = inner[:-1].rstrip()
    return inner


def _parse_count(value: str) -> int | None:
    value = value.lower()
    if value.isdigit():
        return int(value)
    return _NUMBER_WORDS.get(value)


def _sentence_count(text: str) -> int:
    """Count sentences while protecting common periods that are not stops."""
    protected = text.strip()
    if not protected:
        return 0
    protected = re.sub(r"(?<=\d)\.(?=\d)", "<DOT>", protected)
    protected = re.sub(
        r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e)\.",
        lambda match: match.group(0).replace(".", "<DOT>"),
        protected,
        flags=re.IGNORECASE,
    )
    protected = re.sub(
        r"\b(?:[A-Za-z]\.){2,}",
        lambda match: match.group(0).replace(".", "<DOT>"),
        protected,
    )
    parts = re.split(r"(?<=[.!?])[\"'\u2019\u201d)\]]*\s+", protected)
    return len([part for part in parts if part.strip()])


def _summary_constraints(prompt: str) -> tuple[int | None, int | None, int | None, int | None]:
    """Return min/max sentences, exact bullets, and per-bullet word cap."""
    lower = prompt.lower()
    exact = re.search(r"exactly\s+(\d+|[a-z]+)\s+sentences?\b", lower)
    if exact:
        count = _parse_count(exact.group(1))
        min_sentences = max_sentences = count
    else:
        sentence_range = re.search(
            r"\b(\d+|one|two|three)\s*(?:-|\u2013|to)\s*"
            r"(\d+|one|two|three)\s+sentences?\b",
            lower,
        )
        if sentence_range:
            min_sentences = _parse_count(sentence_range.group(1))
            max_sentences = _parse_count(sentence_range.group(2))
        else:
            min_sentences = max_sentences = None

    bullet_match = re.search(r"exactly\s+(\d+|[a-z]+)\s+bullet\s+points?\b", lower)
    exact_bullets = _parse_count(bullet_match.group(1)) if bullet_match else None
    cap_match = re.search(
        r"each\s+(?:no\s+longer\s+than|at\s+most)\s+(\d+)\s+words?\b", lower
    )
    word_cap = int(cap_match.group(1)) if cap_match else None
    return min_sentences, max_sentences, exact_bullets, word_cap


def _assess_summary(prompt: str, text: str, normalized: bool) -> AnswerAssessment:
    minimum, maximum, exact_bullets, word_cap = _summary_constraints(prompt)
    checks: list[str] = []

    if exact_bullets is not None:
        nonempty_lines = [line for line in text.splitlines() if line.strip()]
        bullets = [_BULLET.fullmatch(line) for line in nonempty_lines]
        checks.append("bullet_count")
        if len(nonempty_lines) != exact_bullets or not all(bullets):
            return _failed(
                text,
                f"expected exactly {exact_bullets} marked bullet points",
                *checks,
                normalized=normalized,
            )
        if word_cap is not None:
            checks.append("bullet_word_limit")
            too_long = [
                index + 1
                for index, match in enumerate(bullets)
                if len(_WORD.findall(match.group(1))) > word_cap
            ]
            if too_long:
                return _failed(
                    text,
                    f"bullet(s) {too_long} exceed the {word_cap}-word limit",
                    *checks,
                    normalized=normalized,
                )

    if minimum is not None and maximum is not None:
        checks.append("sentence_count")
        actual = _sentence_count(text)
        if not minimum <= actual <= maximum:
            expected = str(minimum) if minimum == maximum else f"{minimum}-{maximum}"
            return _failed(
                text,
                f"expected {expected} sentence(s), found {actual}",
                *checks,
                normalized=normalized,
            )

    return _passed(text, *checks, normalized=normalized)


def _json_keys_from_prompt(prompt: str) -> tuple[str, ...]:
    match = re.search(
        r"\bkeys?\s+(.+?)\s*,?\s*each\s+(?:a|an)\s+list\b",
        prompt,
        re.IGNORECASE | re.DOTALL,
    )
    return tuple(re.findall(r'["\u201c]([^"\u201d]+)["\u201d]', match.group(1))) if match else ()


def _explicit_location_cues(prompt: str) -> tuple[str, ...]:
    """Extract high-precision location cues for verification, never generation."""
    sentence = prompt.rsplit("Sentence:", 1)[-1]
    matches = re.findall(
        r"\b([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]*)-based\b", sentence
    )
    return tuple(dict.fromkeys(matches))


def _assess_ner(prompt: str, text: str, normalized: bool) -> AnswerAssessment:
    lower = prompt.lower()
    if "json object" in lower:
        candidate = text
        fenced = _unwrap_fence(candidate, "json")
        if fenced is None:
            fenced = _unwrap_fence(candidate)
        if fenced is not None and "only" in lower:
            candidate = fenced
            normalized = normalized or candidate != text
        try:
            value = json.loads(candidate)
        except (TypeError, ValueError):
            return _failed(candidate, "expected a valid JSON object", "json", normalized=normalized)
        if not isinstance(value, dict):
            return _failed(candidate, "JSON response is not an object", "json", normalized=normalized)

        required_keys = _json_keys_from_prompt(prompt)
        if required_keys:
            if set(value) != set(required_keys):
                return _failed(
                    candidate,
                    f"JSON keys must be exactly {list(required_keys)}",
                    "json",
                    "json_schema",
                    normalized=normalized,
                )
            if any(
                not isinstance(value[key], list)
                or any(not isinstance(item, str) for item in value[key])
                for key in required_keys
            ):
                return _failed(
                    candidate,
                    "each requested JSON field must be a list of strings",
                    "json",
                    "json_schema",
                    normalized=normalized,
                )
        locations = value.get("locations")
        if isinstance(locations, list):
            missing_cues = [
                cue
                for cue in _explicit_location_cues(prompt)
                if not any(
                    cue.casefold() in item.casefold()
                    for item in locations
                    if isinstance(item, str)
                )
            ]
            if missing_cues:
                return _failed(
                    candidate,
                    "locations list omits explicit X-based cue(s): "
                    + ", ".join(missing_cues),
                    "json",
                    "json_schema",
                    "location_cues",
                    normalized=normalized,
                )
        return _passed(candidate, "json", "json_schema", normalized=normalized)

    label_match = re.search(
        r"label\s+each\s+as\s+(.+?)(?:\n|:|\.|$)", prompt, re.IGNORECASE
    )
    if label_match:
        allowed = re.findall(r"\b[A-Z][A-Z_]+\b", label_match.group(1))
        if allowed and not re.search(
            r"\b(?:" + "|".join(map(re.escape, allowed)) + r")\b", text, re.IGNORECASE
        ):
            return _failed(
                text,
                "named entities are not annotated with any requested label",
                "entity_labels",
                normalized=normalized,
            )
        return _passed(text, "entity_labels", normalized=normalized)

    return _passed(text, normalized=normalized)


def _assess_sentiment(prompt: str, text: str, normalized: bool) -> AnswerAssessment:
    lower_prompt = prompt.lower()
    labels = [label for label in ("positive", "negative", "neutral", "mixed") if label in lower_prompt]
    if not labels:
        return _passed(text, normalized=normalized)

    if "exactly one word" in lower_prompt:
        plain = text.strip().strip("*_`")
        if plain.lower() not in labels:
            return _failed(
                text,
                f"expected exactly one of {labels}",
                "sentiment_label",
                normalized=normalized,
            )
        canonical = plain.lower()
        return _passed(
            canonical,
            "sentiment_label",
            normalized=normalized or canonical != text,
        )

    # For label-plus-reason tasks, only require an explicit classification.
    # Words such as "positive" later in the reason are evidence, not labels.
    plain = re.sub(r"^[*_`]+|[*_`]+$", "", text.strip())
    label_pattern = "|".join(map(re.escape, labels))
    classified = re.match(
        rf"^(?:(?:the\s+)?sentiment(?:\s+is)?\s*[:\-\u2014]?\s*)?"
        rf"(?:\*\*)?({label_pattern})(?:\*\*)?\b",
        plain,
        re.IGNORECASE,
    )
    if not classified:
        return _failed(
            text,
            "response does not begin with an allowed sentiment classification",
            "sentiment_label",
            normalized=normalized,
        )
    return _passed(text, "sentiment_label", normalized=normalized)


def _assess_number_only(prompt: str, text: str, normalized: bool) -> AnswerAssessment:
    candidate = _strip_answer_wrapper(text)
    normalized = normalized or candidate != text
    if not _NUMBER_ONLY.fullmatch(candidate):
        return _failed(
            candidate,
            "expected a number only",
            "number_only",
            normalized=normalized,
        )
    return _passed(candidate, "number_only", normalized=normalized)


def _assess_term_only(text: str, normalized: bool) -> AnswerAssessment:
    candidate = _strip_answer_wrapper(text)
    normalized = normalized or candidate != text
    if "\n" in candidate or "\r" in candidate:
        return _failed(
            candidate,
            "expected a single name or term without explanation",
            "term_only",
            normalized=normalized,
        )
    return _passed(candidate, "term_only", normalized=normalized)


def _assess_code_generation(prompt: str, text: str, normalized: bool) -> AnswerAssessment:
    lower = prompt.lower()
    if "```python code block" not in lower:
        return _passed(text, normalized=normalized)

    code = _unwrap_fence(text, "python")
    normalize_fence = False
    if code is None:
        embedded = re.findall(
            r"```python\s*\n?(.*?)\n?```", text, re.IGNORECASE | re.DOTALL
        )
        if len(embedded) == 1:
            code = embedded[0].strip()
            normalize_fence = True
        else:
            code = text
            normalize_fence = True
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return _failed(
            text,
            "response is not syntactically valid Python",
            "python_code_block",
            "python_syntax",
            normalized=normalized,
        )

    name_match = re.search(r"write\s+a\s+python\s+function\s+([A-Za-z_]\w*)\s*\(", prompt, re.IGNORECASE)
    if name_match:
        required = name_match.group(1)
        functions = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if required not in functions:
            return _failed(
                text,
                f"Python response does not define required function {required}",
                "python_code_block",
                "python_syntax",
                "function_name",
                normalized=normalized,
            )
    formatted = f"```python\n{code.strip()}\n```" if normalize_fence else text
    return _passed(
        formatted,
        "python_code_block",
        "python_syntax",
        *(('function_name',) if name_match else ()),
        normalized=normalized or normalize_fence,
    )


def _assess_code_debugging(prompt: str, text: str, normalized: bool) -> AnswerAssessment:
    lower = prompt.lower()
    if "output only" not in lower:
        return _passed(text, normalized=normalized)
    fenced = _unwrap_fence(text)
    if fenced is not None:
        return _passed(
            fenced,
            "output_only",
            normalized=True,
        )
    if re.match(r"^(?:the\s+)?output\s+(?:is|will\s+be)\s*:", text, re.IGNORECASE):
        return _failed(
            text,
            "expected program output only, without explanation",
            "output_only",
            normalized=normalized,
        )
    return _passed(text, "output_only", normalized=normalized)


def assess_answer(
    prompt: str,
    category: str | None,
    text: str | None,
    finish_reason: str | None = None,
) -> AnswerAssessment:
    """Assess deterministic answer structure without calling or emulating an LLM.

    Unknown categories and prompts without an explicit machine-checkable format
    constraint are usable by design.  This avoids turning a formatting heuristic
    into a semantic judge.
    """
    if not isinstance(text, str):
        return _failed("", "answer text is missing", "nonempty")
    stripped = text.strip()
    normalized = stripped != text
    if not stripped:
        return _failed(stripped, "answer text is empty", "nonempty", normalized=normalized)
    if _ERROR_MARKER.match(stripped):
        return _failed(
            stripped,
            "answer is an API error marker",
            "nonempty",
            "error_marker",
            normalized=normalized,
        )
    if (finish_reason or "").lower() in {"length", "content_filter"}:
        return _failed(
            stripped,
            f"completion ended with finish_reason={finish_reason}",
            "finish_reason",
            normalized=normalized,
        )

    category = _category_name(category)
    lower_prompt = prompt.lower()
    if category == "summarization":
        return _assess_summary(prompt, stripped, normalized)
    if category == "ner":
        return _assess_ner(prompt, stripped, normalized)
    if category == "sentiment":
        return _assess_sentiment(prompt, stripped, normalized)
    if category == "math_reasoning" and "number only" in lower_prompt:
        return _assess_number_only(prompt, stripped, normalized)
    if category in {"logic_puzzles", "factual_knowledge"} and re.search(
        r"answer\s+with\s+(?:the\s+)?(?:name|name\s+or\s+term|term)\s+only",
        lower_prompt,
    ):
        return _assess_term_only(stripped, normalized)
    if category == "code_generation":
        return _assess_code_generation(prompt, stripped, normalized)
    if category == "code_debugging":
        return _assess_code_debugging(prompt, stripped, normalized)
    return _passed(stripped, normalized=normalized)


__all__ = [
    "AnswerAssessment",
    "VERIFICATION_POLICY_VERSION",
    "assess_answer",
    "verification_policy_hash",
]
