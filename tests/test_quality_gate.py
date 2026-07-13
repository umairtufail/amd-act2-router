"""Focused tests for the deterministic, zero-token answer quality gate."""

from agent.quality_gate import assess_answer


def test_global_failures_and_unknown_prompts_fail_open():
    assert not assess_answer("anything", "factual_knowledge", "").usable
    assert not assess_answer(
        "anything", "factual_knowledge", "ERROR: model call failed (timeout)"
    ).usable
    assert not assess_answer(
        "anything", "factual_knowledge", "partial answer", finish_reason="length"
    ).usable

    unknown = assess_answer("Explain this topic.", "new_category", "A nuanced answer.")
    assert unknown.usable
    assert unknown.text == "A nuanced answer."
    assert unknown.checks == ()


def test_sentiment_checks_format_without_guessing_label():
    one_word = assess_answer(
        "Classify as positive, negative, or neutral. Respond with exactly one word.",
        "sentiment_classification",
        " **Positive** ",
    )
    assert one_word.usable
    assert one_word.text == "positive"
    assert one_word.normalized

    assert not assess_answer(
        "Classify as positive, negative, or neutral. Respond with exactly one word.",
        "sentiment",
        "Positive because it worked.",
    ).usable

    mixed_reason = assess_answer(
        "Classify as Positive, Negative, or Neutral and give a one-sentence reason.",
        "sentiment",
        "Neutral — it contains both positive and negative evidence.",
    )
    assert mixed_reason.usable


def test_ner_json_is_checked_and_fence_removal_is_content_preserving():
    prompt = (
        'Respond with ONLY a JSON object with keys "persons", "organizations", '
        'and "locations", each a list of strings.'
    )
    result = assess_answer(
        prompt,
        "named_entity_recognition",
        '```json\n{"persons": ["Ada"], "organizations": [], "locations": []}\n```',
    )
    assert result.usable
    assert result.normalized
    assert result.text.startswith("{")

    wrong_shape = assess_answer(
        prompt,
        "ner",
        '{"persons": "Ada", "organizations": [], "locations": []}',
    )
    assert not wrong_shape.usable
    assert "list of strings" in wrong_shape.reason

    missing_key = assess_answer(prompt, "ner", '{"persons": [], "locations": []}')
    assert not missing_key.usable


def test_ner_json_flags_an_omitted_explicit_based_location():
    prompt = (
        'Respond with ONLY a JSON object with keys "persons", "organizations", '
        'and "locations", each a list of strings.\n\nSentence: Paris-based '
        'Paris Systems appointed Paris Morgan to lead its office in Lyon.'
    )
    missed = assess_answer(
        prompt,
        "ner",
        '{"persons":["Paris Morgan"],"organizations":["Paris Systems"],'
        '"locations":["Lyon"]}',
    )
    assert not missed.usable
    assert "Paris" in missed.reason
    complete = assess_answer(
        prompt,
        "ner",
        '{"persons":["Paris Morgan"],"organizations":["Paris Systems"],'
        '"locations":["Paris","Lyon"]}',
    )
    assert complete.usable


def test_labeled_ner_requires_at_least_one_requested_label():
    prompt = "Extract entities and label each as PERSON, ORGANIZATION, LOCATION, or DATE: Ada met Acme."
    assert assess_answer(prompt, "ner", "Ada — PERSON\nAcme — ORGANIZATION").usable
    assert not assess_answer(prompt, "ner", "Ada; Acme").usable


def test_exact_summary_sentence_and_bullet_constraints():
    two_sentence_prompt = "Summarize the passage in exactly two sentences: passage"
    assert assess_answer(two_sentence_prompt, "summarization", "First point. Second point.").usable
    assert not assess_answer(two_sentence_prompt, "summarization", "Only one point.").usable

    range_prompt = "Summarize in 1-2 sentences: passage"
    assert assess_answer(range_prompt, "summarization", "Version 2.0 shipped. Adoption grew.").usable

    bullets_prompt = (
        "Summarize in exactly three bullet points, each no longer than 4 words: passage"
    )
    assert assess_answer(
        bullets_prompt,
        "text_summarization",
        "- Remote work adds flexibility.\n- Collaboration remains difficult.\n- Offices support social work.",
    ).usable
    assert not assess_answer(
        bullets_prompt,
        "summarization",
        "- This bullet contains far too many unnecessary words.\n- Short item.\n- Last item.",
    ).usable
    assert not assess_answer(
        bullets_prompt,
        "summarization",
        "Remote work adds flexibility.\nCollaboration remains difficult.\nOffices support social work.",
    ).usable


def test_math_and_term_only_normalization_does_not_solve_task():
    math = assess_answer(
        "What is 47 + 47? Answer with the number only.",
        "mathematical_reasoning",
        "The answer is 94.",
    )
    assert math.usable
    assert math.text == "94"
    assert not assess_answer(
        "Answer with the number only.", "math_reasoning", "About 94"
    ).usable

    # The gate preserves even a factually wrong answer; it is not a local judge.
    factual = assess_answer(
        "What is the capital of France? Answer with the name or term only.",
        "factual_knowledge",
        "The answer is London.",
    )
    assert factual.usable
    assert factual.text == "London"


def test_python_code_contract_checks_syntax_and_function_name():
    prompt = (
        "Write a Python function is_even(n). Return the complete function in a "
        "```python code block. Do not include example usage."
    )
    good = assess_answer(prompt, "code_generation", "```python\ndef is_even(n):\n    return n % 2 == 0\n```")
    assert good.usable
    assert not good.normalized

    raw = assess_answer(prompt, "code_generation", "def is_even(n): return True")
    assert raw.usable
    assert raw.normalized
    assert raw.text.startswith("```python\n")
    prose = assess_answer(
        prompt,
        "code_generation",
        "Here is the function:\n\n```python\ndef is_even(n):\n    return n % 2 == 0\n```",
    )
    assert prose.usable
    assert prose.normalized
    assert prose.text.startswith("```python\n")
    assert "Here is" not in prose.text
    assert not assess_answer(prompt, "code_generation", "```python\ndef other(n):\n    return True\n```").usable
    assert not assess_answer(prompt, "code_generation", "```python\ndef is_even(:\n```" ).usable


def test_debug_output_fence_is_safely_removed():
    prompt = "What is the exact output? Answer with the output only, no explanation."
    fenced = assess_answer(prompt, "code_debugging", "```text\n[2, 2, 2]\n```")
    assert fenced.usable
    assert fenced.text == "[2, 2, 2]"
    assert fenced.normalized
    assert not assess_answer(prompt, "code_debugging", "The output is: [2, 2, 2]").usable
