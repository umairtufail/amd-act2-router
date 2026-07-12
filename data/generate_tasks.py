"""Generate the raw task dataset across 8 categories.

Design principles (following the official tutorial's methodology, extended):
- Ground truths are deterministic and auditable: math answers are computed in
  Python, logic puzzles are brute-force checked for solution uniqueness,
  code-debugging outputs are captured by actually running the snippet,
  code-generation specs carry executable tests, and factual answers come from
  a curated bank of stable facts.
- Tasks span difficulty pools (trivial/medium/hard/adversarial) so the
  multi-tier labeling step has a real spread to measure.

Output: data/tasks_raw.jsonl (one JSON object per line).
"""

import argparse
import itertools
import json
import random
import subprocess
import sys
from pathlib import Path

from data.schema import TaskExample

OUT_PATH = Path(__file__).parent / "tasks_raw.jsonl"

NAME_POOL = ["Ivy", "Jude", "Kai", "Lena", "Moss", "Nora", "Omar", "Pia", "Quinn", "Rui"]


# ---------------------------------------------------------------------------
# math_reasoning — answers computed in Python, so they are correct by construction
# ---------------------------------------------------------------------------

def _gen_math(rng: random.Random, n: int) -> list[dict]:
    tasks = []
    makers = [_math_trivial, _math_medium, _math_hard, _math_adversarial]
    for i in range(n):
        pool_maker = makers[i % len(makers)]
        prompt, answer, pool = pool_maker(rng)
        tasks.append(dict(category="math_reasoning", difficulty_pool=pool,
                          prompt=prompt, ground_truth=str(answer)))
    return tasks


def _math_trivial(rng):
    a, b = rng.randint(7, 99), rng.randint(7, 99)
    if rng.random() < 0.5:
        return f"What is {a} + {b}? Answer with the number only.", a + b, "trivial"
    return f"What is {a} * {b}? Answer with the number only.", a * b, "trivial"


def _math_medium(rng):
    price = rng.randint(40, 400)
    disc = rng.choice([10, 20, 25, 50])
    tax = rng.choice([5, 8, 10])
    final = price * (100 - disc) / 100 * (100 + tax) / 100
    prompt = (f"An item costs {price} dollars. It is discounted by {disc}%, and then "
              f"{tax}% sales tax is applied to the discounted price. What is the final "
              f"price in dollars? Answer with the number only (round to 2 decimals).")
    return prompt, round(final, 2), "medium"


def _math_hard(rng):
    start = rng.randint(300, 800)
    d1, t1 = rng.randint(5, 12), rng.randint(10, 20)
    r, t2 = rng.randint(8, 15), rng.randint(10, 25)
    d2, t3 = rng.randint(3, 8), rng.randint(5, 15)
    final = start - d1 * t1 + r * t2 - d2 * t3
    prompt = (f"A tank starts with {start} liters. It drains at {d1} liters per minute "
              f"for {t1} minutes, then is refilled at {r} liters per minute for {t2} "
              f"minutes, then drains again at {d2} liters per minute for {t3} minutes. "
              f"How many liters are in the tank now? Answer with the number only.")
    return prompt, final, "hard"


def _math_adversarial(rng):
    # Multi-step with an irrelevant distractor number in the middle.
    workers_a, hours_a = rng.randint(3, 6), rng.randint(8, 16)
    total_units = workers_a * hours_a * rng.randint(4, 9)
    rate = total_units // (workers_a * hours_a)
    workers_b = rng.randint(7, 12)
    distractor = rng.randint(11, 19)
    hours_b = total_units / (workers_b * rate)
    prompt = (f"{workers_a} workers produce {total_units} units in {hours_a} hours, all "
              f"working at the same constant rate. The factory cafeteria serves lunch at "
              f"{distractor}:00. How many hours would {workers_b} workers at the same "
              f"rate need to produce {total_units} units? Answer with the number only "
              f"(round to 2 decimals).")
    return prompt, round(hours_b, 2), "adversarial"


# ---------------------------------------------------------------------------
# logic_puzzles — brute-force verified to have exactly one solution
# ---------------------------------------------------------------------------

def _solve_line(names, constraint_fns):
    """Return up to 2 permutations satisfying all constraints (early exit)."""
    sols = []
    for perm in itertools.permutations(names):
        pos = {name: i + 1 for i, name in enumerate(perm)}
        if all(fn(pos) for fn in constraint_fns):
            sols.append(perm)
            if len(sols) > 1:
                break
    return sols


def _candidate_constraints(names, pos, rng):
    """All constraints that are TRUE of the target arrangement, as (text, fn) pairs."""
    cands = []
    for a in names:
        for b in names:
            if a == b:
                continue
            if pos[a] + 1 == pos[b]:
                cands.append((f"{a} is immediately to the left of {b}.",
                              lambda p, a=a, b=b: p[a] + 1 == p[b]))
            if pos[a] < pos[b]:
                cands.append((f"{a} is somewhere to the left of {b}.",
                              lambda p, a=a, b=b: p[a] < p[b]))
    # A couple of absolute anchors so uniqueness is always reachable.
    for a in rng.sample(names, 2):
        cands.append((f"{a} is at position {pos[a]}.",
                      lambda p, a=a, i=pos[a]: p[a] == i))
    rng.shuffle(cands)
    return cands


def _gen_logic(rng: random.Random, n: int) -> list[dict]:
    tasks = []
    pools = [("trivial", 3), ("medium", 4), ("hard", 5), ("adversarial", 5)]
    for i in range(n):
        pool, n_people = pools[i % len(pools)]
        names = rng.sample(NAME_POOL, n_people)
        target = rng.sample(names, n_people)
        pos = {name: j + 1 for j, name in enumerate(target)}

        chosen_texts, chosen_fns = [], []
        for text, fn in _candidate_constraints(names, pos, rng):
            chosen_texts.append(text)
            chosen_fns.append(fn)
            if len(_solve_line(names, chosen_fns)) == 1:
                break
        assert _solve_line(names, chosen_fns) == [tuple(target)], "puzzle not unique"

        ask = rng.randint(1, n_people)
        answer = target[ask - 1]
        clue_list = "\n".join(f"- {t}" for t in chosen_texts)
        prompt = (f"{n_people} people ({', '.join(sorted(names))}) stand in a line, "
                  f"positions numbered 1 (leftmost) to {n_people} (rightmost).\n"
                  f"{clue_list}\n"
                  f"Who is at position {ask}? Answer with the name only.")
        tasks.append(dict(category="logic_puzzles", difficulty_pool=pool,
                          prompt=prompt, ground_truth=answer))
    return tasks


# ---------------------------------------------------------------------------
# sentiment
# ---------------------------------------------------------------------------

_SENTIMENT_TEMPLATES = {
    "trivial": [
        ("I absolutely loved this product, it exceeded every expectation!", "positive"),
        ("Terrible experience. Broken on arrival and support ignored me.", "negative"),
        ("Best purchase I've made all year, works flawlessly.", "positive"),
        ("Complete waste of money, it stopped working after two days.", "negative"),
    ],
    "medium": [
        ("The package arrived on Tuesday and contained the items listed.", "neutral"),
        ("It does what it says. Nothing more, nothing less.", "neutral"),
        ("Decent build quality, though the battery life is only average.", "neutral"),
        ("The manual explains the setup steps in order.", "neutral"),
    ],
    "hard": [
        ("Oh great, another update that deletes my settings. Just what I needed.", "negative"),
        ("Sure, waiting 45 minutes on hold was exactly how I wanted to spend my evening.", "negative"),
        ("Wow, a charger that melts. Truly innovative engineering.", "negative"),
        ("Fantastic — the third replacement is broken too. Impressive consistency.", "negative"),
    ],
    "adversarial": [
        ("This is not bad at all — honestly I can't find anything to complain about.", "positive"),
        ("I expected to hate it, but I couldn't have been more wrong.", "positive"),
        ("It's not that I dislike it; I just can't say it impressed me either.", "neutral"),
        ("Far from perfect, yet nothing I'd actually call a problem.", "positive"),
    ],
}


def _gen_sentiment(rng: random.Random, n: int) -> list[dict]:
    tasks = []
    pools = list(_SENTIMENT_TEMPLATES.keys())
    for i in range(n):
        pool = pools[i % len(pools)]
        text, label = _SENTIMENT_TEMPLATES[pool][(i // len(pools)) % len(_SENTIMENT_TEMPLATES[pool])]
        prompt = (f'Classify the sentiment of this review as positive, negative, or '
                  f'neutral. Respond with exactly one word.\n\nReview: "{text}"')
        tasks.append(dict(category="sentiment", difficulty_pool=pool,
                          prompt=prompt, ground_truth=label))
    return tasks


# ---------------------------------------------------------------------------
# summarization — judge checks that the key facts survive the summary
# ---------------------------------------------------------------------------

_SUMMARIZATION_ITEMS = [
    ("trivial",
     "The city library will close at 3 PM on Friday instead of the usual 8 PM "
     "because of scheduled electrical maintenance. Normal hours resume Saturday.",
     "Key facts: library closes early at 3 PM on Friday; reason is electrical "
     "maintenance; normal hours resume Saturday."),
    ("trivial",
     "Registration for the spring marathon opens March 1st and costs $40. "
     "Participants who register before March 15th receive a free race shirt.",
     "Key facts: registration opens March 1st; costs $40; registering before "
     "March 15th gets a free shirt."),
    ("medium",
     "Researchers at the institute tested three battery chemistries over 2,000 "
     "charge cycles. The lithium-iron-phosphate cells retained 91% capacity, "
     "outperforming the nickel-manganese-cobalt cells at 78% and the older "
     "lithium-cobalt-oxide design at 64%. The team attributes the gap mainly to "
     "cathode degradation at high temperatures.",
     "Key facts: three battery chemistries tested over 2,000 cycles; LFP retained "
     "91%, NMC 78%, LCO 64%; gap attributed to cathode degradation at high "
     "temperatures."),
    ("medium",
     "The quarterly report shows revenue grew 12% year over year to $4.8 million, "
     "driven mostly by the subscription segment, while hardware sales declined 5%. "
     "Operating costs rose 8%, leaving net margin roughly flat at 14%.",
     "Key facts: revenue grew 12% YoY to $4.8M; subscriptions drove growth; "
     "hardware declined 5%; costs rose 8%; net margin flat at about 14%."),
    ("hard",
     "The committee initially proposed relocating the bus depot to the north site, "
     "but after the flood-risk assessment came back unfavorable, it reversed course "
     "and recommended the east site, despite that option costing $2 million more "
     "and requiring an extra year of construction. The final vote was 6 to 3.",
     "Key facts: committee reversed from north site to east site; reason was "
     "unfavorable flood-risk assessment; east site costs $2M more and takes one "
     "extra year; vote was 6 to 3."),
    ("hard",
     "Although early trials suggested the drug reduced symptoms by 30%, the larger "
     "phase-3 study found only an 11% reduction, which fell short of statistical "
     "significance. The company will not seek approval and instead plans a "
     "reformulated version targeting a narrower patient group.",
     "Key facts: early trials showed 30% reduction; phase-3 showed only 11%, not "
     "statistically significant; company will not seek approval; plans a "
     "reformulated version for a narrower group."),
]


def _gen_summarization(rng: random.Random, n: int) -> list[dict]:
    tasks = []
    for i in range(n):
        pool, passage, key_facts = _SUMMARIZATION_ITEMS[i % len(_SUMMARIZATION_ITEMS)]
        prompt = f"Summarize the following passage in 1-2 sentences:\n\n{passage}"
        tasks.append(dict(category="summarization", difficulty_pool=pool,
                          prompt=prompt, ground_truth=key_facts))
    return tasks


# ---------------------------------------------------------------------------
# ner — extraction graded by the judge against a JSON ground truth
# ---------------------------------------------------------------------------

_NER_ITEMS = [
    ("trivial",
     "Maria Chen, the CEO of Brightpath Labs, opened a new office in Toronto.",
     {"persons": ["Maria Chen"], "organizations": ["Brightpath Labs"],
      "locations": ["Toronto"]}),
    ("trivial",
     "Last week, Daniel Okafor joined Vertex Analytics in Berlin.",
     {"persons": ["Daniel Okafor"], "organizations": ["Vertex Analytics"],
      "locations": ["Berlin"]}),
    ("medium",
     "During the summit in Geneva, representatives from Helios Energy and the "
     "World Trade Organization met with minister Ana Sofia Duarte.",
     {"persons": ["Ana Sofia Duarte"],
      "organizations": ["Helios Energy", "World Trade Organization"],
      "locations": ["Geneva"]}),
    ("medium",
     "The merger between Northgate Systems and Kite Robotics was announced in "
     "Austin by spokesperson Liam O'Neill.",
     {"persons": ["Liam O'Neill"],
      "organizations": ["Northgate Systems", "Kite Robotics"],
      "locations": ["Austin"]}),
    ("hard",
     "Reporting from Sao Paulo, journalist Priya Nair noted that neither Apex "
     "Capital nor its subsidiary Apex Ventures had commented, though sources in "
     "Lisbon suggested regulator ANATEL was already involved.",
     {"persons": ["Priya Nair"],
      "organizations": ["Apex Capital", "Apex Ventures", "ANATEL"],
      "locations": ["Sao Paulo", "Lisbon"]}),
    ("adversarial",
     "Jordan River, an analyst at Amazon, said the Amazon river project in Paris, "
     "Texas would not affect operations in Paris, France.",
     {"persons": ["Jordan River"], "organizations": ["Amazon"],
      "locations": ["Amazon river", "Paris, Texas", "Paris, France"]}),
]


def _gen_ner(rng: random.Random, n: int) -> list[dict]:
    tasks = []
    for i in range(n):
        pool, text, entities = _NER_ITEMS[i % len(_NER_ITEMS)]
        prompt = ("Extract all named entities from the sentence below. Respond with "
                  "ONLY a JSON object with keys \"persons\", \"organizations\", and "
                  f"\"locations\", each a list of strings.\n\nSentence: {text}")
        tasks.append(dict(category="ner", difficulty_pool=pool,
                          prompt=prompt, ground_truth=json.dumps(entities)))
    return tasks


# ---------------------------------------------------------------------------
# code_debugging — expected output captured by ACTUALLY running the snippet
# ---------------------------------------------------------------------------

_DEBUG_SNIPPETS = [
    ("medium", "funcs = []\nfor i in range(3):\n    funcs.append(lambda: i)\nprint([f() for f in funcs])\n"),
    ("medium", "def add_item(item, items=[]):\n    items.append(item)\n    return items\nprint(add_item(1))\nprint(add_item(2))\n"),
    ("hard", "gen = (x * x for x in range(4))\nprint(sum(gen))\nprint(sum(gen))\n"),
    ("hard", "a = [[0] * 3] * 2\na[0][1] = 5\nprint(a)\n"),
    ("adversarial", "x = 0.1 + 0.2\nprint(x == 0.3)\nprint(round(x, 1) == 0.3)\n"),
    ("adversarial", "d = {1: 'a', True: 'b', 1.0: 'c'}\nprint(len(d), d[1])\n"),
    ("hard", "s = 'abcdef'\nprint(s[::-2])\n"),
    ("medium", "nums = [1, 2, 3, 4]\nresult = filter(lambda n: n % 2, nums)\nnums.append(5)\nprint(list(result))\n"),
]


def _run_snippet(code: str) -> str:
    """Execute a snippet we wrote ourselves and capture its exact stdout."""
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, f"debug snippet crashed:\n{code}\n{result.stderr}"
    return result.stdout.strip()


def _gen_code_debugging(rng: random.Random, n: int) -> list[dict]:
    tasks = []
    for i in range(n):
        pool, code = _DEBUG_SNIPPETS[i % len(_DEBUG_SNIPPETS)]
        expected = _run_snippet(code)
        prompt = ("What is the exact output printed by this Python program? "
                  "Answer with the output only, no explanation.\n\n"
                  f"```python\n{code}```")
        tasks.append(dict(category="code_debugging", difficulty_pool=pool,
                          prompt=prompt, ground_truth=expected))
    return tasks


# ---------------------------------------------------------------------------
# code_generation — specs with executable tests (graded by data/code_exec.py)
# ---------------------------------------------------------------------------

_CODEGEN_SPECS = [
    ("trivial", "Write a Python function is_even(n) that returns True if the integer n is even.",
     {"function_name": "is_even",
      "tests": [{"args": [4], "expected": True}, {"args": [7], "expected": False},
                {"args": [0], "expected": True}, {"args": [-3], "expected": False}]}),
    ("trivial", "Write a Python function reverse_words(s) that reverses the order of words in the string s (words are separated by single spaces).",
     {"function_name": "reverse_words",
      "tests": [{"args": ["hello world"], "expected": "world hello"},
                {"args": ["a b c"], "expected": "c b a"},
                {"args": ["single"], "expected": "single"}]}),
    ("medium", "Write a Python function second_largest(nums) that returns the second largest distinct value in a list of integers, or None if it doesn't exist.",
     {"function_name": "second_largest",
      "tests": [{"args": [[3, 1, 4, 4, 2]], "expected": 3},
                {"args": [[5, 5, 5]], "expected": None},
                {"args": [[2, 9]], "expected": 2}]}),
    ("medium", "Write a Python function is_balanced(s) that returns True if the brackets ()[]{} in the string s are balanced and properly nested, ignoring other characters.",
     {"function_name": "is_balanced",
      "tests": [{"args": ["(a[b]{c})"], "expected": True},
                {"args": ["([)]"], "expected": False},
                {"args": [""], "expected": True},
                {"args": ["((("], "expected": False}]}),
    ("hard", "Write a Python function levenshtein(a, b) that returns the Levenshtein edit distance between strings a and b.",
     {"function_name": "levenshtein",
      "tests": [{"args": ["kitten", "sitting"], "expected": 3},
                {"args": ["", "abc"], "expected": 3},
                {"args": ["flaw", "lawn"], "expected": 2},
                {"args": ["same", "same"], "expected": 0}]}),
    ("hard", "Write a Python function merge_intervals(intervals) that merges overlapping intervals. Input is a list of [start, end] lists; return the merged list sorted by start.",
     {"function_name": "merge_intervals",
      "tests": [{"args": [[[1, 3], [2, 6], [8, 10]]], "expected": [[1, 6], [8, 10]]},
                {"args": [[[1, 4], [4, 5]]], "expected": [[1, 5]]},
                {"args": [[[5, 6]]], "expected": [[5, 6]]}]}),
    ("adversarial", "Write a Python function rle_decode(s) that decodes a run-length encoded string like 'a3b1c2' into 'aaabcc'. Counts can be multi-digit (e.g. 'x12').",
     {"function_name": "rle_decode",
      "tests": [{"args": ["a3b1c2"], "expected": "aaabcc"},
                {"args": ["x12"], "expected": "xxxxxxxxxxxx"},
                {"args": ["a1b10"], "expected": "abbbbbbbbbb"}]}),
    ("adversarial", "Write a Python function next_permutation(nums) that returns the next lexicographic permutation of the list of integers nums as a new list, or the sorted list if nums is the last permutation.",
     {"function_name": "next_permutation",
      "tests": [{"args": [[1, 2, 3]], "expected": [1, 3, 2]},
                {"args": [[3, 2, 1]], "expected": [1, 2, 3]},
                {"args": [[1, 1, 5]], "expected": [1, 5, 1]}]}),
]


def _gen_code_generation(rng: random.Random, n: int) -> list[dict]:
    tasks = []
    for i in range(n):
        pool, prompt_text, spec = _CODEGEN_SPECS[i % len(_CODEGEN_SPECS)]
        prompt = (prompt_text + " Return the complete function in a ```python code "
                  "block. Do not include example usage.")
        tasks.append(dict(category="code_generation", difficulty_pool=pool,
                          prompt=prompt, ground_truth=json.dumps(spec)))
    return tasks


# ---------------------------------------------------------------------------
# factual_knowledge — curated, stable facts with short deterministic answers
# ---------------------------------------------------------------------------

# Items are interleaved by difficulty so any prefix remains approximately
# balanced. Ground truth is a canonical answer with optional accepted aliases
# separated by ``||``; see data.judge.grade_factual_answer.
_FACTUAL_ITEMS = [
    ("trivial", "What is the capital city of France?", "Paris"),
    ("medium", "What is the SI unit of electrical resistance?", "ohm"),
    ("hard", "Which chemical element has atomic number 74?", "tungsten||wolfram"),
    ("adversarial", "Sydney and Melbourne are major Australian cities, but what is the capital of Australia?", "Canberra"),
    ("trivial", "What is the largest planet in the Solar System?", "Jupiter"),
    ("medium", "Who painted the anti-war mural Guernica?", "Pablo Picasso||Picasso"),
    ("hard", "Which moon of Saturn has a dense atmosphere made mostly of nitrogen?", "Titan"),
    ("adversarial", "Despite its name, in which country did the Panama hat originate?", "Ecuador"),
    ("trivial", "What is the chemical formula for water?", "H2O"),
    ("medium", "Who wrote the novel 1984?", "George Orwell"),
    ("hard", "To which language family does Hungarian belong?", "Uralic"),
    ("adversarial", "Which country gave the Statue of Liberty to the United States?", "France"),
    ("trivial", "What is the capital city of Japan?", "Tokyo"),
    ("medium", "What is the largest organ of the human body?", "skin"),
    ("hard", "What charter was sealed by King John of England in 1215?", "Magna Carta"),
    ("adversarial", "Which sea is defined by ocean currents rather than by land boundaries?", "Sargasso Sea"),
    ("trivial", "Which planet is commonly called the Red Planet?", "Mars"),
    ("medium", "Which chemical element has the symbol Au?", "gold"),
    ("hard", "Which continent lies in all four hemispheres?", "Africa"),
    ("adversarial", "Which planet has the shortest rotation period, and therefore the shortest day, in the Solar System?", "Jupiter"),
    ("trivial", "What is the largest ocean on Earth?", "Pacific Ocean||Pacific"),
    ("medium", "Which organelle is commonly called the powerhouse of the cell?", "mitochondrion||mitochondria"),
    ("hard", "What collective name is given to the 1648 treaties that ended the Thirty Years' War?", "Peace of Westphalia"),
    ("adversarial", "Which sovereign country is completely surrounded by South Africa?", "Lesotho"),
]


def _gen_factual_knowledge(rng: random.Random, n: int) -> list[dict]:
    del rng  # The curated fact bank is deterministic by design.
    tasks = []
    for i in range(n):
        pool, question, answer = _FACTUAL_ITEMS[i % len(_FACTUAL_ITEMS)]
        prompt = f"{question} Answer with the name or term only."
        tasks.append(dict(category="factual_knowledge", difficulty_pool=pool,
                          prompt=prompt, ground_truth=answer))
    return tasks


# ---------------------------------------------------------------------------

_GENERATORS = {
    "math_reasoning": _gen_math,
    "logic_puzzles": _gen_logic,
    "sentiment": _gen_sentiment,
    "summarization": _gen_summarization,
    "ner": _gen_ner,
    "code_debugging": _gen_code_debugging,
    "code_generation": _gen_code_generation,
    "factual_knowledge": _gen_factual_knowledge,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate raw router-training tasks.")
    parser.add_argument("--per-category", type=int, default=22,
                        help="number of tasks per category (default 22, ~176 total)")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed for reproducibility")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    all_tasks: list[TaskExample] = []
    for category, gen in _GENERATORS.items():
        raw = gen(rng, args.per_category)
        for j, t in enumerate(raw):
            all_tasks.append(TaskExample(id=f"{category}_{j:03d}", **t))
        print(f"  {category}: {len(raw)} tasks")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for task in all_tasks:
            f.write(task.model_dump_json() + "\n")
    print(f"Wrote {len(all_tasks)} tasks to {OUT_PATH}")


if __name__ == "__main__":
    # Human: run this locally or on the AMD pod when ready to generate tasks:
    #   python -m data.generate_tasks
    main()
