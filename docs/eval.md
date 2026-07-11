# Evaluation Guide

## Running

```bash
python -m eval.evaluate_strategies
# optionally, to also measure the LLM prompt baseline (one Fireworks call per example):
python -m eval.evaluate_strategies --include-prompt-baseline
```

Prerequisites: `data/labeled_multitier.jsonl` exists (labeling done) and, for the
`multitier_router` row, a trained checkpoint in `router/checkpoints/`.

## What each strategy means

| Strategy | Routing decision | Routing token cost |
|---|---|---|
| `always_tier0` | none — always cheapest model | 0 |
| `always_tier3` | none — always strongest model | 0 |
| `multitier_router` | local fine-tuned classifier | 0 |
| `prompt_baseline` | LLM classifies the query into a tier | paid per query |

## How outcomes are computed

Answer outcomes are **replayed from labeling data**, not re-generated: during
labeling we recorded, per tier, whether the answer passed grading and how many
tokens it cost. Evaluation just looks up those results for whichever tier a
strategy picks.

One caveat: default labeling stops at the first passing tier, so stronger tiers
were never attempted for that task. For those the evaluator assumes stronger
tiers also pass and reuses the nearest measured token count. The report prints
how many outcomes were *assumed* vs *measured* per strategy. To eliminate the
assumption, run labeling as:

```bash
python -m data.label_multitier --all-tiers
```

## Reading the results

- **Accuracy** — fraction of tasks whose selected tier's answer passed grading.
- **Answer tokens** — Fireworks tokens spent generating answers.
- **Routing tokens** — extra tokens spent *deciding* (only nonzero for
  `prompt_baseline`). This gap is the core argument for a local router.
- **Tok/ex** — total tokens per example; the headline efficiency number.
- **Tier usage** — how many tasks each strategy sent to each tier. A healthy
  multi-tier router should send most tasks to tier0/tier1 and escalate rarely.
- `eval/results.json` additionally contains per-category and per-difficulty
  accuracy — check these before tuning: if a single category drags accuracy
  down, that's where to add training data.

## What "good" looks like

The router wins if it (a) matches or beats `always_tier3` accuracy while using
far fewer tokens, and (b) matches `prompt_baseline` accuracy while spending
zero routing tokens. If `always_tier0` already passes nearly everything, that
itself is a finding (the tutorial hit exactly this) — report it honestly and
consider whether the tier spread of chosen models is wide enough to matter.
