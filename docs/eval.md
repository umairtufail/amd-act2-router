# Evaluation Guide

## Running

```bash
python -m eval.evaluate_strategies
# optionally, to also measure the LLM prompt baseline (one Fireworks call per example):
python -m eval.evaluate_strategies --include-prompt-baseline
```

Prerequisites: `data/labeled_multitier.jsonl` exists and the default binary
checkpoint has been trained with `python -m router.train_binary_router`. The
legacy `multitier_router` row appears when its separate checkpoint is available;
missing router checkpoints are skipped with a warning.

## What each strategy means

| Strategy | Routing decision | Routing token cost |
|---|---|---|
| `always_tier0` | none — always cheapest model | 0 |
| `always_tier3` | none — always strongest model | 0 |
| `binary_router` | easy-category rules plus local `P(cheap_ok)` confidence gate; selects tier0/tier3 | 0 |
| `binary_router_tau_sweep` | binary policy at tau `0.6`, `0.7`, `0.8`, and `0.9` | 0 |
| `multitier_router` | legacy local four-class classifier for A/B comparison | 0 |
| `prompt_baseline` | LLM classifies the query into a tier | paid per query |

The default binary strategy uses `BINARY_ROUTER_TAU` (default `0.8`) and lets
`NER_BINARY_TAU` override it for NER; when the NER variable is unset, it inherits
the general threshold. Easy categories always stay on tier0. In the sweep, each
listed tau is applied to both routed categories so the comparison is independent
of environment overrides; easy-category decisions remain unchanged. Binary
escalation always targets tier3, never tier1 or tier2.

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
  binary router should send most tasks to tier0 and reserve tier3 for cases where
  `P(cheap_ok)` falls below the configured threshold.
- `eval/results.json` additionally contains per-category and per-difficulty
  accuracy — check these before tuning: if a single category drags accuracy
  down, that's where to add training data.
- Holdout results include `per_category` accuracy, correct, and total counts for
  all eight required categories.

The binary sweep prints `tau`, accuracy, total tokens, tokens per example, and
tier0/tier3 percentages. Full sweep metrics are written under the
`binary_router_tau_sweep` key in `eval/results.json`; use the trade-off table to
choose a threshold without changing the easy-category rules.

## Holdout generalization

```bash
python -m eval.holdout_generalization --strategy all
python -m eval.holdout_generalization --strategy all --fireworks
```

The default uses one configured `LOCAL_LLM_MODEL` for every routed tier, so it is
suited to checking routing flow, grading, and tier usage without Fireworks cost;
it does not measure tier0-versus-tier3 answer quality. Use `--fireworks` for a
real per-tier comparison. With `--strategy all`, learned strategies whose
checkpoint is unavailable are skipped with a warning; requesting `binary` or
`multitier` explicitly still requires that checkpoint.

If judging failed or was improved after an expensive Fireworks answer run,
regrade only the saved failures without regenerating any answers:

```bash
python -m eval.holdout_generalization --regrade --fireworks
```

This reads answer text from `eval/holdout_results.json`, uses the Fireworks judge
with a schema-constrained Boolean verdict, updates the saved accuracies, and makes
zero tier0/tier3 answer calls. It still consumes judge tokens for unique failures.

## What "good" looks like

The binary router wins if it preserves most of `always_tier3` accuracy while
staying close to `always_tier0` token use, and matches the optional prompt baseline
without paid routing tokens. If `always_tier0` already passes nearly everything,
that is itself a finding: prefer a conservative threshold justified by holdout
results rather than forcing unnecessary tier3 traffic.
