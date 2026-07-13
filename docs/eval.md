# Evaluation Guide

## Two evaluation paths

The project separates arm replay from cascade evaluation.

```bash
# Free: replay recorded arm outcomes and build measured-only frontiers
python -m eval.evaluate_strategies

# Paid: generate fresh answers for the deployed cascade
python -m eval.holdout_generalization --strategy verified_tier0 --fireworks
```

The second command makes Fireworks answer and grader calls. Review the dataset,
policy, and expected spend before running it.

## Strategy definitions

| Strategy | Behavior | Runtime status |
|---|---|---|
| `verified_tier0` | Call tier0, run deterministic verification, then call tier3 once if verification definitely fails | Submission default |
| `always_tier0` | Call tier0 with no verifier fallback | Baseline |
| `always_tier3` | Call tier3 with no verifier fallback | Baseline |
| `binary` | Use the local DistilBERT tier0/tier3 classifier | Offline analysis only |
| `multitier` | Use the legacy local four-arm classifier | Offline analysis only |
| prompt baseline | Ask a Fireworks model to classify the route | Optional paid analysis |

The default verifier is deliberately narrow. It detects API/error markers,
truncation, explicit JSON and schema requirements, sentence/bullet/word limits,
allowed labels, number/name-only formats, Python syntax/function names, output-only
requirements, and high-precision `X-based` location cues. It is not a semantic
judge. Ambiguous or unsupported checks pass through, and it never generates
answer content.

The default runtime allows at most one fallback and counts both calls. The
baseline modes disable fallback so their token totals remain interpretable.
Answer requests use policy v2 (`e785...`) with `max_tokens=700`; the unchanged
deterministic verifier remains `verified-tier0-v1`. Request and verifier hashes
are tracked independently.

## Measured-only arm replay

`eval/evaluate_strategies.py` collapses exact normalized duplicates into prompt
groups. Repeated calls for one prompt become repeated observations of one group;
they do not count as independent examples or cross a train/validation split.

Each model is an independent arm. The evaluator uses only the recorded pass/fail
and raw `total_tokens` for the selected arm. An uncalled arm is unknown: it is not
assumed to pass and cannot borrow another model's token count. Partial strategies
report measured coverage but leave complete accuracy and token totals unknown.

The output has two scopes:

- historical replay, which includes all measured calls and may mix old request
  policies; and
- `current_request_policy_frontier`, which excludes every stale or untagged call.

The previously reported 12-group arm slice was generated under request policy v1
(`c2ac...`): tier0 and tier3 each passed 12/12, using 3,345 and 4,326 tokens.
After adopting v2 (`e785...`), those observations are stale and cannot populate
the v2 current-policy frontier. Re-run `eval.evaluate_strategies` after a refresh;
until then, complete v2 arm-frontier accuracy and token totals are unknown.

## Request-policy v2 evidence

The prompt-only A/B changed the request text while keeping the evaluated NER
cases and other controls fixed:

| Policy | NER accuracy | Raw tokens | Accuracy delta | Token delta |
|---|---:|---:|---:|---:|
| v1 (`c2ac...`) | 25/28 | 7,923 | baseline | baseline |
| v2 (`e785...`) | 26/28 | 8,134 | +3.57 pp | +2.66% |

On a separate 10-case external natural/public NER check, both policies passed
10/10; v2 used 2,902 tokens and v1 used 2,828. On the six locked NER mining-stress
prompts, both passed 6/6. These samples support the targeted NER contract but are
small internal checks, not official grading.

## Cascade evidence

The `verified_tier0` cascade must be evaluated with
`eval/holdout_generalization.py`, because arm-only replay cannot reconstruct a
verification decision and its second call from aggregate pass labels.

The completed fresh v2 Fireworks holdout is the primary internal cascade result:

| Tasks | Accuracy | Prompt tokens | Completion tokens | Raw answer tokens | Tokens/task | Attempts | Fallbacks |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 72 | 71/72 (98.611%) | 12,083 | 7,852 | 19,935 | 276.875 | 72 tier0 | 0 |

Code debugging, code generation, factual knowledge, logic, math, NER, and
summarization each passed 9/9. Sentiment passed 8/9. This is a locked internal
holdout result, not official leaderboard grading.

The only failure, `holdout_sentiment_03`, asks for one word on a factual delivery
review: "The package arrived on Tuesday and contained the three items listed on
the invoice." Ground truth is `neutral`; tier0 answered `positive`. The response
passed deterministic format verification because it is a valid allowed label;
the verifier intentionally does not infer sentiment semantics. A deterministic
regrade of the saved answer remained FAIL and made zero answer calls. Do not tune
the prompt, verifier, or router on this locked failure.

A retrospective replay of 63 saved holdout responses produced:

| Strategy | Accuracy | Raw tokens | Fallbacks |
|---|---:|---:|---:|
| `verified_tier0` | 63/63 | 18,793 | 1 |
| `always_tier3` | 63/63 | 19,261 | 0 |

The verifier caught the single failed tier0 NER response and recovered with one
fallback. Those v1 answers were generated under an older request policy, so the
result remains stale supporting history; it must not be mixed with the fresh v2
72-task result.

The full 10-task public fixture was also run locally with v2. Manual rubric review
showed an apparent 10/10, 3,200 raw tokens, zero fallbacks, and 3.2 seconds. The
comparable v1 run used 3,740 tokens, so v2 saved 540 tokens (14.4%). "Apparent" is
intentional: this was manual/local validation, not official leaderboard grading.

## Hashes and freshness

Evaluation artifacts record stable identities rather than relying on filenames:

- `dataset_sha256` identifies the task set;
- `request_policy_sha256` identifies prompt construction, temperature, token cap,
  category aliases, system instructions, and reasoning effort;
- `verification_policy_sha256` identifies the versioned deterministic verifier;
- `results_sha256` detects missing or edited result content;
- training artifacts additionally record source, grouped dataset, and split
  hashes.

Holdout regrade refuses dataset, request-policy, verifier-policy, or result hash
mismatches. When the verifier changes, bumping its policy version prevents an old
cascade result from being presented as current.

The canonical request hashes are
`c2acb787fc151252a7b536dda2ebbc1b466030c1af2519c2f8f6c46be7b58506`
for v1 and
`e78563784f3eff4ab0cb9aa8b1924897673ec0619e8c106cd23caf4935836838`
for v2. Environment overrides intentionally produce a different hash. V2 keeps
the 700-token cap. The verifier is still version `verified-tier0-v1`, so its hash
does not change merely because request text changed.

The fresh 72-task artifact is bound to:

- request policy:
  `e78563784f3eff4ab0cb9aa8b1924897673ec0619e8c106cd23caf4935836838`;
- verifier policy:
  `30a16cbbf509e24a035319218f6824c9a24ac18efcf463df06960d5da3a22c17`; and
- results:
  `d90d73b1973c2e05bc93a9d202cfc973497486a79857b328b8373f6b0af017c2`.

Any policy, dataset, or saved-result change invalidates that evidence boundary.

## Outputs

`eval/results.json` includes duplicate counts, arm coverage, unknown outcomes,
per-category raw-token statistics, empirical frontiers, Wilson bounds, policy
hashes, binary threshold sweeps when a checkpoint exists, and offline logistic/k-NN
shadow results. The shadow routers are diagnostics only and never enter the
submission runtime.

`eval/holdout_results.json` includes every task's initial arm, final arm,
attempted arms, fallback flag, verifier reason, raw tokens across all attempts,
grade, and the artifact hashes above.

## Hard-case mining workflow

Round 1 is a precommitted NER/summarization screen, not an adaptive rewrite of
the public or natural holdout sets. The generator creates 24 train candidates
(eight prompt families) and 12 locked stress candidates (four families). Whole
families are assigned before any model call, with 12 train + 6 stress prompts per
category. The public validation fixture and natural holdout remain evaluation-only
and are excluded from mining/training; generated prompts are also checked against
the original and natural-holdout prompts.

Generate the immutable split locally:

```bash
python -m data.generate_hard_candidates
```

The generator refuses to overwrite an existing split by default. Reuse and
audit an existing manifest; never regenerate or `--force` the split after model
outcomes have been observed.

Before the first paid probe, inspect both partitions:

```bash
python -m data.label_multitier \
  --input data/mining/train_candidates.jsonl \
  --output data/mining/train_labeled.jsonl \
  --expected-split train --tier tier0 --dry-run

python -m data.label_multitier \
  --input data/mining/stress_candidates.jsonl \
  --output data/mining/stress_labeled.jsonl \
  --expected-split stress_holdout --tier tier0 --dry-run
```

Removing `--dry-run` from those two reviewed commands probes tier0 on all 36
candidates. With Fireworks judging, that paid stage is exactly 36 answer calls
plus 36 judge calls. The train/stress assignment is already fixed, so outcomes
cannot influence the split.

All mining observations used for selection must carry the v2 request hash. If a
label file contains v1 calls, use the reviewed
`--fill-missing-tiers --refresh-policy` path before screening; the selector
rejects stale rows.

After complete current-policy tier0 coverage, select failures offline only:

```bash
python -m data.select_hard_cases --screen-only
```

This command makes no calls and fails closed on missing, stale, malformed, or
infrastructure-error observations. Its screen output contains only genuine tier0
quality failures, separately for train and locked stress.

Only those failures may receive higher-arm calls. Preview the exact second-stage
scope first:

```bash
python -m data.label_multitier \
  --input data/mining/selected/train_hard.jsonl \
  --output data/mining/train_labeled.jsonl \
  --expected-split train --fill-missing-tiers --refresh-policy \
  --tier tier1 --tier tier2 --tier tier3 --dry-run

python -m data.label_multitier \
  --input data/mining/selected/stress_hard.jsonl \
  --output data/mining/stress_labeled.jsonl \
  --expected-split stress_holdout --fill-missing-tiers --refresh-policy \
  --tier tier1 --tier tier2 --tier tier3 --dry-run
```

Review and approve those failure-only calls before removing `--dry-run`. Then
rebuild the final selection from the updated labeled files:

```bash
python -m data.select_hard_cases --force
```

A tier0 failure enters the final mined set only when at least one higher arm has
a valid current-request-policy passing result. The final manifest preserves the
original family split and records the recovery audit.

Binary training combines the original labels with mined **train** failures:

```bash
python -m router.train_binary_router \
  --data-path data/labeled_multitier.jsonl \
  --data-path data/mining/selected/train_hard.jsonl
```

Never pass `stress_labeled.jsonl` or `selected/stress_hard.jsonl` to training.
The loader rejects every `dataset_split=stress_holdout` row; that partition stays
locked for external generalization measurement.

Request policy v2 is now the frozen serving/mining policy. Do not alter its system
prompts, category instructions, reasoning settings, or `max_tokens=700` during
the round. Another change creates a third request hash, invalidates v2 probe
labels and cascade evidence, and requires fresh validation.

## Binary analysis gate

The historical training file has 176 rows, 110 exact prompt groups, and roughly
three conservative hard groups. Overall accuracy is therefore dominated by the
constant tier0 class. Binary reports must include hard precision, recall, F1,
balanced accuracy, confusion matrices, and the constant baseline.

```bash
# Writes a candidate under router/checkpoints/candidates/
python -m router.train_binary_router

# Explicit promotion attempt; expected to fail with insufficient hard groups
python -m router.train_binary_router --promote
```

Promotion requires at least 10 unique hard groups plus hard recall/F1,
balanced-accuracy, and overall-accuracy guards. Even a promoted checkpoint would
remain outside the current submission image until the container policy were
deliberately changed and revalidated.

## Targeted collection

Inspect calls needed to refresh the stale v1 NER/summarization arm slice to v2
without spending:

```bash
python -m data.label_multitier --fill-missing-tiers --refresh-policy \
  --category ner --category summarization --tier tier0 --tier tier3 \
  --unique-prompts --dry-run
```

Only remove `--dry-run` after reviewing and approving the paid scope. The labeler
updates selected stale/missing arms atomically and never assumes monotonic tiers.
