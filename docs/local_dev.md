# Local Development Guide

## Install

PowerShell:

```powershell
cd C:\Users\ro\Documents\amd-act2-router
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

The dependency files are intentionally separated:

- `requirements.txt`: minimal submission runtime; no ML or UI packages;
- `requirements-dev.txt`: tests, evaluation, DistilBERT training, and legacy
  analysis; and
- `requirements-demo.txt`: development dependencies plus Streamlit and pandas.

Fill `.env` with the Fireworks key and model IDs returned by the hackathon's
`ALLOWED_MODELS` command. The normal local default is
`ROUTER_MODE=verified_tier0`, matching the container.

## Default cascade

`verified_tier0` calls tier0 first. `agent/quality_gate.py` then checks only
deterministic requirements that can be proven from the prompt and response. These
include explicit formats, counts, JSON schema, requested code structure,
truncation/errors, and high-precision location cues such as `X-based` in NER.

Unsupported semantic questions pass through. The verifier does not judge factual
correctness, create entities, rewrite reasoning, or supply an answer. If a check
definitely fails, the default `ANSWER_FALLBACKS=1` permits one different
Fireworks arm. Set `ANSWER_FALLBACKS=0` only for a controlled no-fallback baseline.

The adopted answer policy is v2
(`e78563784f3eff4ab0cb9aa8b1924897673ec0619e8c106cd23caf4935836838`)
with `max_tokens=700`. The verifier did not change and remains
`verified-tier0-v1`; its separate hash continues to identify the same local
checks.

Serving, labeling, the demo, and Fireworks holdout evaluation share
`agent/request_policy.py`. Changing the system prompt, temperature, maximum
tokens, category behavior, or reasoning effort changes the request-policy hash
and makes previous measurements stale. The versioned verifier has a separate
verification-policy hash for the same reason.

## Evidence audit

```powershell
python -m eval.evaluate_strategies
python -m tests.smoke_test
```

The controlled 28-case NER prompt A/B gave v1 (`c2ac...`) 25/28 at 7,923 raw
tokens and v2 (`e785...`) 26/28 at 8,134: +3.57 percentage points for +2.66%
tokens. On a separate 10-case external natural/public NER check, both passed
10/10 (v2 2,902 tokens versus v1 2,828); both also passed 6/6 locked NER mining
stress prompts. These are internal checks, not official grading.

The earlier 12-group tier0/tier3 arm refresh (12/12 each; 3,345 versus 4,326
tokens) used v1 and is stale after adopting v2. It is no longer current-policy
frontier evidence and must be refreshed before arm comparisons are reported.

A retrospective 63-task stale-policy replay caught one bad tier0 NER response,
fell back once, and achieved 63/63 with 18,793 tokens versus 19,261 for
`always_tier3`. Treat that as a mechanism check, not the final score.

The full 10-task public fixture produced an apparent manual 10/10 under v2 with
3,200 tokens, zero fallbacks, and 3.2 seconds. V1 used 3,740 tokens, so v2 saved
540 (14.4%). This was a local/manual rubric check, not official leaderboard
grading.

## Run the agent locally

```powershell
New-Item -ItemType Directory -Force out | Out-Null
$env:INPUT_PATH = "tests\fixtures\container_input\tasks.json"
$env:OUTPUT_PATH = "out\results.json"
$env:ROUTER_MODE = "verified_tier0"
$env:ANSWER_FALLBACKS = "1"
python -m agent.agent
Get-Content out\results.json
```

The run uses Fireworks for both the primary answer and any fallback. Calls are
concurrent, output order matches input order, and token logs include all attempts.

## Fresh holdout validation

Free local mode checks plumbing only; one local model cannot compare Fireworks
arms:

```powershell
$env:DEV_LOCAL_ANSWERS = "1"
python -m eval.holdout_generalization --strategy verified_tier0
```

The production-like command is paid:

```powershell
Remove-Item Env:DEV_LOCAL_ANSWERS -ErrorAction SilentlyContinue
python -m eval.holdout_generalization --strategy verified_tier0 --fireworks
```

Review the holdout and expected spend before running it. The saved artifact
includes dataset, request-policy, verification-policy, and result hashes. Regrade
rejects stale or modified artifacts and uses judge tokens even though it makes no
new answer calls.

Optional local Ollama configuration for development judging:

```env
LOCAL_LLM_BASE_URL=http://localhost:11434/v1
LOCAL_LLM_MODEL=llama3.2
JUDGE_BACKEND=auto
```

Never set `DEV_LOCAL_ANSWERS` in the submission environment.

## Targeted arm collection

The old NER/summarization arm refresh is v1-stale. Preview the calls required to
refresh it to v2:

```powershell
python -m data.label_multitier --fill-missing-tiers --refresh-policy `
  --category ner --category summarization --tier tier0 --tier tier3 `
  --unique-prompts --dry-run
```

The dry run makes no API calls. Removing `--dry-run` is a paid action; do so only
after reviewing the selected prompt groups and arm calls. The labeler updates rows
atomically, tags v2 observations, and keeps missing arm outcomes unknown.

## Hard-case mining round 1

Hard-case mining expands the analysis data without contaminating evaluation.
Round 1 contains 24 train candidates and 12 locked stress candidates across NER
and summarization. The generator assigns whole prompt families first (eight train,
four stress), before any answer is observed. Public validation tasks and the
natural holdout are not mining inputs or training rows.

Generate the deterministic split for free:

```powershell
python -m data.generate_hard_candidates
```

Existing split artifacts are immutable after calls begin. If the manifest is
already present, inspect and reuse it; do not regenerate it with `--force` after
observing outcomes.

Dry-run the tier0 screen for both partitions:

```powershell
python -m data.label_multitier `
  --input data/mining/train_candidates.jsonl `
  --output data/mining/train_labeled.jsonl `
  --expected-split train --tier tier0 --dry-run

python -m data.label_multitier `
  --input data/mining/stress_candidates.jsonl `
  --output data/mining/stress_labeled.jsonl `
  --expected-split stress_holdout --tier tier0 --dry-run
```

The reviewed paid probe is the same two commands without `--dry-run`: 36 tier0
answer calls and, with Fireworks judging, 36 judge calls. Do not change the seed,
manifest, partitions, request policy, or outputs after observing results.

Screening accepts only observations carrying the current v2 request hash. Any v1
mining output must be refreshed with the reviewed
`--fill-missing-tiers --refresh-policy` workflow before `--screen-only` can
succeed.

Once all 36 tier0 observations are complete and current, screen failures locally:

```powershell
python -m data.select_hard_cases --screen-only
```

Screening makes no calls. It excludes tier0 passes as well as missing, stale,
invalid, or infrastructure-error results. Only the selected tier0 failures move
to higher-arm measurement. Preview that scope before spending:

```powershell
python -m data.label_multitier `
  --input data/mining/selected/train_hard.jsonl `
  --output data/mining/train_labeled.jsonl `
  --expected-split train --fill-missing-tiers --refresh-policy `
  --tier tier1 --tier tier2 --tier tier3 --dry-run

python -m data.label_multitier `
  --input data/mining/selected/stress_hard.jsonl `
  --output data/mining/stress_labeled.jsonl `
  --expected-split stress_holdout --fill-missing-tiers --refresh-policy `
  --tier tier1 --tier tier2 --tier tier3 --dry-run
```

After reviewing and running the paid failure-only calls, finalize offline:

```powershell
python -m data.select_hard_cases --force
```

Final selection requires both a current-policy tier0 failure and a valid
current-policy pass from at least one higher arm. A failure with no confirmed
recovery is not a binary routing example.

Train a candidate from the original data plus mined train failures only:

```powershell
python -m router.train_binary_router `
  --data-path data/labeled_multitier.jsonl `
  --data-path data/mining/selected/train_hard.jsonl
```

The locked `stress_labeled.jsonl` and `selected/stress_hard.jsonl` files must
never be training inputs. The trainer rejects `stress_holdout` rows even if they
are passed accidentally.

Keep adopted request policy v2 and verifier v1 unchanged during this round. In
particular, do not change category system prompts or `max_tokens=700`: another
change produces a third request hash, makes v2 observations stale, and resets the
current validation evidence. Further prompt/token optimization must be a separate,
freshly measured experiment.

## Offline learned-router analysis

DistilBERT, the legacy multitier model, logistic regression, and k-NN are
development experiments only. The 176 historical rows collapse to 110 prompt
groups with roughly three conservative hard groups, below the binary promotion
minimum.

```powershell
# Candidate only; does not alter the default runtime
python -m router.train_binary_router
python -m router.train_multitier_router
```

Binary thresholds are relevant only when explicitly running source analysis with
`ROUTER_MODE=binary`. No checkpoint or ML dependency is copied into the
submission image.

## Container candidate

```powershell
python -m scripts.preflight_submission --source
docker build --platform linux/amd64 -t amd-act2-router:candidate .
python -m scripts.preflight_submission --image amd-act2-router:candidate
```

The preflight checks the exact CMD, architecture, model defaults, allowlist
resolution, minimal dependency set, absence of ML/model artifacts and secrets,
offline mocked contract, and image size below 500 MB. Keep a known-good tag until
the candidate passes.
