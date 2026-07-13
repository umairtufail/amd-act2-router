# Container Guide

## Track 1 contract

The submission image is Linux/amd64 and starts with
`python -m agent.agent`. It reads `/input/tasks.json`, writes
`/output/results.json`, and exits after the fixed batch.

Input:

```json
[
  {"task_id": "1", "prompt": "Explain overfitting.", "category": "factual_knowledge"}
]
```

Output:

```json
[
  {"task_id": "1", "answer": "..."}
]
```

`task_id` and a non-empty `prompt` are required; `category` is optional. The
agent rejects invalid/duplicate IDs, preserves input order across concurrent
calls, and replaces the output file atomically.

## Packaged policy

The image defaults to `ROUTER_MODE=verified_tier0`:

1. call the configured tier0 Fireworks model;
2. build the answer with request policy v2 (`e785...`) and
   `max_tokens=700`;
3. run deterministic, zero-token verifier policy v1; and
4. if an explicit structure or high-precision location-cue requirement fails,
   make at most one fallback call (tier3 first in the default order).

The verifier also detects empty/error responses and truncation. It is fail-open
for unsupported or ambiguous semantic checks, never generates answer content,
and only performs content-preserving formatting normalization. Every substantive
answer attempt comes from an allowed Fireworks model, and logs count primary plus
fallback tokens.

The submission image contains no DistilBERT or legacy router code, Torch,
Transformers, NumPy, Pydantic, checkpoint, tokenizer, encoder configuration,
training/evaluation data, baseline router, demo, pandas, or Streamlit. Those tools
remain available from the source checkout through the development requirements.

Runtime dependencies are limited to pinned HTTP, YAML, environment-loading, and
HTTP transitive packages. The base Python image is pinned by patch version and
digest, selective `COPY` statements exclude repository trees, and
`PIP_NO_INDEX=1` disables runtime package downloads.

## Environment

| Variable | Required | Purpose |
|---|---:|---|
| `FIREWORKS_API_KEY` | yes | Injected at runtime; never baked into the image |
| `FIREWORKS_BASE_URL` | image default | Fireworks inference endpoint |
| `MODEL_TIER0..MODEL_TIER3` | image defaults | Independent allowed Fireworks arms |
| `ALLOWED_MODELS` | optional | JSON or comma-separated guard checked when an arm resolves |
| `ROUTER_MODE` | image default | `verified_tier0` for submission |
| `ANSWER_MAX_TOKENS` | no | Shared answer-policy cap; default 700 |
| `ANSWER_TEMPERATURE` | no | Shared answer-policy temperature; default 0 |
| `ANSWER_REASONING_EFFORT` | no | Optional shared reasoning-effort override |
| `ANSWER_WORKERS` | no | Concurrent answer workers; default 4 |
| `ANSWER_FALLBACKS` | no | Additional Fireworks calls; submission default 1 |
| `ANSWER_FALLBACK_TIERS` | no | Optional comma-separated fallback-arm order |
| `INPUT_PATH` / `OUTPUT_PATH` | no | Local contract path overrides |

Keep `ANSWER_FALLBACKS=1` for the evidenced submission policy. Increasing it
creates a different cost and runtime policy that needs fresh validation. The tier
names are configuration slots, not a monotonic quality claim.

## Evidence boundary

In a controlled 28-case NER prompt-only A/B, v1 passed 25/28 using 7,923 raw
tokens and v2 passed 26/28 using 8,134 (+3.57 percentage points, +2.66% tokens).
On a separate 10-case external natural/public NER check, both passed 10/10; v2
used 2,902 tokens versus 2,828 for v1. Both passed 6/6 locked NER mining-stress
prompts. These are small internal checks, not official grading.

The full 10-task public fixture appeared to pass 10/10 under manual v2 review,
used 3,200 tokens, made zero fallbacks, and completed in 3.2 seconds. V1 used
3,740 tokens, so the v2 run saved 540 (14.4%). This is a local/manual result and
must not be presented as an official leaderboard verdict.

The earlier 12-group tier0/tier3 frontier (12/12 each; 3,345 versus 4,326 tokens)
was measured under v1 (`c2ac...`) and became stale after v2 (`e785...`) was
adopted. It is not current container evidence.

The 63-task cascade replay (63/63, 18,793 tokens, one fallback versus 19,261 for
`always_tier3`) used stale-policy saved responses. It supports the verifier
mechanism but is not a final container benchmark.

## Build and preflight

Install development tools on the host, not in the image:

```bash
pip install -r requirements-dev.txt
python -m tests.smoke_test
python -m scripts.preflight_submission --source
```

Build a disposable candidate tag and inspect it:

```bash
docker build --platform linux/amd64 -t amd-act2-router:candidate .
python -m scripts.preflight_submission --image amd-act2-router:candidate
```

The source preflight checks the pinned base digest, exact minimal requirements,
verified model defaults, secret exclusions, selective copies, and contract
fixture. The build runs a network-free mocked contract through the exact agent
entrypoint. Image preflight verifies Linux/amd64 metadata, exact CMD, working
directory, `verified_tier0` default, no credentials, no ML/model/data trees, no
forbidden packages or weights, offline mocked execution, and size below 500 MB.
The final local candidate measured 140,350,921 bytes (about 140.35 MB), 93.4%
smaller than the 2.128 GB rollback control.

Do not overwrite or push a known-good submission tag until every candidate check
passes.

## Run locally

PowerShell:

```powershell
New-Item -ItemType Directory -Force out | Out-Null
docker run --rm --platform linux/amd64 `
  --mount "type=bind,source=${PWD}\tests\fixtures\container_input,target=/input,readonly" `
  --mount "type=bind,source=${PWD}\out,target=/output" `
  --env-file .env `
  amd-act2-router:candidate
Get-Content out\results.json
```

Bash:

```bash
mkdir -p out
docker run --rm --platform linux/amd64 \
  -v "$PWD/tests/fixtures/container_input:/input:ro" \
  -v "$PWD/out:/output" \
  --env-file .env \
  amd-act2-router:candidate
cat out/results.json
```

`python -m scripts.preflight_submission --image ...` starts its runtime test with
`--network none` and mocks only the Fireworks call. A real contract run needs the
runtime API key and Fireworks network access.
