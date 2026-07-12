# Container Guide

## Submission contract (Track 1)

- Reads `/input/tasks.json`
- Writes `/output/results.json`
- Exits with code 0
- Boots in under 60 seconds

## Input / output schema

`/input/tasks.json`:

```json
[
  {"task_id": "1", "prompt": "Explain overfitting.", "category": "summarization"},
  {"task_id": "2", "prompt": "Fix this Python function...", "category": "code_debugging"}
]
```

- `task_id` (required), `prompt` (required), `category` (optional — the router
  uses an "unknown" bucket when absent).

`/output/results.json`:

```json
[
  {"task_id": "1", "answer": "..."},
  {"task_id": "2", "answer": "..."}
]
```

## Environment variables

| Variable | Required | Notes |
|---|---|---|
| `FIREWORKS_API_KEY` | yes | never baked into the image |
| `FIREWORKS_BASE_URL` | yes | `https://api.fireworks.ai/inference/v1` |
| `MODEL_TIER0..MODEL_TIER3` | yes | IDs from the hackathon `ALLOWED_MODELS` list |
| `ALLOWED_MODELS` | no | optional harness allowlist; every selected tier must appear in it |
| `ROUTER_MODE` | no | `binary` (default) / `multitier` / `prompt_baseline` / `always_tier0` / `always_tier3` |
| `BINARY_ROUTER_TAU` | no | general `P(cheap_ok)` gate; default `0.8` |
| `NER_BINARY_TAU` | no | NER-specific gate; inherits `BINARY_ROUTER_TAU` when unset |
| `ANSWER_MAX_TOKENS` | no | default 700 |
| `INPUT_PATH` / `OUTPUT_PATH` | no | override container paths for local testing |

In binary mode, `factual_knowledge`, `math_reasoning`, `logic_puzzles`,
`code_debugging`, `code_generation`, and `sentiment` are treated as easy
categories and go directly to `tier0`. NER and summarization use the local
classifier: `P(cheap_ok) >= tau` selects `tier0`; otherwise the request escalates
directly to `tier3`. A higher threshold therefore escalates more requests.
Missing or unknown categories use the classifier with the general threshold.

## Build

**Important:** train the binary router first. The Docker build copies
`router/checkpoints/binary_router.pt`, `binary_router_config.json`, the tokenizer,
and the encoder configuration into the image. The checkpoint is gitignored, so it
must exist locally (or be copied from the AMD pod) before `docker build`. Without
it the agent still runs but warns and falls back to tier0 for every task.

```bash
python -m router.train_binary_router   # if not already done
python -m tests.smoke_test             # sanity check
docker build --platform linux/amd64 -t amd-act2-router .
```

For a legacy A/B image, run `python -m router.train_multitier_router` as well and
set `ROUTER_MODE=multitier` at runtime.

## Run locally

```bash
mkdir -p out
docker run --rm --platform linux/amd64 \
  -v "$PWD/tests/fixtures/container_input:/input:ro" \
  -v "$PWD/out:/output" \
  --env-file .env \
  amd-act2-router
cat out/results.json
```

PowerShell:

```powershell
New-Item -ItemType Directory -Force out | Out-Null
docker run --rm --platform linux/amd64 `
  --mount "type=bind,source=${PWD}\tests\fixtures\container_input,target=/input,readonly" `
  --mount "type=bind,source=${PWD}\out,target=/output" `
  --env-file .env `
  amd-act2-router
Get-Content out\results.json
```

## Boot time

Everything the router needs (weights, tokenizer, encoder config) is loaded
from local files baked into the image — no HuggingFace downloads at boot.
Expect model load in a few seconds; the 60-second budget is comfortable.

## Compliance notes

- The only network calls the container makes are answer generations (and, in
  `prompt_baseline` mode only, classification calls) to `FIREWORKS_BASE_URL`.
- The default binary classifier never generates answers — it only selects
  `tier0` or `tier3`, with zero Fireworks routing tokens.
