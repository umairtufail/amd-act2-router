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
| `ROUTER_MODE` | no | `multitier` (default) / `prompt_baseline` / `always_tier0` / `always_tier3` |
| `ANSWER_MAX_TOKENS` | no | default 700 |
| `INPUT_PATH` / `OUTPUT_PATH` | no | override container paths for local testing |

## Build

**Important:** train the router first — the checkpoint in `router/checkpoints/`
is copied into the image. Without it the agent still runs but falls back to
tier0 for every task (a warning is logged).

```bash
python -m router.train_multitier_router   # if not already done
python -m tests.smoke_test                # sanity check
docker build -t amd-act2-router .
```

## Run locally

```bash
mkdir -p out
docker run --rm \
  -v "$PWD/sample_input:/input:ro" \
  -v "$PWD/out:/output" \
  -e FIREWORKS_API_KEY \
  -e FIREWORKS_BASE_URL \
  -e MODEL_TIER0 -e MODEL_TIER1 -e MODEL_TIER2 -e MODEL_TIER3 \
  amd-act2-router
cat out/results.json
```

(PowerShell: replace `$PWD` with `${PWD}` and line continuations `\` with `` ` ``.)

## Boot time

Everything the router needs (weights, tokenizer, encoder config) is loaded
from local files baked into the image — no HuggingFace downloads at boot.
Expect model load in a few seconds; the 60-second budget is comfortable.

## Compliance notes

- The only network calls the container makes are answer generations (and, in
  `prompt_baseline` mode only, classification calls) to `FIREWORKS_BASE_URL`.
- The local classifier never generates answers — it only picks the tier.
