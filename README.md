# amd-act2-router

**Multi-tier token-efficient LLM router for AMD Developer Hackathon ACT II — Track 1
("Hybrid Token-Efficient Routing Agent").**

Routes each query to the *cheapest* Fireworks-hosted model that can still answer it
correctly. The routing decision is made by a small **local** fine-tuned classifier
(DistilBERT, 4 classes: tier0..tier3), so routing costs **zero Fireworks tokens** —
only the answer itself does.

- Official Track 1 tutorial: <https://lablab.ai/ai-tutorials/fine-tune-llm-query-router-amd>
- Hackathon page: <https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii>

## Architecture Overview

Dataset → labeling → router → agent → container:

1. **Dataset** (`data/generate_tasks.py`) — generates tasks across 7 categories
   (math, logic, sentiment, summarization, NER, code debugging, code generation)
   with programmatically verified ground truths.
2. **Labeling** (`data/label_multitier.py`) — for each task, tries tier0 → tier3 on
   Fireworks and records the *cheapest tier that passes* grading (LLM judge for text,
   real test execution for code). Resumable; consumes Fireworks tokens.
3. **Router** (`router/`) — DistilBERT + numeric-feature classifier trained on the
   labeled data. Predicts `tier0..tier3` locally, zero tokens per decision.
4. **Agent** (`agent/agent.py`) — hackathon entrypoint: reads `/input/tasks.json`,
   routes each task locally, generates the answer via Fireworks, writes
   `/output/results.json`.
5. **Container** (`Dockerfile`) — slim Python 3.11 image, boots in well under 60 s.

## How this project extends the official tutorial

- **Multi-class routing** — 4 model tiers instead of binary easy/hard, so cost is
  optimized at a finer granularity.
- **Richer dataset** — parametrized generators with brute-force-verified logic
  puzzles and executable code tests, across trivial/medium/hard/adversarial pools.
- **Cheapest-tier-that-passes labels** — each label reflects an actual measured
  outcome, not a guess about difficulty.
- **Strategy evaluation** — `eval/evaluate_strategies.py` compares the router
  against always-cheapest, always-strongest, and a prompt-based baseline on both
  accuracy and tokens.

## Track 1 compliance

- All answer-generating inference goes through Fireworks (`FIREWORKS_BASE_URL`)
  using models from `ALLOWED_MODELS` (configured via env vars, never hardcoded).
- Local model inference is used **only** for the routing decision.
- Container contract: read `/input/tasks.json`, write `/output/results.json`,
  exit 0, boot < 60 s.

## Required environment variables

| Variable | Purpose |
|---|---|
| `FIREWORKS_API_KEY` | Fireworks API key (never committed) |
| `FIREWORKS_BASE_URL` | e.g. `https://api.fireworks.ai/inference/v1` |
| `MODEL_TIER0` | Cheapest allowed model ID |
| `MODEL_TIER1` | Mid-low tier model ID |
| `MODEL_TIER2` | Mid-high tier model ID |
| `MODEL_TIER3` | Strongest allowed model ID |
| `MODEL_JUDGE` | Model used to grade answers during labeling |
| `ROUTER_MODE` | `multitier` (default) \| `prompt_baseline` \| `always_tier0` \| `always_tier3` |

Copy `.env.example` to `.env` and fill in values (the `.env` file is gitignored).
Use model IDs from the hackathon's `ALLOWED_MODELS` list (MiniMax / Kimi K series
per the participant FAQ).

## Workflow

```bash
pip install -r requirements.txt

# 1. Generate tasks (free, local)
python -m data.generate_tasks

# 2. Label with real Fireworks calls (consumes tokens; resumable)
python -m data.label_multitier

# 3. Train the local router (~1 min, CPU is fine)
python -m router.train_multitier_router

# 4. Try it
python -m router.infer_multitier_router "Explain transformers vs CNNs" --category summarization

# 5. Compare strategies
python -m eval.evaluate_strategies

# 6. Sanity-check before building the container
python -m tests.smoke_test
```

See [docs/container.md](docs/container.md) for building and running the submission
container, and [docs/eval.md](docs/eval.md) for interpreting evaluation output.

## AMD / ROCm / Fireworks usage

- **Answer generation** runs on Fireworks per Track 1 rules.
- **Router training/eval** runs anywhere PyTorch runs — including AMD MI300X via
  ROCm. The training script is device-agnostic: it picks `cuda` (which is what
  ROCm-enabled PyTorch reports), `mps`, or `cpu` automatically. No code changes
  are needed between a laptop and an AMD pod.
- **Router inference** in the container runs on CPU; DistilBERT (66M params) needs
  no GPU at inference time.

### Running on the AMD pod

```bash
git pull
pip install -r requirements.txt
export FIREWORKS_API_KEY=...   # plus MODEL_TIER0..3, MODEL_JUDGE
python -m data.label_multitier          # labeling (network + tokens)
python -m router.train_multitier_router # training (uses MI300X via ROCm if present)
```

## Demo

```bash
pip install -r requirements-demo.txt
streamlit run demo/app.py
```

Shows the router's live tier decision, the model it maps to, and token cost
compared against an always-strongest strategy.
