# amd-act2-router

**Binary token-efficient LLM router for AMD Developer Hackathon ACT II — Track 1
("Hybrid Token-Efficient Routing Agent").**

Routes each query to the *cheapest* Fireworks-hosted model that can still answer it
correctly. The default policy sends known easy categories directly to `tier0`; for
NER and summarization, a small **local** fine-tuned DistilBERT classifier estimates
`P(cheap_ok)`. It selects `tier0` when that probability meets the confidence
threshold and `tier3` otherwise. Routing therefore costs **zero Fireworks tokens** —
only the answer itself does. The original four-tier router remains available for
legacy A/B comparisons.

- Official Track 1 tutorial: <https://lablab.ai/ai-tutorials/fine-tune-llm-query-router-amd>
- Hackathon page: <https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii>

## Architecture Overview

Dataset → labeling → router → agent → container:

1. **Dataset** (`data/generate_tasks.py`) — generates tasks across 8 categories
   (factual knowledge, math, logic, sentiment, summarization, NER, code
   debugging, code generation)
   with programmatically verified ground truths.
2. **Labeling** (`data/label_multitier.py`) — for each task, tries tier0 → tier3 on
   Fireworks and records the *cheapest tier that passes* grading (LLM judge for text,
   real test execution for code). Resumable; consumes Fireworks tokens.
3. **Router** (`router/`) — maps the measured labels to `cheap_ok` versus
   `needs_strong`, then trains a DistilBERT + numeric-feature binary classifier.
   Category rules and confidence gating produce only `tier0` or `tier3` decisions.
4. **Agent** (`agent/agent.py`) — hackathon entrypoint: reads `/input/tasks.json`,
   routes each task locally, generates the answer via Fireworks, writes
   `/output/results.json`.
5. **Container** (`Dockerfile`) — slim Python 3.11 image, boots in well under 60 s.

## How this project extends the official tutorial

- **Category-aware binary routing** — factual knowledge, math reasoning, logic
  puzzles, code debugging, code generation, and sentiment always use `tier0`;
  NER and summarization use the local classifier and can escalate directly to
  `tier3`.
- **Confidence control** — `BINARY_ROUTER_TAU` sets the general `P(cheap_ok)` gate,
  while `NER_BINARY_TAU` can override it for NER.
- **Richer dataset** — parametrized generators with brute-force-verified logic
  puzzles and executable code tests, across trivial/medium/hard/adversarial pools.
- **Cheapest-tier-that-passes labels** — each label reflects an actual measured
  outcome, not a guess about difficulty. Four-tier labels and the legacy multitier
  checkpoint remain useful for offline analysis and A/B evaluation.
- **Strategy evaluation** — `eval/evaluate_strategies.py` compares the router
  against always-cheapest, always-strongest, the legacy multitier router, and an
  optional prompt baseline, and includes a binary threshold sweep.

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
| `MODEL_TIER1` | Mid-low tier model ID used by labeling and legacy A/B modes |
| `MODEL_TIER2` | Mid-high tier model ID used by labeling and legacy A/B modes |
| `MODEL_TIER3` | Strongest allowed model ID |
| `MODEL_JUDGE` | Model used to grade answers during labeling |
| `ALLOWED_MODELS` | Optional judging-harness allowlist; configured tiers are validated against it |
| `ROUTER_MODE` | `binary` (default) \| `multitier` \| `prompt_baseline` \| `always_tier0` \| `always_tier3` |
| `BINARY_ROUTER_TAU` | General `P(cheap_ok)` threshold; default `0.8` |
| `NER_BINARY_TAU` | Optional NER threshold; inherits `BINARY_ROUTER_TAU` when unset |

Copy `.env.example` to `.env` and fill in values (the `.env` file is gitignored).
Use the exact model IDs reported by the hackathon's `ALLOWED_MODELS` command.

## Workflow

```bash
pip install -r requirements.txt

# 1. Generate tasks (free, local)
python -m data.generate_tasks

# 2. Label with real Fireworks calls (consumes tokens; resumable)
python -m data.label_multitier

# 3. Train the default binary router (~1 min, CPU is fine)
python -m router.train_binary_router

# 4. Try it
python -m router.infer_binary_router "Extract all people from: Dr. Jane Smith met Bob." --category ner

# 5. Compare strategies, including the binary tau sweep
python -m eval.evaluate_strategies

# 6. Sanity-check before building the container
python -m tests.smoke_test
```

To train and inspect the legacy four-class router for A/B comparison, use
`python -m router.train_multitier_router` and
`python -m router.infer_multitier_router "..." --category summarization`.

The trained `router/checkpoints/binary_router.pt` is intentionally gitignored.
Train it locally or copy it from the AMD pod before building the container; the
Docker build packages it together with `binary_router_config.json`, the tokenizer,
and the encoder configuration for fully offline router startup.

See [docs/container.md](docs/container.md) for building and running the submission
container, [docs/eval.md](docs/eval.md) for interpreting evaluation output, and
[docs/local_dev.md](docs/local_dev.md) for running locally with optional Ollama.

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
python -m data.label_multitier        # labeling (network + tokens)
python -m router.train_binary_router  # training (uses MI300X via ROCm if present)
```

## Demo

```bash
pip install -r requirements-demo.txt
streamlit run demo/app.py
```

Shows the router's live tier decision, the model it maps to, and token cost
compared against an always-strongest strategy.

## License

Released under the [MIT License](LICENSE).
