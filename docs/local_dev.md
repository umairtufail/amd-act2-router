# Local Development Guide

## Two environments, one codebase

| Purpose | Answers | Judge / labeling | Router |
|---------|---------|------------------|--------|
| **Submission** (`agent.py` in Docker) | Fireworks only | N/A at runtime | Local CPU |
| **Local dev** | Fireworks or local | Local or Fireworks | Local CPU/GPU |

Track 1 rules apply to **submission**: scored answers must come from Fireworks.
Everything else (labeling judges, holdout tests, extra data) can use a **local LLM**
to save tokens and iterate faster.

## 1. Setup (Windows)

```powershell
cd C:\Users\ro\Documents\amd-act2-router
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env
# Fill in FIREWORKS_API_KEY, MODEL_TIER0..3 for submission testing
```

### Optional: Ollama for free local LLM

1. Install [Ollama](https://ollama.com/)
2. `ollama pull llama3.2`
3. In `.env`:

```env
LOCAL_LLM_BASE_URL=http://localhost:11434/v1
LOCAL_LLM_MODEL=llama3.2
JUDGE_BACKEND=auto
```

With `JUDGE_BACKEND=auto`, summarization/NER grading during labeling uses Ollama
when `LOCAL_LLM_BASE_URL` is set. Factual knowledge and the other closed-answer
categories are graded locally.

## 2. Sync full dataset from pod / GitHub

The default eight-category generator produces 176 tasks. Pull from GitHub, or copy
`data/labeled_multitier.jsonl` and `router/checkpoints/` from the pod if not in git:

```powershell
git pull origin master
```

If checkpoint is missing (gitignored):

```bash
# On pod
tar czf /tmp/checkpoints.tar.gz -C router checkpoints
# Download to local, then:
tar xzf checkpoints.tar.gz -C router/
```

## 3. Train and evaluate locally

```powershell
python -m router.train_binary_router
# Optional legacy A/B mode after a category-vocabulary change:
python -m router.train_multitier_router
python -m eval.evaluate_strategies
python -m tests.smoke_test
```

## 4. Run agent without Docker

```powershell
mkdir out -Force
$env:INPUT_PATH = "sample_input\tasks.json"
$env:OUTPUT_PATH = "out\results.json"
# Uses Fireworks for answers (default)
python -m agent.agent
Get-Content out\results.json
```

## 5. Generalization test (holdout prompts)

Edit `data/holdout_tasks.json` with **new** prompts not in `tasks_raw.jsonl`.

**Free local answers** (dev only):

```powershell
$env:DEV_LOCAL_ANSWERS = "1"
python -m eval.holdout_generalization
```

**Fireworks answers** (closer to production):

```powershell
python -m eval.holdout_generalization --fireworks
```

## 6. Labeling with local judge

```powershell
# Ollama running + LOCAL_LLM_BASE_URL in .env
python -m data.label_multitier --limit 5 --sleep 3
```

Factual/math/logic/sentiment/code_debug still grade locally (no LLM).
Summarization/NER judge calls go to Ollama when configured.

## 7. Docker (local Windows only)

Pods usually have no Docker. Build on your PC:

```powershell
docker build --platform linux/amd64 -t amd-act2-router .
docker run --rm --platform linux/amd64 -v "${PWD}/tests/fixtures/container_input:/input:ro" -v "${PWD}/out:/output" --env-file .env amd-act2-router
```

Do **not** set `DEV_LOCAL_ANSWERS` in the submission image.

## 8. Push to remote when ready

```powershell
git add -A
git commit -m "Add local dev LLM backend and holdout eval"
git push origin master
```

On pod: `git pull` and continue from there.
