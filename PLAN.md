# PLAN — Multi-Tier Token-Efficient LLM Router

**Hackathon:** AMD Developer Hackathon: ACT II — Track 1 "Hybrid Token-Efficient Routing Agent"

**Pitch:** Route each query to the *cheapest* Fireworks model that passes the accuracy
threshold, using a local fine-tuned 4-class classifier (tier0..tier3) so routing
decisions cost **zero** Fireworks tokens.

**Compliance rules (must hold at all times):**
- All answer-generating inference goes through Fireworks (`FIREWORKS_BASE_URL`) using
  models from `ALLOWED_MODELS`.
- Local models are used **only** for routing decisions.
- Container contract: read `/input/tasks.json`, write `/output/results.json`, exit 0,
  boot in < 60 seconds.
- No API keys in the repo — env vars only.

---

## Phase Checklist

- [x] **Phase 0** — Repo scan, PLAN.md, .gitignore
- [x] **Phase 1** — Project structure, requirements.txt, README scaffold, `__init__.py` files
- [x] **Phase 2** — `config/models.yaml`, config loader, `agent/fireworks_client.py`
      (timeouts, retries, graceful errors)
- [x] **Phase 3** — Data pipeline: `data/schema.py`, `data/generate_tasks.py`,
      `data/code_exec.py`, `data/judge.py`, `data/label_multitier.py` (resumable)
- [x] **Phase 4** — Router: `router/features.py`, `router/model.py`,
      `router/train_multitier_router.py`, `router/infer_multitier_router.py`
- [x] **Phase 5** — `baseline/baseline_router.py`, `eval/evaluate_strategies.py`, `docs/eval.md`
- [x] **Phase 6** — `agent/agent.py` (ROUTER_MODE toggle + logging), `Dockerfile`,
      `docs/container.md`, `tests/smoke_test.py`
- [x] **Phase 7** — `demo/app.py`, AMD/ROCm docs, README polish

## Human Runbook (things only the human runs)

- [ ] Set env vars locally and on the AMD pod: `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`,
      `MODEL_TIER0..3`, `MODEL_JUDGE` (see `.env.example`)
- [ ] `python -m data.generate_tasks` → creates `data/tasks_raw.jsonl` (free, local)
- [ ] `python -m data.label_multitier` → creates `data/labeled_multitier.jsonl`
      (**consumes Fireworks tokens**; resumable — safe to stop/restart)
- [ ] `python -m router.train_multitier_router` → trains classifier, saves checkpoint
- [ ] `python -m eval.evaluate_strategies` → strategy comparison table
- [ ] `python -m tests.smoke_test` → sanity checks before container build
- [ ] `docker build -t amd-act2-router .` and test run with mounted `/input` + `/output`

## Open Questions for Human

1. **Model selection:** Which 4 model IDs from the hackathon's `ALLOWED_MODELS`
   (MiniMax / Kimi K series) map to tier0..tier3? Needed before labeling.
2. **Judge model:** Which model grades answers during labeling? (Recommend the
   strongest allowed model, or tier3.)
3. **Dataset size:** Default generator makes ~150 tasks. Increase per category?
4. **AMD pod:** Is git sync already configured between this repo and the pod?
