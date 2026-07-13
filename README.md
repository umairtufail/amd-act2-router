# AMD ACT II Cost-Aware Router

A Track 1 Fireworks agent that minimizes raw answer tokens with a conservative
verify-then-escalate cascade. The submission default is `verified_tier0`: call
tier0 first, run a zero-token deterministic verifier, and make at most one
Fireworks fallback call when an explicit requirement is definitely broken.

The four configured models are independent arms. The project never assumes that
a later tier is more accurate or copies an unmeasured arm's outcome. All
substantive answers, including fallbacks, come from allowed Fireworks models.

- Official Track 1 tutorial: <https://lablab.ai/ai-tutorials/fine-tune-llm-query-router-amd>
- Hackathon page: <https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii>

## Submission runtime

1. `agent/agent.py` validates `/input/tasks.json`.
2. Tier0 answers each task through request policy v2 (`e785...`) with
   `max_tokens=700`.
3. `agent/quality_gate.py` remains verifier policy v1 and checks only
   machine-verifiable requirements: API and
   truncation failures, requested JSON/schema, counts and limits, allowed labels,
   code syntax/function names, and high-precision explicit location cues such as
   `X-based` in NER prompts. Unsupported or ambiguous checks pass through.
4. If verification fails, the default permits one call to a different configured
   Fireworks arm. The verifier never writes or semantically repairs an answer.
5. Results are preserved in input order and written atomically to
   `/output/results.json`.

This path spends no routing tokens and normally makes one answer call. The
DistilBERT binary router, legacy multitier router, logistic regression, and k-NN
remain offline analysis tools. No ML framework, router checkpoint, tokenizer, or
model weight is present in the submission image.

## Evidence, with limits

The fresh internal v2 Fireworks holdout ran all 72 locked tasks through
`verified_tier0`: 71/72 (98.611%) with 19,935 raw answer tokens (12,083 prompt +
7,852 completion), or 276.875 per task. All 72 attempts stayed on tier0 and the
deterministic verifier triggered zero fallbacks. Seven categories passed 9/9;
sentiment passed 8/9. This is an internal holdout, not official leaderboard
grading.

The sole failure was locked task `holdout_sentiment_03`: a neutral factual
delivery statement was answered `positive`. The answer satisfied the requested
one-word format, so the non-semantic verifier correctly passed it through. A
deterministic regrade of the saved answer stayed FAIL and made zero answer calls.
The project does not tune prompts or routing rules on this locked failure.

A controlled prompt-only A/B used the same 28 NER cases. Request policy v1
(`c2ac...`) passed 25/28 with 7,923 raw tokens; v2 (`e785...`) passed 26/28
with 8,134. That is +3.57 percentage points for +2.66% tokens. On a separate
10-case external natural/public NER check, both policies passed 10/10; v2 used
2,902 tokens versus 2,828 for v1. Both also passed all 6 locked NER mining-stress
cases. These are small internal evaluations, not official grading.

On the full 10-task public fixture, v2 appeared to pass 10/10 under manual rubric
review, used 3,200 tokens, triggered zero fallbacks, and completed in 3.2 seconds.
The comparable v1 run used 3,740 tokens, so v2 saved 540 tokens (14.4%). This is
a local/manual result, not an official leaderboard verdict.

The earlier 12-group tier0/tier3 frontier (12/12 for each arm; 3,345 versus
4,326 tokens) belongs to v1 and became stale when v2 was adopted. It must not be
described as current-policy evidence. The 63-task cascade replay also used older
responses and remains stale, mechanism-only evidence. The fresh holdout is bound
to request hash `e785...`, verifier hash `30a16...`, and results hash `d90d...`.

The historical training file still contains 176 rows but only 110 exact prompt
groups and roughly three conservative hard groups. That is insufficient to
promote a learned binary router; duplicate-safe training and hard precision,
recall, F1, balanced accuracy, and the constant baseline remain available for
analysis.

## Evaluation safeguards

- Exact duplicate prompts collapse to one group and cannot leak across splits.
- Missing arm outcomes remain unknown; metrics are measured-only raw-token
  frontiers with explicit coverage.
- 95% Wilson lower bounds expose small-sample uncertainty.
- Request-policy, verification-policy, dataset, split, and results hashes detect
  stale or edited artifacts.
- Holdout evaluation counts every primary and fallback token.
- Learned and shadow routers are analysis-only until they pass their explicit
  data and promotion gates.

## Setup

```bash
# Source evaluation, ML analysis, and tests
pip install -r requirements-dev.txt

# Free local audits
python -m eval.evaluate_strategies
python -m tests.smoke_test
python -m scripts.preflight_submission --source

# Optional paid fresh holdout: review before running
# python -m eval.holdout_generalization --strategy verified_tier0 --fireworks
```

Inspect the v2 refresh of stale v1 arm labels without calls:

```bash
python -m data.label_multitier --fill-missing-tiers --refresh-policy \
  --category ner --category summarization --tier tier0 --tier tier3 \
  --unique-prompts --dry-run
```

Removing `--dry-run`, using `--fireworks`, running a Fireworks judge, or enabling
the prompt baseline consumes tokens.

For the optional demo:

```bash
pip install -r requirements-demo.txt
streamlit run demo/app.py
```

Copy `.env.example` to `.env`, add the runtime key, and use only model IDs
reported by the hackathon's `ALLOWED_MODELS` command. See
[docs/eval.md](docs/eval.md), [docs/local_dev.md](docs/local_dev.md), and
[docs/container.md](docs/container.md) for details.

## Track 1 compliance

- Every answer-generating call uses Fireworks; local code only selects, validates,
  and performs content-preserving formatting normalization.
- The Linux/amd64 container reads `/input/tasks.json`, writes
  `/output/results.json`, and uses the exact `python -m agent.agent` CMD.
- The image has no secret, local LLM, ML package, checkpoint, or runtime download
  path. Model IDs remain environment-configurable and allowlist-checkable.
- The final local candidate is about 140.35 MB, 93.4% smaller than the 2.128 GB
  rollback control.

## License

Released under the [MIT License](LICENSE).
