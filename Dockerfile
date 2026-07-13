# Pin the Python patch release used by the validated linux/amd64 image.
# Build with: docker build --platform linux/amd64 -t amd-act2-router:candidate .
FROM python:3.11.15-slim@sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3

WORKDIR /app

# The submission path only needs HTTP, YAML configuration, and environment
# loading. Training, router inference, tests, and the Streamlit demo are kept
# in their dedicated local requirement files and never enter this image.
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --no-compile -r requirements.txt \
    && pip check \
    && rm -rf /root/.cache/pip

# Selective copies keep source datasets, local checkpoints, training code,
# demo dependencies, and repository metadata out of the submitted artifact.
COPY config/__init__.py config/models.yaml ./config/
COPY agent/__init__.py agent/agent.py agent/fireworks_client.py \
     agent/llm_backend.py agent/local_llm_client.py agent/request_policy.py \
     agent/quality_gate.py ./agent/
COPY scripts/preflight_submission.py ./scripts/preflight_submission.py
COPY tests/fixtures/container_input/tasks.json /opt/contract/tasks.json

ENV ROUTER_MODE=verified_tier0
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_INDEX=1

# The evaluator injects FIREWORKS_API_KEY. These overrideable defaults are
# approved Fireworks model IDs; no credential or local .env file is copied.
ENV FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
ENV MODEL_TIER0=accounts/fireworks/models/gpt-oss-120b
ENV MODEL_TIER1=accounts/fireworks/models/kimi-k2p6
ENV MODEL_TIER2=accounts/fireworks/models/glm-5p2
ENV MODEL_TIER3=accounts/fireworks/models/deepseek-v4-pro

# Network-free build gate: imports the exact entrypoint and exercises the
# /input/tasks.json -> /output/results.json contract with a mocked API call.
RUN python -m scripts.preflight_submission --runtime

LABEL org.opencontainers.image.title="AMD ACT2 Cost-Aware Router" \
      org.opencontainers.image.description="Verified tier-0 Fireworks agent with deterministic quality fallback" \
      org.opencontainers.image.source="https://github.com/umairtufail/amd-act2-router"

CMD ["python", "-m", "agent.agent"]
