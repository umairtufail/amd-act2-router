FROM python:3.11-slim

WORKDIR /app

# CPU-only torch keeps the image far smaller and installs faster; the router
# needs no GPU at inference time.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY config/ ./config/
COPY agent/ ./agent/
COPY baseline/ ./baseline/
COPY data/__init__.py data/schema.py ./data/
COPY router/ ./router/

ENV ROUTER_MODE=binary
ENV BINARY_ROUTER_TAU=0.8
ENV NER_BINARY_TAU=0.9

# The evaluation harness injects the Fireworks credential, but it does not
# load the submitter's local .env file. Keep the approved model IDs as
# overrideable image defaults so routing can start in the harness.
ENV FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
ENV MODEL_TIER0=accounts/fireworks/models/gpt-oss-120b
ENV MODEL_TIER1=accounts/fireworks/models/kimi-k2p6
ENV MODEL_TIER2=accounts/fireworks/models/glm-5p2
ENV MODEL_TIER3=accounts/fireworks/models/deepseek-v4-pro

CMD ["python", "-m", "agent.agent"]
