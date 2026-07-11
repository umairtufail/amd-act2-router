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

ENV ROUTER_MODE=multitier
CMD ["python", "-m", "agent.agent"]
