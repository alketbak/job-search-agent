# syntax=docker/dockerfile:1.6
# JobRadar production image - multi-stage, non-root.
# Same image serves both the `api` and `scheduler` docker-compose services;
# they differ only by the `command` each one runs.

# ---- builder: install Python deps into a venv, nothing else ----
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# build-essential: kept defensively in case a future dependency needs a
# compiler on some architecture. Nothing in requirements.txt currently
# needs it - all current deps ship prebuilt
# wheels for linux/amd64 and linux/arm64 - but it's cheap insurance and
# this layer never reaches the runtime image anyway.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# torch (a sentence-transformers dependency) defaults to CUDA-enabled
# wheels on linux/aarch64, pulling ~1.5GB of NVIDIA packages that are
# useless in a CPU-only container - install the CPU-only build explicitly
# first so sentence-transformers finds it already satisfied. Caught by
# actually building this image, not assumed.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && /opt/venv/bin/pip install -r requirements.txt

# ---- runtime: slim, non-root, only what's needed to run ----
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    DEBIAN_FRONTEND=noninteractive \
    NODE_VERSION=20.x

WORKDIR /app

# Node.js is needed at RUNTIME, not just build time - the MCP memory
# server is spawned via `npx` on every process start (memory/mcp_client.py).
# curl/gnupg are only needed transiently for the NodeSource setup script;
# purging them in this same RUN keeps them out of the final layer.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get purge -y --auto-remove curl gnupg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY . .

RUN mkdir -p output data cv \
    && useradd --create-home --uid 1000 --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app

USER appuser

# Pre-fetch the MCP memory server so the first pipeline run doesn't stall
# on npm downloads. Best-effort - if the registry is unreachable here, the
# runtime code path still falls back to in-memory mode (memory/mcp_client.py).
RUN npx -y @modelcontextprotocol/server-memory --help >/dev/null 2>&1 || true

EXPOSE 8000
CMD ["python", "main.py"]
