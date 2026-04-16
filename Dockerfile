# syntax=docker/dockerfile:1.6
# JobRadar production image.
# Python base + Node.js 20 (for npx'ing the MCP memory server) + Chromium.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    NODE_VERSION=20.x

# --- system dependencies ---
# curl: for nodesource install
# chromium + chromium-driver: for Playwright's fallback browser
# ca-certificates / gnupg: for HTTPS + NodeSource signing
# build-essential: some wheels (faiss-cpu) need a compiler on some arches
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        gnupg \
        git \
        chromium \
        chromium-driver \
        fonts-liberation \
        build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# --- app setup ---
WORKDIR /app

# Install Python deps first so Docker can cache this layer.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the codebase.
COPY . .

# Pre-fetch the MCP memory server so the first pipeline run doesn't stall
# on npm downloads. This is best-effort; if the registry is unreachable the
# runtime code path still falls back to in-memory mode.
RUN npx -y @modelcontextprotocol/server-memory --help >/dev/null 2>&1 || true

# Directories for runtime artefacts — ./output and ./cv/cv.faiss etc.
RUN mkdir -p output data cv

EXPOSE 8000
CMD ["python", "main.py"]
