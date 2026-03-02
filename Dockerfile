# ──────────────────────────────────────────────
# MedQCNN Dockerfile — Multi-stage build
# Target: Edge-compute (Raspberry Pi 5 / Kali Linux)
# ──────────────────────────────────────────────

# Stage 1: Dependencies
FROM python:3.11-slim AS deps

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (CPU-only)
RUN uv sync --frozen --no-dev

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from deps stage
COPY --from=deps /app/.venv /app/.venv

# Copy application code
COPY medqcnn/ medqcnn/
COPY scripts/ scripts/
COPY main.py .
COPY settings/ settings/

# Copy checkpoints if present
COPY checkpoints/ checkpoints/

# Set environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Default: run the API server
CMD ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000"]
