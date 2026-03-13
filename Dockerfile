# ──────────────────────────────────────────────
# MedQCNN Dockerfile — Multi-stage build
# Target: Edge-compute (Raspberry Pi 5 / Kali Linux)
#
# Build:  docker build -t medqcnn .
# Run:    docker run -p 8000:8000 medqcnn
# ──────────────────────────────────────────────

# Stage 1: Dependencies
FROM python:3.11-slim AS deps

WORKDIR /app

# Install system build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy only dependency files (layer cache)
COPY pyproject.toml uv.lock ./

# Install production dependencies only
RUN uv sync --frozen --no-dev

# ──────────────────────────────────────────────
# Stage 2: Runtime
# ──────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from deps stage
COPY --from=deps /app/.venv /app/.venv

# Copy application code only (no data, checkpoints, or docs)
COPY medqcnn/ medqcnn/
COPY scripts/ scripts/
COPY main.py .

# Copy settings if present
COPY settings/ settings/

# Create directories for volume mounts
RUN mkdir -p checkpoints data logs

# Set environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV DATABASE_URL="sqlite:///medqcnn.db"

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: run the API server
CMD ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000"]
