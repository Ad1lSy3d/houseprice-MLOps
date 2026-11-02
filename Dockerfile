# Multi-stage production Dockerfile for EstateIQ House Price Intelligence Platform
# Uses Python 3.12 and uv for reproducible, fast dependency management

FROM python:3.12-slim-bookworm

# Metadata
LABEL maintainer="EstateIQ Team" \
      description="Production image for EstateIQ House Price Intelligence Platform" \
      version="0.1.0"

# Install uv from official Astral distribution
COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /uvx /bin/

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Step 1: Copy dependency definitions for optimal Docker layer caching
COPY pyproject.toml uv.lock ./

# Step 2: Install dependencies reproducibly based on locked versions
RUN uv sync --frozen --no-install-project

# Step 3: Copy application source, pre-trained models, evaluation reports, and frontend assets
COPY backend/ ./backend/
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY models/ ./models/
COPY reports/ ./reports/
COPY README.md ./

# Step 4: Finalize project synchronization
RUN uv sync --frozen

# Step 5: Configure non-privileged user for enhanced container security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose backend port
EXPOSE 8000

# Container healthcheck querying the /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start FastAPI application bound to 0.0.0.0:8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

