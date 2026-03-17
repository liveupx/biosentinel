# ──────────────────────────────────────────────────────────────
# BioSentinel — Production Dockerfile
# Developer: Liveupx Pvt. Ltd. / Mohit Chaprana
# github.com/liveupx/biosentinel
# ──────────────────────────────────────────────────────────────
# Multi-stage build: builder installs deps, runtime is minimal.
# Usage:
#   docker build -t biosentinel .
#   docker run -p 8000:8000 -v $(pwd)/data:/app/data biosentinel
# ──────────────────────────────────────────────────────────────

# ── Stage 1: Builder ─────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN groupadd -r biosentinel && useradd -r -g biosentinel -m biosentinel

WORKDIR /app

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /root/.local /home/biosentinel/.local

# Copy application source
COPY app.py .
COPY biosentinel_dashboard.html .
COPY biosentinel_patient_view.html . 2>/dev/null || true

# Create data directory for SQLite DB persistence
RUN mkdir -p /app/data && chown -R biosentinel:biosentinel /app

# Switch to non-root
USER biosentinel

ENV PATH=/home/biosentinel/.local/bin:$PATH
ENV DATABASE_URL=sqlite:////app/data/biosentinel.db

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "warning"]
