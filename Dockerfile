# ──────────────────────────────────────────────────────────────────────────────
# BioSentinel v2.1 — Production Dockerfile
# Developer: Liveupx Pvt. Ltd. / Mohit Chaprana
# github.com/liveupx/biosentinel
#
# Multi-stage build — builder installs deps, runtime image is minimal.
#
# Build:  docker build -t biosentinel:2.1.0 .
# Run:    docker run -p 8000:8000 -v $(pwd)/data:/app/data \
#           -e ANTHROPIC_API_KEY=sk-ant-... biosentinel:2.1.0
# ──────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System build deps + Tesseract OCR + Poppler (for pdf2image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="BioSentinel"
LABEL org.opencontainers.image.version="2.1.0"
LABEL org.opencontainers.image.description="AI-powered longitudinal health monitoring"
LABEL org.opencontainers.image.url="https://github.com/liveupx/biosentinel"
LABEL org.opencontainers.image.authors="Mohit Chaprana <mohit@liveupx.com>"
LABEL org.opencontainers.image.licenses="MIT"

# Non-root user for security
RUN groupadd -r biosentinel && useradd -r -g biosentinel -m biosentinel

WORKDIR /app

# Runtime system deps:
#   curl         — health check
#   tesseract-ocr — traditional OCR (pdfplumber fallback)
#   poppler-utils — pdf2image for Claude Vision PDF handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /root/.local /home/biosentinel/.local

# Copy application source
COPY app.py .
COPY claude_ai.py .
COPY scheduler.py .
COPY biosentinel_dashboard.html .
COPY biosentinel_patient_portal.html .
COPY biosentinel_patient_view.html .

# Create data directory for SQLite DB persistence
RUN mkdir -p /app/data && chown -R biosentinel:biosentinel /app

# Switch to non-root
USER biosentinel

ENV PATH=/home/biosentinel/.local/bin:$PATH
ENV DATABASE_URL=sqlite:////app/data/biosentinel.db
ENV LOG_FORMAT=json
ENV LOG_LEVEL=INFO

# Health check — wait up to 90s for ML model training on cold start
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# 2 workers — increase to 4 on a machine with 4+ CPU cores
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "warning"]
