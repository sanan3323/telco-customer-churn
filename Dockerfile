# syntax=docker/dockerfile:1.6
# ─────────────────────────────────────────────────────────────────────────
# Telco Churn Prediction API — production image
#
# Multi-stage build: the "builder" stage installs dependencies, the final
# stage copies only what's needed to run. This keeps the final image small
# and free of build-time tools.
# ─────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ─────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Don't buffer stdout/stderr — logs appear in real time.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install requirements into a venv so the final stage can copy just the venv.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# System libs scikit-learn / LightGBM / XGBoost need at runtime. libgomp1
# is the big one — sklearn parallelism fails to import without it.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user so the container doesn't run as root (security).
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Copy the venv from the builder stage. This gives us all installed
# dependencies without the build toolchain.
COPY --from=builder /opt/venv /opt/venv

# Copy the application code. Ordered so frequently changing files come
# last, which keeps Docker's layer cache hot during iteration.
COPY --chown=app:app src/ ./src/
COPY --chown=app:app models/churn_pipeline.joblib ./models/churn_pipeline.joblib
COPY --chown=app:app app.py .

# Tell Docker the container listens on port 8000. This is documentation;
# the actual port binding happens at `docker run -p ...`.
EXPOSE 8000

# Health check: Docker pings /health every 30s. If it fails 3 times in a
# row, the container is marked unhealthy and can be auto-restarted.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD sh -c 'python -c "import urllib.request, sys, os; port=os.environ.get(\"PORT\",\"8000\"); sys.exit(0) if urllib.request.urlopen(f\"http://localhost:{port}/health\", timeout=3).status == 200 else sys.exit(1)"'

# Start uvicorn bound to all interfaces (not just localhost) so the host
# can reach the container via port mapping.
# Bind to $PORT if set (Render, Heroku, Fly.io all set this), else 8000.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]