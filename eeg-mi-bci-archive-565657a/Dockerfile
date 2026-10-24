# =============================================================================
# NeuroMove Enterprise MI-BCI - Multi-stage Dockerfile
# Optimized for production with security hardening
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Build Dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libopenblas-dev \
    libomp-dev \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies with pip cache
COPY requirements.txt api/requirements.txt mlflow/requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r api/requirements.txt && \
    pip install --no-cache-dir -r mlflow/requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Runtime Dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    libomp5 \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r neuromove && useradd -r -g neuromove -u 1000 -m -s /bin/bash neuromove

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=neuromove:neuromove src/ ./src/
COPY --chown=neuromove:neuromove config/ ./config/
COPY --chown=neuromove:neuromove dashboard/ ./dashboard/
COPY --chown=neuromove:neuromove api/ ./api/
COPY --chown=neuromove:neuromove mlflow/ ./mlflow/
COPY --chown=neuromove:neuromove scripts/ ./scripts/
COPY --chown=neuromove:neuromove config.yaml ./
COPY --chown=neuromove:neuromove pyproject.toml ./

# Create necessary directories
RUN mkdir -p /app/models /app/artifacts /app/logs /app/mlruns /app/data && \
    chown -R neuromove:neuromove /app

# Switch to non-root user
USER neuromove

# Environment variables
ENV NEUROMOVE_ROOT=/app
ENV NEUROMOVE_PIP_PACKAGES=""
ENV PYTHONPATH=/app/src:/app
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=file:/app/mlruns
ENV MLFLOW_REGISTRY_URI=file:/app/mlruns

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/ready || exit 1

# Default command (can be overridden)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]