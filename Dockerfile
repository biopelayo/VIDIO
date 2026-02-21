# ============================================================
#  VIDIO — Biomedical Image Analysis Platform
#  Docker container for production deployment.
#
#  Build:
#    docker build -t vidio:latest .
#
#  Run:
#    docker run -d -p 7070:7070 \
#      -e DB_HOST=host.docker.internal \
#      -e DB_PASSWORD=vidio2025 \
#      -e AUTH_SECRET=your-secret-key \
#      -v /path/to/images:/data/repo \
#      vidio:latest
#
#  With GPU (NVIDIA):
#    docker run --gpus all -d -p 7070:7070 \
#      -e DB_HOST=host.docker.internal \
#      vidio:latest
# ============================================================

FROM python:3.12-slim AS base

# --- System dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python dependencies (cached layer) ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Application code ---
COPY . .

# --- Configuration ---
ENV VIDIO_HOST=0.0.0.0
ENV VIDIO_PORT=7070
ENV DB_HOST=localhost
ENV DB_PORT=5432
ENV DB_NAME=vidio
ENV DB_USER=admin_vidio
ENV DB_PASSWORD=vidio2025
ENV AUTH_SECRET=change-me-in-production
ENV REPO_PATH=/data/repo

# --- Create directories ---
RUN mkdir -p /data/repo /data/models

# --- Generate runtime cfg.json from env vars ---
RUN echo '#!/bin/sh' > /app/entrypoint.sh && \
    echo 'cat > /app/cfg.json << EOFCFG' >> /app/entrypoint.sh && \
    echo '{' >> /app/entrypoint.sh && \
    echo '  "db": {"host": "'"\$DB_HOST"'", "port": '$DB_PORT', "db": "'"\$DB_NAME"'", "user": "'"\$DB_USER"'", "password": "'"\$DB_PASSWORD"'"},' >> /app/entrypoint.sh && \
    echo '  "system": {"user": null},' >> /app/entrypoint.sh && \
    echo '  "repository": {"location": "'"\$REPO_PATH"'", "location_windows": "'"\$REPO_PATH"'"},' >> /app/entrypoint.sh && \
    echo '  "server": {"host": "'"\$VIDIO_HOST"'", "port": '$VIDIO_PORT'},' >> /app/entrypoint.sh && \
    echo '  "auth": {"secret_key": "'"\$AUTH_SECRET"'", "algorithm": "HS256", "expiration_days": 7},' >> /app/entrypoint.sh && \
    echo '  "pipelines": {"retinal": {"models_dir": "/data/models/retinal"}, "histology": {"models_dir": "/data/models/histology", "tile_size": 256, "tissue_threshold": 0.5, "tcga_api_url": "https://api.gdc.cancer.gov"}, "radiology": {"models_dir": "/data/models/radiology"}, "spatial": {"min_genes_per_spot": 200, "min_spots_per_gene": 10, "n_top_genes": 2000}},' >> /app/entrypoint.sh && \
    echo '  "processing": {"max_concurrent_jobs": 4, "gpu_device": "cuda:0"}' >> /app/entrypoint.sh && \
    echo '}' >> /app/entrypoint.sh && \
    echo 'EOFCFG' >> /app/entrypoint.sh && \
    echo 'exec python app.py' >> /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

EXPOSE 7070

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import requests; r=requests.get('http://localhost:7070/auth'); exit(0 if r.status_code in [200,400,405] else 1)"

ENTRYPOINT ["/app/entrypoint.sh"]
