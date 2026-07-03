# Full-project image — reproduces the whole MoE energy-offloading project:
# Task 4/5, online baseline comparison, offloading simulation, the live Snakemake
# DAG, and the MoE inference service. One image, dispatched via docker-entrypoint.sh.
#
# Measurement data (work/, ~500 MB) is NOT baked in — it belongs to the separate
# data-collection project and is mounted read-only at /data/work at run time.
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MODEL_PATH=/app/models/moe_linear.pkl \
    PORT=8800

# git is needed by snakemake's workflow tooling; build-essential kept minimal.
RUN apt-get update && apt-get install -y --no-install-recommends \
        bash git \
    && rm -rf /var/lib/apt/lists/*

# Deps first for layer caching.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Project code + the pre-exported model artifact.
COPY moe/ /app/moe/
COPY moe_export/ /app/moe_export/
COPY feature_moe/ /app/feature_moe/
COPY offloading/ /app/offloading/
COPY snakemake_integration/ /app/snakemake_integration/
COPY inference/ /app/inference/
COPY models/moe_linear.pkl /app/models/moe_linear.pkl
COPY models/online_base.pkl /app/models/online_base.pkl
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh /app/snakemake_integration/run.sh

# data.py resolves work/ as a sibling of the project dir; with /app as the project
# we expect the data mounted at /data/work, so symlink it in.
RUN ln -s /data/work /work
WORKDIR /app

EXPOSE 8800

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8800\")}/health')" || exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["serve"]
