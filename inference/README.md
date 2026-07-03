# MoE Energy-Prediction Inference Service (Docker)

A slim, deployable HTTP service that serves the trained MoE energy predictor.
It loads `models/moe_linear.pkl` (raw scikit-learn gate + per-workload experts)
and answers prediction requests. Depends only on numpy + scikit-learn, so the
image is small (~545 MB) and needs none of the training-side packages.

## Build

```sh
cd energy-offloading
docker build -t energy-moe-inference:latest .
```

## Run

```sh
# plain docker
docker run -d --name energy-moe -p 8800:8800 energy-moe-inference:latest

# or docker compose
docker compose up -d
```

The service listens on port **8800**. A container `HEALTHCHECK` hits `/health`
every 30s (status shows `healthy` in `docker ps`).

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | `{"status":"ok"}` |
| GET | `/info` | — | model metadata (features, labels, expert class) |
| POST | `/predict` | `{"features": {...}}` | `{"energy_wh": float, "expert": label}` |
| POST | `/predict_batch` | `{"samples": [{...}, ...]}` | `{"predictions": [float, ...]}` |

`features` is a flat dict of process-metric values; missing keys default to 0.
The gate routes each sample to its workload expert automatically — callers do not
need to know the workload.

## Example calls

```sh
curl -s localhost:8800/health
# {"status": "ok"}

curl -s localhost:8800/info
# {"model_type":"moe","expert_class":"linear","n_features":16, ... ,"labels":["DAW1","DAW2","Phoronix","Stress"]}

curl -s -X POST localhost:8800/predict -H 'Content-Type: application/json' \
  -d '{"features":{"delta_cpu_ns":5e8,"delta_instructions":2e9}}'
# {"energy_wh": 244.38, "expert": "DAW1"}

curl -s -X POST localhost:8800/predict_batch -H 'Content-Type: application/json' \
  -d '{"samples":[{"delta_cpu_ns":0},{"delta_cpu_ns":9e8,"delta_instructions":3e9}]}'
# {"predictions": [239.80, 246.94]}
```

## Swap the model

The image bakes in `moe_linear.pkl`. To serve a different artifact, mount it and
point `MODEL_PATH` at it (no rebuild needed):

```sh
# export an RF-expert MoE first (from the project venv):
#   PYTHONPATH=. python -m moe_export.export_moe --expert rf --out models/moe_rf.pkl
docker run -d --name energy-moe -p 8800:8800 \
  -v "$PWD/models/moe_rf.pkl:/app/models/custom.pkl" \
  -e MODEL_PATH=/app/models/custom.pkl \
  energy-moe-inference:latest
```

## Stop / clean

```sh
docker rm -f energy-moe         # or: docker compose down
```

## Scope / honesty

This is verified by **building and running the container locally** (curl outputs
above are real). It is not deployed to any remote cluster — that needs a target
host / registry, which isn't part of this repo. To deploy elsewhere: push the
image to a registry (`docker tag` + `docker push`) and run it there, or wrap it
in a Kubernetes Deployment + Service using the same image and port.

## Files

| File | Role |
|---|---|
| `inference/predictor.py` | loads the MoE artifact, gate-routes + predicts (sklearn-only) |
| `inference/server.py` | stdlib HTTP server exposing the endpoints |
| `inference/requirements.txt` | pinned numpy / scipy / scikit-learn |
| `Dockerfile` | slim `python:3.12-slim` image |
| `docker-compose.yml` | one-command build + run |
| `.dockerignore` | keeps the build context to inference/ + the model |
