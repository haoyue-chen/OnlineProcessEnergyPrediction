# Energy-Offloading — Full-Project Docker Image

One image that reproduces the **whole project** from the Phase-2 plan:
process-energy MoE (Task 4), online learning under drift (Task 5), the online/offline
baseline comparison, the energy-aware offloading simulation, the live Snakemake
offloading DAG, and the MoE inference HTTP service.

The measurement data (`work/`, ~500 MB) is **not** baked into the image — it
belongs to the data-collection project and is mounted read-only at run time.

## Build

```sh
cd energy-offloading
docker build -t energy-offloading:latest .
```

## Run a task

The image takes a subcommand. Mount the data at `/data/work` (read-only) and,
optionally, a host `results/` dir to capture outputs.

```sh
DATA=/home/hujiao/MPDS/work          # adjust to your work/ path

# Task 4 — MoE vs Single Model (5-fold CV)
docker run --rm -v $DATA:/data/work:ro energy-offloading:latest task4 --expert linear

# Task 5 — Static vs Online MoE under workload drift (ARF online expert)
docker run --rm -v $DATA:/data/work:ro energy-offloading:latest task5

# Online/offline baseline comparison (the 7-family table) — write results to host
docker run --rm -v $DATA:/data/work:ro -v $PWD/results/moe:/app/results/moe \
  energy-offloading:latest online

# Energy-aware offloading simulation
docker run --rm -v $DATA:/data/work:ro energy-offloading:latest offload --expert linear

# Live Snakemake offloading DAG (strategy comparison)
docker run --rm -v $DATA:/data/work:ro energy-offloading:latest snakemake compare

# Everything (task4 + task5 + online + offload)
docker run --rm -v $DATA:/data/work:ro -v $PWD/results:/app/results \
  energy-offloading:latest all
```

### Subcommands

| Command | What it runs |
|---|---|
| `serve` (default) | MoE inference HTTP service on port 8800 |
| `serve-online` | live online-learning service (`/predict` + `/update`) on 8800 |
| `online-workflow` | closed loop: job runner auto-calls `/predict` + `/update` per job |
| `task4` | MoE vs Single Model, 5-fold CV (`--expert linear|rf|hgb|...`) |
| `task5` | Static vs Online MoE under drift (ARF expert) |
| `online` | online/offline baseline comparison (7-family table) |
| `compare` | full batch+online model survey |
| `offload` | energy-aware offloading simulation |
| `snakemake` | live Snakemake offloading DAG (`compare` or a single strategy) |
| `export` | (re)export the MoE artifact used by `serve` |
| `all` | task4 + task5 + online + offload, writing `results/` |
| `bash` | shell inside the container |

## Inference service

```sh
# start (docker compose reads ../work and ./results automatically)
docker compose up -d
# or plain docker:
docker run -d --name energy-offloading -p 8800:8800 \
  -v /home/hujiao/MPDS/work:/data/work:ro energy-offloading:latest serve

curl -s localhost:8800/health
# {"status": "ok"}
curl -s -X POST localhost:8800/predict -H 'Content-Type: application/json' \
  -d '{"features":{"delta_cpu_ns":5e8,"delta_instructions":2e9}}'
# {"energy_wh": 244.38, "expert": "DAW1"}
```

Endpoints: `GET /health`, `GET /info`, `POST /predict`, `POST /predict_batch`.

## Three serving modes (what's implemented vs what needs production wiring)

| Mode | Command | Model updates? | Feedback source | Status |
|---|---|---|---|---|
| **Static inference** | `serve` | no — `/predict` only | n/a | implemented, verified |
| **Live online API** | `serve-online` | yes — manual `/predict` + `/update` | manual `curl` | implemented, verified |
| **Online workflow loop** | `online-workflow` | yes — auto `/predict` + `/update` per job | job runner (here: measured-data records) | implemented as a **prototype**, verified |

- **Static** (`serve`): one-shot predictions, model frozen.
- **Live online API** (`serve-online`): the model *can* learn, but you must call
  `/update` yourself when a job's true energy is known.
- **Online workflow loop** (`online-workflow`): a job runner drives the loop
  automatically — it calls `/predict` before each job and `/update` after, with no
  manual curl. This is the closed-loop **prototype**. The "true energy" fed back is
  taken from the measured workload records (simulating a real job-completion
  signal); wiring `/update` to an actual Snakemake hook / cluster energy monitor is
  the remaining production step.

## Live online learning API (`serve-online`)

The default `serve` mode is **static inference**: `/predict` always returns the
same answer for the same input — the model never changes. `serve-online` is a
**live online-learning** service: after a job finishes you report its true energy
to `/update`, the corresponding workload expert learns from it incrementally, the
new state is persisted, and the *next* `/predict` uses the updated model.

Intended loop:

```text
job starts → POST /predict → {prediction_id, energy_wh, expert}
job finishes, true energy observed
→ POST /update {prediction_id, true_energy_wh}
→ that expert updates incrementally, state saved, model_version++
→ next /predict reflects the update
```

Online experts are **River ARF** (Adaptive Random Forest) per workload — the
strongest online model in our baseline survey — warm-started on each workload.

### Online workflow loop (`online-workflow`) — automatic feedback

Runs the whole closed loop in one container: it launches `serve-online` internally,
then a job runner walks a real workflow and for **each job** calls `/predict`
before dispatch and `/update` after completion — no manual curl. Each job record
captures `job_id, expert, predicted_energy_wh, true_energy_wh, prediction_id,
model_version_before, model_version_after, update_success`.

```sh
docker run --rm \
  -e ONLINE_STATE_PATH=/app/models/state/online_state.pkl \
  -v /home/hujiao/MPDS/work:/data/work:ro \
  -v "$PWD/online_state:/app/models/state" \
  -v "$PWD/results:/app/results" \
  energy-offloading:latest online-workflow --jobs-per-workload 5 \
    --out /app/results/online_workflow_run.json
```

It prints a job-by-job table + a summary (`num_updates 0 → N`, all updates ok,
and the MAE of the first vs second half of the run — online learning typically
lowers it). The updated state lands in the mounted volume, so a later
`serve-online` continues from it. The `work/` mount is required because the jobs
(and their measured "true energy" feedback) come from the real workload data.

Standalone driver (against an already-running `serve-online`):

```sh
PYTHONPATH=. python -m offloading.online_workflow --url http://localhost:8800 \
  --jobs-per-workload 5 --out results/online_workflow/run.json
```

### Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | `{"status":"ok"}` |
| GET | `/info` | — | metadata + `num_updates`, `model_version` |
| POST | `/predict` | `{"features":{...}}` | `{prediction_id, energy_wh, expert, model_version}` |
| POST | `/update` | `{prediction_id\|features, true_energy_wh, [expert]}` | `{updated_expert, num_updates, model_version}` |
| POST | `/predict_then_update` | `{"features":{...}, "true_energy_wh":float}` | predict + update combined |

### Run (persist state via a mounted volume)

```sh
# state lives at ONLINE_STATE_PATH; mount a host dir so it survives restarts
mkdir -p ./online_state
docker run -d --name energy-online -p 8800:8800 \
  -e ONLINE_STATE_PATH=/app/models/state/online_state.pkl \
  -v "$PWD/online_state:/app/models/state" \
  energy-offloading:latest serve-online
```

`serve-online` needs **no** `work/` mount — the warm-started base model
(`online_base.pkl`) is baked into the image. The data is only needed if you want
to *rebuild* that base (`export-online`).

### Example

```sh
# predict
P=$(curl -s -X POST localhost:8800/predict -H 'Content-Type: application/json' \
     -d '{"features":{"delta_cpu_ns":5e8,"delta_instructions":2e9}}')
echo "$P"   # {"prediction_id":"...","energy_wh":268.19,"expert":"DAW1","model_version":0}

# feed back the true energy after the job finishes
PID=$(echo "$P" | python3 -c "import sys,json;print(json.load(sys.stdin)['prediction_id'])")
curl -s -X POST localhost:8800/update -H 'Content-Type: application/json' \
  -d "{\"prediction_id\":\"$PID\",\"true_energy_wh\":250.0}"
# {"updated_expert":"DAW1","num_updates":1,"model_version":1}
```

### Persistence & concurrency

- State is written atomically to `online_state.pkl` after every update (temp file
  + replace), under a lock so concurrent `/update` calls cannot corrupt it.
- On start the service loads `online_state.pkl` if present (continues prior
  learning), else the baked-in `online_base.pkl`. **Mount a volume** at the state
  path or updates are lost when the container is removed.

### Single-instance only (scaling caveat)

The in-process lock + local-file state make this safe for a **single instance**.
It is **not** safe to run multiple replicas against the same `online_state.pkl`:
two pods would each hold their own in-memory model and overwrite each other's
state file, losing updates and corrupting learning.

- **Docker:** run one container for the online service.
- **Kubernetes:** keep the online Deployment at **`replicas: 1`** (the provided
  `deploy/k8s/deployment.yaml` runs `serve` static inference, which *is* safe to
  scale; a `serve-online` Deployment must stay at 1 — see `deploy/k8s/README.md`).
- To scale online learning horizontally you'd need **external shared state with
  locking** (e.g. a single writer service, a database/Redis-backed model store, or
  a streaming update queue) instead of a local pickle — out of scope for v1.1.

### Rebuild the warm-started base (optional)

```sh
docker run --rm -v /home/hujiao/MPDS/work:/data/work:ro \
  -v "$PWD/models:/app/models" energy-offloading:latest export-online arf
```

### Honest scope

This is a real, working live-update service (verified: predict → update changes
the model, state survives container restart — see `deploy/online_smoke_test.sh`).
What it does **not** include: an automatic feedback pipeline. In production the
true `energy_wh` must be fed back automatically — i.e. a **Snakemake hook / job
runner / cluster energy monitor** would, when a job finishes, read its measured
energy and call `POST /update` (with the `prediction_id` returned at scheduling
time, or the job's features). In this v1.1 that feedback step is supplied
**manually via `curl`** to simulate the real job-completion signal; wiring it to
an actual job runner is the remaining integration work.

## Feature-based (resource-grouped) MoE — new architecture (`feature_moe/`)

Per the team's new design docs, there is a **second MoE** that groups experts by
**resource type** (CPU / Memory / I/O / Network) with a **learned soft gate**
(weighted fusion), coexisting with the workload-grouped MoE used by the service
above. It is the offline research core (Modules 4–5 of the new architecture) and
runs directly from the project — no Docker/entrypoint change needed:

```sh
docker run --rm -v /home/hujiao/MPDS/work:/data/work:ro \
  energy-offloading:latest bash -c \
  "PYTHONPATH=. python -m feature_moe.run_resource_moe --expert linear"
```

Result (full data, 5-fold CV): resource-MoE ≈ single global under RF experts
(0.960 vs 0.964) and *below* it under linear experts (0.738 vs 0.792) — the
workload-grouped MoE remains strongest here (RF 0.972, linear 0.843). The resource
design is implemented and sound but CPU-dominated data limits its benefit; see
`feature_moe/README.md` for the honest analysis. Wiring this resource-MoE into
`serve-online` / offloading is a follow-up; the service currently serves the
workload-grouped MoE.

## Verified results (run in-container)

| Task | Result |
|---|---|
| task4 (linear) | overall R² single 0.792 → MoE 0.888 |
| task5 (ARF) | Online-MoE R² 0.93 / MAE 6.18 vs static 0.71 / 18.3 |
| online | Adaptive RF best online R² 0.936 |
| offload | MoE saves 6.2%, closes 87% of single→oracle gap |
| snakemake | moe saves 6.0%, beats random 5.0%, tracks oracle |
| serve | `/predict` → 244.38 Wh, expert DAW1 |

## Scope / honesty

Verified by **building and running the image locally** (the outputs above are
real). It is not deployed to a remote cluster — that needs a target host/registry,
which isn't part of this repo. To deploy elsewhere: `docker tag` + `docker push`
to a registry and run it there, or wrap the same image in a Kubernetes
Deployment + Service (the inference service listens on 8800).

## Deploy to a remote server

Two common ways to get the image onto a server:

**A. Save / copy / load** (no registry needed — simplest for one server):

```sh
# on this machine
docker save energy-offloading:latest -o energy-offloading.tar
scp energy-offloading.tar user@server:/path/

# on the server
ssh user@server
docker load -i /path/energy-offloading.tar
# copy the work/ data over too (only needed for batch tasks, not for `serve`):
#   scp -r work user@server:/path/work
docker run -d --name energy-offloading -p 8800:8800 \
  -v /path/work:/data/work:ro energy-offloading:latest serve
curl -s localhost:8800/health
```

**B. Registry** (better for multiple hosts / CI):

```sh
docker tag energy-offloading:latest <registry>/energy-offloading:latest
docker push <registry>/energy-offloading:latest
# on the server:
docker run -d -p 8800:8800 <registry>/energy-offloading:latest serve
```

For Kubernetes, see `deploy/k8s/README.md` (optional templates).

## Smoke test

A one-shot script exercises the batch tasks + the service end-to-end:

```sh
./deploy/smoke_test.sh
# runs: docker run task4, docker run offload, start service, curl /health, curl /predict
# DATA=/path/to/work PORT=8801 ./deploy/smoke_test.sh   # overrides
```

The live online-learning service has its own smoke test (predict → update →
restart → state persists):

```sh
./deploy/online_smoke_test.sh
# PORT=8821 ./deploy/online_smoke_test.sh   # override port
```

The closed-loop workflow has its own smoke test (runs a small workflow, checks
every job called `/predict` + `/update`, num_updates grew, state persists + reloads):

```sh
./deploy/online_workflow_smoke_test.sh
# DATA=/path/to/work PORT=8841 ./deploy/online_workflow_smoke_test.sh
```

For just the lightweight inference service (numpy+sklearn only), see
`inference/README.md`.
