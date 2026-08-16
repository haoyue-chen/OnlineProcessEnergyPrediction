# Kubernetes Deployment (optional template)

Optional manifests for running the MoE energy service on Kubernetes. **These are
templates — they have not been applied to any cluster** (this project was built
and verified with Docker locally; no cluster was available). They're provided so
the service *can* be deployed when a cluster exists.

## What's here

| File | Resources |
|---|---|
| `deployment.yaml` | `Deployment` (inference `serve`, 1 replica) + `Service` (ClusterIP, port 8800) |
| `data-and-job.yaml` | `PersistentVolumeClaim` for `work/` + a one-off `Job` template for batch tasks |

The Deployment runs the inference API (`serve`). The model artifact is baked into
`energy-offloading:latest`, so **the Deployment needs no data volume**. The batch
tasks (task4/task5/online/offload/snakemake) read the dataset, so they're run as a
`Job` with `work/` mounted (see below).

## Prerequisites

The cluster must be able to pull `energy-offloading:latest`:

- **Local cluster (kind / minikube):** load the locally built image into the cluster
  ```sh
  # kind:
  kind load docker-image energy-offloading:latest
  # minikube:
  minikube image load energy-offloading:latest
  ```
- **Real cluster:** push to a registry and update the `image:` field
  ```sh
  docker tag energy-offloading:latest <registry>/energy-offloading:latest
  docker push <registry>/energy-offloading:latest
  # then set image: <registry>/energy-offloading:latest and imagePullPolicy: Always
  ```

## Deploy the inference service

```sh
kubectl apply -f deploy/k8s/deployment.yaml

kubectl get pods -l app=energy-offloading
kubectl get svc energy-offloading

# test from your machine via port-forward:
kubectl port-forward svc/energy-offloading 8800:8800 &
curl -s localhost:8800/health
curl -s -X POST localhost:8800/predict -H 'Content-Type: application/json' \
  -d '{"features":{"delta_cpu_ns":5e8,"delta_instructions":2e9}}'
```

Health checks are built in: a **readiness** and **liveness** probe both hit
`GET /health` on port 8800 (see `deployment.yaml`).

## Static `serve` vs live `serve-online` (scaling)

`deployment.yaml` runs **`serve`** (static inference). That is stateless, so it is
safe to scale: bump `replicas` freely.

The live **`serve-online`** mode is **stateful** — it keeps the evolving model in
memory and persists it to a single local `online_state.pkl`. It must therefore run
as a **single replica**: do not point multiple pods at one state file, or they
will overwrite each other's updates. If you add a `serve-online` Deployment:

- set `replicas: 1`, `strategy.type: Recreate`, and back the state path with a
  `ReadWriteOnce` PVC mounted at `/app/models/state`;
- horizontal scaling of online learning would require external shared state with
  locking (single-writer service / DB / Redis / update queue) — out of scope for v1.1.

## Providing the `work/` data for batch tasks

The dataset (~500 MB) is **not** in the image. Two ways to mount it:

**A. PersistentVolumeClaim (real cluster)** — create the PVC, populate it once,
then the Job mounts it read-only:

```sh
kubectl apply -f deploy/k8s/data-and-job.yaml   # creates the PVC + a task4 Job

# populate the PVC once (e.g. via a temporary pod):
#   kubectl cp ./work <helper-pod>:/data/work
```

**B. hostPath (single-node local cluster)** — if `work/` already lives on the
node, swap the Job's `work-data` volume for the commented `hostPath` block in
`data-and-job.yaml` and point it at the absolute path of your dataset directory.

Run a different task by editing the Job's `args` (e.g. `["task5"]`,
`["offload","--expert","linear"]`) and re-applying. Watch it:

```sh
kubectl logs -f job/energy-task4
```

## Clean up

```sh
kubectl delete -f deploy/k8s/deployment.yaml
kubectl delete -f deploy/k8s/data-and-job.yaml
```

## Honest scope

- These manifests are **valid and ready to apply, but were never applied** — no
  Kubernetes cluster was available in this environment (`kubectl` isn't installed
  here). They are an optional deployment template, not a verified live deployment.
- The Docker path (`DOCKER.md`) **is** verified end-to-end locally; prefer it for
  reproducing the project. Use these manifests only when you actually have a cluster.
