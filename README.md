# Online Process Energy Prediction

This project extends the ProcessEnergyAccounting framework with online and incremental machine learning techniques for process-level energy prediction.

## Project Objectives

1. Evaluate online learning methods against the current baseline model.
2. Investigate workload-dependent model performance.
3. Explore adaptive and mixture-of-experts approaches.
4. Prepare integration into the Snakemake offloading middleware.

## Current Status

- [x] Baseline framework analysis
- [x] Repository setup
- [ ] Benchmark evaluation
- [ ] Online learning model implementation
- [ ] Comparative experiments
- [ ] Middleware integration

# Original Process Energy Accounting

This repository collects **machine power measurements** together with **per-process runtime metrics** and stores them in **InfluxDB**. Its main purpose is to create datasets for later energy analysis and attribution.

In short, the project does this:

1. read power data from a smart meter
2. collect process-level metrics from the machine
3. store both in InfluxDB at fixed time intervals
4. export the recorded session to parquet for analysis

## What this is for

Use this project when you want to record how much activity different processes generate over time and relate that to measured node power.

The repository is mainly focused on:

- **data collection**
- **session recording**
- **dataset export**

## Requirements

You need:

- Linux (Tested on Ubuntu 24.04.4 LTS)
- Python `>=3.10,<3.14`
- Poetry
- InfluxDB 2.x
- a compatible smart meter reachable over HTTP or HTTPS
- root privileges for monitoring on most systems

Some monitoring features rely on **BCC/eBPF** and Linux performance counters from **perf**, so the machine must support that setup.

Required Python packages:

- `pandas`
- `numpy`
- `psutil`
- `influxdb-client`
- `requests`
- `pyyaml`
- `cvxpy` 
- `scikit-learn` 
- `matplotlib`

For Linux perf counters, a permissive setup is often needed. Example:

- `echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid`

Depending on your system, you may also want to make this persistent through `sysctl`.

## Installation

### 1. Set up the Python environment

The Python dependencies are defined in `pyproject.toml`, and the repository includes a helper script for creating and validating a Python environment:

- `scripts/py-env.sh`

If you run the script from the project directory, it can usually figure out the correct setup on its own. Since this repository uses Poetry, the script will normally auto-detect that from `pyproject.toml`.

Recommended order:

1. create the Python environment with `bash scripts/py-env.sh`
2. activate that environment using the **source/home/<user>.../.venv.bin/activate** command
3. run `bash scripts/install-deps.sh` (make scripts executable) so the system dependecies such as BCC Python bindings can also be installed into the active environment

Recommended setup:

- `bash scripts/py-env.sh`

If you want to recreate the environment from scratch:

- `bash scripts/py-env.sh --clean --force`

At the end, the script prints the environment location and how to activate it. For a Poetry environment, that is typically one of these:

- `source "$(poetry env info -p)/bin/activate"`
- `python ...` or if elevated priviledges are needed use `sudo + python venv path`

A practical setup flow is therefore:

- `bash scripts/py-env.sh`
- `source "$(poetry env info -p)/bin/activate"`

### 2. Install system dependencies

The repository includes an installation helper:

- `scripts/install-deps.sh`

This script installs:

- Docker / Docker Compose
- build tools
- LLVM/Clang dependencies
- Python tooling
- Linux headers
- BCC build requirements

Run it with:

- `bash scripts/install-deps.sh`

This script is mostly aimed at Debian/Ubuntu-like systems.

Important: this script does more than install system packages. If you already have an active virtual environment or active Poetry environment, it will also try to install the Python `bcc` bindings into that environment so that `import bcc` works there as well. That matters for this project, because the monitor depends on BCC from inside the Python environment you use to run `delta_aggregator.py`.

### 3. Start InfluxDB

A minimal local InfluxDB setup is included in `docker-compose.yml`.

Start it with:

- `docker compose up -d`

The default local configuration uses:

- URL: `http://localhost:8086`
- org: `myorg`
- bucket: `mybucket`

## Configuration

Create a local `.env` file. You can start from `example.env`.

Important variables:

- `INFLUX_URL`
- `INFLUX_TOKEN`
- `INFLUX_ORG`
- `INFLUX_BUCKET`
- `SMARTMETER_HOST`
- `SMARTMETER_USER`
- `SMARTMETER_PASSWORD`
- `SMARTMETER_SSL`

Example:

```ProcessEnergyAccounting/example.env#L1-8
INFLUX_URL=http://localhost:8086
INFLUX_TOKEN=my-super-secret-auth-token
INFLUX_ORG=myorg
INFLUX_BUCKET=mybucket

SMARTMETER_HOST=replace-with-smart-meter-host
SMARTMETER_USER=replace-with-smart-meter-user
SMARTMETER_PASSWORD=replace-with-smart-meter-password
```

## Quick start

### 1. Start the monitor

Run:

- `sudo $(which python) delta_aggregator.py --interval 2 --sample-rate 0.5`

This starts the main monitoring loop.

It will:

- query the smart meter repeatedly during each interval
- collect per-process metrics from the machine
- compute interval-level values
- write the result to InfluxDB

Root privileges are often required because the monitor uses BPF and performance counters.

Example output:

```ProcessEnergyAccounting/README.md#L1-17
sudo $(which python) delta_aggregator.py
Starting DeltaAggregator: interval=2.0, sample_rate=2.0
Influx: http://localhost:8086 (org=myorg, bucket=mybucket)
Monitoring started. Press Ctrl+C to stop.
PIDs before cleanup:  58
Found 57 unique PIDs in BPF tables (after cleanup)
Items in Syscall Count Table:  95
Items in Syscall Count Table after cleanup:  95
Total CPU time (sum):  22480672
Total CPU time (bpf):  22191007
Process list generated in 0.01 seconds
PIDs before cleanup:  66
Found 65 unique PIDs in BPF tables (after cleanup)
Items in Syscall Count Table:  103
Items in Syscall Count Table after cleanup:  103
Total CPU time (sum):  195661425
Total CPU time (bpf):  195387613
Process list generated in 0.01 seconds
[11:02:35] delta count: 57, avg_power: 676.0, interval_energy: 1352.126519203186
```

### 2. Run your workload

While the monitor is running, start whatever workload you want to measure.

That can be:

- your own application
- a benchmark
- a script
- an optional automated workload generator from this repository

For example, `scripts/run_daw_load_generation.sh` creates a reproducible session by running a sequence of pipeline workloads, inserting idle periods between runs, optionally adding short `stress-ng` bursts, and writing session metadata under `runs/`. Conceptually, it is just a way to generate a varied mix of busy and idle machine phases so the monitor records a broader range of behavior over time.

### 3. Stop the monitor

When the workload is finished, stop `delta_aggregator.py`.

At that point, the recorded interval data is stored in InfluxDB.

### 4. Data exports

To export a recorded session to parquet, use:

- `scripts/export_process_dataset.sh`

Example:

- `bash scripts/export_process_dataset.sh --session-dir runs/<session-id>`

This writes a process-level dataset such as:

- `runs/<session-id>/datasets/process_interval_data.parquet`

You can also use the Python loader directly through `estimation/data/data_loader.py` if you want full control over time range and aggregation window.

### 5. Analysis and modeling

The repository contains a few standalone scripts for feature exploration and modeling that works on the exported datasets of the prior step:

- `estimation/feature_selection/feature_selection.py` — exploratory feature analysis, correlations, lag checks, etc.

- `estimation/feature_selection/sfs.py` — sequential forward selection that greedily adds features based on test R² improvement
- `cvxpy_estimator.py` — a direct CVXPY-based estimator workflow that trains a sparse interval-energy model and produces diagnostic plots


## Optional workload generation

The repository also contains:

- `scripts/run_daw_load_generation.sh`

This can be used to create repeatable workload sessions and store run metadata under `runs/`.

This is optional and not required for basic monitoring.

## Repository overview

Important files and directories:

- `delta_aggregator.py` — main monitor
- `monitoring/` — process and kernel-level metric collection
- `database/client.py` — InfluxDB read/write logic
- `smart_meter/client.py` — smart meter API client
- `scripts/install-deps.sh` — system dependency setup helper
- `scripts/export_process_dataset.sh` — export monitored data to parquet
- `scripts/run_daw_load_generation.sh` — optional workload generator
- `docker-compose.yml` — local InfluxDB setup
- `example.env` — configuration template

## Common issues

### `ModuleNotFoundError: No module named 'bcc'`

Your Python environment cannot see the BCC bindings.

Things to check:

- whether BCC is installed on the system
- whether your Poetry environment can access system packages
- whether `scripts/install-deps.sh` completed successfully

### permission errors

Always check file and directory and user permissions if you run into errors.

### smart meter connection problems

Check:

- `SMARTMETER_HOST`
- `SMARTMETER_USER`
- `SMARTMETER_PASSWORD`
- `SMARTMETER_SSL`

If the device only supports HTTPS, set `SMARTMETER_SSL=true`.
