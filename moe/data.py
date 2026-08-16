"""Shared data layer for the Mixture-of-Experts energy estimation experiments.

The raw datasets are per-process, per-interval rows (one parquet per workload run,
collected under the dataset root — see ``_WORK_ROOT`` below). The energy target
``interval_energy`` is a node-level value repeated across every process row of the
same ``_time``. To frame this as a supervised problem we aggregate per interval:
process features are summed over the interval, and the target is the (single) node
energy for that interval.

This mirrors the framing already used by the linear CVXPY baseline
(``feature_selection.py`` / ``cvxpy_estimator.py``) so results are comparable.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Dataset root. The measurement data is produced by the separate data-collection
# project (ProcessEnergyAccounting); this project only reads it. By default we
# look for ``datasets/`` inside the repository checkout; set the ``PEA_DATA_ROOT``
# environment variable to read the data from anywhere else (the Docker image sets
# ``PEA_DATA_ROOT=/data/work``, where the host dataset is mounted read-only).
# The expected layout is unchanged:
#   <root>/<run_dir>/runs/<run_id>/datasets/process_interval_data.parquet
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_ROOT = _PROJECT_ROOT / "datasets"
_WORK_ROOT = Path(os.environ.get("PEA_DATA_ROOT", _DEFAULT_DATA_ROOT))

TARGET = "interval_energy"
TIME = "_time"

# Process metrics summed per interval to form the regression features. These are
# the numeric counters available in every dataset; they cover CPU, memory, IO,
# network, context switches and the syscall-class breakdown referenced in the plan.
FEATURES = [
    "delta_cpu_ns",
    "delta_instructions",
    "delta_cache_misses",
    "delta_branch_instructions",
    "delta_cycles",
    "delta_io_bytes",
    "delta_net_send_bytes",
    "delta_rss_memory",
    "context_switches",
    "syscall_count",
    "syscall_class_file",
    "syscall_class_network",
    "syscall_class_memory",
    "syscall_class_process",
    "syscall_class_other",
    "syscall_class_sched",
]

# Friendly workload label for each run directory. These are the four datasets the
# Phase-1 summary refers to (DAW1, DAW2, Phoronix, Stress).
WORKLOAD_LABELS = {
    "baseline-daw-4h-01": "DAW1",
    "baseline-daw-clean-perf-03": "DAW2",
    "baseline-phoronix-clean-perf-01": "Phoronix",
    "baseline-stress-clean-perf-01": "Stress",
}


@dataclass
class IntervalDataset:
    """Interval-aggregated data for a single workload."""

    workload: str
    X: pd.DataFrame  # index = _time, columns = FEATURES
    y: pd.Series  # index = _time, value = interval_energy

    def __len__(self) -> int:
        return len(self.y)


def _find_parquet(run_dir: str) -> Path | None:
    """Locate the process_interval_data.parquet for a given run directory name."""
    matches = glob.glob(
        str(_WORK_ROOT / run_dir / "runs" / "*" / "datasets" / "process_interval_data.parquet")
    )
    return Path(matches[0]) if matches else None


def aggregate_intervals(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Collapse per-process rows into one row per interval.

    Features are summed across all processes active in the interval; the target is
    the node energy, which is constant within an interval (we take the first value).
    """
    df = df[[TIME, TARGET] + FEATURES].copy()
    df[FEATURES] = df[FEATURES].fillna(0.0)
    grouped = df.groupby(TIME)
    X = grouped[FEATURES].sum()
    y = grouped[TARGET].first()
    # Keep only intervals with a valid, positive energy reading.
    valid = y.notna() & (y > 0)
    X, y = X[valid], y[valid]
    # Preserve chronological order — the online experiments stream in time order.
    X = X.sort_index()
    y = y.loc[X.index]
    return X, y


def load_workload(run_dir: str) -> IntervalDataset:
    """Load and interval-aggregate a single workload run."""
    path = _find_parquet(run_dir)
    if path is None:
        raise FileNotFoundError(f"No parquet found for run directory '{run_dir}' under {_WORK_ROOT}")
    df = pd.read_parquet(path, columns=[TIME, TARGET] + FEATURES)
    X, y = aggregate_intervals(df)
    return IntervalDataset(workload=WORKLOAD_LABELS.get(run_dir, run_dir), X=X, y=y)


def load_all() -> dict[str, IntervalDataset]:
    """Load every known workload, keyed by friendly label (DAW1, DAW2, …)."""
    datasets: dict[str, IntervalDataset] = {}
    for run_dir, label in WORKLOAD_LABELS.items():
        if _find_parquet(run_dir) is not None:
            datasets[label] = load_workload(run_dir)
    if not datasets:
        raise FileNotFoundError(f"No workload parquet files found under {_WORK_ROOT}")
    return datasets


def combined_frame(datasets: dict[str, IntervalDataset]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Stack all workloads into a single (X, y, workload-label) view.

    The returned index is a clean RangeIndex; the per-row workload label is
    returned separately so the gate has classification targets.
    """
    xs, ys, labels = [], [], []
    for label, ds in datasets.items():
        xs.append(ds.X.reset_index(drop=True))
        ys.append(ds.y.reset_index(drop=True))
        labels.append(pd.Series([label] * len(ds), name="workload"))
    X = pd.concat(xs, ignore_index=True)
    y = pd.concat(ys, ignore_index=True)
    w = pd.concat(labels, ignore_index=True)
    return X, y, w
