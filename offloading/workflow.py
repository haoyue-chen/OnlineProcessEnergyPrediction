"""Workflow model: segment real measured intervals into a sequence of jobs.

A "workflow" here is a list of ``Job``s, each built by grouping a contiguous run
of measured intervals from one workload. For each job we keep:

* ``features`` — the summed interval features (a single feature row the MoE can
  score, formed by summing the member intervals just like ``data.aggregate_intervals``
  sums processes within an interval),
* ``true_energy_wh`` — the measured node energy of the job on the primary cluster
  (sum of member ``interval_energy``); this is the ground truth,
* ``runtime_s`` — the wall-clock span of the job on the primary, derived from the
  interval timestamps.

Segmenting the real stream (rather than inventing a synthetic DAG) keeps every
job's features, energy and runtime grounded in measured data, so the MoE's
per-job predictions are meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from moe import data


@dataclass
class Job:
    job_id: int
    workload: str
    interval_features: pd.DataFrame  # member intervals' FEATURE rows (trained domain)
    true_energy_wh: float    # measured primary energy (ground truth)
    runtime_s: float         # measured primary runtime
    n_intervals: int


def _median_interval_seconds(index: pd.DatetimeIndex) -> float:
    """Typical spacing between consecutive intervals (seconds)."""
    if len(index) < 2:
        return 2.0  # datasets are sampled ~every 2s; safe fallback
    deltas = np.diff(index.values).astype("timedelta64[ns]").astype(float) / 1e9
    deltas = deltas[(deltas > 0) & (deltas < 3600)]  # drop gaps/restarts
    return float(np.median(deltas)) if len(deltas) else 2.0


def build_workflow(
    datasets: dict[str, data.IntervalDataset],
    jobs_per_workload: int = 25,
    seed: int = 0,
) -> list[Job]:
    """Build a workflow by chopping each workload into ``jobs_per_workload`` jobs.

    Each job is a contiguous block of intervals (chronological), so its summed
    features and energy are internally consistent. Jobs from all workloads are
    concatenated to form one heterogeneous workflow — exactly the mix the gate
    must route correctly.
    """
    jobs: list[Job] = []
    jid = 0
    for label, ds in datasets.items():
        X, y = ds.X, ds.y
        dt = _median_interval_seconds(X.index)
        blocks = np.array_split(np.arange(len(X)), jobs_per_workload)
        for block in blocks:
            if len(block) == 0:
                continue
            sl = slice(block[0], block[-1] + 1)
            jobs.append(
                Job(
                    job_id=jid,
                    workload=label,
                    interval_features=X.iloc[sl].reset_index(drop=True),
                    true_energy_wh=float(y.iloc[sl].sum()),
                    runtime_s=float(len(block) * dt),
                    n_intervals=int(len(block)),
                )
            )
            jid += 1
    return jobs


def predict_job_energy(jobs: list[Job], predict_fn) -> np.ndarray:
    """Predict each job's total energy = sum of its interval-level predictions.

    The predictors are trained on *interval* rows, so we score each interval and
    sum within a job, rather than feeding a (much larger) summed feature row that
    lies outside the trained domain. One batched call keeps it fast.
    """
    frames = [j.interval_features for j in jobs]
    big = pd.concat(frames, ignore_index=True)
    preds = predict_fn(big)
    out = np.empty(len(jobs), dtype=float)
    pos = 0
    for i, f in enumerate(frames):
        n = len(f)
        out[i] = float(preds[pos:pos + n].sum())
        pos += n
    return out
