"""MoE-driven offloading decision engine (Objective 3).

Energy-first formulation
------------------------
The secondary cluster is greener (``energy_factor < 1``) but has **limited
capacity** — it is a rented slice, so it can only absorb jobs whose total
secondary-runtime fits a capacity budget. Offloading job *j* to the secondary
saves ``(1 - energy_factor) * energy(j)`` Wh but consumes ``runtime_secondary(j)``
of that budget. Choosing the subset that maximises energy saved within the budget
is a 0/1 knapsack; we solve it greedily by *energy-saved per unit secondary time*
(i.e. offload the most energy-dense jobs first), which is the standard, near-optimal
heuristic for this ratio problem and mirrors the greedy heuristic in the offloading
paper.

The catch: the engine does **not** know each job's true energy at decision time —
it must *predict* it. That is exactly where the MoE comes in. A strategy is just a
way of scoring jobs:

* ``moe``      — rank by MoE-predicted energy (gate routes each job to its expert),
* ``single``   — rank by the single global model's prediction,
* ``oracle``   — rank by the *true* measured energy (upper bound on any predictor),
* ``random``   — random order (lower-information baseline),
* ``all_primary`` — offload nothing (the single-cluster baseline; 0 energy saved).

All strategies are then scored on the **actual measured** energy/cost/makespan, so
a better predictor translates directly into more real energy saved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .clusters import Cluster
from .workflow import Job


@dataclass
class Outcome:
    strategy: str
    n_offloaded: int
    total_energy_wh: float
    energy_saved_wh: float       # vs all-primary
    energy_saved_pct: float
    total_cost: float
    makespan_s: float
    deadline_met: bool


def _ranking(strategy: str, jobs, pred_energy, rng) -> np.ndarray:
    """Return job indices ordered best-to-offload-first for the given strategy."""
    true_energy = np.array([j.true_energy_wh for j in jobs])
    runtime = np.array([j.runtime_s for j in jobs])
    if strategy == "all_primary":
        return np.array([], dtype=int)
    if strategy == "random":
        order = np.arange(len(jobs))
        rng.shuffle(order)
        return order
    score_energy = {"moe": pred_energy, "single": pred_energy, "oracle": true_energy}[strategy]
    # Energy saved per unit secondary time → offload energy-dense jobs first.
    density = score_energy / np.maximum(runtime, 1e-9)
    return np.argsort(-density)


def select_offloaded(
    jobs: list[Job],
    pred_energy: np.ndarray,
    secondary: Cluster,
    strategy: str,
    capacity_frac: float,
    seed: int = 0,
) -> np.ndarray:
    """Decide which jobs to offload — the single source of truth for the policy.

    Returns a boolean mask (True = offload to secondary). Both the in-process
    simulator (``decide``) and the Snakemake planner call this, so the live
    workflow and the experiment use *identical* decision logic.
    """
    rng = np.random.default_rng(seed)
    runtime = np.array([j.runtime_s for j in jobs])
    capacity_budget = capacity_frac * float(runtime.sum())

    order = _ranking(strategy, jobs, pred_energy, rng)
    offloaded = np.zeros(len(jobs), dtype=bool)
    used_secondary = 0.0
    for idx in order:
        sec_rt = secondary.runtime_of(runtime[idx])
        if used_secondary + sec_rt <= capacity_budget:
            offloaded[idx] = True
            used_secondary += sec_rt
    return offloaded


def decide(
    jobs: list[Job],
    pred_energy: np.ndarray,
    primary: Cluster,
    secondary: Cluster,
    strategy: str,
    capacity_frac: float,
    deadline_frac: float,
    seed: int = 0,
) -> Outcome:
    """Run one offloading strategy and score it on the *actual* measured numbers.

    ``capacity_frac`` sets the secondary budget as a fraction of the all-primary
    makespan; ``deadline_frac`` sets the makespan deadline likewise.
    """
    runtime = np.array([j.runtime_s for j in jobs])
    true_energy = np.array([j.true_energy_wh for j in jobs])

    total_primary_runtime = float(runtime.sum())
    deadline = deadline_frac * total_primary_runtime

    offloaded = select_offloaded(jobs, pred_energy, secondary, strategy, capacity_frac, seed)
    used_secondary = float(
        np.array([secondary.runtime_of(rt) for rt in runtime[offloaded]]).sum()
    )

    # Actual outcome (ground-truth energy, not predictions).
    primary_energy = true_energy[~offloaded].sum()
    secondary_energy = np.array(
        [secondary.energy_of(e) for e in true_energy[offloaded]]
    ).sum()
    total_energy = float(primary_energy + secondary_energy)

    primary_cost = (true_energy[~offloaded] / 1000.0 * primary.cost_per_kwh).sum()
    secondary_cost = np.array(
        [secondary.cost_of(e) for e in true_energy[offloaded]]
    ).sum()
    total_cost = float(primary_cost + secondary_cost)

    primary_makespan = float(runtime[~offloaded].sum())
    makespan = max(primary_makespan, used_secondary)

    baseline_energy = float(true_energy.sum())  # all-primary energy
    saved = baseline_energy - total_energy
    return Outcome(
        strategy=strategy,
        n_offloaded=int(offloaded.sum()),
        total_energy_wh=total_energy,
        energy_saved_wh=saved,
        energy_saved_pct=100.0 * saved / baseline_energy if baseline_energy else 0.0,
        total_cost=total_cost,
        makespan_s=makespan,
        deadline_met=makespan <= deadline + 1e-6,
    )
