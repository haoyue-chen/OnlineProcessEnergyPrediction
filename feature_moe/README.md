# Feature-based (Resource-grouped) MoE — `feature_moe/`

Implements the **new architecture** from `pipeline_design_architecture.md` /
`AI_Development_Guide.md` (Modules 4–5): experts grouped **by resource type**
(CPU / Memory / I/O / Network) with a **learned soft gate** (weighted fusion), as
opposed to the workload-grouped MoE in `moe/` (experts = DAW1/DAW2/Phoronix/Stress,
hard gate). The two coexist for comparison.

## Architecture (Module 5)

```text
features ──► group by resource ──► 4 experts (each on its group's columns)
                                          │  CPU exp   Mem exp   IO exp   Net exp
                                          ▼
                              learned gate (non-neg least-squares weights, sum=1)
                                          ▼
                          predicted energy = Σ gate_weight × expert_pred   (weighted fusion)
```

- **Experts**: one regressor per resource group, trained *only* on that group's
  features. Pluggable via `moe.registry` (rf / linear / hgb / ...).
- **Gate** (learned, the team's answer to doc open-question #2): non-negative
  least-squares weights over the expert-prediction matrix (stacking), normalized
  to sum 1. Data-driven soft routing, not hand-set. An importance-weighted gate
  (`gate="importance"`) is included as an ablation.

## Feature grouping (Module 4.2) — `groups.py`

| Group | Features |
|---|---|
| cpu | delta_cpu_ns, delta_cycles, delta_instructions, delta_branch_instructions, delta_cache_misses + system (context_switches, syscall_count, syscall_class_process/sched/other) |
| memory | delta_rss_memory, syscall_class_memory |
| io | delta_io_bytes, syscall_class_file |
| network | delta_net_send_bytes, syscall_class_network |

"System" syscall/context-switch features are folded into CPU (CPU activity
dominates them); the doc's example uses exactly 4 experts.

## Run

```sh
cd energy-offloading
PYTHONPATH=. python -m feature_moe.run_resource_moe --expert rf --plot results/feature_moe/cmp_rf.png
PYTHONPATH=. python -m feature_moe.run_resource_moe --expert linear
```

5-fold paired CV comparing: single global model, workload-grouped MoE (existing),
resource-MoE (learned gate), resource-MoE (importance gate). Prints the resource-
importance table (Module 4.3), the per-model R²/MAE, and the learned gate weights.
Outputs: `results/feature_moe/resource_moe_comparison.{csv,txt}` + plot.

## Results (full data, 5-fold CV)

**Resource importance (Module 4.3):** CPU ~93–99%, Memory ~0%, I/O ~1%,
Network ~2% — CPU dominates these workloads, consistent across methods.

| Model | RF experts R² | Linear experts R² |
|---|---|---|
| single global | 0.964 | 0.792 |
| workload-grouped MoE | 0.972 | 0.843 |
| resource-MoE (learned gate) | 0.960 | 0.738 |
| resource-MoE (importance gate) | 0.960 | 0.738 |

**Honest findings:**
- The learned gate collapses to CPU≈0.97–0.99 because CPU dominates the
  permutation importance — Memory/IO/Network experts get ~0 weight. So on *these
  workloads* the resource split mostly reduces to "CPU expert + a little IO/Net",
  and the weighted fusion ≈ the CPU expert alone.
- Under **RF** experts, resource-MoE (0.960) ≈ single global (0.964) — essentially
  tied; a strong learner already captures the structure.
- Under **linear** experts, resource-MoE (0.738) is **below** single global (0.792)
  and well below workload-grouped MoE (0.843). The resource grouping does *not* help
  here: each per-group linear expert sees fewer features and the CPU-dominated gate
  can't recover what the global linear model captures.

**What this means for the architecture:** the feature-based resource MoE is sound
and matches the design, but on this CPU-dominated dataset its benefit is limited —
the workload-grouped MoE remains the stronger design here. The resource-based
design's value would show on workloads where Memory/IO/Network genuinely share
prediction (more balanced resource mixes). The learned-vs-importance gates are
nearly identical (importance is static; learned folds to the same CPU-dominated
weights).

> Note: an earlier draft reported resource-MoE "winning" under linear experts —
> that was an artifact of an 8k subsample cap (overfits the small sample). The
> numbers above are on full data and are the honest ones. The runner now defaults
> to full data.

## Files

| File | Role |
|---|---|
| `groups.py` | feature → resource-group mapping (Module 4.2) |
| `importance.py` | permutation-importance → resource shares (Module 4.3, ablation gate) |
| `moe.py` | `ResourceMoE` (4 experts + learned gate) + `SingleGlobalModel` baseline |
| `run_resource_moe.py` | 5-fold CV evaluation + comparison |

## Honest scope

This delivers the new architecture's **core** (Modules 4–5) as a standalone,
evaluated package. Modules 1–3 = the existing `work/` data (upstream pipeline).
Wiring the resource-MoE into `serve-online` / offloading / the online loop
(Modules 6–7) is a follow-up once the core is validated — the workload-MoE already
has those. No Docker/entrypoint changes; runnable directly with `PYTHONPATH=.`
from the project root.
