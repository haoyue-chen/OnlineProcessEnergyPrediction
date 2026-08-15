# LearnedResourceMoE -- Final Controlled-Data Evaluation

## 1. Data source
`/home/ting/OnlineProcessEnergyPrediction/data/controlled_workload_labeled_final.parquet` -- 18 verified controlled phases from the fresh gpu02 collection (replaces the lost 5-run pilot dataset). numa_imbalance excluded (single-socket host, N/A by design). No network-stress phase was collected (no reachable iperf3 target) -- network-dimension conclusions below should be read with that gap in mind.

## 2. Two leave-one-out designs
- **Phase-level LORO** (18 folds): tests generalization to an unseen specific workload pattern within a resource type.
- **Group-level LOGO** (4 folds: cpu/memory/io/mixed): tests whether the gate can route a sample from an entirely unseen resource TYPE -- the more direct test of objective 2's MoE-suitability question. No network group (see above).

## 3. Results -- averages across folds
| Design | single RF R2 | old ResourceMoE R2 | LearnedResourceMoE R2 |
|---|---|---|---|
| Phase LORO (18-fold) | -1.172 | -1.088 | -1.178 |
| Group LOGO (4-fold) | -0.844 | -1.398 | -2.057 |

## 4. Hyperparameters
alpha=0.01 beta=0.0 gamma=0.01 (grid mean R2=-1.897, tuned via group-level 4-fold search, reused as-is for the phase-level 18-fold evaluation)

## 5. Known caveats to state explicitly in the report
- `memory_bw` and `mixed_compute_mem` are grouped under **cpu**, not memory -- `delta_rss_memory` captures footprint, not bandwidth, so these show as CPU-dominant in the raw feature data despite their names.
- `thermal_hysteresis` is tentatively grouped under **cpu** (heat-half only); its group assignment is less certain than the others and worth a footnote.
- No network-dominant phase exists in this dataset; network-related gate/routing results should be caveated, not asserted as validated.
- `idle` is included in phase-level LORO but excluded from group-level LOGO (baseline condition, not a resource-dominant regime).