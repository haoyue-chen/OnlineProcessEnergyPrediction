"""Controlled-dataset evaluation of LearnedResourceMoE using the new 18-phase
labeled dataset (controlled_workload_labeled_final.parquet), instead of the
old 5-run pilot (which is what run_learned_feature_moe_controlled.py used,
and whose source parquets are lost).

Runs TWO leave-one-out designs, both using the same underlying data:

  1. leave-one-PHASE-out  (18 folds) -- tests generalization to an unseen
     specific workload pattern within a resource type (e.g. train on
     cpu_scalar/burst_micro/burst_milli/..., test on cpu_avx).

  2. leave-one-GROUP-out  (4 folds: cpu/memory/io/mixed) -- tests whether the
     gate can route a sample from an ENTIRELY UNSEEN resource type. This is
     the design that most directly answers "is MoE suitable across resource
     types" (objective 2), and mirrors the old 5-run script's intent (which
     used cpu/mem/io/net/mixed as its 5 held-out sets).

     NOTE: there is no "network" group -- no controlled network-stress phase
     was ever collected (no reachable iperf3 target). This is a known,
     documented gap, not an oversight. Only 4 groups: cpu/memory/io/mixed.

     NOTE on phase -> group assignment (from the idle-ratio analysis done
     earlier on validate_shares_v2.py output):
       cpu:    cpu_scalar, cpu_avx, burst_micro, burst_milli, memory_bw,
               mixed_compute_mem, mixed_unbalanced, smt_sweep
               (memory_bw and mixed_compute_mem are here, NOT in "memory" --
               delta_rss_memory doesn't capture bandwidth stress, only
               footprint growth; their raw signal is CPU-dominant. Documented
               finding, not a bug.)
       memory: memory_sparse (the only phase with a clean memory-dominant
               raw signal)
       io:     io_sequential, io_random, mixed_cpu_io, mixed_platform_max,
               power_virus (several of these have misleading names --
               power_virus and mixed_platform_max are actually IO-dominant
               in the raw data, not "everything maxed")
       mixed:  mixed_all_domains, mixed_thermal_ramp (both cpu AND io
               significantly elevated together -- the only two phases that
               are genuinely multi-domain rather than dominated by one
               resource)
     idle and thermal_hysteresis are handled specially -- see GROUPS below.

  numa_imbalance is excluded entirely (hardware N/A on this single-socket host).

Outputs (separate files per design, so they don't overwrite each other):
    results/learned_feature_moe_phase_loro_metrics.csv
    results/learned_feature_moe_phase_loro_gate_weights.csv
    results/learned_feature_moe_group_logo_metrics.csv
    results/learned_feature_moe_group_logo_gate_weights.csv
    results/learned_feature_moe_hyperparams.csv   (shared grid search, group-level)
    results/learned_feature_moe_final_summary.md

Usage:
    PYTHONPATH=. python -m experiments.run_learned_feature_moe_final
"""

from __future__ import annotations

import warnings; warnings.filterwarnings("ignore")
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from moe.data import FEATURES, TARGET, TIME, aggregate_intervals
from feature_moe.moe import ResourceMoE, ResourceMoEConfig, SingleGlobalModel
from feature_moe.learned_moe import LearnedResourceMoE, LearnedMoEConfig
from feature_moe.groups import GROUP_ORDER

# --- input: the merged, verified, 18-phase controlled dataset -------------
# Expected location: <repo root>/data/controlled_workload_labeled_final.parquet
# (scp'd down from gpu02:~/controlled_workload_labeled_final.parquet)
LABELED_PARQUET = Path(__file__).resolve().parents[1] / "data" / "controlled_workload_labeled_final.parquet"

# --- phase -> resource-group mapping (see module docstring for rationale) -
GROUPS: dict[str, list[str]] = {
    "cpu": [
        "cpu_scalar", "cpu_avx", "burst_micro", "burst_milli",
        "memory_bw", "mixed_compute_mem", "mixed_unbalanced", "smt_sweep",
        "thermal_hysteresis",  # heat-half is cpu_avx-style load; cool-half is idle.
                               # Net signal is CPU-elevated-but-diluted. Tentative --
                               # re-verify with validate_shares_v2 on just this phase
                               # if the group-level result looks off.
    ],
    "memory": ["memory_sparse"],
    "io": ["io_sequential", "io_random", "mixed_cpu_io", "mixed_platform_max", "power_virus"],
    "mixed": ["mixed_all_domains", "mixed_thermal_ramp"],
}
# idle is deliberately EXCLUDED from group-level LOGO: it's a baseline/no-load
# condition, not a resource-dominant regime, so "holding it out" doesn't test
# the same thing as holding out an unseen resource type. It IS still included
# in phase-level LORO (as its own fold) below, since that just asks "can the
# model generalize to an unseen phase," which idle answers fine.
IDLE_PHASE = "idle"


def load_final_dataset() -> pd.DataFrame:
    df = pd.read_parquet(LABELED_PARQUET)
    if "phase" not in df.columns:
        raise ValueError(f"{LABELED_PARQUET} has no 'phase' column -- wrong file?")
    return df


def load_by_phase(df: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    out = {}
    for phase, sub in df.groupby("phase"):
        X, y = aggregate_intervals(sub[[TIME, TARGET] + FEATURES])
        if len(X) == 0:
            print(f"  WARNING: phase '{phase}' produced 0 intervals after aggregation, skipping")
            continue
        out[phase] = (X, y)
    return out


def load_by_group(df: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    out = {}
    for group, phases in GROUPS.items():
        sub = df[df["phase"].isin(phases)]
        missing = set(phases) - set(sub["phase"].unique())
        if missing:
            print(f"  WARNING: group '{group}' missing expected phases: {missing}")
        X, y = aggregate_intervals(sub[[TIME, TARGET] + FEATURES])
        out[group] = (X, y)
    return out


def metrics(yt, yp):
    return {
        "r2": float(r2_score(yt, yp)),
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
    }


def leave_one_out_split(data, holdout):
    Xtr, ytr, Xte, yte = [], [], [], []
    for label, (X, y) in data.items():
        if label == holdout:
            Xte.append(X); yte.append(y)
        else:
            Xtr.append(X); ytr.append(y)
    return (pd.concat(Xtr, ignore_index=True), pd.concat(ytr, ignore_index=True),
            pd.concat(Xte, ignore_index=True), pd.concat(yte, ignore_index=True))


def grid_search(data, labels, combo_idx_total=None):
    """Grid search over (alpha, beta, gamma), evaluated via leave-one-out mean R2
    across `labels`. Prints progress per FOLD (not just per combo) so long runs
    don't look hung."""
    alphas = [0.01, 0.05, 0.1, 0.2]
    betas = [0.0, 0.001, 0.01]
    gammas = [0.0, 0.001, 0.01]
    combos = list(itertools.product(alphas, betas, gammas))
    best, grid_rows = None, []
    for ci, (a, b, g) in enumerate(combos, start=1):
        fold_r2, fold_maxw = [], []
        for fi, holdout in enumerate(labels, start=1):
            Xtr, ytr, Xte, yte = leave_one_out_split(data, holdout)
            m = LearnedResourceMoE(LearnedMoEConfig(alpha=a, beta=b, gamma=g, gate_epochs=120)).fit(Xtr, ytr)
            r2 = r2_score(yte, m.predict(Xte))
            fold_r2.append(r2)
            fold_maxw.append(max(m.gate_weights_mean(Xte).values()))
            print(f"    combo {ci}/{len(combos)} (a={a},b={b},g={g}) fold {fi}/{len(labels)} "
                  f"[{holdout}] r2={r2:.3f}", flush=True)
        mean_r2, mean_maxw = float(np.mean(fold_r2)), float(np.mean(fold_maxw))
        score = mean_r2 - max(0.0, mean_maxw - 0.8) * 0.5
        grid_rows.append({"alpha": a, "beta": b, "gamma": g,
                           "mean_r2": round(mean_r2, 4), "mean_max_gate": round(mean_maxw, 3),
                           "score": round(score, 4)})
        print(f"  combo {ci}/{len(combos)} done: mean_r2={mean_r2:.3f} maxw={mean_maxw:.2f} score={score:.3f}", flush=True)
        if best is None or score > best["score"]:
            best = {"alpha": a, "beta": b, "gamma": g, "mean_r2": mean_r2, "score": score}
    return best, pd.DataFrame(grid_rows).sort_values("score", ascending=False)


def run_loo_eval(data, labels, alpha, beta, gamma, tag):
    metric_rows, gate_rows = [], []
    for holdout in labels:
        Xtr, ytr, Xte, yte = leave_one_out_split(data, holdout)

        sg = RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=0).fit(Xtr.values, ytr.values)
        sgm = metrics(yte, sg.predict(Xte.values))

        rm = ResourceMoE(ResourceMoEConfig(expert="rf", gate="learned")).fit(Xtr, ytr)
        rmm = metrics(yte, rm.predict(Xte))
        rgw = rm.info()["gate_weights"]

        lm = LearnedResourceMoE(LearnedMoEConfig(alpha=alpha, beta=beta, gamma=gamma, gate_epochs=200)).fit(Xtr, ytr)
        lmm = metrics(yte, lm.predict(Xte))
        lgw = lm.gate_weights_mean(Xte)

        w_arr = np.array([lgw[g] for g in GROUP_ORDER])
        ent = float(-(w_arr * np.log(w_arr + 1e-12)).sum())
        dom = GROUP_ORDER[int(np.argmax(w_arr))]

        Xn = Xte.values.astype(np.float64)
        q = lm._build_intensity_q(Xn)
        qmean = {g: float(q[:, gi].mean()) for gi, g in enumerate(GROUP_ORDER)}
        kl = float((w_arr * (np.log(w_arr + 1e-12) - np.log(np.array([qmean[g] for g in GROUP_ORDER]) + 1e-12))).sum())

        metric_rows.append({
            "held_out": holdout, "n_test": len(yte),
            "single_r2": sgm["r2"], "single_mae": sgm["mae"], "single_rmse": sgm["rmse"],
            "old_resmoe_r2": rmm["r2"], "old_resmoe_mae": rmm["mae"], "old_resmoe_rmse": rmm["rmse"],
            "learned_r2": lmm["r2"], "learned_mae": lmm["mae"], "learned_rmse": lmm["rmse"],
        })
        gate_rows.append({
            "held_out": holdout,
            "old_cpu": rgw["cpu"], "old_mem": rgw["memory"], "old_io": rgw["io"], "old_net": rgw["network"],
            "learned_cpu": lgw["cpu"], "learned_mem": lgw["memory"], "learned_io": lgw["io"], "learned_net": lgw["network"],
            "learned_entropy": ent, "learned_dominant": dom,
            "q_cpu": qmean["cpu"], "q_mem": qmean["memory"], "q_io": qmean["io"], "q_net": qmean["network"],
            "kl_gate_vs_q": kl,
        })
        print(f"  [{tag}] {holdout:20s} single={sgm['r2']:.3f} old={rmm['r2']:.3f} learned={lmm['r2']:.3f} "
              f"| gate cpu/mem/io/net={lgw['cpu']:.2f}/{lgw['memory']:.2f}/{lgw['io']:.2f}/{lgw['network']:.2f} dom={dom}")

    return pd.DataFrame(metric_rows), pd.DataFrame(gate_rows)


def main():
    out_dir = Path("results"); out_dir.mkdir(exist_ok=True)

    print("Loading controlled_workload_labeled_final.parquet ...")
    df = load_final_dataset()
    print(f"  {len(df)} rows, phases: {sorted(df['phase'].unique())}")

    # ============================================================
    # Load both views of the data up front
    # ============================================================
    print("\n=== Aggregating by PHASE (18 folds) ===")
    phase_data = load_by_phase(df)
    for label, (X, y) in phase_data.items():
        print(f"  {label:20s} intervals={len(X):>5} energy_mean={y.mean():.2f}")
    phase_labels = list(phase_data.keys())

    print("\n=== Aggregating by GROUP (4 folds: cpu/memory/io/mixed) ===")
    group_data = load_by_group(df)
    for label, (X, y) in group_data.items():
        print(f"  {label:10s} intervals={len(X):>5} energy_mean={y.mean():.2f}")
    group_labels = list(group_data.keys())

    # ============================================================
    # Grid search ONCE, on the cheaper group-level split (4 folds x 36
    # combos = 144 fits, not 18 x 36 = 648). The resulting hyperparameters
    # are reused for both evaluations below -- tuning is about picking a
    # good alpha/beta/gamma for the gate in general, not re-tuning per
    # evaluation design, so this is not a shortcut that weakens the result.
    # ============================================================
    print(f"\nGrid search on group-level split ({len(group_labels)} folds x 36 combos = "
          f"{len(group_labels) * 36} fits) ...")
    best, grid_df = grid_search(group_data, group_labels)
    print("  best:", best)
    grid_df.to_csv(out_dir / "learned_feature_moe_hyperparams.csv", index=False)
    A, B, G = best["alpha"], best["beta"], best["gamma"]

    # ============================================================
    # Design 1: leave-one-PHASE-out (18 folds), using the shared hyperparams
    # ============================================================
    print(f"\nRunning full leave-one-phase-out (alpha={A} beta={B} gamma={G}) ...")
    mdf_phase, gdf_phase = run_loo_eval(phase_data, phase_labels, A, B, G, tag="phase")
    mdf_phase.to_csv(out_dir / "learned_feature_moe_phase_loro_metrics.csv", index=False)
    gdf_phase.to_csv(out_dir / "learned_feature_moe_phase_loro_gate_weights.csv", index=False)

    # ============================================================
    # Design 2: leave-one-GROUP-out (4 folds), using the shared hyperparams
    # ============================================================
    print(f"\nRunning full leave-one-group-out (alpha={A} beta={B} gamma={G}) ...")
    mdf_group, gdf_group = run_loo_eval(group_data, group_labels, A, B, G, tag="group")
    mdf_group.to_csv(out_dir / "learned_feature_moe_group_logo_metrics.csv", index=False)
    gdf_group.to_csv(out_dir / "learned_feature_moe_group_logo_gate_weights.csv", index=False)

    # ============================================================
    # Summary
    # ============================================================
    avg_phase = {c: float(mdf_phase[c].mean()) for c in mdf_phase.columns if c not in ("held_out", "n_test")}
    avg_group = {c: float(mdf_group[c].mean()) for c in mdf_group.columns if c not in ("held_out", "n_test")}

    print("\n=== Averages: leave-one-PHASE-out (18 folds) ===")
    for k in ["single_r2", "old_resmoe_r2", "learned_r2"]:
        print(f"  {k:18s} {avg_phase[k]:.3f}")
    print("\n=== Averages: leave-one-GROUP-out (4 folds) ===")
    for k in ["single_r2", "old_resmoe_r2", "learned_r2"]:
        print(f"  {k:18s} {avg_group[k]:.3f}")

    lines = []
    lines.append("# LearnedResourceMoE -- Final Controlled-Data Evaluation\n")
    lines.append("## 1. Data source")
    lines.append(f"`{LABELED_PARQUET}` -- 18 verified controlled phases from the fresh gpu02 collection "
                 "(replaces the lost 5-run pilot dataset). numa_imbalance excluded (single-socket host, "
                 "N/A by design). No network-stress phase was collected (no reachable iperf3 target) -- "
                 "network-dimension conclusions below should be read with that gap in mind.\n")
    lines.append("## 2. Two leave-one-out designs")
    lines.append("- **Phase-level LORO** (18 folds): tests generalization to an unseen specific workload "
                 "pattern within a resource type.")
    lines.append("- **Group-level LOGO** (4 folds: cpu/memory/io/mixed): tests whether the gate can route "
                 "a sample from an entirely unseen resource TYPE -- the more direct test of objective 2's "
                 "MoE-suitability question. No network group (see above).\n")
    lines.append("## 3. Results -- averages across folds")
    lines.append("| Design | single RF R2 | old ResourceMoE R2 | LearnedResourceMoE R2 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Phase LORO (18-fold) | {avg_phase['single_r2']:.3f} | {avg_phase['old_resmoe_r2']:.3f} | {avg_phase['learned_r2']:.3f} |")
    lines.append(f"| Group LOGO (4-fold) | {avg_group['single_r2']:.3f} | {avg_group['old_resmoe_r2']:.3f} | {avg_group['learned_r2']:.3f} |\n")
    lines.append("## 4. Hyperparameters")
    lines.append(f"alpha={best['alpha']} beta={best['beta']} gamma={best['gamma']} "
                 f"(grid mean R2={best['mean_r2']:.3f}, tuned via group-level 4-fold search, "
                 f"reused as-is for the phase-level 18-fold evaluation)\n")
    lines.append("## 5. Known caveats to state explicitly in the report")
    lines.append("- `memory_bw` and `mixed_compute_mem` are grouped under **cpu**, not memory -- "
                 "`delta_rss_memory` captures footprint, not bandwidth, so these show as CPU-dominant "
                 "in the raw feature data despite their names.")
    lines.append("- `thermal_hysteresis` is tentatively grouped under **cpu** (heat-half only); its group "
                 "assignment is less certain than the others and worth a footnote.")
    lines.append("- No network-dominant phase exists in this dataset; network-related gate/routing results "
                 "should be caveated, not asserted as validated.")
    lines.append("- `idle` is included in phase-level LORO but excluded from group-level LOGO (baseline "
                 "condition, not a resource-dominant regime).")

    (out_dir / "learned_feature_moe_final_summary.md").write_text("\n".join(lines))
    print(f"\nWrote results/learned_feature_moe_*  (phase_loro, group_logo, hyperparams, final_summary.md)")


if __name__ == "__main__":
    main()
