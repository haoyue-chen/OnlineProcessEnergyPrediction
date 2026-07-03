"""Feature → resource-group mapping (architecture Module 4.2).

The 16 process-metric features are partitioned into 4 resource groups; each group
feeds one expert. This is the core of the *feature-based* (resource-grouped) MoE,
as opposed to the workload-grouped MoE in ``moe/``.

"System" features (context switches, syscall counts) with no single clean home are
attached to the CPU expert — CPU activity dominates these syscall/context-switch
counts in practice, and the architecture doc's example uses exactly 4 experts
(CPU/Memory/IO/Network). Documented here so the choice is visible.
"""

from __future__ import annotations

RESOURCE_GROUPS: dict[str, list[str]] = {
    "cpu": [
        "delta_cpu_ns",
        "delta_cycles",
        "delta_instructions",
        "delta_branch_instructions",
        "delta_cache_misses",
        # System features folded into CPU (see module docstring).
        "context_switches",
        "syscall_count",
        "syscall_class_process",
        "syscall_class_sched",
        "syscall_class_other",
    ],
    "memory": [
        "delta_rss_memory",
        "syscall_class_memory",
    ],
    "io": [
        "delta_io_bytes",
        "syscall_class_file",
    ],
    "network": [
        "delta_net_send_bytes",
        "syscall_class_network",
    ],
}

# Order is stable for indexing (cpu, memory, io, network).
GROUP_ORDER = ["cpu", "memory", "io", "network"]


def features_for(group: str) -> list[str]:
    return RESOURCE_GROUPS[group]


def assert_valid(all_features: list[str]) -> None:
    """Every feature must belong to exactly one group; every group non-empty."""
    seen = []
    for g in GROUP_ORDER:
        gfeats = RESOURCE_GROUPS[g]
        assert gfeats, f"empty group {g}"
        seen.extend(gfeats)
    assert sorted(seen) == sorted(all_features), (
        "group features do not match the dataset features exactly. "
        f"missing={set(all_features)-set(seen)} extra={set(seen)-set(all_features)}"
    )
