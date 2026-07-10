"""Expand (step 6): generate expanded RESOURCE_GROUPS + candidate model config.

Given a discovery proposal (e.g. a new 'gpu' group with delta_gpu_* features),
produce an expanded group map and a candidate config. This ONLY builds a config —
it does NOT touch the live approved model. The candidate is trained & evaluated
later (train_candidate.py, evaluate_candidate.py) and only promoted if it passes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from feature_moe.groups import RESOURCE_GROUPS, GROUP_ORDER


@dataclass
class CandidateSpec:
    """Describes a candidate model to be trained, without building it."""
    candidate_version: str
    kind: str  # "retrain" (same structure) | "expanded" (adds expert(s))
    groups: dict[str, list[str]]        # feature -> group map this candidate uses
    group_order: list[str]
    new_groups: list[str] = field(default_factory=list)   # only for "expanded"
    base_features: list[str] = field(default_factory=list)
    reason: str = ""
    base_model_version: str | None = None
    trigger_reason: str = ""
    trigger_evidence: dict[str, Any] = field(default_factory=dict)
    new_group_features: dict[str, list[str]] = field(default_factory=dict)
    gate_mode: str = "global"
    activation_threshold: float = 1e-6


def expanded_groups(proposals: list[dict]) -> tuple[dict[str, list[str]], list[str]]:
    """Merge current groups with proposed new ones. Returns (groups, order)."""
    groups = {g: list(feats) for g, feats in RESOURCE_GROUPS.items()}
    order = list(GROUP_ORDER)
    for prop in proposals:
        g = prop["group"]
        if g not in groups:
            groups[g] = list(prop["features"])
            order.append(g)
        else:
            for f in prop["features"]:
                if f not in groups[g]:
                    groups[g].append(f)
    return groups, order


def make_retrain_candidate(version: str, all_features: list[str], *,
                           base_model_version: str | None = None,
                           trigger_reason: str = "scheduled retrain on accumulated online labels",
                           trigger_evidence: dict[str, Any] | None = None) -> CandidateSpec:
    """Ordinary retrain: same 4-group structure, just on the latest buffer."""
    return CandidateSpec(
        candidate_version=version, kind="retrain",
        groups={g: list(RESOURCE_GROUPS[g]) for g in GROUP_ORDER},
        group_order=list(GROUP_ORDER),
        base_features=all_features,
        reason=trigger_reason,
        base_model_version=base_model_version,
        trigger_reason=trigger_reason,
        trigger_evidence=trigger_evidence or {},
        gate_mode="global",
        activation_threshold=1e-6,
    )


def make_expanded_candidate(version: str, proposals: list[dict], all_features: list[str], *,
                            base_model_version: str | None = None,
                            trigger_reason: str = "",
                            trigger_evidence: dict[str, Any] | None = None) -> CandidateSpec:
    """Expanded: current groups + one new expert per proposal."""
    groups, order = expanded_groups(proposals)
    new_groups = [p["group"] for p in proposals if p["group"] not in GROUP_ORDER]
    return CandidateSpec(
        candidate_version=version, kind="expanded",
        groups=groups, group_order=order, new_groups=new_groups,
        base_features=all_features,
        reason=trigger_reason or f"new resource dimension detected: {new_groups}",
        base_model_version=base_model_version,
        trigger_reason=trigger_reason or f"new resource dimension detected: {new_groups}",
        trigger_evidence=trigger_evidence or {},
        new_group_features={p["group"]: list(p["features"]) for p in proposals},
        gate_mode="conditional_new_groups",
        activation_threshold=1e-6,
    )
