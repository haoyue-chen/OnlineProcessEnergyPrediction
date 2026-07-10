"""Train candidate (step 7): train a candidate model from a CandidateSpec.

Two candidate kinds:
  * "retrain"  — same structure as the approved model, retrained on the buffer.
  * "expanded" — adds new-group expert(s). Uses ResourceMoE with an expanded
                 group map (a temporary override of feature_moe.groups).

Models here are candidate-local and never change the live approved model until
promotion. Retrain candidates can keep the simple global NNLS gate; expanded
candidates can additionally use conditional or learned per-sample routing.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import nnls
from sklearn.ensemble import RandomForestRegressor

from .expand import CandidateSpec

PENDING_DIR = Path("models/pending")
LEGACY_GROUPS = ["cpu", "memory", "io", "network"]


class _GateMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_groups: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_groups),
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)


def _train_experts(X, y, groups, group_order, n_estimators=120, seed=0):
    experts = {}
    for g in group_order:
        cols = [c for c in groups[g] if c in X.columns]
        if not cols:
            experts[g] = None
            continue
        rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=seed).fit(X[cols].values, y.values)
        experts[g] = (rf, cols)
    return experts


def _expert_matrix(X, experts, group_order):
    cols = []
    valid_groups = []
    for g in group_order:
        e = experts[g]
        if e is None:
            continue
        rf, feat_cols = e
        cols.append(rf.predict(X[feat_cols].values))
        valid_groups.append(g)
    return np.column_stack(cols) if cols else np.zeros((len(X), 0)), valid_groups


def _learn_gate(E, y):
    if E.shape[1] == 0:
        return np.zeros(0)
    w, _ = nnls(E, y.values if hasattr(y, "values") else y)
    s = w.sum()
    return w / s if s > 1e-9 else np.ones(E.shape[1]) / E.shape[1]


def _activation_mask(X: pd.DataFrame, feature_cols: list[str], threshold: float) -> np.ndarray:
    cols = [c for c in feature_cols if c in X.columns]
    if not cols:
        return np.zeros(len(X), dtype=bool)
    return (np.abs(X[cols].values) > threshold).any(axis=1)


def _standardize(X_np: np.ndarray):
    mean = X_np.mean(axis=0)
    std = X_np.std(axis=0)
    Xs = (X_np - mean) / np.where(std > 0, std, 1.0)
    return np.clip(Xs.astype(np.float32), -5.0, 5.0), mean, std


def _build_intensity_q(X: pd.DataFrame, groups: dict[str, list[str]], group_order: list[str], feature_names: list[str], mean: np.ndarray, std: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    Xn = X[feature_names].values.astype(np.float64)
    scaled = (Xn - mean) / np.where(std > 0, std, 1.0)
    scaled = np.log1p(np.abs(scaled)) * np.sign(scaled)
    intens = np.zeros((len(X), len(group_order)), dtype=np.float64)
    for gi, g in enumerate(group_order):
        cols = [c for c in groups[g] if c in feature_names]
        if not cols:
            continue
        idx = [feature_names.index(c) for c in cols]
        intens[:, gi] = scaled[:, idx].mean(axis=1)
    intens = intens / max(temperature, 1e-6)
    intens = intens - intens.max(axis=1, keepdims=True)
    e = np.exp(intens)
    return e / e.sum(axis=1, keepdims=True)


def _train_per_sample_router(X: pd.DataFrame, y: pd.Series, experts, groups: dict[str, list[str]], group_order: list[str], *, gate_hidden: int = 64, gate_epochs: int = 160, gate_lr: float = 1e-3, alpha: float = 0.05, beta: float = 0.01, gamma: float = 0.001, seed: int = 0) -> dict:
    X_for_gate = X.copy()
    feature_names = list(X_for_gate.columns)
    E, valid_groups = _expert_matrix(X_for_gate, experts, group_order)
    Xs, mean, std = _standardize(X_for_gate[feature_names].values.astype(np.float64))
    q = _build_intensity_q(X_for_gate, groups, valid_groups, feature_names, mean, std).astype(np.float32)

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    gate = _GateMLP(Xs.shape[1], gate_hidden, len(valid_groups)).to(device)
    opt = torch.optim.Adam(gate.parameters(), lr=gate_lr, weight_decay=1e-5)

    Xt = torch.from_numpy(Xs).to(device)
    Et = torch.from_numpy(E.astype(np.float32)).to(device)
    yt = torch.from_numpy(y.values.astype(np.float32)).to(device)
    qt = torch.from_numpy(q).to(device)

    n = Xt.shape[0]
    bs = 256
    for _ in range(gate_epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, Eb, yb, qb = Xt[idx], Et[idx], yt[idx], qt[idx]
            w = gate(xb)
            yhat = (w * Eb).sum(dim=1)
            mse = F.mse_loss(yhat, yb)
            kl = (w * (torch.log(w + 1e-8) - torch.log(qb + 1e-8))).sum(dim=1).mean()
            balance = w.max(dim=1).values.mean()
            ent = -(w * torch.log(w + 1e-8)).sum(dim=1).mean()
            loss = mse + alpha * kl + beta * balance - gamma * ent
            opt.zero_grad(); loss.backward(); opt.step()

    return {
        "router": gate,
        "feature_names": feature_names,
        "group_order": valid_groups,
        "x_mean": mean,
        "x_std": std,
    }


class CandidateModel:
    """A trained candidate with candidate-local gate behavior."""

    def __init__(self, spec: CandidateSpec, experts, feature_names, *,
                 group_order=None, gate_weights=None,
                 inactive_gate=None, active_gate=None,
                 router_bundle=None):
        self.spec = spec
        self.experts = experts
        self.feature_names = feature_names
        self.group_order = group_order or []
        self.gate_weights = gate_weights
        self.inactive_gate = inactive_gate
        self.active_gate = active_gate
        self.router_bundle = router_bundle

    def _predict_with_gate(self, X: pd.DataFrame, gate: dict) -> np.ndarray:
        groups = gate["groups"]
        weights = np.asarray(gate["weights"], dtype=float)
        E, valid = _expert_matrix(X, self.experts, groups)
        if E.shape[1] == 0:
            return np.full(len(X), float("nan"))
        if valid != groups:
            aligned = []
            for g in valid:
                idx = groups.index(g)
                aligned.append(weights[idx])
            weights = np.asarray(aligned, dtype=float)
        s = weights.sum()
        if s > 1e-9:
            weights = weights / s
        else:
            weights = np.ones(len(valid), dtype=float) / max(1, len(valid))
        return E @ weights

    def _router_weights(self, X: pd.DataFrame) -> np.ndarray:
        bundle = self.router_bundle
        Xn = X[bundle["feature_names"]].values.astype(np.float64)
        Xs = (Xn - bundle["x_mean"]) / np.where(bundle["x_std"] > 0, bundle["x_std"], 1.0)
        Xs = np.clip(Xs.astype(np.float32), -5.0, 5.0)
        with torch.no_grad():
            Xt = torch.from_numpy(Xs)
            w = bundle["router"](Xt).cpu().numpy()
        if self.spec.kind == "expanded" and "gpu" in self.spec.new_groups:
            gpu_cols = self.spec.new_group_features.get("gpu", [])
            gpu_active = _activation_mask(X, gpu_cols, self.spec.activation_threshold)
            if "gpu" in bundle["group_order"]:
                gi = bundle["group_order"].index("gpu")
                w[~gpu_active, gi] = 0.0
                sums = w.sum(axis=1, keepdims=True)
                sums = np.where(sums > 1e-9, sums, 1.0)
                w = w / sums
        return w

    def predict_with_gate(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
        if self.spec.gate_mode == "conditional_new_groups" and self.active_gate and self.inactive_gate:
            gpu_cols = self.spec.new_group_features.get("gpu", [])
            gpu_active = _activation_mask(X, gpu_cols, self.spec.activation_threshold)
            pred = np.empty(len(X), dtype=float)
            gate_w = np.zeros((len(X), len(self.spec.group_order)), dtype=float)
            if (~gpu_active).any():
                pred[~gpu_active] = self._predict_with_gate(X.loc[~gpu_active], self.inactive_gate)
                inactive_map = {g: w for g, w in zip(self.inactive_gate["groups"], self.inactive_gate["weights"])}
                for gi, g in enumerate(self.spec.group_order):
                    gate_w[~gpu_active, gi] = inactive_map.get(g, 0.0)
            if gpu_active.any():
                pred[gpu_active] = self._predict_with_gate(X.loc[gpu_active], self.active_gate)
                active_map = {g: w for g, w in zip(self.active_gate["groups"], self.active_gate["weights"])}
                for gi, g in enumerate(self.spec.group_order):
                    gate_w[gpu_active, gi] = active_map.get(g, 0.0)
            return pred, gate_w, list(self.spec.group_order)

        if self.spec.gate_mode == "learned_per_sample" and self.router_bundle is not None:
            weights = self._router_weights(X)
            groups = list(self.router_bundle["group_order"])
            E, valid = _expert_matrix(X, self.experts, groups)
            if E.shape[1] == 0:
                return np.full(len(X), float("nan")), np.zeros((len(X), 0)), groups
            if valid != groups:
                idx = [groups.index(g) for g in valid]
                weights = weights[:, idx]
                groups = valid
            yhat = (weights * E).sum(axis=1)
            return yhat, weights, groups

        E, _ = _expert_matrix(X, self.experts, self.group_order)
        if E.shape[1] == 0:
            return np.full(len(X), float("nan")), np.zeros((len(X), 0)), list(self.group_order)
        yhat = E @ self.gate_weights
        gate_w = np.tile(self.gate_weights, (len(X), 1))
        return yhat, gate_w, list(self.group_order)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_with_gate(X)[0]

    def gate_weights_per_sample(self, X: pd.DataFrame) -> list[dict]:
        _, weights, groups = self.predict_with_gate(X)
        return [{g: float(row[i]) for i, g in enumerate(groups)} for row in weights]

    def gate_weights_per_group(self, X: pd.DataFrame | None = None) -> dict:
        if self.spec.gate_mode == "conditional_new_groups":
            active = {g: 0.0 for g in self.spec.group_order}
            inactive = {g: 0.0 for g in self.spec.group_order}
            if self.active_gate:
                for g, w in zip(self.active_gate["groups"], self.active_gate["weights"]):
                    active[g] = float(w)
            if self.inactive_gate:
                for g, w in zip(self.inactive_gate["groups"], self.inactive_gate["weights"]):
                    inactive[g] = float(w)
            return {"overall": active, "active": active, "inactive": inactive}

        if self.spec.gate_mode == "learned_per_sample":
            if X is None:
                raise ValueError("gate_weights_per_group(X) requires X for learned_per_sample candidates")
            _, weights, groups = self.predict_with_gate(X)
            overall = {g: float(weights[:, gi].mean()) for gi, g in enumerate(groups)}
            result = {"overall": overall}
            if "gpu" in groups:
                gpu_cols = self.spec.new_group_features.get("gpu", [])
                gpu_active = _activation_mask(X, gpu_cols, self.spec.activation_threshold)
                gi = groups.index("gpu")
                result["active"] = {"gpu": float(weights[gpu_active, gi].mean()) if gpu_active.any() else 0.0}
                result["inactive"] = {"gpu": float(weights[~gpu_active, gi].mean()) if (~gpu_active).any() else 0.0}
            return result

        out = {}
        gi = 0
        for g in self.group_order:
            if self.experts[g] is not None:
                out[g] = float(self.gate_weights[gi])
                gi += 1
            else:
                out[g] = 0.0
        return out


def train_candidate(spec: CandidateSpec, X: pd.DataFrame, y: pd.Series) -> CandidateModel:
    """Train experts + gate per the spec. Returns a CandidateModel."""
    experts = _train_experts(X, y, spec.groups, spec.group_order)

    if spec.gate_mode == "learned_per_sample":
        router_bundle = _train_per_sample_router(X, y, experts, spec.groups, spec.group_order)
        return CandidateModel(spec, experts, list(X.columns), router_bundle=router_bundle)

    if spec.gate_mode == "conditional_new_groups" and "gpu" in spec.new_groups:
        gpu_cols = spec.new_group_features.get("gpu", [])
        gpu_active = _activation_mask(X, gpu_cols, spec.activation_threshold)

        inactive_groups = [g for g in LEGACY_GROUPS if experts.get(g) is not None]
        active_groups = [g for g in spec.group_order if experts.get(g) is not None]

        if (~gpu_active).any():
            E_inactive, valid_inactive = _expert_matrix(X.loc[~gpu_active], experts, inactive_groups)
            w_inactive = _learn_gate(E_inactive, y.loc[~gpu_active])
        else:
            E_inactive, valid_inactive = _expert_matrix(X, experts, inactive_groups)
            w_inactive = _learn_gate(E_inactive, y)

        if gpu_active.any():
            E_active, valid_active = _expert_matrix(X.loc[gpu_active], experts, active_groups)
            w_active = _learn_gate(E_active, y.loc[gpu_active])
        else:
            E_active, valid_active = _expert_matrix(X, experts, active_groups)
            w_active = _learn_gate(E_active, y)

        return CandidateModel(
            spec,
            experts,
            list(X.columns),
            inactive_gate={"groups": valid_inactive, "weights": w_inactive.tolist()},
            active_gate={"groups": valid_active, "weights": w_active.tolist()},
        )

    E, valid = _expert_matrix(X, experts, spec.group_order)
    gate = _learn_gate(E, y)
    return CandidateModel(spec, experts, list(X.columns), group_order=valid, gate_weights=gate)


def save_candidate(model: CandidateModel, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(model, fh)
    (path.parent / "spec.json").write_text(json.dumps(model.spec.__dict__, indent=2))
