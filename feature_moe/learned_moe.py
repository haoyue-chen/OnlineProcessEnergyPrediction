"""LearnedResourceMoE — per-sample soft-gate resource MoE.

Hybrid architecture that fixes the old ResourceMoE's two failures:
  1. global gate (one weight vector) -> collapses to CPU
  2. (in the all-torch version) MLP experts blow up on held-out workloads whose
     feature magnitudes are far outside train (net bytes=3.4e10 etc.)

Here experts are sklearn RandomForest (robust to extreme inputs, like the single
RF baseline that reaches R²=0.42 on leave-one-run-out), trained per resource group
on that group's features. The GATE is a torch MLP on standardized features that
outputs per-sample softmax weights. Gate training treats expert outputs as fixed
constants and learns weights via:
    loss = mse + alpha * KL(gate || q) + beta * balance + gamma * entropy
where q is a resource-intensity auxiliary target (log1p-scaled group mean, softmax).
This gives per-sample routing (mem/io/net-heavy intervals reach their experts)
without the instability of all-torch experts on tiny, extreme-valued data.

Prediction: y = Σ_g w_g(x) * expert_g(x_group_g),  w = softmax(gate(x))
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor

from .groups import GROUP_ORDER, RESOURCE_GROUPS


@dataclass
class LearnedMoEConfig:
    n_estimators: int = 150
    gate_hidden: int = 64
    gate_epochs: int = 200
    gate_lr: float = 1e-3
    gate_weight_decay: float = 1e-5
    alpha: float = 0.1            # KL(gate || q)
    beta: float = 0.01            # balance (penalize over-concentration)
    gamma: float = 0.001          # entropy regularization
    temperature: float = 1.0      # softmax temperature for q
    device: str = "cpu"
    seed: int = 0


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


class LearnedResourceMoE:
    """RF experts (per group) + per-sample torch soft gate."""

    def __init__(self, cfg: LearnedMoEConfig | None = None):
        self.cfg = cfg or LearnedMoEConfig()
        self._group_idx: dict[str, list[int]] = {}
        self.feature_names: list[str] = []
        self.experts: dict[str, RandomForestRegressor] = {}
        self.gate: _GateMLP | None = None
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._y_mean = 0.0
        self._y_std = 1.0

    def _standardize(self, X_np: np.ndarray) -> np.ndarray:
        Xs = (X_np - self._x_mean) / np.where(self._x_std > 0, self._x_std, 1.0)
        return np.clip(Xs.astype(np.float32), -5.0, 5.0)

    def _build_intensity_q(self, X_np: np.ndarray) -> np.ndarray:
        """Per-sample resource-intensity soft target q (n,4), sums to 1.

        Per group: intensity = MEAN of that group's log1p(|standardized|) features
        (mean, not sum — sum is biased toward groups with more/larger columns).
        q = softmax(intensity / temperature). Scaler fit on TRAIN only.
        """
        scaled = (X_np - self._x_mean) / np.where(self._x_std > 0, self._x_std, 1.0)
        scaled = np.log1p(np.abs(scaled)) * np.sign(scaled)
        intens = np.zeros((X_np.shape[0], len(GROUP_ORDER)))
        for gi, g in enumerate(GROUP_ORDER):
            idx = self._group_idx[g]
            intens[:, gi] = scaled[:, idx].mean(axis=1) if idx else 0.0
        intens = intens / max(self.cfg.temperature, 1e-6)
        intens = intens - intens.max(axis=1, keepdims=True)
        e = np.exp(intens)
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LearnedResourceMoE":
        cfg = self.cfg
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
        self.feature_names = list(X.columns)
        self._group_idx = {
            g: [X.columns.get_loc(c) for c in RESOURCE_GROUPS[g] if c in X.columns]
            for g in GROUP_ORDER
        }
        Xn = X.values.astype(np.float64)
        yn = y.values.astype(np.float64)
        self._x_mean = Xn.mean(axis=0)
        self._x_std = Xn.std(axis=0)
        self._y_mean = float(yn.mean()); self._y_std = float(yn.std()) or 1.0

        # 1) train one RF expert per group (on raw group features, robust to scale)
        self.experts = {}
        for g in GROUP_ORDER:
            idx = self._group_idx[g]
            rf = RandomForestRegressor(n_estimators=cfg.n_estimators, n_jobs=-1,
                                        random_state=cfg.seed).fit(Xn[:, idx], yn)
            self.experts[g] = rf

        # 2) expert predictions on train (constants for gate training)
        E = np.stack([self.experts[g].predict(Xn[:, self._group_idx[g]])
                      for g in GROUP_ORDER], axis=1).astype(np.float32)  # (n,4) in Wh

        # 3) build q and standardized X for the gate
        Xs = self._standardize(Xn)
        q = self._build_intensity_q(Xn).astype(np.float32)

        device = torch.device(cfg.device)
        Xt = torch.from_numpy(Xs).to(device)
        Et = torch.from_numpy(E).to(device)         # expert outputs (frozen)
        yt = torch.from_numpy(yn.astype(np.float32)).to(device)
        qt = torch.from_numpy(q).to(device)

        n_groups = len(GROUP_ORDER)
        gate = _GateMLP(Xs.shape[1], cfg.gate_hidden, n_groups).to(device)
        opt = torch.optim.Adam(gate.parameters(), lr=cfg.gate_lr, weight_decay=cfg.gate_weight_decay)
        n = Xt.shape[0]
        bs = 256
        for ep in range(cfg.gate_epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                xb, Eb, yb, qb = Xt[idx], Et[idx], yt[idx], qt[idx]
                w = gate(xb)                                   # (b,4)
                yhat = (w * Eb).sum(dim=1)                     # weighted fusion (Wh)
                mse = F.mse_loss(yhat, yb)
                kl = (w * (torch.log(w + 1e-8) - torch.log(qb + 1e-8))).sum(dim=1).mean()
                balance = w.max(dim=1).values.mean()
                ent = -(w * torch.log(w + 1e-8)).sum(dim=1).mean()
                loss = mse + cfg.alpha * kl + cfg.beta * balance - cfg.gamma * ent
                opt.zero_grad(); loss.backward(); opt.step()
        self.gate = gate
        self._device = device
        return self

    def predict_with_gate(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        Xn = X.values.astype(np.float64)
        E = np.stack([self.experts[g].predict(Xn[:, self._group_idx[g]])
                      for g in GROUP_ORDER], axis=1).astype(np.float32)  # (n,4) Wh
        Xs = self._standardize(Xn)
        with torch.no_grad():
            Xt = torch.from_numpy(Xs).to(self._device)
            w = self.gate(Xt).cpu().numpy()
        yhat = (w * E).sum(axis=1)
        # A held-out workload can make an out-of-domain expert predict wild values
        # (e.g. io routed to net expert -> -50 R²). Clip to a sane band around the
        # training target so one bad expert can't blow up the whole prediction.
        lo = self._y_mean - 6 * self._y_std
        hi = self._y_mean + 6 * self._y_std
        yhat = np.clip(yhat, lo, hi)
        return yhat, w

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_with_gate(X)[0]

    def gate_weights_mean(self, X: pd.DataFrame) -> dict[str, float]:
        w = self.predict_with_gate(X)[1]
        return {g: float(w[:, gi].mean()) for gi, g in enumerate(GROUP_ORDER)}
