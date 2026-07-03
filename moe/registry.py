"""Model registry — batch (sklearn) and online (river) experts/baselines.

Centralises every model class the project can use, so the MoE experts, the single
baseline, and the comparison scripts all draw from one place. Adding a new model
is a one-line entry here.

Two registries:
  * ``BATCH_MODELS``  — sklearn regressors (offline fit/predict). Used as MoE
    experts, as the single global baseline, and in the batch comparison.
  * ``ONLINE_MODELS`` — river regressors (incremental learn_one/predict_one). Used
    as online MoE experts and in the online comparison.

Model families covered (one representative each, to compare *families* not tune
within one):
  batch : linear, random forest (bagged trees), extra-trees (bagged trees),
          hist-gradient-boosting (boosted trees; the LightGBM stand-in), kNN, MLP, SVR.
  online: linear (Adam), passive-aggressive, Hoeffding tree, Hoeffding adaptive
          tree, adaptive random forest (ARF), kNN.
"""

from __future__ import annotations

from typing import Callable

from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


# --- Batch (sklearn) -------------------------------------------------------

def _linear(cfg):
    return make_pipeline(StandardScaler(), LinearRegression())


def _rf(cfg):
    return RandomForestRegressor(
        n_estimators=cfg.n_estimators, max_depth=cfg.max_depth,
        random_state=cfg.random_state, n_jobs=cfg.n_jobs,
    )


def _extra_trees(cfg):
    return ExtraTreesRegressor(
        n_estimators=cfg.n_estimators, max_depth=cfg.max_depth,
        random_state=cfg.random_state, n_jobs=cfg.n_jobs,
    )


def _hgb(cfg):
    # Histogram gradient boosting — sklearn's native LightGBM-equivalent (boosted
    # trees). Stands in for LightGBM, which is not installed (Phase-1 Finding 4:
    # RF ~= LightGBM, and HGB is the same boosted-tree family).
    return HistGradientBoostingRegressor(random_state=cfg.random_state)


def _knn(cfg):
    return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10))


def _mlp(cfg):
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400,
                     random_state=cfg.random_state),
    )


def _svr(cfg):
    return make_pipeline(StandardScaler(), SVR(C=10.0))


BATCH_MODELS: dict[str, Callable] = {
    "linear": _linear,
    "rf": _rf,
    "extra_trees": _extra_trees,
    "hgb": _hgb,
    "knn": _knn,
    "mlp": _mlp,
    "svr": _svr,
}


def make_batch_model(name: str, cfg):
    if name not in BATCH_MODELS:
        raise ValueError(f"Unknown batch model '{name}'. Options: {list(BATCH_MODELS)}")
    return BATCH_MODELS[name](cfg)


# --- Online (river) --------------------------------------------------------

def _make_online(name: str):
    from river import forest as river_forest
    from river import linear_model as river_linear
    from river import neighbors as river_neighbors
    from river import optim as river_optim
    from river import preprocessing as river_pre
    from river import tree as river_tree

    if name == "linear":
        # Adam, not plain SGD: unscaled ~300 Wh targets make default SGD diverge.
        return river_pre.StandardScaler() | river_linear.LinearRegression(
            optimizer=river_optim.Adam()
        )
    if name == "pa":
        return river_pre.StandardScaler() | river_linear.PARegressor()
    if name == "htr":
        return river_tree.HoeffdingTreeRegressor(leaf_prediction="mean")
    if name == "hatr":
        # Adaptive variant: detects drift and replaces stale branches.
        return river_tree.HoeffdingAdaptiveTreeRegressor(seed=0, leaf_prediction="mean")
    if name == "arf":
        # Adaptive Random Forest — explicitly named in the Phase-2 plan (Task 2).
        return river_pre.StandardScaler() | river_forest.ARFRegressor(n_models=10, seed=0)
    if name == "knn":
        return river_pre.StandardScaler() | river_neighbors.KNNRegressor()
    raise ValueError(f"Unknown online model '{name}'. Options: {ONLINE_MODEL_NAMES}")


ONLINE_MODEL_NAMES = ["linear", "pa", "htr", "hatr", "arf", "knn"]


# Online-learning capability — which model *families* can update incrementally
# (predict → receive truth → update, without retraining on all past data) vs which
# are inherently batch/offline. This is the distinction that matters under workload
# drift (DAW1 → DAW2 → Phoronix → Stress): only online-capable models adapt as the
# stream's distribution shifts. The batch registry holds the offline families
# (used as MoE experts and for the offline accuracy survey); the online registry
# holds river models that genuinely support learn_one/predict_one.
#
# Note the same *family* can be either, depending on implementation: sklearn's
# batch OLS "linear" is offline, but river's SGD-style "linear" updates online.
#
# family         online?  representative here
# -------------  -------  -----------------------------------------
# linear (OLS)   no       batch "linear"
# SGD-linear     yes      online "linear" (river LinearRegression + Adam)
# random forest  no       batch "rf"
# LightGBM/boost no       batch "hgb" (HistGradientBoosting stand-in)
# Hoeffding tree yes      online "htr" / "hatr"
# Adaptive RF    yes      online "arf"
#
# By construction every batch-registry model is offline and every online-registry
# model is online; the helper below answers it unambiguously given the registry.

def is_online_capable(name: str, kind: str) -> bool:
    """Whether a model supports incremental (streaming) learning.

    ``kind`` must be "batch" or "online" to disambiguate names shared by both
    registries (e.g. linear, knn). Batch-registry models are offline by
    construction; online-registry models are online by construction.
    """
    if kind == "online":
        if name not in ONLINE_MODEL_NAMES:
            raise ValueError(f"Unknown online model '{name}'")
        return True
    if kind == "batch":
        if name not in BATCH_MODELS:
            raise ValueError(f"Unknown batch model '{name}'")
        return False
    raise ValueError("kind must be 'batch' or 'online'")


def make_online_model(name: str):
    return _make_online(name)
