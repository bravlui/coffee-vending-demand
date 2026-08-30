"""Leakage-safe, bounded LightGBM tuning for rolling-origin forecasting."""

from __future__ import annotations

from itertools import product

import pandas as pd

from coffee_intel.config import Config
from coffee_intel.models.backtest import _fold_origins, run_backtest, summarise_backtest_cycle


def candidate_params(cfg: Config) -> list[dict]:
    tuning = cfg.forecasting.tuning
    grid = [
        {
            "objective": objective,
            "num_leaves": leaves,
            "learning_rate": rate,
            "min_child_samples": child,
        }
        for objective, leaves, rate, child in product(
            tuning.objectives,
            tuning.num_leaves,
            tuning.learning_rates,
            tuning.min_child_samples,
        )
    ]
    # Deterministic coverage of the grid; bounded for fast iteration on small data.
    if len(grid) <= tuning.max_candidates:
        return grid
    indexes = [
        round(i * (len(grid) - 1) / (tuning.max_candidates - 1))
        for i in range(tuning.max_candidates)
    ]
    return [grid[i] for i in indexes]


def temporal_split(
    feat: pd.DataFrame, cfg: Config
) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    origins = _fold_origins(feat["date"], cfg)
    holdout = cfg.forecasting.tuning.holdout_folds
    if holdout < 1 or holdout >= len(origins):
        raise ValueError("tuning.holdout_folds must be between 1 and n_folds - 1")
    return origins[:-holdout], origins[-holdout:]


def tune_forecaster(feat: pd.DataFrame, cfg: Config) -> tuple[dict, pd.DataFrame]:
    """Choose parameters only on development origins; never inspect holdout folds."""
    development_origins, _ = temporal_split(feat, cfg)
    rows: list[dict] = []
    for candidate_id, params in enumerate(candidate_params(cfg)):
        preds, _ = run_backtest(feat, cfg, development_origins, params)
        score = summarise_backtest_cycle(preds, ["ml_lightgbm"]).loc["ml_lightgbm"]
        rows.append({"candidate_id": candidate_id, **params, **score.to_dict()})
    results = pd.DataFrame(rows)
    results["abs_bias"] = results["bias"].abs()
    results = results.sort_values(["wape", "abs_bias"]).drop(columns="abs_bias")
    parameter_names = ["objective", "num_leaves", "learning_rate", "min_child_samples"]
    best = results.iloc[0][parameter_names].to_dict()
    best["num_leaves"] = int(best["num_leaves"])
    best["min_child_samples"] = int(best["min_child_samples"])
    return best, results
