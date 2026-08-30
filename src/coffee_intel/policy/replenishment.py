"""Turn a demand forecast into a concrete refill recommendation.

This is a deliberately transparent base-stock (order-up-to) policy:

    target_stock = expected_demand(cycle + lead_time)
                   + z(service_level) * sigma_demand * sqrt(cycle + lead_time)
    order_qty    = max(0, target_stock - on_hand)

`sigma_demand` is taken from the backtest error of the chosen model, so the
safety buffer reflects how wrong the forecast has actually been, not a guess.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from coffee_intel.config import Config


def _z(service_level: float) -> float:
    """Inverse standard-normal CDF (stdlib, no scipy needed)."""
    return NormalDist().inv_cdf(service_level)


def daily_error_std(backtest_preds: pd.DataFrame, model_col: str) -> pd.Series:
    """Per-product std of the daily forecast error from the backtest."""
    err = backtest_preds[model_col] - backtest_preds["y_true"]
    return err.groupby(backtest_preds["product"], observed=True).std().rename("error_std")


def empirical_cycle_buffer(
    backtest_preds: pd.DataFrame, model_col: str, service_level: float
) -> pd.Series:
    """Per-product quantile of positive cycle under-forecast errors."""
    cycle = backtest_preds.groupby(["fold", "product"], observed=True)[["y_true", model_col]].sum()
    underforecast = (cycle["y_true"] - cycle[model_col]).clip(lower=0.0)
    return (
        underforecast.groupby("product", observed=True)
        .quantile(service_level)
        .rename("empirical_buffer")
    )


def build_recommendation(
    forecast: pd.DataFrame,
    error_std: pd.Series,
    cfg: Config,
    on_hand: pd.Series | None = None,
    empirical_buffer: pd.Series | None = None,
) -> pd.DataFrame:
    """`forecast` has columns date, product, forecast_units for the next cycle."""
    cycle = cfg.forecasting.horizon_days
    protection = cycle + cfg.replenishment.lead_time_days
    z = _z(cfg.replenishment.service_level)

    agg = (
        forecast.groupby("product", observed=True)["forecast_units"]
        .sum()
        .rename("expected_demand_cycle")
        .to_frame()
    )
    agg["daily_error_std"] = error_std.reindex(agg.index).fillna(error_std.mean())
    # Scale the cycle demand to the full protection interval.
    agg["expected_demand_protection"] = agg["expected_demand_cycle"] * (protection / cycle)
    normal_buffer = z * agg["daily_error_std"] * np.sqrt(protection)
    if cfg.replenishment.safety_stock_method == "empirical" and empirical_buffer is not None:
        fallback = float(empirical_buffer.mean()) if not empirical_buffer.empty else 0.0
        agg["safety_stock"] = empirical_buffer.reindex(agg.index).fillna(fallback)
    else:
        agg["safety_stock"] = normal_buffer
    agg["target_stock"] = np.ceil(agg["expected_demand_protection"] + agg["safety_stock"])

    on_hand = (
        on_hand.reindex(agg.index).fillna(0.0)
        if on_hand is not None
        else pd.Series(0.0, index=agg.index)
    )
    agg["on_hand"] = on_hand
    agg["order_qty"] = np.maximum(0.0, agg["target_stock"] - agg["on_hand"])
    agg["service_level"] = cfg.replenishment.service_level

    return agg.round(2).reset_index().sort_values("expected_demand_cycle", ascending=False)


def backtest_service_levels(
    predictions: pd.DataFrame,
    model_col: str,
    service_levels: tuple[float, ...] = (0.90, 0.95, 0.99),
) -> pd.DataFrame:
    """Walk-forward evaluation of nominal service levels.

    Each fold is calibrated only with errors from earlier folds. Results are
    diagnostic: without inventory/stockout telemetry they are not realised KPIs.
    """
    rows: list[dict[str, float | int | str]] = []
    folds = sorted(predictions["fold"].unique())
    for fold in folds[1:]:
        prior = predictions[predictions["fold"] < fold]
        current = predictions[predictions["fold"] == fold]
        sigma = daily_error_std(prior, model_col)
        cycle = current.groupby("product", observed=True)[["y_true", model_col]].sum()
        fallback = float(sigma.mean()) if not sigma.empty else 0.0
        cycle["sigma"] = sigma.reindex(cycle.index).fillna(fallback)
        horizon = int(current["date"].nunique())
        for service_level in service_levels:
            normal = _z(service_level) * cycle["sigma"] * np.sqrt(horizon)
            empirical = empirical_cycle_buffer(prior, model_col, service_level)
            empirical_fallback = float(empirical.mean()) if not empirical.empty else 0.0
            empirical = empirical.reindex(cycle.index).fillna(empirical_fallback)
            for method, buffer in (("normal", normal), ("empirical", empirical)):
                target = cycle[model_col] + buffer
                shortage = (cycle["y_true"] - target).clip(lower=0.0)
                excess = (target - cycle["y_true"]).clip(lower=0.0)
                rows.append(
                    {
                        "fold": int(fold),
                        "method": method,
                        "service_level_nominal": service_level,
                        "empirical_coverage": float((cycle["y_true"] <= target).mean()),
                        "shortage_units": float(shortage.sum()),
                        "excess_units": float(excess.sum()),
                    }
                )
    return pd.DataFrame(rows)
