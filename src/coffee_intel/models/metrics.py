"""Forecast error metrics.

WAPE is the headline metric: it is scale-free, aggregates cleanly across
products of very different volume, and maps directly to "how much total volume
did we get wrong", which is what a replenishment planner cares about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype="float64")


def wape(y_true, y_pred) -> float:
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denom) if denom else np.nan


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(_arr(y_true) - _arr(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((_arr(y_true) - _arr(y_pred)) ** 2)))


def bias(y_true, y_pred) -> float:
    """Mean signed error. Positive => model over-forecasts."""
    return float(np.mean(_arr(y_pred) - _arr(y_true)))


def smape(y_true, y_pred) -> float:
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    return float(np.mean(2 * np.abs(y_pred - y_true)[mask] / denom[mask])) if mask.any() else np.nan


def all_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "wape": wape(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }


def metrics_by_group(df: pd.DataFrame, group: str, y_true: str, y_pred: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group, observed=True):
        m = all_metrics(g[y_true], g[y_pred])
        m[group] = key
        m["volume"] = float(g[y_true].sum())
        rows.append(m)
    return pd.DataFrame(rows).set_index(group).sort_values("volume", ascending=False)
