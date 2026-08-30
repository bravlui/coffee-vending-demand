"""Naive forecasting baselines.

Any model has to beat these to justify its operational and maintenance cost.
Both are computed per product and expect the feature frame from
`features.forecasting.add_features`.
"""

from __future__ import annotations

import pandas as pd


def seasonal_naive(df: pd.DataFrame) -> pd.Series:
    """Predict each day as the same weekday one week ago (units_same_dow_1w)."""
    return df["units_same_dow_1w"].fillna(df.get("units_roll_mean_7", 0.0)).clip(lower=0.0)


def moving_average(df: pd.DataFrame, window: int = 28) -> pd.Series:
    """Predict each day as the trailing N-day mean (already lagged by horizon)."""
    col = f"units_roll_mean_{window}"
    if col not in df.columns:
        raise KeyError(f"{col} not in feature frame; add {window} to rolling_windows")
    return df[col].fillna(0.0).clip(lower=0.0)


BASELINES = {
    "seasonal_naive": seasonal_naive,
    "moving_average_28": lambda df: moving_average(df, 28),
}


def _croston_level(values: pd.Series, alpha: float = 0.1) -> float:
    """Classic Croston estimate for a non-negative intermittent series."""
    y = values.to_numpy(dtype="float64")
    nonzero = y.nonzero()[0]
    if len(nonzero) == 0:
        return 0.0
    demand = y[nonzero[0]]
    interval = float(nonzero[0] + 1)
    previous = nonzero[0]
    for idx in nonzero[1:]:
        demand += alpha * (y[idx] - demand)
        gap = float(idx - previous)
        interval += alpha * (gap - interval)
        previous = idx
    return max(0.0, demand / max(interval, 1e-9))


def _tsb_level(values: pd.Series, alpha: float = 0.1, beta: float = 0.1) -> float:
    """Teunter-Syntetos-Babai estimate, including obsolescence probability."""
    y = values.to_numpy(dtype="float64")
    if len(y) == 0:
        return 0.0
    nonzero_values = y[y > 0]
    size = float(nonzero_values[0]) if len(nonzero_values) else 0.0
    probability = 1.0 if y[0] > 0 else 0.0
    for value in y[1:]:
        occurred = float(value > 0)
        probability += beta * (occurred - probability)
        if occurred:
            size += alpha * (value - size)
    return max(0.0, probability * size)


def intermittent_forecast(
    train: pd.DataFrame, test: pd.DataFrame, method: str, alpha: float = 0.1
) -> pd.Series:
    """Constant per-product horizon forecast fitted only on training history."""
    estimator = _croston_level if method == "croston" else _tsb_level
    levels = train.groupby("product", observed=True)["units"].apply(
        lambda values: estimator(values, alpha=alpha)
    )
    return test["product"].map(levels).fillna(0.0).clip(lower=0.0)
