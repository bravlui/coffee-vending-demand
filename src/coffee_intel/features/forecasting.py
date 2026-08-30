"""Build the daily product-level demand panel and its features.

Design choice: a *direct* multi-horizon setup. Every feature (lags, rolling
means) is shifted by at least `horizon_days`, so the exact same feature row is
valid for any day within the next cycle and no recursive prediction is needed.
That keeps the model simple, fast to backtest, and free of error accumulation.
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from coffee_intel.config import Config
from coffee_intel.logging_utils import get_logger

logger = get_logger(__name__)

TARGET_COL = "units"


def build_daily_panel(transactions: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One row per (date, product) with units and revenue, zero-filled.

    Missing days inside the observed range are real zero-demand days (the machine
    was up but that product did not sell) and must be modelled as such.
    """
    daily = (
        transactions.groupby(["date", "product"], observed=True)
        .agg(units=("price", "size"), revenue=("price", "sum"), avg_price=("price", "mean"))
        .reset_index()
    )

    full_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    products = daily["product"].unique()
    grid = pd.MultiIndex.from_product([full_dates, products], names=["date", "product"])

    panel = daily.set_index(["date", "product"]).reindex(grid).reset_index()
    panel["units"] = panel["units"].fillna(0.0)
    panel["revenue"] = panel["revenue"].fillna(0.0)
    # Carry the last known price forward for zero-demand days.
    panel["avg_price"] = panel.groupby("product", observed=True)["avg_price"].ffill().bfill()
    panel = panel.sort_values(["product", "date"]).reset_index(drop=True)
    logger.info(
        "Daily panel: %d rows (%d days x %d products)",
        len(panel),
        len(full_dates),
        len(products),
    )
    return panel


def mask_machine_inactive_days(
    panel: pd.DataFrame, transactions: pd.DataFrame
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """Mark full-machine no-sale dates as unknown instead of zero demand."""
    calendar = pd.date_range(panel["date"].min(), panel["date"].max(), freq="D")
    active = pd.DatetimeIndex(transactions["date"].dropna().unique())
    inactive = list(calendar.difference(active))
    masked = panel.copy()
    masked.loc[masked["date"].isin(inactive), ["units", "revenue"]] = np.nan
    return masked, inactive


def _calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df["date"].dt
    df["dow"] = d.dayofweek.astype("int16")
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    df["day_of_month"] = d.day.astype("int16")
    df["week_of_year"] = d.isocalendar().week.astype("int16")
    df["month"] = d.month.astype("int16")
    df["days_since_start"] = (df["date"] - df["date"].min()).dt.days.astype("int32")

    years = range(df["date"].dt.year.min(), df["date"].dt.year.max() + 1)
    ua_holidays = holidays.Ukraine(years=list(years))
    df["is_holiday"] = df["date"].isin(ua_holidays).astype("int8")
    return df


def add_features(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add calendar, lag, rolling and price features to the daily panel."""
    df = panel.copy()
    df = _calendar_features(df)

    h = cfg.forecasting.horizon_days
    g = df.groupby("product", observed=True)["units"]

    for lag in cfg.forecasting.lags:
        df[f"units_lag_{lag}"] = g.shift(lag)

    for win in cfg.forecasting.rolling_windows:
        # shift(h) first so the window only uses information available at forecast time
        df[f"units_roll_mean_{win}"] = (
            g.shift(h)
            .rolling(win, min_periods=max(2, win // 2))
            .mean()
            .reset_index(level=0, drop=True)
        )
        df[f"units_roll_std_{win}"] = (
            g.shift(h)
            .rolling(win, min_periods=max(2, win // 2))
            .std()
            .reset_index(level=0, drop=True)
        )

    # Same weekday one and two weeks ago — strong signal for this data.
    df["units_same_dow_1w"] = g.shift(7)
    df["units_same_dow_2w"] = g.shift(14)

    # Adaptive demand LEVEL: EWMA of recent daily units (lagged by the horizon).
    # The model predicts a multiplicative factor on top of this, so the trend /
    # level always comes from recent data and the tree only learns the shape
    # (weekday, holiday, price). Tree models cannot extrapolate a trend on their
    # own; this is how we let them keep up with one.
    df["_u_shift"] = g.shift(h)
    df["level"] = (
        df.groupby("product", observed=True)["_u_shift"]
        .transform(lambda s: s.ewm(span=14, min_periods=3).mean())
        .clip(lower=0.05)
    )
    df = df.drop(columns="_u_shift")

    # Price relative to the product's trailing average = a crude promo/repricing signal.
    df["price_vs_trailing"] = df["avg_price"] / (
        df.groupby("product", observed=True)["avg_price"]
        .shift(h)
        .rolling(28, min_periods=7)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["product_code"] = df["product"].astype("category").cat.codes.astype("int16")

    feature_cols = feature_columns(cfg)
    df = df.dropna(subset=[*[c for c in feature_cols if c.startswith("units_lag")], "level"])
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    logger.info("Feature frame: %d rows, %d features", len(df), len(feature_cols))
    return df


def extend_panel_with_future(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Append `horizon_days` future rows per product (units/revenue = 0 placeholder).

    Price is carried forward as the last observed value; a planner can override
    it if a repricing is scheduled. Feature building then fills lag/rolling
    columns for these rows from real history.
    """
    horizon = cfg.forecasting.horizon_days
    last_date = panel["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    last_price = panel.sort_values("date").groupby("product", observed=True)["avg_price"].last()

    rows = [
        {"date": d, "product": p, "units": 0.0, "revenue": 0.0, "avg_price": last_price[p]}
        for p in panel["product"].unique()
        for d in future_dates
    ]
    future = pd.DataFrame(rows)
    out = pd.concat([panel, future], ignore_index=True)
    return out.sort_values(["product", "date"]).reset_index(drop=True)


def split_future(feat: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a feature frame into (history, next-cycle future rows)."""
    horizon = cfg.forecasting.horizon_days
    cutoff = feat["date"].max() - pd.Timedelta(days=horizon)
    return feat[feat["date"] <= cutoff].copy(), feat[feat["date"] > cutoff].copy()


def feature_columns(cfg: Config) -> list[str]:
    cols = [
        "product_code",
        "dow",
        "is_weekend",
        "day_of_month",
        "week_of_year",
        "month",
        "days_since_start",
        "is_holiday",
        "avg_price",
        "price_vs_trailing",
        "units_same_dow_1w",
        "units_same_dow_2w",
    ]
    cols += [f"units_lag_{lag}" for lag in cfg.forecasting.lags]
    for win in cfg.forecasting.rolling_windows:
        cols += [f"units_roll_mean_{win}", f"units_roll_std_{win}"]
    return cols
