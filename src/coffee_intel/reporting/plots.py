"""Matplotlib figures for the EDA and the model reports.

All functions save a PNG and return its path so pipelines can collect them.
Plain matplotlib, one chart per figure, no styling cleverness.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def daily_revenue(transactions: pd.DataFrame, out_dir: Path) -> Path:
    s = transactions.groupby("date")["price"].sum()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(s.index, s.to_numpy(), lw=0.8)
    ax.plot(s.index, s.rolling(14).mean().to_numpy(), lw=2, label="14-day mean")
    ax.set_title("Daily revenue")
    ax.set_ylabel("revenue")
    ax.legend()
    return _save(fig, out_dir, "eda_daily_revenue")


def demand_profiles(transactions: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    by_hour = transactions.groupby("hour").size()
    axes[0].bar(by_hour.index, by_hour.to_numpy())
    axes[0].set_title("Transactions by hour of day")
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_dow = transactions.groupby("dow").size()
    axes[1].bar([dow_names[i] for i in by_dow.index], by_dow.to_numpy())
    axes[1].set_title("Transactions by day of week")
    return _save(fig, out_dir, "eda_demand_profiles")


def product_mix(transactions: pd.DataFrame, out_dir: Path) -> Path:
    s = transactions["product"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(s.index.tolist(), s.to_numpy())
    ax.set_title("Units sold by product")
    return _save(fig, out_dir, "eda_product_mix")


def backtest_by_model(summary: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(summary.index.tolist(), summary["wape"].to_numpy())
    ax.set_title("Backtest WAPE by model (lower is better)")
    ax.set_ylabel("WAPE")
    ax.tick_params(axis="x", rotation=20)
    return _save(fig, out_dir, "backtest_wape_by_model")


def forecast_vs_actual(preds: pd.DataFrame, model_col: str, out_dir: Path) -> Path:
    g = preds.groupby("date").agg(y_true=("y_true", "sum"), y_pred=(model_col, "sum"))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(g.index, g["y_true"].to_numpy(), label="actual", lw=1.5)
    ax.plot(g.index, g["y_pred"].to_numpy(), label="forecast", lw=1.5, ls="--")
    ax.set_title("Backtest: total daily units, actual vs forecast")
    ax.legend()
    return _save(fig, out_dir, "backtest_forecast_vs_actual")


def feature_importance(importance: pd.Series, out_dir: Path) -> Path:
    top = importance.head(15).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.index.tolist(), top.to_numpy())
    ax.set_title("LightGBM feature importance (top 15)")
    return _save(fig, out_dir, "forecast_feature_importance")
