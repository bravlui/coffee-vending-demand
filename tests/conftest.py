"""Shared fixtures. Tests run on synthetic data and never need the raw file."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from coffee_intel.config import Config

PRODUCTS = ["Latte", "Americano", "Cappuccino", "Cocoa"]
PRICES = {"Latte": 35.0, "Americano": 26.0, "Cappuccino": 36.0, "Cocoa": 35.0}


@pytest.fixture
def synthetic_transactions() -> pd.DataFrame:
    """~1 year of plausible vending transactions with weekday seasonality."""
    rng = np.random.default_rng(0)
    start = pd.Timestamp("2024-01-01")
    rows = []
    for day in range(400):
        date = start + pd.Timedelta(days=day)
        base = 10 + 3 * np.sin(day / 7 * 2 * np.pi)  # weekly cycle
        base *= 0.8 if date.dayofweek >= 5 else 1.0
        n = max(0, int(rng.poisson(base)))
        for _ in range(n):
            product = rng.choice(PRODUCTS, p=[0.4, 0.3, 0.2, 0.1])
            hour = int(np.clip(rng.normal(11, 3), 6, 22))
            ts = date + pd.Timedelta(hours=hour, minutes=int(rng.integers(0, 60)))
            is_card = rng.random() < 0.95
            cust = f"ANON-{rng.integers(1, 120):04d}" if is_card else pd.NA
            price = PRICES[product] * (1 + rng.normal(0, 0.02))
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "datetime": ts.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "cash_type": "card" if is_card else "cash",
                    "card": cust,
                    "money": round(price, 2),
                    "coffee_name": product,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def raw_csv(tmp_path, synthetic_transactions) -> pathlib.Path:  # noqa: F821
    path = tmp_path / "index_1.csv"
    synthetic_transactions.to_csv(path, index=False)
    return path


@pytest.fixture
def config(tmp_path, raw_csv) -> Config:
    """A Config pointing at the synthetic raw file and a temp workspace."""
    return Config.model_validate(
        {
            "paths": {
                "raw_dir": str(raw_csv.parent),
                "interim_dir": str(tmp_path / "interim"),
                "processed_dir": str(tmp_path / "processed"),
                "figures_dir": str(tmp_path / "figures"),
                "metrics_dir": str(tmp_path / "metrics"),
            },
            "data": {
                "primary_file": "index_1.csv",
                "secondary_file": "index_2.csv",
                "use_secondary": False,
            },
            "forecasting": {
                "horizon_days": 7,
                "backtest": {"n_folds": 3, "step_days": 7, "min_train_days": 120},
                "lightgbm": {"n_estimators": 60},
                "tuning": {"enabled": False},
            },
        }
    )
