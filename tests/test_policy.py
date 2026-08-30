"""Tests for the replenishment policy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from coffee_intel.policy.replenishment import (
    _z,
    backtest_service_levels,
    build_recommendation,
    daily_error_std,
    empirical_cycle_buffer,
)


def test_z_matches_known_service_levels():
    assert _z(0.5) == pytest.approx(0.0, abs=1e-9)
    assert _z(0.95) == pytest.approx(1.645, abs=1e-3)
    assert _z(0.975) == pytest.approx(1.960, abs=1e-3)


def test_higher_service_level_means_more_safety_stock(config):
    forecast = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=7).tolist() * 2,
            "product": ["Latte"] * 7 + ["Americano"] * 7,
            "forecast_units": [3.0] * 14,
        }
    )
    err_std = pd.Series({"Latte": 1.0, "Americano": 2.0})

    config.replenishment.service_level = 0.90
    low = build_recommendation(forecast, err_std, config).set_index("product")
    config.replenishment.service_level = 0.99
    high = build_recommendation(forecast, err_std, config).set_index("product")

    assert (high["safety_stock"] > low["safety_stock"]).all()
    assert (high["target_stock"] >= low["target_stock"]).all()


def test_order_qty_never_negative_and_respects_on_hand(config):
    forecast = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=7),
            "product": ["Latte"] * 7,
            "forecast_units": [5.0] * 7,
        }
    )
    err_std = pd.Series({"Latte": 1.0})
    rec = build_recommendation(forecast, err_std, config)
    assert (rec["order_qty"] >= 0).all()
    assert rec.loc[0, "expected_demand_cycle"] == pytest.approx(35.0)


def test_daily_error_std_per_product():
    preds = pd.DataFrame(
        {
            "product": ["A"] * 4 + ["B"] * 4,
            "y_true": [1, 2, 3, 4, 1, 1, 1, 1],
            "model": [1, 2, 3, 4, 2, 2, 2, 2],
        }
    )
    std = daily_error_std(preds, "model")
    assert std["A"] == pytest.approx(0.0)
    assert std["B"] == pytest.approx(np.std([1, 1, 1, 1], ddof=1))


def test_policy_backtest_uses_prior_folds():
    rows = []
    for fold in range(3):
        for product in ["A", "B"]:
            rows.append(
                {
                    "fold": fold,
                    "date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=fold),
                    "product": product,
                    "y_true": 2.0,
                    "model": 1.5,
                }
            )
    result = backtest_service_levels(pd.DataFrame(rows), "model", (0.90, 0.95))
    assert set(result["fold"]) == {1, 2}
    assert set(result["method"]) == {"normal", "empirical"}
    assert result["empirical_coverage"].between(0, 1).all()
    assert (result[["shortage_units", "excess_units"]] >= 0).all().all()


def test_empirical_buffer_uses_underforecast_quantile():
    preds = pd.DataFrame(
        {
            "fold": [0, 1, 2],
            "product": ["A", "A", "A"],
            "y_true": [10.0, 12.0, 8.0],
            "model": [8.0, 8.0, 9.0],
        }
    )
    buffer = empirical_cycle_buffer(preds, "model", 0.5)
    assert buffer["A"] == pytest.approx(2.0)
