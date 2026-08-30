"""Smoke tests for the plotting helpers — they must produce a PNG, not crash."""

from __future__ import annotations

import pytest

from coffee_intel.data.clean import clean_transactions
from coffee_intel.data.ingest import load_transactions
from coffee_intel.features.forecasting import add_features, build_daily_panel
from coffee_intel.features.segmentation import build_customer_features
from coffee_intel.models.backtest import run_backtest, summarise_backtest
from coffee_intel.models.forecaster import train_forecaster
from coffee_intel.models.segmentation import segment_customers
from coffee_intel.reporting import plots


@pytest.fixture
def tx(config):
    return clean_transactions(load_transactions(config), config)


def test_eda_plots(tx, tmp_path):
    for fn in (plots.daily_revenue, plots.demand_profiles, plots.product_mix):
        out = fn(tx, tmp_path)
        assert out.exists() and out.stat().st_size > 0


def test_model_plots(config, tx, tmp_path):
    feat = add_features(build_daily_panel(tx, config), config)
    preds, _ = run_backtest(feat, config)
    summary = summarise_backtest(preds, ["ml_lightgbm", "seasonal_naive"])
    assert plots.backtest_by_model(summary, tmp_path).exists()
    assert plots.forecast_vs_actual(preds, "ml_lightgbm", tmp_path).exists()

    model = train_forecaster(feat, config)
    assert plots.feature_importance(model.feature_importance(), tmp_path).exists()


def test_segment_plot(config, tx, tmp_path):
    feats = build_customer_features(tx, config)
    result = segment_customers(feats, config)
    assert plots.segment_scatter(result.customers, tmp_path).exists()
