"""Tests for baselines, the forecaster, backtest selection and segmentation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from coffee_intel.data.clean import clean_transactions
from coffee_intel.data.ingest import load_transactions
from coffee_intel.features.forecasting import add_features, build_daily_panel
from coffee_intel.features.segmentation import build_customer_features
from coffee_intel.models.backtest import (
    run_backtest,
    select_model,
    select_model_robust,
    summarise_backtest_cycle,
)
from coffee_intel.models.baselines import BASELINES, _croston_level, _tsb_level
from coffee_intel.models.forecaster import train_forecaster
from coffee_intel.models.segmentation import segment_customers
from coffee_intel.models.tuning import candidate_params, temporal_split


@pytest.fixture
def feat(config):
    tx = clean_transactions(load_transactions(config), config)
    return add_features(build_daily_panel(tx, config), config)


def test_baselines_non_negative(feat):
    for name, fn in BASELINES.items():
        pred = fn(feat)
        assert (pred >= 0).all(), name
        assert len(pred) == len(feat)


def test_intermittent_baselines_are_non_negative():
    series = pd.Series([0, 0, 2, 0, 0, 3, 0, 0, 0, 1], dtype=float)
    assert _croston_level(series) >= 0
    assert _tsb_level(series) >= 0
    assert _croston_level(pd.Series([0.0, 0.0])) == 0


def test_tuning_keeps_latest_origins_as_holdout(config, feat):
    config.forecasting.tuning.enabled = True
    config.forecasting.tuning.holdout_folds = 1
    development, holdout = temporal_split(feat, config)
    assert max(development) < min(holdout)
    assert len(development) + len(holdout) == config.forecasting.backtest.n_folds
    config.forecasting.tuning.max_candidates = 2
    assert len(candidate_params(config)) == 2


def test_forecaster_predicts_non_negative(config, feat):
    model = train_forecaster(feat, config)
    preds = model.predict(feat)
    assert (preds >= 0).all()
    assert len(preds) == len(feat)
    assert model.feature_importance().sum() > 0


def test_backtest_runs(config, feat):
    preds, fold_metrics = run_backtest(feat, config)
    assert set(preds.columns) >= {"fold", "date", "product", "y_true", "ml_lightgbm", "ensemble"}
    assert fold_metrics["fold"].nunique() == config.forecasting.backtest.n_folds
    assert (preds[["ml_lightgbm", "ensemble"]] >= 0).all().all()


def test_select_model_parsimony_rule():
    summary = pd.DataFrame(
        {"wape": [0.30, 0.315, 0.34]},
        index=["ml_lightgbm", "seasonal_naive", "moving_average_28"],
    )
    # ml wins by only 0.015 < tol 0.02 -> keep the simple reference
    assert select_model(summary, "seasonal_naive", 0.02) == "seasonal_naive"
    # ml wins by 0.05 > tol -> adopt ml
    summary.loc["ml_lightgbm", "wape"] = 0.265
    assert select_model(summary, "seasonal_naive", 0.02) == "ml_lightgbm"


def test_robust_selection_requires_stability_and_bias():
    summary = pd.DataFrame(
        {"wape": [0.25, 0.30], "bias": [0.3, 0.2]},
        index=["ml_lightgbm", "seasonal_naive"],
    )
    stability = pd.DataFrame({"model": ["ml_lightgbm"], "folds_won": [4], "folds": [6]})
    chosen, decision = select_model_robust(summary, stability, "seasonal_naive", 0.02, 0.5, 0.25)
    assert chosen == "ml_lightgbm"
    assert decision["passed"]


def test_summarise_cycle_aggregates(config, feat):
    preds, _ = run_backtest(feat, config)
    cyc = summarise_backtest_cycle(preds, ["ml_lightgbm", "seasonal_naive"])
    assert cyc["wape"].is_monotonic_increasing  # sorted best-first
    assert (cyc["wape"] >= 0).all()


def test_segmentation_assigns_every_customer(config):
    tx = clean_transactions(load_transactions(config), config)
    feats = build_customer_features(tx, config)
    result = segment_customers(feats, config)
    assert result.customers["segment"].notna().all()
    assert result.k in config.segmentation.kmeans.k_candidates
    assert np.isclose(result.profile["revenue_share"].sum(), 1.0, atol=1e-6)
    assert 0 <= result.stability_ari <= 1
