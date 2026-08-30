"""Tests for feature engineering — the leakage checks matter most here."""

from __future__ import annotations

import numpy as np
import pandas as pd

from coffee_intel.data.clean import clean_transactions
from coffee_intel.data.ingest import load_transactions
from coffee_intel.features.forecasting import (
    add_features,
    build_daily_panel,
    extend_panel_with_future,
    feature_columns,
    mask_machine_inactive_days,
    split_future,
)
from coffee_intel.features.segmentation import build_customer_features


def _panel(config):
    tx = clean_transactions(load_transactions(config), config)
    return tx, build_daily_panel(tx, config)


def test_daily_panel_is_dense_and_zero_filled(config):
    _, panel = _panel(config)
    n_days = panel["date"].nunique()
    n_products = panel["product"].nunique()
    assert len(panel) == n_days * n_products
    assert (panel["units"] >= 0).all()


def test_inactive_day_sensitivity_masks_full_machine_zeros(config):
    tx, panel = _panel(config)
    missing_day = panel["date"].max() + pd.Timedelta(days=1)
    products = panel["product"].unique()
    extension = pd.DataFrame(
        {
            "date": [missing_day] * len(products),
            "product": products,
            "units": 0.0,
            "revenue": 0.0,
            "avg_price": 30.0,
        }
    )
    masked, inactive = mask_machine_inactive_days(pd.concat([panel, extension]), tx)
    assert missing_day in inactive
    assert masked.loc[masked["date"] == missing_day, "units"].isna().all()
    assert panel["units"].eq(0).any()  # some product/day had no sale


def test_panel_units_match_transactions(config):
    tx, panel = _panel(config)
    assert panel["units"].sum() == len(tx)


def test_features_have_no_future_leakage(config):
    """Every lag/rolling feature must be computable from data >= horizon days old.

    Check: for a given (product, date), the lag_7 feature equals the actual
    units 7 days earlier — never the same day.
    """
    _, panel = _panel(config)
    feat = add_features(panel, config)
    one = feat[feat["product"] == feat["product"].iloc[0]].sort_values("date")
    merged = one.merge(
        one[["date", "units"]].assign(date=lambda d: d["date"] + pd.Timedelta(days=7)),
        on="date",
        suffixes=("", "_7d_ago"),
    )
    assert np.allclose(merged["units_lag_7"], merged["units_7d_ago"])


def test_feature_columns_present(config):
    _, panel = _panel(config)
    feat = add_features(panel, config)
    for col in feature_columns(config):
        assert col in feat.columns
    assert feat[feature_columns(config)].notna().all().all()


def test_future_rows_are_horizon_days_and_have_features(config):
    _, panel = _panel(config)
    full = extend_panel_with_future(panel, config)
    feat = add_features(full, config)
    _, future = split_future(feat, config)
    per_product_days = future.groupby("product")["date"].nunique()
    assert (per_product_days == config.forecasting.horizon_days).all()
    assert future["date"].min() > panel["date"].max()
    assert future[feature_columns(config)].notna().all().all()


def test_customer_features_only_card_transactions(config):
    tx = clean_transactions(load_transactions(config), config)
    feats = build_customer_features(tx, config)
    assert feats["frequency"].min() >= config.segmentation.min_transactions
    assert (feats["recency_days"] >= 0).all()
    assert (feats["monetary"] > 0).all()
    # monetary total cannot exceed card revenue
    assert feats["monetary"].sum() <= tx.loc[tx["customer_id"].notna(), "price"].sum() + 1e-6
