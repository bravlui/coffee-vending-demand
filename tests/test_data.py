"""Tests for ingest, cleaning and validation."""

from __future__ import annotations

import pandas as pd
import pytest

from coffee_intel.data.clean import clean_transactions
from coffee_intel.data.ingest import CANONICAL_COLUMNS, load_transactions
from coffee_intel.data.validate import validate_transactions


def test_load_transactions_schema(config):
    df = load_transactions(config)
    assert list(df.columns) == CANONICAL_COLUMNS
    assert df["ts"].is_monotonic_increasing
    assert df["payment_type"].isin(["card", "cash"]).all()
    # cash rows carry no customer id
    assert df.loc[df["payment_type"] == "cash", "customer_id"].isna().all()


def test_load_transactions_missing_file(config):
    config.data.primary_file = "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_transactions(config)


def test_clean_drops_out_of_range_prices(config):
    df = load_transactions(config)
    df.loc[df.index[0], "price"] = 999.0
    df.loc[df.index[1], "price"] = 0.5
    cleaned = clean_transactions(df, config)
    assert cleaned["price"].between(config.cleaning.min_price, config.cleaning.max_price).all()
    assert len(cleaned) == len(df) - 2


def test_clean_deduplicates(config):
    df = load_transactions(config)
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    cleaned = clean_transactions(dup, config)
    assert not cleaned.duplicated(subset=["ts", "product", "price", "customer_id"]).any()


def test_clean_normalises_product_names(config):
    df = load_transactions(config)
    df.loc[df.index[0], "product"] = "  latte  "
    cleaned = clean_transactions(df, config)
    assert "Latte" in set(cleaned["product"])
    assert "  latte  " not in set(cleaned["product"])


def test_validate_flags_bad_price(config):
    df = clean_transactions(load_transactions(config), config)
    df.loc[df.index[0], "price"] = 10_000
    report = validate_transactions(df, config)
    assert not report.ok
    assert any(c.name == "price_in_range" and not c.passed for c in report.checks)


def test_validate_passes_on_clean_data(config):
    df = clean_transactions(load_transactions(config), config)
    report = validate_transactions(df, config)
    assert report.ok, [c for c in report.checks if not c.passed]


def test_validate_missing_schema_returns_fatal_report(config):
    report = validate_transactions(pd.DataFrame({"price": [10.0]}), config)
    assert report.has_fatal_errors
    assert any(c.name == "required_columns" and not c.passed for c in report.checks)
