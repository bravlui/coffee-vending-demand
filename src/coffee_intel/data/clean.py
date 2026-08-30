"""Clean the raw transaction frame into the analysis-ready `transactions` table."""

from __future__ import annotations

import pandas as pd

from coffee_intel.config import Config
from coffee_intel.logging_utils import get_logger

logger = get_logger(__name__)

# Known case / spelling variants across the two source files -> canonical name.
_PRODUCT_ALIASES = {
    "americano with milk": "Americano with Milk",
}


def _canonical_product(name: str) -> str:
    key = name.strip().lower()
    if key in _PRODUCT_ALIASES:
        return _PRODUCT_ALIASES[key]
    return name.strip().title()


def clean_transactions(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    n0 = len(df)

    # 1. Price sanity filter (removes data-entry errors, not genuine promos).
    price_ok = df["price"].between(cfg.cleaning.min_price, cfg.cleaning.max_price)
    dropped_price = int((~price_ok).sum())
    df = df[price_ok]

    # 2. Product name normalisation.
    if cfg.cleaning.normalise_product_names:
        df["product"] = df["product"].map(_canonical_product).astype("string")

    # 3. Payment type normalisation + derived flag.
    df["payment_type"] = df["payment_type"].str.lower()
    df["is_card"] = df["payment_type"].eq("card")

    # 4. Exact duplicate rows (same second, product, price, customer) are almost
    #    certainly double-logged events, not two real purchases.
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["ts", "product", "price", "customer_id"])
    dropped_dupes = before_dedup - len(df)

    # 5. Calendar helpers used everywhere downstream.
    df["date"] = df["ts"].dt.normalize()
    df["hour"] = df["ts"].dt.hour.astype("int16")
    df["dow"] = df["ts"].dt.dayofweek.astype("int16")

    df = df.sort_values("ts").reset_index(drop=True)
    logger.info(
        "Cleaned: %d -> %d rows (dropped %d out-of-range prices, %d duplicates)",
        n0,
        len(df),
        dropped_price,
        dropped_dupes,
    )
    return df
