"""Build the per-customer feature table used for RFM segmentation.

Only card transactions carry a customer id, so cash sales (~2.5% of volume) are
excluded here by construction. That is acceptable for a loyalty-oriented view
and is called out as a limitation in the report.
"""

from __future__ import annotations

import pandas as pd

from coffee_intel.config import Config
from coffee_intel.logging_utils import get_logger

logger = get_logger(__name__)


def build_customer_features(transactions: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    card = transactions.loc[transactions["customer_id"].notna()].copy()

    # Snapshot = the moment we score from. Default: just after the last event,
    # so recency is always >= 0.
    snapshot = (
        pd.Timestamp(cfg.segmentation.snapshot_date)
        if cfg.segmentation.snapshot_date
        else card["ts"].max()
    )

    fav_product = (
        card.groupby(["customer_id", "product"], observed=True)
        .size()
        .reset_index(name="n")
        .sort_values(["customer_id", "n"], ascending=[True, False])
        .drop_duplicates("customer_id")
        .set_index("customer_id")["product"]
    )

    grp = card.groupby("customer_id", observed=True)
    feats = grp.agg(
        first_purchase=("ts", "min"),
        last_purchase=("ts", "max"),
        frequency=("ts", "size"),
        monetary=("price", "sum"),
        avg_ticket=("price", "mean"),
        distinct_days=("date", "nunique"),
        distinct_products=("product", "nunique"),
        morning_share=("hour", lambda s: (s < 12).mean()),
    )

    feats["recency_days"] = (snapshot - feats["last_purchase"]).dt.days
    feats["tenure_days"] = (feats["last_purchase"] - feats["first_purchase"]).dt.days
    feats["purchases_per_active_day"] = feats["frequency"] / feats["distinct_days"]
    feats["favourite_product"] = fav_product
    feats["snapshot_date"] = snapshot

    feats = feats[feats["frequency"] >= cfg.segmentation.min_transactions]
    logger.info(
        "Customer features: %d customers (snapshot %s)",
        len(feats),
        snapshot.date(),
    )
    return feats.reset_index()
