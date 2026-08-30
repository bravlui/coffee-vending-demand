"""Load the raw transaction CSV(s) into a single normalised transaction frame.

Output schema (one row = one purchase):
    ts            datetime64[ns]  transaction timestamp (naive, local time)
    date          datetime64[ns]  calendar date (midnight)
    payment_type  string          "card" | "cash"
    customer_id   string          anonymised card id, or <NA> for cash
    price         float64         amount paid for the item
    product       string          product name (raw, cleaning happens later)
    source_file   string          provenance
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from coffee_intel.config import Config
from coffee_intel.logging_utils import get_logger

logger = get_logger(__name__)

_RENAME = {
    "datetime": "ts",
    "cash_type": "payment_type",
    "card": "customer_id",
    "money": "price",
    "coffee_name": "product",
}
CANONICAL_COLUMNS = [
    "ts",
    "date",
    "payment_type",
    "customer_id",
    "price",
    "product",
    "source_file",
]


def _read_one(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw file not found: {path.resolve()}. See the README 'Data' section."
        )
    df = pd.read_csv(path)
    missing = {"datetime", "cash_type", "money", "coffee_name"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing expected columns: {sorted(missing)}")
    if "card" not in df.columns:
        df["card"] = pd.NA
    df = df.rename(columns=_RENAME)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["date"] = df["ts"].dt.normalize()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    for col in ("payment_type", "customer_id", "product"):
        df[col] = df[col].astype("string").str.strip()
    df["customer_id"] = df["customer_id"].replace({"": pd.NA})
    df["source_file"] = path.name
    return df[CANONICAL_COLUMNS]


def load_transactions(cfg: Config) -> pd.DataFrame:
    """Load and concatenate the configured raw files, sorted by timestamp."""
    raw_dir = Path(cfg.paths.raw_dir)
    frames = [_read_one(raw_dir / cfg.data.primary_file)]
    if cfg.data.use_secondary:
        frames.append(_read_one(raw_dir / cfg.data.secondary_file))
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    logger.info(
        "Loaded %d transactions from %s (%s to %s)",
        len(df),
        ", ".join(f.name for f in [raw_dir / cfg.data.primary_file]),
        df["date"].min().date(),
        df["date"].max().date(),
    )
    return df
