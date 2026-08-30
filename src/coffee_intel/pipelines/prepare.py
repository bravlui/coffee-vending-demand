"""Stage 1: raw CSV -> validated, cleaned `transactions` table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from coffee_intel.config import Config
from coffee_intel.data.clean import clean_transactions
from coffee_intel.data.ingest import load_transactions
from coffee_intel.data.validate import validate_transactions
from coffee_intel.io_utils import write_json, write_parquet
from coffee_intel.logging_utils import get_logger

logger = get_logger(__name__)


def run_prepare(cfg: Config) -> pd.DataFrame:
    cfg.ensure_dirs()
    raw = load_transactions(cfg)

    report = validate_transactions(raw, cfg)
    write_json(report.to_dict(), Path(cfg.paths.metrics_dir) / "data_quality.json")
    if report.has_fatal_errors and cfg.data.strict_validation:
        raise ValueError("Fatal data-quality checks failed; see data_quality.json")
    if not report.ok:
        logger.warning("Non-fatal data-quality checks reported issues; continuing.")

    clean = clean_transactions(raw, cfg)
    out = write_parquet(clean, Path(cfg.paths.processed_dir) / "transactions.parquet")
    logger.info("Wrote %s", out)
    return clean


def load_clean(cfg: Config) -> pd.DataFrame:
    """Load the cleaned table, running prepare if it is missing."""
    for ext in ("parquet", "csv"):
        p = Path(cfg.paths.processed_dir) / f"transactions.{ext}"
        if p.exists():
            if ext == "parquet":
                try:
                    return pd.read_parquet(p)
                except ImportError:
                    logger.warning(
                        "Parquet exists but no engine is installed; trying CSV fallback."
                    )
                    continue
            return pd.read_csv(p, parse_dates=["ts", "date"])
    return run_prepare(cfg)
