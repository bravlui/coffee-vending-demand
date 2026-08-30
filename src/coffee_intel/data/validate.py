"""Lightweight data-quality checks.

This is a stand-in for a proper contract tool (Great Expectations / Pandera /
dbt tests) in production. It returns a structured report instead of only raising,
so the pipeline can log every issue and decide what is fatal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from coffee_intel.config import Config
from coffee_intel.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    severity: str = "error"


@dataclass
class ValidationReport:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "", severity: str = "error") -> None:
        passed = bool(passed)  # normalise numpy.bool_ so JSON stays true/false
        self.checks.append(Check(name, passed, detail, severity))
        level = logger.info if passed else logger.warning
        level("check %-28s %s %s", name, "OK" if passed else "FAIL", detail)

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def has_fatal_errors(self) -> bool:
        return any(not c.passed and c.severity == "error" for c in self.checks)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "checks": [c.__dict__ for c in self.checks],
        }


def validate_transactions(df: pd.DataFrame, cfg: Config) -> ValidationReport:
    r = ValidationReport()

    r.add("non_empty", len(df) > 0, f"{len(df)} rows")
    required = {"ts", "price", "product", "payment_type", "customer_id", "date"}
    missing = required - set(df.columns)
    r.add("required_columns", not missing, f"missing={sorted(missing)}" if missing else "")
    if missing:
        return r
    r.add("ts_not_null", df["ts"].notna().all(), f"{df['ts'].isna().sum()} nulls")
    r.add("price_not_null", df["price"].notna().all(), f"{df['price'].isna().sum()} nulls")
    r.add("product_not_null", df["product"].notna().all(), f"{df['product'].isna().sum()} nulls")

    in_range = df["price"].between(cfg.cleaning.min_price, cfg.cleaning.max_price)
    r.add(
        "price_in_range",
        bool(in_range.all()),
        f"{(~in_range).sum()} outside [{cfg.cleaning.min_price}, {cfg.cleaning.max_price}]",
    )

    r.add(
        "payment_type_domain",
        set(df["payment_type"].dropna().unique()).issubset({"card", "cash"}),
        f"values={sorted(df['payment_type'].dropna().unique())}",
    )

    dup = df.duplicated(subset=["ts", "product", "price", "customer_id"]).sum()
    r.add("no_exact_duplicates", dup == 0, f"{dup} duplicate rows")

    span = (df["date"].max() - df["date"].min()).days + 1
    covered = df["date"].nunique()
    r.add(
        "date_coverage",
        covered / span > 0.8,
        f"{covered}/{span} days have >=1 transaction",
        severity="warning",
    )

    return r
