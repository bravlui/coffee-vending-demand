"""Stage 3: customer segmentation (complementary analytical layer)."""

from __future__ import annotations

from pathlib import Path

from coffee_intel.config import Config
from coffee_intel.features.segmentation import build_customer_features
from coffee_intel.io_utils import write_csv, write_json
from coffee_intel.logging_utils import get_logger
from coffee_intel.models.segmentation import segment_customers
from coffee_intel.pipelines.prepare import load_clean

logger = get_logger(__name__)


def run_segmentation(cfg: Config, make_plots: bool = True) -> dict:
    metrics_dir = Path(cfg.paths.metrics_dir)
    processed_dir = Path(cfg.paths.processed_dir)
    figures_dir = Path(cfg.paths.figures_dir)

    transactions = load_clean(cfg)
    features = build_customer_features(transactions, cfg)
    result = segment_customers(features, cfg)

    write_csv(result.customers, processed_dir / "customer_segments.csv")
    write_csv(result.profile.reset_index(), metrics_dir / "segment_profile.csv")

    card_rev = transactions.loc[transactions["customer_id"].notna(), "price"].sum()
    summary = {
        "k": result.k,
        "silhouette": round(result.silhouette, 4),
        "stability_ari": round(result.stability_ari, 4),
        "n_customers": len(result.customers),
        "card_revenue_covered": float(card_rev),
        "segments": result.profile.reset_index().to_dict(orient="records"),
    }
    write_json(summary, metrics_dir / "segmentation_summary.json")

    if make_plots:
        from coffee_intel.reporting import plots

        plots.segment_scatter(result.customers, figures_dir)

    logger.info("Segmentation pipeline complete (k=%d).", result.k)
    return summary
