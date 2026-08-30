"""Command-line entry point.

coffee-intel prepare        # raw CSV -> cleaned transactions + data-quality report
coffee-intel forecast       # backtest + train + next-cycle forecast + replenishment
coffee-intel segment        # RFM customer segmentation
coffee-intel run-all        # the three above, in order
"""

from __future__ import annotations

from pathlib import Path

import typer

from coffee_intel import __version__
from coffee_intel.config import DEFAULT_CONFIG_PATH, Config
from coffee_intel.logging_utils import get_logger

app = typer.Typer(add_completion=False, help="Coffee vending demand & customer intelligence.")
logger = get_logger("coffee_intel.cli")

ConfigOpt = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c", help="Path to config YAML.")
NoPlotsOpt = typer.Option(False, "--no-plots", help="Skip figure generation.")


def _load(config: Path) -> Config:
    cfg = Config.load(config)
    cfg.ensure_dirs()
    return cfg


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def prepare(config: Path = ConfigOpt) -> None:
    """Ingest, validate and clean the raw transactions."""
    from coffee_intel.pipelines.prepare import run_prepare

    run_prepare(_load(config))


@app.command()
def forecast(config: Path = ConfigOpt, no_plots: bool = NoPlotsOpt) -> None:
    """Backtest, train the final model and produce the replenishment plan."""
    from coffee_intel.pipelines.forecasting import run_forecasting

    res = run_forecasting(_load(config), make_plots=not no_plots)
    typer.echo(
        f"shipped_model={res['shipped_model']}  total_order_qty={res['total_order_qty']:.0f}"
    )


@app.command()
def segment(config: Path = ConfigOpt, no_plots: bool = NoPlotsOpt) -> None:
    """Build RFM features and cluster customers into segments."""
    from coffee_intel.pipelines.segmentation import run_segmentation

    res = run_segmentation(_load(config), make_plots=not no_plots)
    typer.echo(f"k={res['k']}  silhouette={res['silhouette']}  customers={res['n_customers']}")


@app.command(name="run-all")
def run_all(config: Path = ConfigOpt, no_plots: bool = NoPlotsOpt) -> None:
    """Run prepare -> forecast -> segment end to end."""
    from coffee_intel.pipelines.forecasting import run_forecasting
    from coffee_intel.pipelines.prepare import run_prepare
    from coffee_intel.pipelines.segmentation import run_segmentation
    from coffee_intel.reporting.snapshot import publish_evidence_snapshot

    cfg = _load(config)
    run_prepare(cfg)
    run_forecasting(cfg, make_plots=not no_plots)
    run_segmentation(cfg, make_plots=not no_plots)
    publish_evidence_snapshot(cfg)
    logger.info("run-all complete.")


if __name__ == "__main__":
    app()
