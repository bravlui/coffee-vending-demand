"""End-to-end smoke tests: each pipeline runs and writes its artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from coffee_intel.cli import app
from coffee_intel.pipelines.forecasting import run_forecasting
from coffee_intel.pipelines.prepare import run_prepare
from coffee_intel.pipelines.segmentation import run_segmentation

runner = CliRunner()


def test_prepare_writes_clean_table(config):
    df = run_prepare(config)
    assert len(df) > 0
    out = list(Path(config.paths.processed_dir).glob("transactions.*"))
    assert out
    assert (Path(config.paths.metrics_dir) / "data_quality.json").exists()


def test_forecasting_pipeline_produces_recommendation(config):
    run_prepare(config)
    res = run_forecasting(config, make_plots=False)
    assert res["shipped_model"] in {
        "seasonal_naive",
        "moving_average_28",
        "ml_lightgbm",
        "ensemble",
        "croston",
        "tsb",
    }
    rec = Path(config.paths.processed_dir) / "replenishment_recommendation.csv"
    assert rec.exists()
    summary = json.loads((Path(config.paths.metrics_dir) / "forecasting_summary.json").read_text())
    assert summary["total_order_qty"] >= 0


def test_segmentation_pipeline_writes_segments(config):
    run_prepare(config)
    res = run_segmentation(config, make_plots=False)
    assert res["n_customers"] > 0
    assert (Path(config.paths.processed_dir) / "customer_segments.csv").exists()


def test_cli_run_all(config, tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    import yaml

    cfg_path.write_text(yaml.safe_dump(json.loads(config.model_dump_json())))
    result = runner.invoke(app, ["run-all", "-c", str(cfg_path), "--no-plots"])
    assert result.exit_code == 0, result.output
