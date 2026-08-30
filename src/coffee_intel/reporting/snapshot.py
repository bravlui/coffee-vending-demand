"""Publish a compact, versionable evidence snapshot of a pipeline run."""

from __future__ import annotations

import hashlib
import shutil
from datetime import UTC, datetime
from pathlib import Path

from coffee_intel import __version__
from coffee_intel.config import Config
from coffee_intel.io_utils import write_json

EVIDENCE_FILES = [
    "forecasting_summary.json",
    "development_summary_cycle.csv",
    "backtest_summary_cycle.csv",
    "lightgbm_tuning_results.csv",
    "backtest_stability.csv",
    "policy_backtest.csv",
    "downtime_sensitivity.csv",
    "segmentation_summary.json",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_evidence_snapshot(cfg: Config) -> Path:
    metrics = Path(cfg.paths.metrics_dir)
    destination = metrics.parent / "evidence"
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in EVIDENCE_FILES:
        source = metrics / name
        if source.exists():
            shutil.copy2(source, destination / name)
            copied.append(name)

    raw_path = Path(cfg.paths.raw_dir) / cfg.data.primary_file
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "package_version": __version__,
        "config_sha256": hashlib.sha256(cfg.model_dump_json().encode()).hexdigest(),
        "raw_data_sha256": _sha256(raw_path) if raw_path.exists() else None,
        "evidence_files": copied,
    }
    write_json(manifest, destination / "manifest.json")
    return destination
