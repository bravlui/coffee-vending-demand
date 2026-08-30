"""Tests for configuration loading."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from coffee_intel.config import Config


def test_load_real_config_file():
    cfg = Config.load("config/config.yaml")
    assert cfg.forecasting.horizon_days > 0
    assert 0 < cfg.replenishment.service_level < 1
    assert cfg.forecasting.model_selection.reference_model


def test_load_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        Config.load("config/nope.yaml")


def test_ensure_dirs_creates_paths(config):
    config.ensure_dirs()
    for p in config.paths.model_dump().values():
        assert p.exists()


def test_config_rejects_temporal_leakage(config):
    raw = config.model_dump()
    raw["forecasting"]["lags"] = [1, 7]
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_config_rejects_invalid_service_level(config):
    raw = config.model_dump()
    raw["replenishment"]["service_level"] = 1.0
    with pytest.raises(ValidationError):
        Config.model_validate(raw)
