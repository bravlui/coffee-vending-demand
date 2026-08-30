"""Typed configuration loaded from a single YAML file.

Keeping all knobs in one validated object means a pipeline run is reproducible
from the config alone, and a bad value fails fast with a clear message.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

DEFAULT_CONFIG_PATH = Path("config/config.yaml")


class Paths(BaseModel):
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    figures_dir: Path
    metrics_dir: Path


class DataCfg(BaseModel):
    primary_file: str
    secondary_file: str
    use_secondary: bool = False
    timezone: str = "Europe/Kiev"
    currency: str = "UAH"
    strict_validation: bool = True


class CleaningCfg(BaseModel):
    min_price: float = 5.0
    max_price: float = 100.0
    normalise_product_names: bool = True


class BacktestCfg(BaseModel):
    n_folds: int = 8
    step_days: int = 7
    min_train_days: int = 180


class LightGBMCfg(BaseModel):
    objective: str = "regression_l2"
    tweedie_variance_power: float = 1.2
    n_estimators: int = 400
    learning_rate: float = 0.03
    num_leaves: int = 15
    min_child_samples: int = 30
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.8
    random_state: int = 42


class TuningCfg(BaseModel):
    enabled: bool = True
    holdout_folds: int = Field(default=2, ge=1)
    max_candidates: int = Field(default=12, ge=2)
    objectives: list[str] = Field(default_factory=lambda: ["regression_l2", "poisson", "tweedie"])
    num_leaves: list[int] = Field(default_factory=lambda: [7, 15, 31])
    learning_rates: list[float] = Field(default_factory=lambda: [0.02, 0.05])
    min_child_samples: list[int] = Field(default_factory=lambda: [20, 50])


class ModelSelectionCfg(BaseModel):
    reference_model: str = "seasonal_naive"
    parsimony_tolerance: float = Field(default=0.02, ge=0)
    min_fold_win_rate: float = Field(default=0.5, ge=0, le=1)
    max_abs_bias_increase: float = Field(default=0.25, ge=0)
    holdout_max_regression: float = Field(default=0.02, ge=0)


class ForecastingCfg(BaseModel):
    target: str = "units"
    horizon_days: int = Field(default=7, ge=1)
    lags: list[int] = Field(default_factory=lambda: [7, 14, 28])
    rolling_windows: list[int] = Field(default_factory=lambda: [7, 28])
    target_transform: str = "ratio_to_level"
    model_selection: ModelSelectionCfg = Field(default_factory=ModelSelectionCfg)
    backtest: BacktestCfg = Field(default_factory=BacktestCfg)
    lightgbm: LightGBMCfg = Field(default_factory=LightGBMCfg)
    tuning: TuningCfg = Field(default_factory=TuningCfg)

    @model_validator(mode="after")
    def validate_temporal_design(self) -> ForecastingCfg:
        if any(lag < self.horizon_days for lag in self.lags):
            raise ValueError("every forecasting lag must be >= horizon_days")
        if self.tuning.enabled and self.tuning.holdout_folds >= self.backtest.n_folds:
            raise ValueError("holdout_folds must be smaller than backtest.n_folds")
        allowed = {"regression_l2", "poisson", "tweedie"}
        unknown = set(self.tuning.objectives) - allowed
        if unknown:
            raise ValueError(f"unsupported LightGBM objectives: {sorted(unknown)}")
        return self


class ReplenishmentCfg(BaseModel):
    service_level: float = Field(default=0.95, gt=0, lt=1)
    lead_time_days: int = Field(default=1, ge=0)
    safety_stock_method: str = Field(default="empirical", pattern="^(normal|empirical)$")


class KMeansCfg(BaseModel):
    k_candidates: list[int] = Field(default_factory=lambda: [3, 4, 5, 6])
    random_state: int = 42
    n_init: int = 10


class SegmentationCfg(BaseModel):
    snapshot_date: str | None = None
    min_transactions: int = 1
    features: list[str] = Field(
        default_factory=lambda: [
            "recency_days",
            "frequency",
            "monetary",
            "tenure_days",
            "avg_ticket",
        ]
    )
    kmeans: KMeansCfg = Field(default_factory=KMeansCfg)


class Config(BaseModel):
    paths: Paths
    data: DataCfg
    cleaning: CleaningCfg = Field(default_factory=CleaningCfg)
    forecasting: ForecastingCfg = Field(default_factory=ForecastingCfg)
    replenishment: ReplenishmentCfg = Field(default_factory=ReplenishmentCfg)
    segmentation: SegmentationCfg = Field(default_factory=SegmentationCfg)
    random_seed: int = 42

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path.resolve()}")
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return cls.model_validate(raw)

    def ensure_dirs(self) -> None:
        for p in self.paths.model_dump().values():
            Path(p).mkdir(parents=True, exist_ok=True)
