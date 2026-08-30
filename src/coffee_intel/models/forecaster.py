"""LightGBM demand forecaster.

A single global model across all products (product enters as a categorical
feature). With only ~380 days of history, one pooled model learns shared
weekday / holiday / price dynamics far more reliably than eight per-product
models, and it extends to new products / machines without retraining from
scratch.

Target transform (`ratio_to_level`): the model is trained on
``units / level``, where ``level`` is an adaptive EWMA of recent demand. The
prediction is ``ratio_hat * level``. This keeps the trend / level in a component
that adapts to recent data and leaves the tree to learn only the seasonal shape,
which is what tree models are good at (and lets them keep up with a trend they
cannot extrapolate on their own).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from coffee_intel.config import Config
from coffee_intel.features.forecasting import TARGET_COL, feature_columns
from coffee_intel.logging_utils import get_logger

logger = get_logger(__name__)

LEVEL_COL = "level"


@dataclass
class ForecastModel:
    booster: lgb.LGBMRegressor
    features: list[str]
    transform: str

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.booster.predict(df[self.features])
        if self.transform == "ratio_to_level":
            raw = raw * df[LEVEL_COL].to_numpy()
        return np.clip(raw, a_min=0.0, a_max=None)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.booster_.save_model(str(path))

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.booster.feature_importances_, index=self.features).sort_values(
            ascending=False
        )


def _params(cfg: Config, overrides: dict | None = None) -> dict:
    lg = cfg.forecasting.lightgbm
    params = {
        "objective": lg.objective,
        "n_estimators": lg.n_estimators,
        "learning_rate": lg.learning_rate,
        "num_leaves": lg.num_leaves,
        "min_child_samples": lg.min_child_samples,
        "subsample": lg.subsample,
        "subsample_freq": lg.subsample_freq,
        "colsample_bytree": lg.colsample_bytree,
        "random_state": lg.random_state,
        "n_jobs": -1,
        "verbose": -1,
    }
    if overrides:
        params.update(overrides)
    if params["objective"] == "tweedie":
        params["tweedie_variance_power"] = lg.tweedie_variance_power
    return params


def train_forecaster(
    train_df: pd.DataFrame, cfg: Config, overrides: dict | None = None
) -> ForecastModel:
    features = feature_columns(cfg)
    transform = cfg.forecasting.target_transform

    y = train_df[TARGET_COL].to_numpy(dtype="float64")
    if transform == "ratio_to_level":
        y = y / train_df[LEVEL_COL].to_numpy(dtype="float64")

    model = lgb.LGBMRegressor(**_params(cfg, overrides))
    model.fit(train_df[features], y)
    logger.info(
        "Trained LightGBM on %d rows, %d features (transform=%s)",
        len(train_df),
        len(features),
        transform,
    )
    return ForecastModel(booster=model, features=features, transform=transform)
