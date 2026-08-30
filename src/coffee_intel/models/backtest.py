"""Rolling-origin backtest for the demand forecaster.

Expanding train window, fixed `horizon_days` test window, stepped forward
`step_days` at a time. This mimics how the model is actually used: retrain on
everything known so far, forecast the next cycle, move on.
"""

from __future__ import annotations

import pandas as pd

from coffee_intel.config import Config
from coffee_intel.features.forecasting import TARGET_COL
from coffee_intel.logging_utils import get_logger
from coffee_intel.models.baselines import BASELINES, intermittent_forecast
from coffee_intel.models.forecaster import train_forecaster
from coffee_intel.models.metrics import all_metrics

logger = get_logger(__name__)


def _fold_origins(dates: pd.Series, cfg: Config) -> list[pd.Timestamp]:
    bt = cfg.forecasting.backtest
    unique_days = pd.Series(sorted(dates.unique()))
    last_day = unique_days.iloc[-1]
    origins = []
    for i in range(bt.n_folds):
        cutoff = last_day - pd.Timedelta(days=cfg.forecasting.horizon_days + i * bt.step_days)
        if (cutoff - unique_days.iloc[0]).days < bt.min_train_days:
            break
        origins.append(cutoff)
    return sorted(origins)


def run_backtest(
    feat: pd.DataFrame,
    cfg: Config,
    origins: list[pd.Timestamp] | None = None,
    model_overrides: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (predictions_long, fold_metrics).

    predictions_long: date, product, y_true, and one column per model.
    fold_metrics: one row per (fold, model) with all error metrics.
    """
    horizon = pd.Timedelta(days=cfg.forecasting.horizon_days)
    origins = origins if origins is not None else _fold_origins(feat["date"], cfg)
    if not origins:
        raise RuntimeError("Not enough history for a single backtest fold; lower min_train_days.")

    all_preds: list[pd.DataFrame] = []
    fold_rows: list[dict] = []

    for fold, origin in enumerate(origins):
        train = feat[feat["date"] <= origin]
        test = feat[(feat["date"] > origin) & (feat["date"] <= origin + horizon)].copy()
        if test.empty:
            continue

        model = train_forecaster(train, cfg, model_overrides)
        test["ml_lightgbm"] = model.predict(test)
        for name, fn in BASELINES.items():
            test[name] = fn(test).to_numpy()
        test["croston"] = intermittent_forecast(train, test, "croston").to_numpy()
        test["tsb"] = intermittent_forecast(train, test, "tsb").to_numpy()
        # Simple 50/50 blend of the ML model and the strongest baseline.
        test["ensemble"] = 0.5 * test["ml_lightgbm"] + 0.5 * test["seasonal_naive"]

        test["fold"] = fold
        model_names = ["ml_lightgbm", "ensemble", *BASELINES.keys(), "croston", "tsb"]
        keep = ["fold", "date", "product", TARGET_COL, *model_names]
        all_preds.append(test[keep].rename(columns={TARGET_COL: "y_true"}))

        fold_scores: dict[str, float] = {}
        for name in model_names:
            m = all_metrics(test[TARGET_COL], test[name])
            fold_scores[name] = m["wape"]
            row: dict[str, object] = dict(m)
            row.update(fold=fold, model=name, origin=str(origin.date()), n=len(test))
            fold_rows.append(row)

        logger.info(
            "fold %d | origin %s | train=%d test=%d | LGBM WAPE=%.3f seasonal=%.3f",
            fold,
            origin.date(),
            len(train),
            len(test),
            fold_scores["ml_lightgbm"],
            fold_scores.get("seasonal_naive", float("nan")),
        )

    preds = pd.concat(all_preds, ignore_index=True)
    metrics = pd.DataFrame(fold_rows)
    return preds, metrics


def summarise_backtest(preds: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    """Pooled daily metrics across all folds."""
    rows = []
    for name in model_cols:
        row: dict[str, object] = dict(all_metrics(preds["y_true"], preds[name]))
        row["model"] = name
        rows.append(row)
    return pd.DataFrame(rows).set_index("model").sort_values("wape")


def summarise_backtest_cycle(preds: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    """Pooled metrics on per-(fold, product) cycle totals.

    This is the decision-relevant view: replenishment cares about the total
    over the cycle, not which day inside it. Errors partly cancel across days,
    so these numbers are lower and more meaningful than the daily ones.
    """
    cyc = preds.groupby(["fold", "product"], observed=True)[["y_true", *model_cols]].sum()
    rows = []
    for name in model_cols:
        row: dict[str, object] = dict(all_metrics(cyc["y_true"], cyc[name]))
        row["model"] = name
        rows.append(row)
    return pd.DataFrame(rows).set_index("model").sort_values("wape")


def select_model(cycle_summary: pd.DataFrame, reference: str, tolerance: float) -> str:
    """Pick the model to ship, applying a parsimony rule.

    The best model by cycle WAPE is only adopted if it beats the simple
    `reference` model by more than `tolerance`; otherwise the reference wins.
    """
    best = cycle_summary.index[0]
    if best == reference or reference not in cycle_summary.index:
        return best
    gain = cycle_summary.loc[reference, "wape"] - cycle_summary.loc[best, "wape"]
    return best if gain > tolerance else reference


def fold_stability(preds: pd.DataFrame, model_cols: list[str], reference: str) -> pd.DataFrame:
    """Cycle-WAPE distribution and wins against the reference across folds."""
    rows: list[dict[str, object]] = []
    for fold, fold_df in preds.groupby("fold"):
        summary = summarise_backtest_cycle(fold_df, model_cols)
        reference_wape = float(summary.loc[reference, "wape"])
        for model in model_cols:
            rows.append(
                {
                    "fold": int(fold),
                    "model": model,
                    "cycle_wape": float(summary.loc[model, "wape"]),
                    "beats_reference": float(summary.loc[model, "wape"]) < reference_wape,
                }
            )
    detail = pd.DataFrame(rows)
    return (
        detail.groupby("model", observed=True)
        .agg(
            median_cycle_wape=("cycle_wape", "median"),
            q25_cycle_wape=("cycle_wape", lambda s: s.quantile(0.25)),
            q75_cycle_wape=("cycle_wape", lambda s: s.quantile(0.75)),
            folds_won=("beats_reference", "sum"),
            folds=("fold", "nunique"),
        )
        .reset_index()
    )


def select_model_robust(
    cycle_summary: pd.DataFrame,
    stability: pd.DataFrame,
    reference: str,
    tolerance: float,
    min_fold_win_rate: float,
    max_abs_bias_increase: float,
) -> tuple[str, dict[str, object]]:
    """Apply accuracy, stability and bias gates on development data only."""
    candidate = str(cycle_summary.index[0])
    if candidate == reference:
        return reference, {"candidate": candidate, "passed": True, "reason": "reference_best"}
    gain = float(cycle_summary.loc[reference, "wape"] - cycle_summary.loc[candidate, "wape"])
    row = stability.set_index("model").loc[candidate]
    win_rate = float(row["folds_won"] / row["folds"])
    bias_increase = float(
        abs(cycle_summary.loc[candidate, "bias"]) - abs(cycle_summary.loc[reference, "bias"])
    )
    gates = {
        "wape_gain": gain,
        "wape_gate": gain > tolerance,
        "fold_win_rate": win_rate,
        "stability_gate": win_rate >= min_fold_win_rate,
        "abs_bias_increase": bias_increase,
        "bias_gate": bias_increase <= max_abs_bias_increase,
    }
    passed = bool(gates["wape_gate"] and gates["stability_gate"] and gates["bias_gate"])
    return (candidate if passed else reference), {"candidate": candidate, "passed": passed, **gates}
