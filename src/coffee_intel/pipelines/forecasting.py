"""Stage 2: demand forecasting + replenishment recommendation.

Steps: build daily panel -> features -> rolling backtest (baselines vs LightGBM)
-> pick best model -> retrain on all history -> forecast next cycle -> convert to
an order-up-to recommendation. Every artifact lands in reports/ or data/processed/.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from coffee_intel.config import Config
from coffee_intel.features.forecasting import (
    add_features,
    build_daily_panel,
    extend_panel_with_future,
    mask_machine_inactive_days,
    split_future,
)
from coffee_intel.io_utils import write_csv, write_json
from coffee_intel.logging_utils import get_logger
from coffee_intel.models.backtest import (
    fold_stability,
    run_backtest,
    select_model_robust,
    summarise_backtest,
    summarise_backtest_cycle,
)
from coffee_intel.models.baselines import BASELINES, intermittent_forecast
from coffee_intel.models.forecaster import train_forecaster
from coffee_intel.models.metrics import metrics_by_group
from coffee_intel.models.tuning import temporal_split, tune_forecaster
from coffee_intel.pipelines.prepare import load_clean
from coffee_intel.policy.replenishment import (
    backtest_service_levels,
    build_recommendation,
    daily_error_std,
    empirical_cycle_buffer,
)

logger = get_logger(__name__)

MODEL_COLS = ["ml_lightgbm", "ensemble", *BASELINES.keys(), "croston", "tsb"]


def run_forecasting(cfg: Config, make_plots: bool = True) -> dict:
    figures_dir = Path(cfg.paths.figures_dir)
    metrics_dir = Path(cfg.paths.metrics_dir)
    processed_dir = Path(cfg.paths.processed_dir)

    transactions = load_clean(cfg)
    panel = build_daily_panel(transactions, cfg)
    feat_hist = add_features(panel, cfg)

    # ---- backtest ------------------------------------------------------
    model_overrides: dict = {}
    tuning_results = None
    if cfg.forecasting.tuning.enabled:
        model_overrides, tuning_results = tune_forecaster(feat_hist, cfg)
        development_origins, evaluation_origins = temporal_split(feat_hist, cfg)
        development_preds, _ = run_backtest(
            feat_hist, cfg, origins=development_origins, model_overrides=model_overrides
        )
        preds, fold_metrics = run_backtest(
            feat_hist, cfg, origins=evaluation_origins, model_overrides=model_overrides
        )
        write_csv(tuning_results, metrics_dir / "lightgbm_tuning_results.csv")
    else:
        preds, fold_metrics = run_backtest(feat_hist, cfg)
        development_preds = preds
    # Model choice is frozen on development folds. Holdout is reporting only.
    development_summary = summarise_backtest(development_preds, MODEL_COLS)
    development_cycle_summary = summarise_backtest_cycle(development_preds, MODEL_COLS)
    summary = summarise_backtest(preds, MODEL_COLS)
    cycle_summary = summarise_backtest_cycle(preds, MODEL_COLS)
    ms = cfg.forecasting.model_selection
    top_model = development_cycle_summary.index[0]
    stability = fold_stability(development_preds, MODEL_COLS, ms.reference_model)
    best_model, promotion = select_model_robust(
        development_cycle_summary,
        stability,
        ms.reference_model,
        ms.parsimony_tolerance,
        ms.min_fold_win_rate,
        ms.max_abs_bias_increase,
    )
    if (
        best_model != ms.reference_model
        and cycle_summary.loc[best_model, "wape"]
        > cycle_summary.loc[ms.reference_model, "wape"] + ms.holdout_max_regression
    ):
        promotion["holdout_gate"] = False
        promotion["holdout_reason"] = "candidate_regressed_beyond_tolerance"
        best_model = ms.reference_model
    else:
        promotion["holdout_gate"] = True
    logger.info(
        "Top by cycle WAPE: %s (%.3f). Shipping: %s (parsimony rule vs %s, tol %.2f)",
        top_model,
        development_cycle_summary.loc[top_model, "wape"],
        best_model,
        ms.reference_model,
        ms.parsimony_tolerance,
    )
    per_product = metrics_by_group(preds, "product", "y_true", best_model)
    if cfg.forecasting.tuning.enabled:
        holdout_for_policy = preds.copy()
        holdout_for_policy["fold"] += development_preds["fold"].nunique()
        policy_history = pd.concat([development_preds, holdout_for_policy], ignore_index=True)
    else:
        policy_history = preds
    policy_backtest = backtest_service_levels(policy_history, best_model)

    # Sensitivity: treat full-machine no-sale dates as unknown, not true zeroes.
    masked_panel, inactive_dates = mask_machine_inactive_days(panel, transactions)
    masked_features = add_features(masked_panel, cfg).dropna(subset=["units"])
    sensitivity_preds, _ = run_backtest(
        masked_features,
        cfg,
        origins=evaluation_origins if cfg.forecasting.tuning.enabled else None,
        model_overrides=model_overrides,
    )
    sensitivity_summary = summarise_backtest_cycle(sensitivity_preds, MODEL_COLS)
    sensitivity_comparison = (
        cycle_summary[["wape"]]
        .rename(columns={"wape": "zero_assumption_wape"})
        .join(sensitivity_summary[["wape"]].rename(columns={"wape": "unknown_day_wape"}))
    )
    sensitivity_comparison["wape_delta"] = (
        sensitivity_comparison["unknown_day_wape"] - sensitivity_comparison["zero_assumption_wape"]
    )

    write_csv(preds, processed_dir / "backtest_predictions.csv")
    write_csv(fold_metrics, metrics_dir / "backtest_fold_metrics.csv")
    write_csv(summary.reset_index(), metrics_dir / "backtest_summary_daily.csv")
    write_csv(cycle_summary.reset_index(), metrics_dir / "backtest_summary_cycle.csv")
    write_csv(development_summary.reset_index(), metrics_dir / "development_summary_daily.csv")
    write_csv(
        development_cycle_summary.reset_index(),
        metrics_dir / "development_summary_cycle.csv",
    )
    write_csv(per_product.reset_index(), metrics_dir / "backtest_per_product.csv")
    write_csv(stability, metrics_dir / "backtest_stability.csv")
    write_csv(policy_backtest, metrics_dir / "policy_backtest.csv")
    write_csv(sensitivity_comparison.reset_index(), metrics_dir / "downtime_sensitivity.csv")

    # ---- final model + next-cycle forecast ---------------------------
    full_panel = extend_panel_with_future(panel, cfg)
    feat_all = add_features(full_panel, cfg)
    _, future = split_future(feat_all, cfg)

    final_model = train_forecaster(feat_hist, cfg, model_overrides)
    future = future.copy()
    future["ml_lightgbm"] = final_model.predict(future)
    for name, fn in BASELINES.items():
        future[name] = fn(future).to_numpy()
    future["croston"] = intermittent_forecast(feat_hist, future, "croston").to_numpy()
    future["tsb"] = intermittent_forecast(feat_hist, future, "tsb").to_numpy()
    future["ensemble"] = 0.5 * future["ml_lightgbm"] + 0.5 * future["seasonal_naive"]

    # Ship the model that won the cycle-level backtest.
    future["forecast_units"] = future[best_model]
    forecast_out = future[["date", "product", "avg_price", "forecast_units", *MODEL_COLS]].copy()
    forecast_out[["forecast_units", *MODEL_COLS]] = forecast_out[
        ["forecast_units", *MODEL_COLS]
    ].round(2)
    write_csv(forecast_out, processed_dir / "forecast_next_cycle.csv")

    # ---- replenishment recommendation ------------------------------
    err_std = daily_error_std(policy_history, best_model)
    empirical_buffer = empirical_cycle_buffer(
        policy_history, best_model, cfg.replenishment.service_level
    )
    recommendation = build_recommendation(
        forecast_out, err_std, cfg, empirical_buffer=empirical_buffer
    )
    write_csv(recommendation, processed_dir / "replenishment_recommendation.csv")

    importance = final_model.feature_importance().rename("importance")
    importance_df = importance.rename_axis("feature").reset_index()
    write_csv(importance_df, metrics_dir / "feature_importance.csv")

    result = {
        "shipped_model": best_model,
        "top_model_by_cycle_wape": top_model,
        "backtest_summary_daily": summary.to_dict(orient="index"),
        "backtest_summary_cycle": cycle_summary.to_dict(orient="index"),
        "development_summary_cycle": development_cycle_summary.to_dict(orient="index"),
        "n_folds": int(fold_metrics["fold"].nunique()),
        "development_folds": int(development_preds["fold"].nunique()),
        "horizon_days": cfg.forecasting.horizon_days,
        "forecast_period": [
            str(forecast_out["date"].min().date()),
            str(forecast_out["date"].max().date()),
        ],
        "total_forecast_units": float(forecast_out["forecast_units"].sum()),
        "total_order_qty": float(recommendation["order_qty"].sum()),
        "lightgbm_best_params": model_overrides,
        "promotion_decision": promotion,
        "inactive_dates_sensitivity_count": len(inactive_dates),
        "downtime_sensitivity": sensitivity_comparison.to_dict(orient="index"),
        "evaluation_scope": "final_holdout" if cfg.forecasting.tuning.enabled else "all_folds",
        "evaluation_folds": int(fold_metrics["fold"].nunique()),
        "policy_backtest": policy_backtest.groupby(
            ["method", "service_level_nominal"], observed=True, as_index=False
        )
        .agg(
            empirical_coverage=("empirical_coverage", "mean"),
            shortage_units=("shortage_units", "sum"),
            excess_units=("excess_units", "sum"),
        )
        .to_dict(orient="records"),
    }
    write_json(result, metrics_dir / "forecasting_summary.json")

    if make_plots:
        from coffee_intel.reporting import plots

        plots.backtest_by_model(summary, figures_dir)
        plots.forecast_vs_actual(preds, best_model, figures_dir)
        plots.feature_importance(importance, figures_dir)
        plots.daily_revenue(transactions, figures_dir)
        plots.demand_profiles(transactions, figures_dir)
        plots.product_mix(transactions, figures_dir)

    logger.info("Forecasting pipeline complete.")
    return result
