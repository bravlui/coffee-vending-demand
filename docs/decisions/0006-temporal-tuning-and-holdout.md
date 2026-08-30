# ADR 0006 — Temporal tuning and final holdout

## Status
Accepted

## Context
The initial LightGBM used conservative hand-picked parameters and all eight
rolling folds for comparison. That did not show whether the ML model had
received a fair, leakage-safe tuning process.

## Decision
- Split rolling origins chronologically: oldest folds are development and the
  latest two folds remain an untouched final holdout.
- Run a bounded deterministic search over objective, learning rate, leaves and
  minimum child samples on development folds only.
- Optimise cycle-level WAPE, aligned with the replenishment decision.
- Select the production model, including the parsimony gate, on development
  folds. Use the final holdout for reporting only.
- Add Croston and TSB as intermittent-demand baselines.
- Report median/IQR and folds won against seasonal naive.
- Backtest nominal service levels with walk-forward error calibration.

## Consequences
- Holdout metrics are not used to tune parameters or select the model.
- The bounded search controls compute and reduces overfitting to short history.
- With two final folds, conclusions remain directional and must be read with
  the development-fold stability analysis.
- Policy coverage is a diagnostic simulation, not a realised stockout KPI,
  because the data has no inventory or machine-availability telemetry.
