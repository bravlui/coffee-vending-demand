# ADR 0002 — Forecasting approach

## Status
Accepted

## Context
We need per-product daily demand for a 7-day horizon, on ~380 days of history for
one machine. Demand is highly intermittent (~1 unit/product/day).

## Decision
- **Daily product panel**, zero-filled inside the observed range (a missing day
  is genuine zero demand).
- **Direct multi-horizon** setup: every lag/rolling feature is shifted by at
  least the horizon, so one feature row is valid for the whole next cycle and no
  recursive forecasting is needed.
- **Baselines**: seasonal-naive (same weekday last week) and 28-day moving
  average. Any model must beat these.
- **Model**: a single global **LightGBM** (`regression_l2`) across all products,
  product as a categorical feature.
- **Target transform** `ratio_to_level`: train on `units / level`, where `level`
  is an EWMA of recent demand; predict `ratio x level`. Trees cannot extrapolate
  a trend; this keeps the level in an adaptive component and lets the tree learn
  only the seasonal shape.
- **Backtest**: rolling origin, expanding train window, 7-day test window, 8
  folds. Metrics pooled at daily and cycle level.
- **Ensemble**: 50/50 blend of LightGBM and seasonal-naive, evaluated alongside.

## Alternatives considered
| Option | Why not (now) |
|---|---|
| ARIMA / ETS per product | Per-series, doesn't pool, weak on intermittent low counts, heavier to operate |
| Prophet | Overkill for this signal; poor with intermittency; adds a dependency |
| Recursive multi-step ML | Error accumulation; more complex backtest; no benefit at this horizon |
| Croston / TSB (intermittent demand) | Implemented as extra baselines; they do not win development selection |
| Deep learning (N-BEATS, TFT) | No data volume to justify; operationally heavy |

## Consequences
- On a single machine, seasonal-naive is competitive and often wins on cycle
  WAPE. See ADR 0003 (parsimony rule).
- The ML pipeline is still built and tested, ready for the fleet setting where
  pooling and covariates are expected to make it win.
