# Results (single machine)

Numbers regenerate on every `coffee-intel run-all`; machine-readable evidence
is written to `reports/metrics/`.

## Experimental protocol

- Eight rolling origins with a 7-day horizon.
- Oldest six folds: bounded LightGBM tuning and model selection.
- Latest two folds: untouched final holdout, used for reporting only.
- Primary metric: WAPE on per-product cycle totals.
- Parsimony gate: a complex model must improve development WAPE by more than
  0.02 against seasonal naive.

This avoids selecting hyperparameters or the shipped model on the final result.
Two holdout folds are still a small sample, so conclusions are directional.

## LightGBM tuning

Twelve deterministic candidates were tested on development folds. The best
configuration was:

| Parameter | Value |
|---|---:|
| Objective | Poisson |
| num_leaves | 31 |
| learning_rate | 0.05 |
| min_child_samples | 20 |

Development WAPE for the tuned LightGBM was 0.348. Candidate-level evidence is
in `reports/metrics/lightgbm_tuning_results.csv`.

## Development folds — model selection

| Model | Cycle WAPE | Bias |
|---|---:|---:|
| Ensemble | 0.324 | -1.69 |
| **Seasonal naive (shipped)** | **0.324** | **-1.35** |
| TSB | 0.342 | -3.09 |
| Croston | 0.344 | -2.91 |
| Tuned LightGBM | 0.348 | -2.02 |
| Moving average 28 | 0.390 | -3.48 |

The ensemble advantage over seasonal naive is only 0.0003, far below the 0.02
promotion margin. It wins 3/6 folds and also worsens absolute bias by 0.33,
above the configured 0.25 limit. The multidimensional gate therefore keeps the
simple reference before holdout inspection.

## Untouched final holdout — two latest folds

| Model | Cycle WAPE | Bias |
|---|---:|---:|
| Tuned LightGBM | **0.168** | -0.42 |
| Croston | 0.173 | -0.50 |
| Ensemble | 0.178 | -0.43 |
| TSB | 0.183 | -0.96 |
| Seasonal naive | 0.202 | -0.44 |
| Moving average 28 | 0.202 | +0.24 |

LightGBM performs best in the final period, but this does not retroactively
change the frozen selection rule. It supports continued shadow testing, not
promotion based on only two cycles.

## Sensitivity to full-machine no-sale days

The source contains seven dates with no transaction from any product. Treating
those dates as unknown rather than true zero demand changes holdout WAPE by:

| Model | Zero assumption | Unknown-day assumption | Delta |
|---|---:|---:|---:|
| LightGBM | 0.168 | 0.205 | +0.037 |
| Ensemble | 0.178 | 0.198 | +0.020 |
| Croston | 0.173 | 0.175 | +0.002 |
| TSB | 0.183 | 0.183 | -0.001 |
| Seasonal naive | 0.202 | 0.202 | 0.000 |

The selected model and business conclusion do not change. LightGBM is more
sensitive to this assumption, strengthening the case for uptime telemetry.

## Replenishment-policy diagnostic

Walk-forward calibration, using only errors from earlier folds, produced:

| Method | Nominal level | Coverage | Shortage | Excess |
|---|---:|---:|---:|---:|
| Normal | 90% | 82.1% | 42.0 | 316.2 |
| Normal | 95% | **87.5%** | **32.5** | 397.8 |
| Normal | 99% | 94.6% | 23.7 | 559.8 |
| Empirical quantile | 90% | 82.1% | 43.7 | 356.0 |
| Empirical quantile | 95% | 83.9% | 41.8 | 392.4 |
| Empirical quantile | 99% | 83.9% | 40.4 | 421.7 |

The empirical buffer is unstable with so few cycles and performs worse than the
normal approximation. The operational configuration therefore remains normal.
Even so, nominal 95% achieves only 87.5% coverage, so neither approach is
considered calibrated. These are simulated quantities, not realised stockout
KPIs, because the dataset has neither inventory nor availability telemetry.

The next-cycle output assumes `on_hand=0`; `order_qty` is therefore equal to the
order-up-to requirement, not a literal purchase order. Current inventory must
be integrated before operational use.

## Limitations

- One machine and about 13 months of data.
- Only two final holdout cycles.
- Missing transaction days cannot distinguish zero demand, downtime and
  stockout; the sensitivity analysis shows the selected baseline is unchanged.
- No inventory, waste or stockout ground truth.
- Repricing is confounded with time; price effects are not causal.
