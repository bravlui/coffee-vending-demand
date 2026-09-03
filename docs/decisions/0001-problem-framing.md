# ADR 0001 — Problem framing: replenishment, not raw forecasting

## Status
Accepted

## Context
The Coffee Sales dataset is a transaction log from one vending machine
(Mar 2024 – Mar 2025, ~3.6k rows, 8 products). Many framings are possible:
demand forecasting, price elasticity, customer churn, anomaly detection.

## Decision
Frame the problem as **inventory replenishment decision support**: forecast
per-product demand for the next cycle and convert it into an order-up-to
quantity at a target service level. Keep the scope limited to this operational
decision so the evidence, implementation and presentation tell one story.

## Rationale
- It maps to a real, recurring operational decision with a clear cost of being
  wrong in either direction (stockout vs. waste).
- It is the framing that generalises cleanly to a fleet, which is the realistic
  business context for a coffee company.
- The forecasting is "light" time series — tabular regression with calendar and
  lag features — so the solution stays simple.
- Price elasticity was rejected: repricing events in the data are system-wide
  and fully confounded with time, so no credible causal estimate is possible.
- Customer segmentation was excluded from this iteration because it addresses a
  different stakeholder and would dilute the replenishment decision being validated.

## Consequences
- Success is measured with WAPE at the **cycle** level (decision-relevant), not
  only daily error.
- The safety-stock layer needs a forecast-error estimate, which the backtest
  provides.
