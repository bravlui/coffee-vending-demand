# ADR 0004 — Replenishment policy

## Status
Accepted

## Context
A forecast is not a decision. Field ops needs a number: how many units of each
product to load.

## Decision
A transparent **base-stock (order-up-to) policy**:

```
protection      = cycle_days + lead_time_days
target_stock    = expected_demand(protection) + z(service_level) * sigma * sqrt(protection)
order_qty       = max(0, target_stock - on_hand)
```

- `expected_demand` comes from the shipped forecast model.
- `sigma` is the **per-product daily forecast-error std measured in the
  backtest** — the buffer reflects how wrong we have actually been, not a guess.
- `z(service_level)` uses the normal approximation (`statistics.NormalDist`,
  no scipy dependency).
- `service_level` is a config knob (default 0.95), tunable per product class.

## Alternatives considered
- **Newsvendor with explicit understock/overstock costs**: better in theory, but
  this dataset has no reliable waste/stockout cost. Deferred until KPIs are
  measured.
- **Quantile forecast → directly read the safety stock**: cleaner statistically;
  planned as an evolution (quantile LightGBM).

## Consequences
- The policy is a pure, unit-tested function, independent of the model — easy to
  reason about, easy to override, easy to roll back.
- Higher service level ⇒ more safety stock ⇒ more waste risk; the trade-off is
  explicit and owned by the business.
