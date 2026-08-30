# ADR 0003 — Parsimony rule for model selection

## Status
Accepted

## Context
In the backtest on this single machine, LightGBM is best on daily WAPE (~0.69 vs
0.71 for seasonal-naive) but seasonal-naive is best on **cycle** WAPE
(~0.29 vs ~0.36). The differences are small and within the noise of 8 folds.

## Decision
The pipeline selects the model to ship with a **parsimony rule**: adopt a more
complex model only if it beats the simple reference model (`seasonal_naive`) by
more than a configurable margin (`parsimony_tolerance`, default 0.02) on
cycle-level WAPE. Otherwise ship the reference.

## Rationale
- A model that only ties the baseline is not worth its operational cost:
  retraining, monitoring, drift handling, on-call surface.
- The rule is explicit and in config, so the trade-off is visible and tunable,
  not buried in a notebook.
- It makes the "when does ML win?" question concrete: the fleet rollout flips the
  reference once the global model clears the margin fleet-wide.

## Consequences
- On this dataset, the shipped model is `seasonal_naive`.
- The replenishment layer consumes whichever model is selected; nothing
  downstream changes.
- The shipped result stays honest and the decision framework is explicit — which
  matters more here than a marginal, unstable accuracy gain.
