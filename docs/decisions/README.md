# Architecture Decision Records

Short records of the decisions that shaped this solution and why. Format: one
decision per file, numbered.

| # | Decision |
|---|---|
| [0001](0001-problem-framing.md) | Problem framing: replenishment decision support, not raw forecasting |
| [0002](0002-forecasting-approach.md) | Forecasting approach: pooled LightGBM + baselines, direct multi-horizon |
| [0003](0003-parsimony-model-selection.md) | Parsimony rule for choosing the model to ship |
| [0004](0004-replenishment-policy.md) | Base-stock replenishment policy with backtest-derived safety stock |
| [0006](0006-temporal-tuning-and-holdout.md) | Temporal tuning and final holdout |
