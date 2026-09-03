# %% [markdown]
# # Coffee Sales — Exploratory Data Analysis
#
# Run as a script (`python notebooks/01_eda.py`) or cell-by-cell in an editor
# that understands the `# %%` format (VS Code, PyCharm, Jupytext).
#
# Goal: understand the data well enough to justify the problem framing and the
# modelling choices (see `docs/decisions/`).

# %%
from pathlib import Path

import pandas as pd

from coffee_intel.config import Config
from coffee_intel.data.clean import clean_transactions
from coffee_intel.data.ingest import load_transactions
from coffee_intel.reporting import plots

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

cfg = Config.load("config/config.yaml")
tx = clean_transactions(load_transactions(cfg), cfg)
print(tx.shape)
tx.head()

# %% [markdown]
# ## 1. Coverage and volume

# %%
span = (tx["date"].max() - tx["date"].min()).days + 1
print(f"period: {tx['date'].min().date()} -> {tx['date'].max().date()}  ({span} days)")
print(f"days with >=1 sale: {tx['date'].nunique()}  ({tx['date'].nunique() / span:.1%})")
print(
    f"transactions/day: mean {tx.groupby('date').size().mean():.1f}, "
    f"median {tx.groupby('date').size().median():.0f}"
)
print(
    tx.groupby(tx["date"].dt.to_period("M"))
    .agg(tx=("price", "size"), revenue=("price", "sum"))
    .round(0)
)

# %% [markdown]
# ## 2. Products and price

# %%
print((tx["product"].value_counts(normalize=True) * 100).round(1))
# Price changes over time -> repricing events, not per-transaction promotions.
print(tx.groupby("product")["price"].agg(["min", "max", "nunique", "mean"]).round(2))

# %% [markdown]
# ## 3. Seasonality — hour of day is strong, weekday is mild

# %%
print("by hour:\n", tx.groupby("hour").size())
print("\nby weekday (0=Mon):\n", tx.groupby("dow").size())

# %% [markdown]
# ## 4. Figures (written to reports/figures/)

# %%
figdir = Path(cfg.paths.figures_dir)
figdir.mkdir(parents=True, exist_ok=True)
for f in (
    plots.daily_revenue(tx, figdir),
    plots.demand_profiles(tx, figdir),
    plots.product_mix(tx, figdir),
):
    print("wrote", f)

# %% [markdown]
# ## Takeaways
#
# 1. One machine, ~13 months, almost always operating (few zero days).
# 2. Demand is **intermittent** at product-day level (~1 unit) — favours pooled,
#    simple models and cycle-level evaluation over daily.
# 3. Strong **intraday** and **holiday** structure; mild weekday effect; a visible
#    **upward trend / level shift** in early 2025 — the model must adapt to level,
#    not extrapolate a slope.
# 4. Prices move in discrete steps (repricing) and are confounded with time →
#    no causal pricing analysis from this data.
