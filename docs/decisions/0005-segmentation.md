# ADR 0005 — Customer segmentation

## Status
Accepted

## Context
~59% of carded customers bought only once; ~78% of revenue comes from repeat
customers. The commercial team wants a small number of actionable groups.

## Decision
- Build a per-customer **RFM feature table** (recency, frequency, monetary,
  tenure, average ticket, plus favourite product and morning share).
- Cluster with **K-means** on log-transformed, standardised features; choose `k`
  from `k_candidates` by **silhouette**.
- **Label clusters from their own RFM profile** with neutral descriptive names:
  High value / Repeat engaged / Recent low-frequency / Inactive / Low engagement.
- Measure stability across extra K-means seeds with Adjusted Rand Index.
- Only card transactions are used (cash has no id); this is stated as a
  limitation.

## Alternatives considered
- **Rule-based RFM quintiles**: the industry-standard, very interpretable —
  worth adding as a cross-check; K-means chosen here to also demonstrate the
  unsupervised workflow and let the data set the boundaries.
- **Repeat-purchase / churn model**: more valuable but needs more data than one
  machine offers; flagged as the main evolution for the customer layer.

## Consequences
- Output is descriptive: a segment per customer + a profile table with revenue
  share. It informs targeting; it is not a production scoring model yet.
- Silhouette is modest (~0.35) because the one-time majority does not cluster
  crisply — expected, and not a blocker for the intended use.
