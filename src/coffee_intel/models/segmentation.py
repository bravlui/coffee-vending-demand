"""RFM-style customer segmentation via K-means, with business-readable labels.

The clustering is unsupervised and descriptive: its job is to give the loyalty
team a small number of actionable groups, not to predict anything. Labels are
assigned from each cluster's RFM profile, not hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from coffee_intel.config import Config
from coffee_intel.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class SegmentationResult:
    customers: pd.DataFrame  # input features + `cluster` + `segment`
    profile: pd.DataFrame  # per-segment summary
    k: int
    silhouette: float
    stability_ari: float


def _log_transform(x: pd.Series) -> pd.Series:
    return np.log1p(x.clip(lower=0))


def _choose_k(x: np.ndarray, cfg: Config) -> tuple[int, float]:
    best_k: int = cfg.segmentation.kmeans.k_candidates[0]
    best_score = -1.0
    for k in cfg.segmentation.kmeans.k_candidates:
        if k >= len(x):
            continue
        km = KMeans(
            n_clusters=k,
            random_state=cfg.segmentation.kmeans.random_state,
            n_init=cfg.segmentation.kmeans.n_init,
        )
        labels = km.fit_predict(x)
        score = silhouette_score(x, labels)
        logger.info("k=%d silhouette=%.3f", k, score)
        if score > best_score:
            best_k, best_score = k, score
    return best_k, best_score


def _label_segments(profile: pd.DataFrame) -> dict[int, str]:
    """Assign neutral, profile-consistent names rather than churn claims."""
    if len(profile) == 5:
        remaining = set(profile.index)
        high_value = profile["monetary"].idxmax()
        remaining.remove(high_value)
        inactive = profile.loc[list(remaining), "recency_days"].idxmax()
        remaining.remove(inactive)
        recent = profile.loc[list(remaining), "recency_days"].idxmin()
        remaining.remove(recent)
        repeat = profile.loc[list(remaining), "frequency"].idxmax()
        remaining.remove(repeat)
        low_engagement = remaining.pop()
        return {
            high_value: "High value",
            repeat: "Repeat engaged",
            recent: "Recent low-frequency",
            inactive: "Inactive",
            low_engagement: "Low engagement",
        }

    r = profile["recency_days"].rank(ascending=True)  # lower recency = better
    f = profile["frequency"].rank(ascending=False)
    m = profile["monetary"].rank(ascending=False)
    score = (r + f + m) / 3
    order = score.sort_values().index.tolist()

    names_by_count = {
        1: ["All customers"],
        2: ["Champions", "Occasional"],
        3: ["Champions", "Promising", "Dormant"],
        4: ["Champions", "Loyal", "Promising", "Dormant"],
        5: ["Champions", "Loyal", "Promising", "At risk", "Dormant"],
        6: ["Champions", "Loyal", "Promising", "New", "At risk", "Dormant"],
    }
    names = names_by_count.get(len(order), [f"Segment {i}" for i in range(len(order))])
    return {cluster: names[i] for i, cluster in enumerate(order)}


def segment_customers(features: pd.DataFrame, cfg: Config) -> SegmentationResult:
    df = features.copy()
    matrix = pd.DataFrame(index=df.index)
    for col in cfg.segmentation.features:
        if col not in df.columns:
            raise KeyError(f"segmentation feature '{col}' missing from customer features")
        matrix[col] = _log_transform(df[col]) if col != "recency_days" else df[col]

    x = StandardScaler().fit_transform(matrix.to_numpy())
    k, sil = _choose_k(x, cfg)

    km = KMeans(
        n_clusters=k,
        random_state=cfg.segmentation.kmeans.random_state,
        n_init=cfg.segmentation.kmeans.n_init,
    )
    base_labels = km.fit_predict(x)
    df["cluster"] = base_labels
    stability_scores = []
    for offset in range(1, 6):
        challenger = KMeans(
            n_clusters=k,
            random_state=cfg.segmentation.kmeans.random_state + offset,
            n_init=cfg.segmentation.kmeans.n_init,
        ).fit_predict(x)
        stability_scores.append(adjusted_rand_score(base_labels, challenger))
    stability_ari = float(np.mean(stability_scores))

    profile = (
        df.groupby("cluster")
        .agg(
            customers=("customer_id", "size"),
            recency_days=("recency_days", "median"),
            frequency=("frequency", "median"),
            monetary=("monetary", "median"),
            avg_ticket=("avg_ticket", "median"),
            tenure_days=("tenure_days", "median"),
            total_revenue=("monetary", "sum"),
        )
        .round(2)
    )
    labels = _label_segments(profile)
    df["segment"] = df["cluster"].map(labels)
    profile["segment"] = profile.index.map(labels)
    profile["revenue_share"] = (profile["total_revenue"] / profile["total_revenue"].sum()).round(3)

    logger.info("Segmented %d customers into %d groups (silhouette %.3f)", len(df), k, sil)
    return SegmentationResult(
        customers=df,
        profile=profile.reset_index().set_index("segment"),
        k=k,
        silhouette=sil,
        stability_ari=stability_ari,
    )
