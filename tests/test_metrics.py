"""Tests for forecast error metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from coffee_intel.models.metrics import bias, mae, metrics_by_group, wape


def test_wape_perfect_forecast():
    y = [1, 2, 3, 4]
    assert wape(y, y) == 0.0


def test_wape_known_value():
    # abs errors sum = 2, actual sum = 10 -> 0.2
    assert wape([2, 4, 4], [3, 3, 4]) == 0.2


def test_bias_sign():
    assert bias([1, 1, 1], [2, 2, 2]) > 0  # over-forecast
    assert bias([2, 2, 2], [1, 1, 1]) < 0  # under-forecast


def test_mae():
    assert mae([0, 0], [1, 3]) == 2.0


def test_metrics_by_group_orders_by_volume():
    df = pd.DataFrame(
        {
            "product": ["A", "A", "B", "B"],
            "y_true": [10, 10, 1, 1],
            "y_pred": [9, 11, 1, 1],
        }
    )
    out = metrics_by_group(df, "product", "y_true", "y_pred")
    assert out.index.tolist() == ["A", "B"]
    assert np.isclose(out.loc["B", "wape"], 0.0)
