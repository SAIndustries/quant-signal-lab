"""Walk-forward (expanding or rolling) validation.

At each step the model is trained on the past only and predicts the next block of
unseen days; the blocks are concatenated into one out-of-sample prediction series.
A single random or 80/20 split is not enough for time series: it tests one regime,
and random splits leak the future into training.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.base import clone


def walk_forward_splits(
    n: int,
    min_train: int = 756,
    step: int = 63,
    window: int | None = None,
    embargo: int = 1,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` integer index arrays.

    * ``min_train``: samples required before the first prediction (756 ~ 3 years).
    * ``step``: days predicted per refit (63 ~ one quarter).
    * ``window``: if set, use a rolling training window of this length, else expanding.
    * ``embargo``: samples dropped between train and test. The label is a 1-day-ahead
      return so 1 is sufficient; raise it if you extend the label horizon.
    """
    if min_train <= embargo:
        raise ValueError("min_train must exceed embargo")
    start = min_train
    while start < n:
        end = min(start + step, n)
        train_end = start - embargo
        train_start = 0 if window is None else max(0, train_end - window)
        yield np.arange(train_start, train_end), np.arange(start, end)
        start = end


def walk_forward_predict(model, X: pd.DataFrame, y: pd.Series, **split_kwargs) -> pd.Series:
    """Out-of-sample predictions (NaN before the first test block)."""
    preds = pd.Series(np.nan, index=X.index, name="pred", dtype=float)
    for train_idx, test_idx in walk_forward_splits(len(X), **split_kwargs):
        fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        preds.iloc[test_idx] = np.asarray(fitted.predict(X.iloc[test_idx])).ravel()
    return preds
