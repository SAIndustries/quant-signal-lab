import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from qsl.validation import walk_forward_predict, walk_forward_splits


def test_splits_never_overlap_and_respect_embargo():
    for train, test in walk_forward_splits(1000, min_train=300, step=50, embargo=2):
        assert train.max() < test.min()
        assert test.min() - train.max() - 1 >= 2  # embargo gap
        assert len(set(train) & set(test)) == 0


def test_splits_cover_all_out_of_sample_days_exactly_once():
    tests = np.concatenate([te for _, te in walk_forward_splits(1000, min_train=300, step=70)])
    assert np.array_equal(tests, np.arange(300, 1000))


def test_rolling_window_limits_training_length():
    for train, _ in walk_forward_splits(1000, min_train=300, step=50, window=200):
        assert len(train) <= 200


class _Spy(BaseEstimator, RegressorMixin):
    """Fails loudly if it is ever asked to predict a date it could have trained on."""

    def fit(self, X, y):
        self.last_train_ = X.index.max()
        return self

    def predict(self, X):
        assert X.index.min() > self.last_train_, "test rows overlap or precede training rows"
        return np.zeros(len(X))


def test_walk_forward_never_predicts_inside_training_window():
    idx = pd.bdate_range("2015-01-01", periods=900)
    X = pd.DataFrame({"a": np.arange(900.0)}, index=idx)
    y = pd.Series(np.zeros(900), index=idx)
    preds = walk_forward_predict(_Spy(), X, y, min_train=300, step=60)
    assert preds.iloc[:300].isna().all()
    assert preds.iloc[300:].notna().all()
