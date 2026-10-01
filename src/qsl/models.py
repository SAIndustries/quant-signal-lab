"""Model zoo. Everything follows the scikit-learn ``fit``/``predict`` interface.

Scalers live *inside* each pipeline, so during walk-forward validation they are
re-fit on each training window only (the notebook this repo replaces fit a
MinMaxScaler on the full series, leaking test-period information).

To plug in your own model (e.g. a PyTorch LSTM), wrap it in a class with
``fit(X, y)`` and ``predict(X)`` and add it to ``get_models``.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class HistoricalMean(BaseEstimator, RegressorMixin):
    """Predicts the training-window mean return. The standard benchmark for OOS R^2.

    Because its sign is almost always positive, its backtest is effectively
    "always long, volatility-targeted" - the baseline a signal has to beat.
    """

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def get_models(seed: int = 0) -> dict:
    return {
        "hist_mean": HistoricalMean(),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=500.0)),
        "gbm": HistGradientBoostingRegressor(
            max_depth=3,
            learning_rate=0.03,
            max_iter=150,
            min_samples_leaf=50,
            l2_regularization=1.0,
            random_state=seed,
        ),
        "mlp": TransformedTargetRegressor(
            regressor=make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(32, 16),
                    alpha=1e-2,
                    early_stopping=True,
                    validation_fraction=0.15,
                    max_iter=200,
                    random_state=seed,
                ),
            ),
            transformer=StandardScaler(),
        ),
    }
