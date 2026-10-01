"""End-to-end checks of the *evaluation machinery* on data where the truth is known.

* Null control: on random-walk data a sound pipeline must find no edge.
* Positive control: with planted return autocorrelation it must find it.
"""
import numpy as np

from qsl.data import simulate_prices
from qsl.features import build_dataset
from qsl.metrics import information_coefficient, oos_r2
from qsl.models import get_models
from qsl.validation import walk_forward_predict

KW = dict(min_train=756, step=126)


def _evaluate(ar_coef, seed):
    ds = build_dataset(simulate_prices(n=3000, seed=seed, ar_coef=ar_coef))
    zoo = get_models(0)
    p = walk_forward_predict(zoo["ridge"], ds.X, ds.y, **KW)
    base = walk_forward_predict(zoo["hist_mean"], ds.X, ds.y, **KW)
    m = p.notna()
    ic, ic_p = information_coefficient(ds.y[m], p[m])
    return oos_r2(ds.y[m], p[m], base[m]), ic, ic_p


def test_null_control_finds_no_edge():
    r2s = [_evaluate(0.0, seed)[0] for seed in range(4)]
    assert np.mean(r2s) < 0.005  # no systematic out-of-sample skill


def test_positive_control_detects_planted_signal():
    results = [_evaluate(0.15, seed) for seed in range(3)]
    assert all(ic > 0.05 for _, ic, _ in results)
    assert all(p < 0.01 for _, _, p in results)
