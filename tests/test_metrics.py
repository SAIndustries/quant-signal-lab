import numpy as np
import pandas as pd

from qsl.metrics import bh_adjust, hit_rate, information_coefficient, max_drawdown, oos_r2, sharpe_stats


def test_oos_r2_perfect_and_benchmark():
    y = np.array([0.01, -0.02, 0.03, -0.01])
    assert np.isclose(oos_r2(y, y, np.zeros(4)), 1.0)
    assert np.isclose(oos_r2(y, np.zeros(4), np.zeros(4)), 0.0)


def test_hit_rate_perfect_predictions():
    y = np.array([0.01, -0.02, 0.03, -0.01, 0.02, -0.03] * 20)
    out = hit_rate(y, y)
    assert out["hit_rate"] == 1.0 and out["hit_p"] < 1e-6


def test_always_up_prediction_is_not_credited_for_market_drift():
    rng = np.random.default_rng(0)
    y = rng.normal(0.001, 0.01, 5000)  # positive drift, no predictability
    out = hit_rate(y, np.full_like(y, 1e-4))
    assert out["hit_p"] > 0.99  # hit rate == base rate == null -> no evidence of skill


def test_ic_constant_prediction_is_nan():
    ic, p = information_coefficient(np.random.randn(100), np.zeros(100))
    assert np.isnan(ic) and np.isnan(p)


def test_max_drawdown():
    r = pd.Series([0.10, -0.50, 0.20])
    assert np.isclose(max_drawdown(r), -0.5)


def test_sharpe_ci_brackets_estimate():
    r = pd.Series(np.random.default_rng(1).normal(0.0005, 0.01, 2000))
    s = sharpe_stats(r)
    assert s["sharpe_lo"] < s["sharpe"] < s["sharpe_hi"]


def test_bh_adjust_is_monotone_and_ignores_nan():
    p = pd.Series([0.001, 0.04, np.nan, 0.2])
    adj = bh_adjust(p)
    assert np.isnan(adj.iloc[2])
    assert (adj.dropna() >= p.dropna()).all()
