import numpy as np
import pandas as pd

from qsl.backtest import BacktestConfig, positions_from_signal, run_backtest


def _toy(n=50, sigma=0.01):
    idx = pd.bdate_range("2020-01-01", periods=n)
    return (
        pd.Series(np.where(np.arange(n) % 2 == 0, 1.0, -1.0) * 0.001, index=idx),  # flips every day
        pd.Series(0.0, index=idx),  # zero market return
        pd.Series(sigma, index=idx),
    )


def test_zero_cost_zero_return_gives_zero_pnl():
    pred, ret, sigma = _toy()
    bt = run_backtest(pred, ret, sigma, BacktestConfig(cost_bps=0.0))
    assert np.allclose(bt["net"], 0.0)


def test_costs_reduce_net_returns_and_scale_with_turnover():
    pred, ret, sigma = _toy()
    cheap = run_backtest(pred, ret, sigma, BacktestConfig(cost_bps=1.0))["net"].sum()
    dear = run_backtest(pred, ret, sigma, BacktestConfig(cost_bps=5.0))["net"].sum()
    assert dear < cheap < 0


def test_position_size_is_inverse_to_volatility_and_capped():
    pred, _, _ = _toy()
    cfg = BacktestConfig(target_vol=0.10, max_leverage=2.0)
    low = positions_from_signal(pred, pd.Series(0.005, index=pred.index), cfg).abs()
    high = positions_from_signal(pred, pd.Series(0.020, index=pred.index), cfg).abs()
    assert (low > high).all()
    assert (low <= 2.0 + 1e-12).all()


def test_position_uses_only_current_information():
    """Changing tomorrow's return must not change today's position."""
    pred, ret, sigma = _toy()
    a = run_backtest(pred, ret, sigma, BacktestConfig())["pos"]
    ret2 = ret.copy()
    ret2.iloc[-1] = 0.5
    b = run_backtest(pred, ret2, sigma, BacktestConfig())["pos"]
    pd.testing.assert_series_equal(a, b)


def test_long_only_never_shorts():
    pred, ret, sigma = _toy()
    pos = run_backtest(pred, ret, sigma, BacktestConfig(long_only=True))["pos"]
    assert (pos >= 0).all()
