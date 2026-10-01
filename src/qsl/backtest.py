"""Cost-aware, volatility-targeted backtest of a prediction series.

Row ``t`` is a *decision date*: the position is set at the close of ``t`` using only
information available then (the prediction and EWMA volatility at ``t``), and earns
the return from ``t`` to ``t+1``. Trading costs are charged on every change in
position (turnover), in basis points of traded notional.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    target_vol: float = 0.10  # annualised volatility target per instrument
    max_leverage: float = 2.0  # cap on |position| as a multiple of capital
    cost_bps: float = 2.0  # one-way cost (spread + slippage + fees), bps of traded notional
    periods: int = 252
    long_only: bool = False


def positions_from_signal(pred: pd.Series, sigma: pd.Series, cfg: BacktestConfig) -> pd.Series:
    """Direction = sign of predicted return; size = target_vol / forecast vol (capped)."""
    direction = np.sign(pred)
    if cfg.long_only:
        direction = direction.clip(lower=0.0)
    daily_target = cfg.target_vol / np.sqrt(cfg.periods)
    size = (daily_target / sigma).clip(upper=cfg.max_leverage)
    return (direction * size).where(pred.notna())


def run_backtest(pred: pd.Series, ret_next: pd.Series, sigma: pd.Series, cfg: BacktestConfig) -> pd.DataFrame:
    """Return a frame with columns ``pos, turnover, gross, cost, net`` over the OOS window."""
    pos = positions_from_signal(pred, sigma, cfg).dropna()
    ret = ret_next.reindex(pos.index)
    turnover = pos.diff().abs()
    turnover.iloc[0] = abs(pos.iloc[0])  # opening the first position is also a trade
    gross = pos * ret
    cost = turnover * cfg.cost_bps / 1e4
    return pd.DataFrame({"pos": pos, "turnover": turnover, "gross": gross, "cost": cost, "net": gross - cost})


def buy_and_hold(ret_next: pd.Series, index: pd.Index, cfg: BacktestConfig) -> pd.Series:
    """Unlevered buy-and-hold over the same OOS dates (cost paid once, on entry)."""
    r = ret_next.reindex(index).copy()
    r.iloc[0] -= cfg.cost_bps / 1e4
    return r.rename("buy_and_hold")
