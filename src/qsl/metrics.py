"""Statistical and performance metrics.

Prediction-quality metrics (do the forecasts contain information?):
    oos_r2, hit_rate, information_coefficient
Performance metrics (does a traded signal make money after costs?):
    sharpe_stats, max_drawdown, performance_summary
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def oos_r2(y, pred, baseline) -> float:
    """Out-of-sample R^2 versus a benchmark forecast (Campbell & Thompson, 2008).

    Positive means the model beats the benchmark (here: the expanding historical
    mean). Daily-return R^2 is tiny even for good signals (~0.1% is notable).
    """
    y, pred, baseline = (np.asarray(a, dtype=float) for a in (y, pred, baseline))
    sse_model = np.sum((y - pred) ** 2)
    sse_base = np.sum((y - baseline) ** 2)
    return float(1.0 - sse_model / sse_base)


def hit_rate(y, pred) -> dict:
    """Directional accuracy with a test against the right null.

    Comparing to 50% would reward a model that always predicts "up" simply because
    markets drift upward. Under independence between prediction and outcome the
    expected hit rate is ``q*b + (1-q)*(1-b)`` where ``q`` is the fraction of "up"
    predictions and ``b`` the fraction of up days; the binomial test uses that null.
    """
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    mask = (y != 0) & (pred != 0)
    y, pred = y[mask], pred[mask]
    n = int(len(y))
    if n == 0:
        return {"hit_rate": np.nan, "up_day_rate": np.nan, "hit_p": np.nan, "n": 0}
    hits = int(np.sum(np.sign(y) == np.sign(pred)))
    q, b = float(np.mean(pred > 0)), float(np.mean(y > 0))
    null = q * b + (1 - q) * (1 - b)
    p = float(stats.binomtest(hits, n, null).pvalue)
    return {"hit_rate": hits / n, "up_day_rate": b, "hit_p": p, "n": n}


def information_coefficient(y, pred) -> tuple[float, float]:
    """Spearman rank correlation between prediction and realised return, with p-value."""
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    if np.std(pred) == 0 or len(y) < 3:
        return np.nan, np.nan
    rho, p = stats.spearmanr(pred, y)
    return float(rho), float(p)


def sharpe_stats(returns: pd.Series, periods: int = 252) -> dict:
    """Annualised Sharpe (risk-free ignored) with a 95% CI (Lo, 2002, i.i.d. approx)."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 3 or np.std(r, ddof=1) == 0:
        return {"sharpe": np.nan, "sharpe_lo": np.nan, "sharpe_hi": np.nan}
    sr_d = np.mean(r) / np.std(r, ddof=1)
    se_d = np.sqrt((1 + 0.5 * sr_d**2) / n)
    k = np.sqrt(periods)
    return {
        "sharpe": sr_d * k,
        "sharpe_lo": (sr_d - 1.96 * se_d) * k,
        "sharpe_hi": (sr_d + 1.96 * se_d) * k,
    }


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + pd.Series(returns).fillna(0.0)).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def performance_summary(returns: pd.Series, turnover: pd.Series | None = None, periods: int = 252) -> dict:
    r = pd.Series(returns).dropna()
    n = len(r)
    equity_end = float((1.0 + r).prod())
    out = {
        "ann_return": equity_end ** (periods / n) - 1.0 if n and equity_end > 0 else np.nan,
        "ann_vol": float(r.std(ddof=1) * np.sqrt(periods)) if n > 1 else np.nan,
        **sharpe_stats(r, periods),
        "max_drawdown": max_drawdown(r),
        "n_days": n,
    }
    if turnover is not None:
        out["avg_daily_turnover"] = float(pd.Series(turnover).dropna().mean())
    return out


def bh_adjust(pvalues: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR adjustment, ignoring NaNs (many tickers x models = many tests)."""
    p = pd.Series(pvalues, dtype=float)
    mask = p.notna()
    out = pd.Series(np.nan, index=p.index)
    if mask.sum():
        out[mask] = stats.false_discovery_control(p[mask].to_numpy(), method="bh")
    return out
