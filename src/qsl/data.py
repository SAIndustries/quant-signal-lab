"""Data access: Yahoo Finance download with on-disk caching, plus synthetic generators.

The synthetic generators exist so the *evaluation machinery itself* can be tested:
on a pure random walk a sound pipeline must find no edge, and on data with a planted
weak autocorrelation it should find it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path("data/cache")


def load_prices(
    ticker: str,
    start: str = "2010-01-01",
    end: str | None = None,
    cache_dir: str | Path = CACHE_DIR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame indexed by date with columns ``close`` and ``volume``.

    Prices are split- and dividend-adjusted (``auto_adjust=True``), so log returns
    are total returns. Downloads are cached as CSV under ``cache_dir``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{ticker}_{start}_{end or 'latest'}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)

    import yfinance as yf  # imported lazily so tests/synthetic runs work offline

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError(f"No data returned for {ticker!r}")
    if isinstance(raw.columns, pd.MultiIndex):  # newer yfinance returns (field, ticker)
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Close", "Volume"]].rename(columns=str.lower).dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    df.to_csv(path)
    return df


def simulate_prices(
    n: int = 3000,
    seed: int = 0,
    ar_coef: float = 0.0,
    daily_vol: float = 0.012,
    start: str = "2012-01-02",
) -> pd.DataFrame:
    """Simulate a price/volume series with GARCH(1,1)-style volatility clustering.

    ``ar_coef`` plants first-order autocorrelation in returns
    (r_t = ar_coef * r_{t-1} + sigma_t * z_t). With ``ar_coef=0`` the series has no
    predictable component in the mean, so any apparent edge is a false positive.
    """
    rng = np.random.default_rng(seed)
    alpha, beta = 0.08, 0.90
    omega = daily_vol**2 * (1 - alpha - beta)
    z = rng.standard_normal(n)
    r = np.zeros(n)
    var = daily_vol**2
    for t in range(1, n):
        var = omega + alpha * (r[t - 1] - ar_coef * (r[t - 2] if t > 1 else 0.0)) ** 2 + beta * var
        r[t] = ar_coef * r[t - 1] + np.sqrt(var) * z[t]
    idx = pd.bdate_range(start, periods=n, name="date")
    close = 100.0 * np.exp(np.cumsum(r))
    volume = np.exp(15 + 0.3 * rng.standard_normal(n) + 5 * np.abs(r))
    return pd.DataFrame({"close": close, "volume": volume}, index=idx)
