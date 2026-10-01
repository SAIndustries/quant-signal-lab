"""Feature and target construction.

Timing convention (the single most important rule in this repo):

* Row ``t`` holds features computed **only from data up to the close of day t**.
* The target on row ``t`` is the log return from close ``t`` to close ``t+1``.
* A position decided on row ``t`` therefore earns the return on row ``t``'s target.

No feature may look forward. ``tests/test_features.py`` enforces this by checking
that features computed on a truncated history equal those computed on the full one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    X: pd.DataFrame  # features known at close of t
    y: pd.Series  # next-day log return (t -> t+1)
    sigma: pd.Series  # EWMA daily volatility known at close of t (for sizing)

    @property
    def ret_next(self) -> pd.Series:
        """Next-day *simple* return, i.e. what a position taken at t actually earns."""
        return np.expm1(self.y).rename("ret_next")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].astype(float)
    volume = df["volume"].astype(float).where(lambda v: v > 0)
    r = np.log(close).diff()

    f = pd.DataFrame(index=df.index)
    f["ret_1"] = r
    for w in (5, 10, 21, 63):
        f[f"ret_{w}"] = r.rolling(w).sum()
    for w in (10, 21, 63):
        f[f"vol_{w}"] = r.rolling(w).std()
    f["mom_21_vol_adj"] = f["ret_21"] / (f["vol_21"] * np.sqrt(21))
    f["ma_gap_50"] = np.log(close / close.rolling(50).mean()) / f["vol_63"]
    f["vol_ratio"] = f["vol_10"] / f["vol_63"]
    f["volume_z"] = np.log(volume / volume.rolling(21).mean())
    return f.replace([np.inf, -np.inf], np.nan)


def build_dataset(df: pd.DataFrame, sigma_span: int = 30) -> Dataset:
    close = df["close"].astype(float)
    r = np.log(close).diff()
    X = build_features(df)
    y = r.shift(-1).rename("target")  # label: the *next* day's return
    sigma = r.ewm(span=sigma_span, min_periods=sigma_span).std().rename("sigma")
    data = pd.concat([X, y, sigma], axis=1).dropna()
    return Dataset(X=data[X.columns], y=data["target"], sigma=data["sigma"])
