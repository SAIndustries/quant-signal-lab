import numpy as np
import pandas as pd

from qsl.data import simulate_prices
from qsl.features import build_dataset, build_features


def test_features_have_no_lookahead():
    """Features at day t must be identical whether or not future data exists."""
    df = simulate_prices(n=600, seed=1)
    full = build_features(df)
    for t in (150, 300, 450):
        truncated = build_features(df.iloc[: t + 1])
        pd.testing.assert_series_equal(full.iloc[t], truncated.iloc[t], check_names=False)


def test_target_is_next_day_return():
    df = simulate_prices(n=400, seed=2)
    ds = build_dataset(df)
    log_ret = np.log(df["close"]).diff()
    t = ds.y.index[10]
    next_day = df.index[df.index.get_loc(t) + 1]
    assert np.isclose(ds.y.loc[t], log_ret.loc[next_day])


def test_no_nans_in_dataset():
    ds = build_dataset(simulate_prices(n=500, seed=3))
    assert not ds.X.isna().any().any()
    assert not ds.y.isna().any()
    assert not ds.sigma.isna().any()
