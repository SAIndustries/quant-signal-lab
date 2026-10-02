# quant-signal-lab

**Leakage-safe walk-forward evaluation and cost-aware backtesting of return-prediction models.**

Most "stock prediction" projects plot a model's predicted price over the real price, see the lines overlap, and report a high accuracy. That overlap is mostly an artifact: prices are highly persistent, so even "tomorrow = today" looks like a good forecast. This repo takes the opposite approach. It asks the question a trading desk asks: **does a model's forecast contain information about *future returns*, and does acting on it make money after costs, out of sample, with proper statistical uncertainty?**

It was built to replace an earlier LSTM price-prediction notebook that had these flaws (price-level target, scaler fit on the full series, single 80/20 split, no baseline, no backtest, mislabeled RMSE).

## What it does

```
prices ──► features (known at close of t) ──► walk-forward models ──► out-of-sample predictions
                                                                           │
            ┌──────────────────────────────────────────────────────────────┤
            ▼                                                              ▼
 prediction quality                                          vol-targeted, cost-aware backtest
 OOS R² vs historical mean                                   Sharpe (with 95% CI), max drawdown,
 rank IC (+ p-value)                                         turnover, cost sensitivity,
 hit rate vs correct null                                    vs. buy-and-hold
 BH-adjusted across all tests
```

- **Target:** next-day log return (not price level).
- **Models:** expanding-mean baseline, ridge, gradient boosting, small MLP (any scikit-learn-style model plugs in).
- **Universe (default):** liquid ETFs across equity indices, sectors, rates and gold (`SPY QQQ IWM XLF XLE XLK XLV XLI TLT GLD`).
- **Validation:** walk-forward, retrained every 63 trading days on an expanding window, first prediction after 3 years of history.
- **Backtest:** sign of the forecast sets direction, volatility targeting sets size, trading costs are charged on turnover.

## Design decisions that matter

| Risk in naive pipelines | How it is handled here |
|---|---|
| Look-ahead in features | Every feature at day *t* uses data up to the close of *t* only. A unit test checks that features computed on truncated history equal those on the full history. |
| Scaler/preprocessing leakage | Scalers live inside each model's pipeline and are re-fit on each training window only. |
| Train/test overlap | Walk-forward splits with an embargo; a test uses a spy model that fails if it is asked to predict inside its training window. |
| Misleading metric (price overlap, "accuracy") | Target is returns. Skill is measured by OOS R² vs the historical-mean forecast, rank IC, and directional accuracy. |
| Drift inflating hit rate | Hit-rate p-value uses the null expected under independence (`q·b + (1−q)(1−b)`), not 50%, so an always-long model is not credited for the market going up. |
| No baseline | Historical mean (≈ always long, vol-targeted) and unlevered buy-and-hold are reported next to every model. |
| Costs ignored | Costs are charged per unit of turnover; the report includes a Sharpe-vs-cost sensitivity table (0 to 10 bps). |
| Backtest overfitting / luck | Sharpe is reported with a 95% confidence interval; p-values are Benjamini-Hochberg adjusted across all instrument × model tests. Hyperparameters are fixed up front, not tuned on the test window. |
| Position sizing ignores risk | Positions scale with `target_vol / forecast_vol` (capped), so risk per trade is roughly constant across regimes. |

## Quick start

```bash
git clone https://github.com/SAIndustries/quant-signal-lab.git
cd quant-signal-lab
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"

python -m pytest -q                         # 21 tests, a few seconds
python -m qsl.cli --synthetic null          # offline sanity check (no edge expected)
python -m qsl.cli --out results/real        # real data via Yahoo Finance (needs internet)
```

Requires Python 3.10+. The `qsl` command is installed as a shortcut for `python -m qsl.cli`.

Useful options: `--tickers SPY AAPL`, `--models ridge gbm`, `--step 126` (faster), `--window 1260` (rolling instead of expanding), `--cost-bps 5`, `--long-only`. Run `python -m qsl.cli --help` for all of them. Outputs go to the folder given by `--out` (default `results/`): `summary.md`, `portfolio.csv`, `per_asset.csv`, `cost_sensitivity.csv`, two PNG charts and `run_config.json` (arguments, library versions and data ranges for reproducibility).

## Results

### 1. Real-market results

Real-market results are not included yet. To generate them (about 10-15 minutes with the default settings, internet required):

```bash
python -m qsl.cli --out results/real
```

This writes `results/real/summary.md`, the portfolio and per-instrument tables, and the equity and cost-sensitivity charts. Read them against the baselines (`hist_mean`, `buy_and_hold`) and the confidence intervals, not the raw Sharpe alone.

### 2. Sanity checks on synthetic data (control experiments)

These do **not** say anything about markets. They verify that the evaluation machinery tells the truth when the answer is known. Reproduce with `python -m qsl.cli --synthetic null --out results/synthetic_null` and `python -m qsl.cli --synthetic ar --out results/synthetic_planted` (4 assets, ~8 years out-of-sample, 2 bps cost, seed 0). Full outputs are in those two folders. Exact figures can vary slightly across library versions; each folder's `run_config.json` records the versions used.

**Null control: random walks with volatility clustering, no predictable component.**
No model should show an edge, and none does:

- Net of 2 bps, the equal-weight Sharpe is negative for all three models (ridge −0.64, gbm −0.49, mlp −0.91). The MLP's 95% confidence interval lies entirely below zero (−1.58 to −0.24): a model trading noise does not merely break even, it pays costs.
- Out-of-sample R² versus the historical mean is negative in all 12 model-asset combinations.
- One of 12 rank-IC tests fell below 5% after multiple-testing adjustment (ridge on one asset), and its IC was negative, which is the wrong direction for a real edge and in line with what chance produces at that many tests.
- The always-long baseline (`hist_mean`) shows a Sharpe of +0.30, but its confidence interval (−0.36 to 0.97) spans zero, so that is noise too.

**Positive control: same data with planted lag-1 return autocorrelation of 0.15** (far stronger than in real markets, chosen so it is clearly detectable).
The pipeline recovers it. Rank IC is positive for every model on every asset, and 11 of the 12 model-asset tests are significant at 5% after multiple-testing adjustment (the exception is the MLP on one asset). At 2 bps the equal-weight net Sharpe is 2.41 for ridge (95% CI 1.74 to 3.08), 1.52 for gbm and 1.28 for mlp, against 0.09 for buy-and-hold. The same signal is worth less as costs rise:

| Net Sharpe vs. cost | 0 bps | 2 bps | 5 bps | 10 bps |
|---|---|---|---|---|
| ridge | 2.86 | 2.41 | 1.72 | 0.59 |
| gbm | 1.96 | 1.52 | 0.87 | −0.23 |
| mlp | 1.65 | 1.28 | 0.73 | −0.18 |

Two takeaways: the simplest model (ridge) extracts the planted signal best, and a strong gross edge can be mostly or entirely consumed by costs, which is why the backtest charges for turnover.

**Calibration of the significance tests** (`python scripts/null_calibration.py --runs 60`; one run, ridge model): on 60 independent random-walk series the IC and hit-rate tests reject at the 5% level in 6.7% of runs and at the 1% level in 1.7% of runs, in line with nominal rates (sampling error on 60 runs is about ±3 points at 5%).

## Project layout

```
src/qsl/
  data.py        Yahoo Finance download + caching; synthetic data generators
  features.py    Feature/target construction with the timing convention documented
  models.py      Baseline + ridge / gradient boosting / MLP (scikit-learn interface)
  validation.py  Walk-forward splits and out-of-sample prediction
  metrics.py     OOS R², rank IC, hit rate, Sharpe + CI, drawdown, BH adjustment
  backtest.py    Vol-targeted, cost-aware backtest and buy-and-hold benchmark
  cli.py         Experiment runner and report writer
tests/           21 tests: look-ahead, leakage, metrics, backtest logic, null/positive controls
scripts/         Null-calibration experiment
results/         Synthetic control runs (synthetic_null/, synthetic_planted/); real-data runs go in results/real/
```

## Limitations (read before trusting any number)

- **Daily bars, closing-price fills.** Real execution happens at prices you cannot trade exactly at the close; costs are a flat bps assumption with no market-impact or capacity model.
- **No financing, borrow or short-sale costs, and risk-free rate ignored** in the Sharpe ratio.
- **Sharpe confidence intervals assume i.i.d. returns** (Lo, 2002), which understates uncertainty when returns are autocorrelated or heteroskedastic.
- **Selection effects.** The default universe is today's liquid ETFs, which is less survivorship-biased than hand-picked stocks but still not a point-in-time universe. Trying several models and settings and reporting the best one overstates performance; this repo reports all of them and adjusts p-values across them, but does not implement a deflated Sharpe ratio.
- **Weak by construction.** Daily return predictability in liquid markets is tiny. Expect modest IC and OOS R² near zero; a result of "no edge after costs" is plausible and should be reported as such.
- **Not investment advice.** This is a research and education project.

## Extending it

- Wrap a PyTorch/TensorFlow LSTM in a class with `fit(X, y)` / `predict(X)` and add it to `get_models` in `models.py`. It is evaluated by exactly the same walk-forward, metrics and backtest code.
- Add features (cross-asset signals, macro, options-implied vol) in `features.py`. Keep the timing rule: nothing after the close of day *t*.
- Add a deflated Sharpe ratio or a stationary bootstrap for dependence-robust confidence intervals.
- Extend to multi-day horizons (raise `embargo` to at least the horizon).

## License

MIT. See `LICENSE`.