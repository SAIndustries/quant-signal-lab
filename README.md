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
git clone https://github.com/saindustries/quant-signal-lab.git
cd quant-signal-lab
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

pytest -q                                   # 21 tests, a few seconds
qsl --synthetic null                        # offline sanity check (no edge expected)
qsl                                         # real data via Yahoo Finance (needs internet)
```

Useful options: `--tickers SPY AAPL`, `--models ridge gbm`, `--step 126` (faster), `--window 1260` (rolling instead of expanding), `--cost-bps 5`, `--long-only`. Run `qsl --help` for all of them. Outputs go to `results/`: `summary.md`, `portfolio.csv`, `per_asset.csv`, `cost_sensitivity.csv`, two PNG charts and `run_config.json` (arguments, library versions and data ranges for reproducibility).

## Results

### 1. Real-market results

> **TODO (fill in after your run):** run `qsl`, then paste the "Equal-weight portfolio" and "Net Sharpe vs. transaction cost" tables from `results/summary.md` here, embed `results/portfolio_equity.png`, and write 3-4 sentences on what you found. Report it plainly, including if no model beats the baseline after costs. That is a legitimate and common result and is more credible than an unexplained high number.

### 2. Sanity checks on synthetic data (control experiments)

These do **not** say anything about markets. They verify that the evaluation machinery tells the truth when the answer is known. Reproduce with `qsl --synthetic null` and `qsl --synthetic ar` (4 assets, ~8 years out-of-sample, 2 bps cost). Full outputs are in `results/synthetic_null/` and `results/synthetic_planted/`.

**Null control: random walks with volatility clustering, no predictable component.**
No model should show an edge. None does: all three models have negative net Sharpe at 2 bps (ridge −0.66, gbm −0.44, mlp −0.65) and negative OOS R² versus the historical mean.

**Positive control: same data with planted lag-1 return autocorrelation of 0.15** (far stronger than in real markets, chosen so it is clearly detectable).
The pipeline recovers it, and shows how costs erode it:

| Net Sharpe vs. cost | 0 bps | 2 bps | 5 bps | 10 bps |
|---|---|---|---|---|
| ridge | 2.75 | 2.30 | 1.63 | 0.50 |
| gbm | 2.18 | 1.75 | 1.10 | 0.03 |
| mlp | 1.60 | 1.23 | 0.67 | −0.26 |

**Calibration of the significance tests** (`python scripts/null_calibration.py --runs 60`): on 60 independent random-walk series the IC and hit-rate tests reject at the 5% level in 6.7% of runs and at the 1% level in 1.7% of runs, in line with nominal rates (sampling error on 60 runs is about ±3 points at 5%).

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
results/         Output of the synthetic control runs (your real run writes here too)
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

