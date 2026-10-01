> **SYNTHETIC DATA** - sanity-check run, not market results.

# Results

Out-of-sample window: 2015-02-20 to 2023-06-29  
Instruments: SYN0, SYN1, SYN2, SYN3  
Cost: 2 bps one-way; vol target 10%; refit every 63 days, expanding window.

## Equal-weight portfolio (net of costs)

| model | ann_return | ann_vol | sharpe | sharpe_lo | sharpe_hi | max_drawdown | avg_daily_turnover |
|---|---|---|---|---|---|---|---|
| hist_mean | 0.014 | 0.051 | 0.302 | -0.364 | 0.969 | -0.094 | 0.019 |
| ridge | -0.034 | 0.052 | -0.641 | -1.307 | 0.026 | -0.282 | 0.216 |
| gbm | -0.026 | 0.051 | -0.491 | -1.158 | 0.175 | -0.233 | 0.355 |
| mlp | -0.047 | 0.051 | -0.910 | -1.577 | -0.243 | -0.347 | 0.396 |
| buy_and_hold | 0.005 | 0.095 | 0.099 | -0.567 | 0.766 | -0.190 |  |

## Net Sharpe vs. transaction cost (bps)

| cost_bps | hist_mean | ridge | gbm | mlp |
|---|---|---|---|---|
| 0.00 | 0.32 | -0.43 | -0.14 | -0.52 |
| 1.00 | 0.31 | -0.54 | -0.32 | -0.72 |
| 2.00 | 0.30 | -0.64 | -0.49 | -0.91 |
| 5.00 | 0.27 | -0.96 | -1.01 | -1.49 |
| 10.00 | 0.23 | -1.48 | -1.88 | -2.45 |

## Per-instrument predictive tests

`ic_p_adj` / `hit_p_adj` are Benjamini-Hochberg adjusted across all instrument x model tests.

| ticker | model | oos_r2 | ic | ic_p_adj | hit_rate | up_day_rate | hit_p_adj | sharpe | max_drawdown |
|---|---|---|---|---|---|---|---|---|---|
| SYN0 | hist_mean | 0.000 | -0.025 |  | 0.522 | 0.478 |  | 0.523 | -0.293 |
| SYN0 | ridge | -0.010 | -0.068 | 0.017 | 0.496 | 0.478 | 0.737 | -0.370 | -0.487 |
| SYN0 | gbm | -0.031 | -0.012 | 0.866 | 0.514 | 0.478 | 0.823 | 0.264 | -0.234 |
| SYN0 | mlp | -0.035 | -0.024 | 0.633 | 0.504 | 0.478 | 0.983 | -0.220 | -0.377 |
| SYN0 | buy_and_hold |  |  |  |  |  |  | -0.480 | -0.707 |
| SYN1 | hist_mean | 0.000 | 0.000 |  | 0.494 | 0.506 |  | -0.501 | -0.411 |
| SYN1 | ridge | -0.002 | -0.001 | 0.991 | 0.493 | 0.506 | 0.891 | -0.487 | -0.419 |
| SYN1 | gbm | -0.021 | -0.011 | 0.866 | 0.491 | 0.506 | 0.823 | -0.475 | -0.525 |
| SYN1 | mlp | -0.051 | -0.000 | 0.991 | 0.507 | 0.506 | 0.823 | -0.631 | -0.502 |
| SYN1 | buy_and_hold |  |  |  |  |  |  | 0.452 | -0.339 |
| SYN2 | hist_mean | 0.000 | 0.004 |  | 0.498 | 0.502 |  | 0.230 | -0.279 |
| SYN2 | ridge | -0.010 | 0.004 | 0.991 | 0.506 | 0.502 | 0.823 | 0.001 | -0.264 |
| SYN2 | gbm | -0.036 | -0.010 | 0.866 | 0.498 | 0.502 | 0.983 | -0.414 | -0.406 |
| SYN2 | mlp | -0.045 | -0.041 | 0.330 | 0.485 | 0.502 | 0.737 | -0.605 | -0.495 |
| SYN2 | buy_and_hold |  |  |  |  |  |  | -0.068 | -0.677 |
| SYN3 | hist_mean | 0.000 | -0.022 |  | 0.513 | 0.513 |  | 0.344 | -0.270 |
| SYN3 | ridge | -0.008 | -0.027 | 0.619 | 0.497 | 0.513 | 0.823 | -0.429 | -0.444 |
| SYN3 | gbm | -0.027 | -0.010 | 0.866 | 0.498 | 0.513 | 0.937 | -0.346 | -0.362 |
| SYN3 | mlp | -0.049 | -0.030 | 0.619 | 0.484 | 0.513 | 0.737 | -0.354 | -0.454 |
| SYN3 | buy_and_hold |  |  |  |  |  |  | 0.294 | -0.500 |
