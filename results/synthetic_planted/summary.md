> **SYNTHETIC DATA** - sanity-check run, not market results.

# Results

Out-of-sample window: 2015-02-20 to 2023-06-29  
Instruments: SYN0, SYN1, SYN2, SYN3  
Cost: 2 bps one-way; vol target 10%; refit every 126 days, expanding window.

## Equal-weight portfolio (net of costs)

| model | ann_return | ann_vol | sharpe | sharpe_lo | sharpe_hi | max_drawdown | avg_daily_turnover |
|---|---|---|---|---|---|---|---|
| hist_mean | 0.018 | 0.052 | 0.373 | -0.294 | 1.039 | -0.103 | 0.019 |
| ridge | 0.126 | 0.052 | 2.303 | 1.633 | 2.973 | -0.046 | 0.465 |
| gbm | 0.095 | 0.053 | 1.746 | 1.077 | 2.414 | -0.062 | 0.447 |
| mlp | 0.065 | 0.052 | 1.232 | 0.564 | 1.899 | -0.093 | 0.386 |
| buy_and_hold | 0.004 | 0.096 | 0.086 | -0.580 | 0.753 | -0.220 |  |

## Net Sharpe vs. transaction cost (bps)

| cost_bps | hist_mean | ridge | gbm | mlp |
|---|---|---|---|---|
| 0.00 | 0.39 | 2.75 | 2.18 | 1.60 |
| 1.00 | 0.38 | 2.53 | 1.96 | 1.42 |
| 2.00 | 0.37 | 2.30 | 1.75 | 1.23 |
| 5.00 | 0.35 | 1.63 | 1.10 | 0.67 |
| 10.00 | 0.30 | 0.50 | 0.03 | -0.26 |

## Per-instrument predictive tests

`ic_p_adj` / `hit_p_adj` are Benjamini-Hochberg adjusted across all instrument x model tests.

| ticker | model | oos_r2 | ic | ic_p_adj | hit_rate | up_day_rate | hit_p_adj | sharpe | max_drawdown |
|---|---|---|---|---|---|---|---|---|---|
| SYN0 | hist_mean | 0.000 | -0.000 |  | 0.522 | 0.478 |  | 0.643 | -0.329 |
| SYN0 | ridge | 0.011 | 0.088 | 0.000 | 0.520 | 0.478 | 0.161 | 0.827 | -0.198 |
| SYN0 | gbm | -0.014 | 0.059 | 0.007 | 0.520 | 0.478 | 0.161 | 0.598 | -0.290 |
| SYN0 | mlp | -0.021 | 0.033 | 0.133 | 0.501 | 0.478 | 0.830 | 0.121 | -0.347 |
| SYN0 | buy_and_hold |  |  |  |  |  |  | -0.570 | -0.763 |
| SYN1 | hist_mean | 0.000 | -0.004 |  | 0.497 | 0.503 |  | -0.559 | -0.446 |
| SYN1 | ridge | 0.024 | 0.126 | 0.000 | 0.529 | 0.503 | 0.013 | 0.887 | -0.155 |
| SYN1 | gbm | -0.006 | 0.065 | 0.004 | 0.519 | 0.503 | 0.108 | 0.766 | -0.143 |
| SYN1 | mlp | -0.015 | 0.085 | 0.000 | 0.527 | 0.503 | 0.023 | 0.644 | -0.257 |
| SYN1 | buy_and_hold |  |  |  |  |  |  | 0.510 | -0.382 |
| SYN2 | hist_mean | 0.000 | -0.007 |  | 0.498 | 0.502 |  | 0.278 | -0.308 |
| SYN2 | ridge | -0.000 | 0.089 | 0.000 | 0.538 | 0.502 | 0.001 | 1.218 | -0.150 |
| SYN2 | gbm | -0.024 | 0.061 | 0.006 | 0.541 | 0.502 | 0.000 | 0.927 | -0.115 |
| SYN2 | mlp | -0.040 | 0.018 | 0.395 | 0.510 | 0.502 | 0.390 | 0.529 | -0.252 |
| SYN2 | buy_and_hold |  |  |  |  |  |  | -0.099 | -0.731 |
| SYN3 | hist_mean | 0.000 | -0.031 |  | 0.511 | 0.511 |  | 0.375 | -0.309 |
| SYN3 | ridge | 0.019 | 0.144 | 0.000 | 0.557 | 0.511 | 0.000 | 1.692 | -0.102 |
| SYN3 | gbm | -0.003 | 0.099 | 0.000 | 0.548 | 0.511 | 0.000 | 1.247 | -0.155 |
| SYN3 | mlp | -0.019 | 0.065 | 0.004 | 0.527 | 0.511 | 0.032 | 1.184 | -0.174 |
| SYN3 | buy_and_hold |  |  |  |  |  |  | 0.326 | -0.553 |
