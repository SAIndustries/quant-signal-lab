> **SYNTHETIC DATA** - sanity-check run, not market results.

# Results

Out-of-sample window: 2015-02-20 to 2023-06-29  
Instruments: SYN0, SYN1, SYN2, SYN3  
Cost: 2 bps one-way; vol target 10%; refit every 63 days, expanding window.

## Equal-weight portfolio (net of costs)

| model | ann_return | ann_vol | sharpe | sharpe_lo | sharpe_hi | max_drawdown | avg_daily_turnover |
|---|---|---|---|---|---|---|---|
| hist_mean | 0.018 | 0.052 | 0.373 | -0.294 | 1.039 | -0.103 | 0.019 |
| ridge | 0.131 | 0.052 | 2.406 | 1.735 | 3.076 | -0.048 | 0.467 |
| gbm | 0.081 | 0.052 | 1.525 | 0.857 | 2.193 | -0.062 | 0.454 |
| mlp | 0.069 | 0.053 | 1.283 | 0.616 | 1.951 | -0.068 | 0.387 |
| buy_and_hold | 0.004 | 0.096 | 0.086 | -0.580 | 0.753 | -0.220 |  |

## Net Sharpe vs. transaction cost (bps)

| cost_bps | hist_mean | ridge | gbm | mlp |
|---|---|---|---|---|
| 0.00 | 0.39 | 2.86 | 1.96 | 1.65 |
| 1.00 | 0.38 | 2.63 | 1.74 | 1.47 |
| 2.00 | 0.37 | 2.41 | 1.52 | 1.28 |
| 5.00 | 0.35 | 1.72 | 0.87 | 0.73 |
| 10.00 | 0.30 | 0.59 | -0.23 | -0.18 |

## Per-instrument predictive tests

`ic_p_adj` / `hit_p_adj` are Benjamini-Hochberg adjusted across all instrument x model tests.

| ticker | model | oos_r2 | ic | ic_p_adj | hit_rate | up_day_rate | hit_p_adj | sharpe | max_drawdown |
|---|---|---|---|---|---|---|---|---|---|
| SYN0 | hist_mean | 0.000 | -0.026 |  | 0.522 | 0.478 |  | 0.643 | -0.329 |
| SYN0 | ridge | 0.011 | 0.088 | 0.000 | 0.521 | 0.478 | 0.138 | 0.840 | -0.191 |
| SYN0 | gbm | -0.014 | 0.063 | 0.005 | 0.524 | 0.478 | 0.077 | 0.793 | -0.198 |
| SYN0 | mlp | -0.018 | 0.045 | 0.038 | 0.516 | 0.478 | 0.290 | 0.528 | -0.261 |
| SYN0 | buy_and_hold |  |  |  |  |  |  | -0.570 | -0.763 |
| SYN1 | hist_mean | 0.000 | 0.002 |  | 0.497 | 0.503 |  | -0.559 | -0.446 |
| SYN1 | ridge | 0.025 | 0.128 | 0.000 | 0.530 | 0.503 | 0.008 | 0.852 | -0.168 |
| SYN1 | gbm | -0.004 | 0.064 | 0.004 | 0.513 | 0.503 | 0.276 | 0.238 | -0.197 |
| SYN1 | mlp | -0.011 | 0.074 | 0.001 | 0.531 | 0.503 | 0.008 | 0.638 | -0.280 |
| SYN1 | buy_and_hold |  |  |  |  |  |  | 0.510 | -0.382 |
| SYN2 | hist_mean | 0.000 | 0.005 |  | 0.498 | 0.502 |  | 0.278 | -0.308 |
| SYN2 | ridge | 0.001 | 0.091 | 0.000 | 0.539 | 0.502 | 0.001 | 1.317 | -0.146 |
| SYN2 | gbm | -0.028 | 0.056 | 0.011 | 0.533 | 0.502 | 0.005 | 0.593 | -0.200 |
| SYN2 | mlp | -0.041 | 0.007 | 0.733 | 0.506 | 0.502 | 0.592 | 0.457 | -0.346 |
| SYN2 | buy_and_hold |  |  |  |  |  |  | -0.099 | -0.731 |
| SYN3 | hist_mean | 0.000 | -0.023 |  | 0.511 | 0.511 |  | 0.375 | -0.309 |
| SYN3 | ridge | 0.019 | 0.145 | 0.000 | 0.559 | 0.511 | 0.000 | 1.815 | -0.096 |
| SYN3 | gbm | -0.000 | 0.106 | 0.000 | 0.550 | 0.511 | 0.000 | 1.444 | -0.114 |
| SYN3 | mlp | -0.022 | 0.065 | 0.004 | 0.526 | 0.511 | 0.035 | 1.008 | -0.144 |
| SYN3 | buy_and_hold |  |  |  |  |  |  | 0.326 | -0.553 |
