"""Command-line entry point: run the full walk-forward experiment and write a report.

Examples
--------
    qsl                                   # default ETF universe, real data (needs internet)
    qsl --tickers SPY AAPL --step 126     # custom universe, faster
    qsl --synthetic null                  # sanity check: random walks must show no edge
    qsl --synthetic ar                    # sanity check: planted signal must be detected
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402

from .backtest import BacktestConfig, buy_and_hold, run_backtest  # noqa: E402
from .data import load_prices, simulate_prices  # noqa: E402
from .features import build_dataset  # noqa: E402
from .metrics import bh_adjust, hit_rate, information_coefficient, oos_r2, performance_summary  # noqa: E402
from .models import get_models  # noqa: E402
from .validation import walk_forward_predict  # noqa: E402

DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "XLF", "XLE", "XLK", "XLV", "XLI", "TLT", "GLD"]
BASELINE = "hist_mean"
COST_GRID = [0, 1, 2, 5, 10]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--models", nargs="+", default=["ridge", "gbm", "mlp"], choices=["ridge", "gbm", "mlp"])
    p.add_argument("--min-train", type=int, default=756, help="days of history before first prediction")
    p.add_argument("--step", type=int, default=63, help="days predicted per refit")
    p.add_argument("--window", type=int, default=None, help="rolling train window (default: expanding)")
    p.add_argument("--cost-bps", type=float, default=2.0, help="one-way trading cost in bps")
    p.add_argument("--target-vol", type=float, default=0.10, help="annualised vol target per instrument")
    p.add_argument("--max-leverage", type=float, default=2.0)
    p.add_argument("--long-only", action="store_true")
    p.add_argument("--synthetic", choices=["null", "ar"], default=None, help="use simulated data instead of Yahoo")
    p.add_argument("--n-synthetic", type=int, default=4, help="number of synthetic assets")
    p.add_argument("--n-synthetic-days", type=int, default=3000)
    p.add_argument("--ar-coef", type=float, default=0.15, help="planted return autocorrelation for --synthetic ar")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")
    p.add_argument("--refresh", action="store_true", help="re-download data instead of using cache")
    return p.parse_args(argv)


def load_universe(args) -> dict[str, pd.DataFrame]:
    if args.synthetic:
        ar = 0.0 if args.synthetic == "null" else args.ar_coef
        return {
            f"SYN{i}": simulate_prices(n=args.n_synthetic_days, seed=args.seed + i, ar_coef=ar)
            for i in range(args.n_synthetic)
        }
    universe = {}
    for t in args.tickers:
        try:
            universe[t] = load_prices(t, start=args.start, end=args.end, refresh=args.refresh)
        except Exception as exc:  # network failure, delisted ticker, ...
            print(f"[warn] skipping {t}: {exc}")
    if not universe:
        raise SystemExit("No data could be loaded. Check your connection or try --synthetic null.")
    return universe


def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) and pd.notna(v) else ("" if pd.isna(v) else str(v)) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def equal_weight(frames: dict[str, pd.Series]) -> pd.Series:
    return pd.DataFrame(frames).mean(axis=1, skipna=True)


def run(args) -> dict[str, pd.DataFrame]:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    universe = load_universe(args)
    zoo = get_models(args.seed)
    names = [BASELINE] + [m for m in args.models if m != BASELINE]
    cfg = BacktestConfig(
        target_vol=args.target_vol, max_leverage=args.max_leverage, cost_bps=args.cost_bps, long_only=args.long_only
    )
    split_kw = dict(min_train=args.min_train, step=args.step, window=args.window, embargo=1)

    rows, datasets, oos_preds = [], {}, {}
    nets: dict[str, dict[str, pd.Series]] = {n: {} for n in names + ["buy_and_hold"]}
    turns: dict[str, dict[str, pd.Series]] = {n: {} for n in names}

    for ticker, df in universe.items():
        print(f"[{ticker}] building dataset ({len(df)} rows)")
        ds = build_dataset(df)
        datasets[ticker] = ds
        preds = {}
        for name in names:
            print(f"[{ticker}] walk-forward: {name}")
            preds[name] = walk_forward_predict(zoo[name], ds.X, ds.y, **split_kw)
        mask = preds[BASELINE].notna()
        y, base = ds.y[mask], preds[BASELINE][mask]

        for name in names:
            p = preds[name][mask]
            oos_preds[(ticker, name)] = p
            bt = run_backtest(p, ds.ret_next, ds.sigma, cfg)
            ic, ic_p = information_coefficient(y, p)
            row = {
                "ticker": ticker,
                "model": name,
                "oos_r2": oos_r2(y, p, base),
                "ic": ic,
                "ic_p": ic_p,
                **hit_rate(y, p),
                **performance_summary(bt["net"], bt["turnover"], cfg.periods),
            }
            rows.append(row)
            nets[name][ticker] = bt["net"]
            turns[name][ticker] = bt["turnover"]

        bh = buy_and_hold(ds.ret_next, preds[BASELINE][mask].index, cfg)
        nets["buy_and_hold"][ticker] = bh
        rows.append({"ticker": ticker, "model": "buy_and_hold", **performance_summary(bh, None, cfg.periods)})

    assets = pd.DataFrame(rows)
    is_model = ~assets["model"].isin([BASELINE, "buy_and_hold"])
    assets["ic_p_adj"] = np.nan
    assets["hit_p_adj"] = np.nan
    assets.loc[is_model, "ic_p_adj"] = bh_adjust(assets.loc[is_model, "ic_p"]).to_numpy()
    assets.loc[is_model, "hit_p_adj"] = bh_adjust(assets.loc[is_model, "hit_p"]).to_numpy()

    # Equal-weight portfolio across instruments
    port_rows, port_net = [], {}
    for name in names + ["buy_and_hold"]:
        net = equal_weight(nets[name])
        port_net[name] = net
        turnover = equal_weight(turns[name]) if name in turns else None
        port_rows.append({"model": name, **performance_summary(net, turnover, cfg.periods)})
    portfolio = pd.DataFrame(port_rows)

    # Cost sensitivity of the equal-weight portfolio
    sens = {}
    for name in names:
        col = {}
        for c in COST_GRID:
            c_cfg = replace(cfg, cost_bps=float(c))
            per = {t: run_backtest(oos_preds[(t, name)], datasets[t].ret_next, datasets[t].sigma, c_cfg)["net"] for t in universe}
            col[c] = performance_summary(equal_weight(per), None, cfg.periods)["sharpe"]
        sens[name] = col
    sensitivity = pd.DataFrame(sens).rename_axis("cost_bps").reset_index()

    # ---- artifacts -------------------------------------------------------------
    assets.to_csv(out_dir / "per_asset.csv", index=False)
    portfolio.to_csv(out_dir / "portfolio.csv", index=False)
    sensitivity.to_csv(out_dir / "cost_sensitivity.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, net in port_net.items():
        ax.plot((1 + net.fillna(0)).cumprod(), label=name, lw=2 if name == "buy_and_hold" else 1.3,
                ls="--" if name == "buy_and_hold" else "-")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title(f"Equal-weight portfolio, out-of-sample, net of {args.cost_bps:g} bps costs")
    ax.set_ylabel("Growth of 1 (log scale)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "portfolio_equity.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in names:
        ax.plot(sensitivity["cost_bps"], sensitivity[name], marker="o", label=name)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("One-way cost (bps)")
    ax.set_ylabel("Net Sharpe (equal-weight portfolio)")
    ax.set_title("How fast does transaction cost erase the edge?")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cost_sensitivity.png", dpi=140)
    plt.close(fig)

    first_oos = min(s.index.min() for s in port_net.values())
    last_oos = max(s.index.max() for s in port_net.values())
    header = "> **SYNTHETIC DATA** - sanity-check run, not market results.\n\n" if args.synthetic else ""
    cols_port = ["model", "ann_return", "ann_vol", "sharpe", "sharpe_lo", "sharpe_hi", "max_drawdown", "avg_daily_turnover"]
    cols_asset = ["ticker", "model", "oos_r2", "ic", "ic_p_adj", "hit_rate", "up_day_rate", "hit_p_adj", "sharpe", "max_drawdown"]
    summary = (
        f"{header}# Results\n\nOut-of-sample window: {first_oos.date()} to {last_oos.date()}  \n"
        f"Instruments: {', '.join(universe)}  \nCost: {args.cost_bps:g} bps one-way; vol target {args.target_vol:.0%}; "
        f"refit every {args.step} days, {'rolling ' + str(args.window) if args.window else 'expanding'} window.\n\n"
        f"## Equal-weight portfolio (net of costs)\n\n{to_markdown(portfolio[cols_port])}\n\n"
        f"## Net Sharpe vs. transaction cost (bps)\n\n{to_markdown(sensitivity, '{:.2f}')}\n\n"
        f"## Per-instrument predictive tests\n\n`ic_p_adj` / `hit_p_adj` are Benjamini-Hochberg adjusted across all "
        f"instrument x model tests.\n\n{to_markdown(assets[cols_asset])}\n"
    )
    (out_dir / "summary.md").write_text(summary)

    meta = {
        "args": vars(args),
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "versions": {m.__name__: m.__version__ for m in (np, pd, matplotlib)},
        "data_ranges": {t: [str(d.index.min().date()), str(d.index.max().date())] for t, d in universe.items()},
    }
    (out_dir / "run_config.json").write_text(json.dumps(meta, indent=2, default=str))

    print("\n" + summary)
    return {"assets": assets, "portfolio": portfolio, "sensitivity": sensitivity}


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
