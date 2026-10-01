"""Calibration check: how often does the pipeline 'find' skill in pure noise?

Runs the ridge walk-forward on many independent random-walk series (no predictable
component) and reports the fraction of runs where the IC / hit-rate tests reject at
5% and 1%. A well-calibrated test rejects ~5% and ~1% respectively.

    python scripts/null_calibration.py --runs 60
"""
import argparse
import warnings

import numpy as np

from qsl.data import simulate_prices
from qsl.features import build_dataset
from qsl.metrics import hit_rate, information_coefficient
from qsl.models import get_models
from qsl.validation import walk_forward_predict

warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=int, default=60)
ap.add_argument("--days", type=int, default=2500)
args = ap.parse_args()

ridge = get_models(0)["ridge"]
ic_p, hit_p = [], []
for s in range(args.runs):
    ds = build_dataset(simulate_prices(n=args.days, seed=100 + s))
    pred = walk_forward_predict(ridge, ds.X, ds.y, min_train=756, step=126)
    m = pred.notna()
    ic_p.append(information_coefficient(ds.y[m], pred[m])[1])
    hit_p.append(hit_rate(ds.y[m], pred[m])["hit_p"])

for name, ps in (("IC test", np.array(ic_p)), ("Hit-rate test", np.array(hit_p))):
    print(f"{name}: reject@5% = {np.mean(ps < 0.05):.1%}   reject@1% = {np.mean(ps < 0.01):.1%}   (n={len(ps)})")
