# -*- coding: utf-8 -*-
"""Backtest and compare the day-ahead predictors of Question 2.

Runs every predictor in a rolling fashion over 2025-02-01 .. 2025-12-31:
each day is planned with the predictor's forecast, then scored against the
actual load/PV with the real settlement rule (emergency power = 5x price).

    python compare_predictors.py            # main table
    python compare_predictors.py --margin   # + safety-margin sweep
    python compare_predictors.py --quantile # + Fourier quantile sweep

All predictors are scored on the same quantity, the net load N = L - P,
because the day-ahead LP and the emergency settlement only depend on it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cost import calculate_cost  # noqa: E402
from data_loader import (  # noqa: E402
    load_fixed_price,
    load_historical_load,
    load_historical_pv,
)
from prediction import (  # noqa: E402
    FourierPredictor,
    FourierIntradayPredictor,
    FourierQuantilePredictor,
    LastWeekPredictor,
    LinearFitPredictor,
    WeightedAveragePredictor,
)

START_DATE = "2025-02-01"
END_DATE = "2025-12-31"
EMERGENCY_MULTIPLIER = 5.0


class ScalingPredictor:
    """Wrap a (load, pv) predictor with a multiplicative safety margin."""

    def __init__(self, base, load_scale=1.0, pv_scale=1.0):
        self.base = base
        self.load_scale = load_scale
        self.pv_scale = pv_scale

    def predict(self, date_str):
        load_pred, pv_pred = self.base.predict(date_str)
        return load_pred * self.load_scale, pv_pred * self.pv_scale


def evaluate(predictor, dates, load, pv, price, verbose=False):
    """Return a dict of accuracy + cost metrics for one predictor."""
    net_actual = load - pv
    forecast_err, actuals = [], []
    daily_cost, daily_mae, stamps = [], [], []
    plan_kwh = emergency_kwh = normal_cost = emergency_cost = 0.0
    t0 = time.time()

    for i, date_str in enumerate(dates):
        load_pred, pv_pred = predictor.predict(date_str)
        load_pred = np.asarray(load_pred, dtype=float)
        pv_pred = np.asarray(pv_pred, dtype=float)

        err = (load_pred - pv_pred) - net_actual[i]
        forecast_err.append(err)
        actuals.append(net_actual[i])
        daily_mae.append(np.abs(err).mean())

        res = calculate_cost(price, load_pred, pv_pred, load[i], pv[i])
        plan_kwh += res.grid_purchase.sum()
        emergency_kwh += res.emergency_purchase.sum()
        normal_cost += res.normal_cost
        emergency_cost += res.emergency_cost
        daily_cost.append(res.total_cost)
        stamps.append(pd.Timestamp(date_str))

    err = np.concatenate(forecast_err)
    actual = np.concatenate(actuals)
    total_cost = normal_cost + emergency_cost
    metrics = {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "bias": float(err.mean()),
        "mape": float(np.mean(np.abs(err) / np.maximum(np.abs(actual), 1e-9)) * 100),
        "plan_kwh": plan_kwh,
        "emergency_kwh": emergency_kwh,
        "normal_cost": normal_cost,
        "emergency_cost": emergency_cost,
        "total_cost": total_cost,
        "emergency_share": emergency_cost / total_cost * 100,
        "daily_cost": np.array(daily_cost),
        "daily_mae": np.array(daily_mae),
        "stamps": pd.DatetimeIndex(stamps),
        "seconds": time.time() - t0,
    }
    if verbose:
        print(f"    (done in {metrics['seconds']:.1f}s)")
    return metrics


def build_predictors(dates_load, load_data, dates_pv, pv_data):
    def base(cls):
        return cls(dates_load, load_data, dates_pv, pv_data)

    return {
        "LastWeek (t-7)": base(LastWeekPredictor),
        "WeightedAverage 4:3:2:1": base(WeightedAveragePredictor),
        "LinearFit (4 same weekdays)": base(LinearFitPredictor),
        "Fourier (12 features)": base(FourierPredictor),
        "Fourier+Intraday (K=6)": FourierIntradayPredictor(
            dates_load, load_data, dates_pv, pv_data, n_intraday=6
        ),
    }


def print_table(results):
    print(f"\n{'model':<28}{'MAE':>9}{'RMSE':>9}{'bias':>9}{'MAPE':>8}"
          f"{'plan 万kWh':>12}{'emg 万kWh':>11}{'total 万元':>12}{'emg%':>7}")
    print("-" * 105)
    for name, m in results.items():
        print(f"{name:<28}{m['mae']:>9.2f}{m['rmse']:>9.2f}{m['bias']:>+9.2f}"
              f"{m['mape']:>7.2f}%{m['plan_kwh'] / 1e4:>12.4f}"
              f"{m['emergency_kwh'] / 1e4:>11.4f}{m['total_cost'] / 1e4:>12.4f}"
              f"{m['emergency_share']:>6.1f}%")


def paired(results, ref_name, other_name):
    ref = results[ref_name]["daily_cost"]
    oth = results[other_name]["daily_cost"]
    d = ref - oth
    se = d.std(ddof=1) / np.sqrt(d.size)
    print(f"  {ref_name} vs {other_name}: mean diff={d.mean():+9.2f} 元/day  "
          f"t={d.mean() / se:+6.2f}  "
          f"(first cheaper on {int((d < 0).sum())}/{d.size} days, "
          f"total {d.sum():+,.2f} 元)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin", action="store_true",
                        help="sweep multiplicative safety margins")
    parser.add_argument("--quantile", action="store_true",
                        help="sweep Fourier newsboy quantiles")
    parser.add_argument("--monthly", action="store_true",
                        help="print monthly net-load MAE")
    args = parser.parse_args()

    print("Loading attachment 2 ...")
    dates_load, load_data = load_historical_load()
    dates_pv, pv_data = load_historical_pv()
    price = load_fixed_price()

    stamps = pd.DatetimeIndex([pd.Timestamp(d) for d in dates_load])
    mask = (stamps >= START_DATE) & (stamps <= END_DATE)
    dates = [stamps[i].strftime("%Y-%m-%d") for i in np.where(mask)[0]]
    load = load_data[mask]
    pv = pv_data[mask]
    print(f"{len(dates)} days: {dates[0]} .. {dates[-1]}\n")

    results = {}
    for name, predictor in build_predictors(dates_load, load_data, dates_pv, pv_data).items():
        print(f"  running {name} ...")
        results[name] = evaluate(predictor, dates, load, pv, price, verbose=True)

    if args.quantile:
        for q in (0.60, 0.75, 5 / 6, 0.90):
            name = f"Fourier + q{q:.3f} margin"
            print(f"  running {name} ...")
            pred = FourierQuantilePredictor(dates_load, load_data, dates_pv, pv_data, quantile=q)
            results[name] = evaluate(pred, dates, load, pv, price, verbose=True)

    if args.margin:
        for scale in (1.02, 1.05, 1.10):
            for base_name, cls in (
                ("LastWeek", LastWeekPredictor),
                ("WeightedAverage", WeightedAveragePredictor),
            ):
                name = f"{base_name} x{scale:.2f} margin"
                print(f"  running {name} ...")
                pred = ScalingPredictor(
                    cls(dates_load, load_data, dates_pv, pv_data), scale, 2.0 - scale
                )
                results[name] = evaluate(pred, dates, load, pv, price, verbose=True)

    print_table(results)

    print("\npaired daily-cost comparisons (negative = first model cheaper):")
    ref = "Fourier (12 features)"
    for name in results:
        if name != ref:
            paired(results, ref, name)

    if args.monthly:
        print("\nmonthly net-load MAE (kW):")
        months = results[ref]["stamps"].month
        print(f"{'month':<8}" + "".join(f"{n[:20]:>22}" for n in results))
        for m in range(2, 13):
            sel = months == m
            if not sel.any():
                continue
            print(f"2025-{m:02d}  ", end="")
            for name in results:
                mae = results[name]["daily_mae"][sel].mean()
                print(f"{mae:>22.2f}", end="")
            print()


if __name__ == "__main__":
    main()
