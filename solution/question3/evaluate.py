# -*- coding: utf-8 -*-
"""消融实验：不同预报时刻组合 / 随机 vs 确定性 / CVaR。

每次运行只跑一种策略（``--only``），便于并行执行后汇总::

    python evaluate.py --only plan_only
    python evaluate.py --only nodes01
    ...
    python evaluate.py --summary        # 汇总已保存的 json

结果写入 ``runs/<name>.json``。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import day_index_of, load_actual, load_price, load_pv_forecast  # noqa: E402
from rolling import run_day  # noqa: E402
from scenarios import load_error_pool, load_forecast_model, pv_error_pool  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = Path(__file__).resolve().parent / "runs"

STRATEGIES = {
    # battery_mode="realtime" + 跨日 SOC 连续 = 参考解的实际执行口径
    "plan_only_det": dict(nodes=(0,), stochastic=False),
    "plan_only": dict(nodes=(0,), stochastic=True),
    "nodes01": dict(nodes=(0, 1), stochastic=True),
    "nodes012": dict(nodes=(0, 1, 2), stochastic=True),
    "nodes0123": dict(nodes=(0, 1, 2, 3), stochastic=True),
    "nodes0123_det": dict(nodes=(0, 1, 2, 3), stochastic=False),
    "nodes0123_cvar": dict(nodes=(0, 1, 2, 3), stochastic=True, cvar_weight=0.3),
    "nodes0123_committed": dict(nodes=(0, 1, 2, 3), stochastic=True, battery_mode="committed",
                                continuous_soc=False),
}
REPORT_DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, choices=sorted(STRATEGIES))
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--start", default="2025-02-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--scenarios", type=int, default=12)
    parser.add_argument("--lookback", type=int, default=90)
    parser.add_argument("--sample", type=int, default=1, help="每隔 sample 天取一天")
    args = parser.parse_args()

    if args.summary:
        print(f"{'strategy':<18}{'days':>6}{'计划万kWh':>12}{'调整万kWh':>12}"
              f"{'紧急万kWh':>12}{'总费用万元':>12}{'元/天':>10}")
        rows = []
        for name in STRATEGIES:
            path = RUNS_DIR / f"{name}.json"
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as fh:
                r = json.load(fh)
            rows.append(r)
            print(f"{name:<18}{r['n_days']:>6}{r['plan_kwh'] / 1e4:>12.4f}"
                  f"{r['adjusted_kwh'] / 1e4:>12.4f}{r['emergency_kwh'] / 1e4:>12.4f}"
                  f"{r['total_cost'] / 1e4:>12.4f}{r['total_cost'] / r['n_days']:>10.2f}")
        if rows:
            base = next((r for r in rows if r["name"] == "plan_only"), None)
            if base:
                print(f"\n相对 plan_only 的节省（万元）:")
                for r in rows:
                    print(f"  {r['name']:<18}{(base['total_cost'] - r['total_cost']) / 1e4:>10.4f}"
                          f"   ({(base['total_cost'] - r['total_cost']) / base['total_cost'] * 100:>6.2f}%)")
        return

    if args.only is None:
        parser.error("--only or --summary is required")

    name = args.only
    cfg = STRATEGIES[name]
    print(f"Loading data for strategy {name} ...")
    dates, load, pv = load_actual()
    _, fc = load_pv_forecast()
    price = load_price()
    model = load_forecast_model(dates, load)
    load_err = load_error_pool(model, dates, load)
    pv_err = tuple(pv_error_pool(pv, fc, i, method="linear") for i in range(4))

    start = day_index_of(dates, args.start)
    end = day_index_of(dates, args.end)
    days = list(range(start, end + 1))[:: args.sample]
    print(f"{len(days)} days, scenarios={args.scenarios}, "
          f"nodes={cfg['nodes']}, stochastic={cfg['stochastic']}")

    t0 = time.perf_counter()
    totals = dict(plan_kwh=0.0, adjusted_kwh=0.0, emergency_kwh=0.0,
                  planned_cost=0.0, reduce_penalty=0.0, increase_cost=0.0,
                  emergency_cost=0.0, total_cost=0.0)
    per_date = {}
    soc_carry = 6000.0
    for i, day in enumerate(days):
        load_fc, _ = model.predict(str(dates[day])[:10])
        cfg_run = dict(cfg)
        battery_mode = cfg_run.pop("battery_mode", "realtime")
        continuous = cfg_run.pop("continuous_soc", True)
        res = run_day(
            day=day, dates=dates, load_actual=load, pv_actual=pv, pv_fc_day=fc[day],
            price=price, load_fc=load_fc, load_err=load_err, pv_err_pools=pv_err,
            n_scenarios=args.scenarios, lookback=args.lookback,
            battery_mode=battery_mode, s_init=soc_carry, **cfg_run,
        )
        soc_carry = float(res["soc"][-1]) if continuous else 6000.0
        for key in totals:
            totals[key] += res[key]
        if res["date"] in REPORT_DATES:
            per_date[res["date"]] = {
                "plan_kwh": res["plan_kwh"], "adjusted_kwh": res["adjusted_kwh"],
                "emergency_kwh": res["emergency_kwh"], "total_cost": res["total_cost"],
            }
        if (i + 1) % 20 == 0:
            rate = (time.perf_counter() - t0) / (i + 1)
            print(f"  {i + 1}/{len(days)}  {rate:.2f}s/day  "
                  f"eta {rate * (len(days) - i - 1) / 60:.1f}min")

    elapsed = time.perf_counter() - t0
    out = dict(name=name, n_days=len(days), sample=args.sample,
               scenarios=args.scenarios, nodes=list(cfg["nodes"]),
               stochastic=cfg["stochastic"], seconds=elapsed,
               per_date=per_date, **totals)
    RUNS_DIR.mkdir(exist_ok=True)
    with (RUNS_DIR / f"{name}.json").open("w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print(f"\n[{name}] {len(days)} days in {elapsed / 60:.1f} min")
    print(f"  计划 {totals['plan_kwh'] / 1e4:.4f} 万kWh | 调整 "
          f"{totals['adjusted_kwh'] / 1e4:.4f} 万kWh | 紧急 "
          f"{totals['emergency_kwh'] / 1e4:.4f} 万kWh")
    print(f"  总费用 {totals['total_cost'] / 1e4:.4f} 万元  "
          f"(计划 {totals['planned_cost'] / 1e4:.2f} + 调减 "
          f"{totals['reduce_penalty'] / 1e4:.2f} + 调增 {totals['increase_cost'] / 1e4:.2f}"
          f" + 紧急 {totals['emergency_cost'] / 1e4:.2f})")
    for d, v in per_date.items():
        print(f"  {d}: 计划 {v['plan_kwh']:.2f} 调整 {v['adjusted_kwh']:.2f} "
              f"紧急 {v['emergency_kwh']:.2f} 费用 {v['total_cost']:.2f} 元")


if __name__ == "__main__":
    main()
