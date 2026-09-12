# -*- coding: utf-8 -*-
"""commitment 引擎的信息集消融（用于论文的信息价值 V6/V12/V18）。

用法（四个更新集合可并行）::

    python ablation_commitment.py --nodes 0
    python ablation_commitment.py --nodes 0,1
    python ablation_commitment.py --nodes 0,1,2
    python ablation_commitment.py --nodes 0,1,2,3

每个配置写出 ``runs/commitment_nodes<n>.json``，便于合并比较。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data_loader import day_index_of, load_actual, load_price, load_pv_forecast  # noqa: E402
from commitment import (  # noqa: E402
    CausalFourierLoadModel,
    causal_load_error_pool,
    run_day_commitment,
)
from scenarios import pv_error_pool  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", default="0,1,2,3")
    ap.add_argument("--start", default="2025-02-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--scenarios", type=int, default=5)
    ap.add_argument("--lookback", type=int, default=90)
    ap.add_argument("--settlement", default="sequential",
                    choices=["sequential", "final_net"])
    args = ap.parse_args()
    nodes = tuple(int(x) for x in args.nodes.split(","))

    dates, load, pv = load_actual()
    _, fc = load_pv_forecast()
    price = load_price()
    model = CausalFourierLoadModel(dates, load)
    load_err = causal_load_error_pool(model, dates, load)
    pv_err = tuple(pv_error_pool(pv, fc, i, method="linear") for i in range(4))

    start = day_index_of(dates, args.start)
    end = day_index_of(dates, args.end)
    t0 = time.perf_counter()
    soc_carry = 6000.0
    tot = dict(planned_cost=0.0, reduce_penalty=0.0, increase_cost=0.0,
               emergency_cost=0.0, total_cost=0.0, plan_kwh=0.0, adjusted_kwh=0.0,
               emergency_kwh=0.0, final_net_total=0.0)
    for day in range(start, end + 1):
        load_fc, _ = model.predict(str(dates[day])[:10])
        res = run_day_commitment(
            day=day, dates=dates, load_actual=load, pv_actual=pv, pv_fc_day=fc[day],
            price=price, load_fc=load_fc, load_err=load_err, pv_err_pools=pv_err,
            n_scenarios=args.scenarios, lookback=args.lookback, nodes=nodes,
            stochastic=True, s_init=soc_carry, settlement=args.settlement,
        )
        for k in ("planned_cost", "reduce_penalty", "increase_cost", "emergency_cost",
                  "total_cost"):
            tot[k] += res[k]
        tot["plan_kwh"] += res["plan_kwh"]
        tot["adjusted_kwh"] += res["adjusted_kwh"]
        tot["emergency_kwh"] += res["emergency_kwh"]
        tot["final_net_total"] += res["final_net_total_cost"]
        soc_carry = float(res["soc"][-1])

    out = dict(engine="commitment", nodes=list(nodes), start=args.start, end=args.end,
               n_days=end - start + 1, n_scenarios=args.scenarios,
               lookback_days=args.lookback, settlement=args.settlement,
               elapsed_s=time.perf_counter() - t0, terminal_soc_kwh=soc_carry, **tot)
    runs = HERE / "runs"
    runs.mkdir(exist_ok=True)
    tag = "".join(str(n) for n in nodes)
    path = runs / f"commitment_nodes{tag}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in
                      ("nodes", "total_cost", "emergency_kwh", "plan_kwh",
                       "adjusted_kwh", "terminal_soc_kwh", "elapsed_s")},
                     ensure_ascii=False))
    print(f"-> {path.name}")


if __name__ == "__main__":
    main()
