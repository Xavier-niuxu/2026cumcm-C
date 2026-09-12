# -*- coding: utf-8 -*-
"""问题 3：滚动随机 MPC 全年求解 + 写出 result3.xlsx。

用法::

    python3 main.py                       # 全年主模型（默认 5 个加权聚类场景）
    python3 main.py --engine legacy --scenarios 12  # 原两阶段实现
    python3 main.py --nodes 0,1           # 只用 0:00 与 6:00 的预报
    python3 main.py --deterministic       # 单场景（点预报）
    python3 main.py --start 2025-06-01 --end 2025-06-30   # 只跑一段时间
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import (  # noqa: E402
    FILE_TEMPLATE,
    N_SLOTS,
    day_index_of,
    load_actual,
    load_price,
    load_pv_forecast,
)
from commitment import (  # noqa: E402
    CausalFourierLoadModel,
    causal_load_error_pool,
    run_day_commitment,
)
from rolling import run_day as run_day_legacy  # noqa: E402
from scenarios import load_error_pool, load_forecast_model, pv_error_pool  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = PROJECT_ROOT / "附件" / "附件5" / "result3.xlsx"
KEEP_DAYS = 6
TIME_BLOCKS = [
    "0:00-4:00", "4:00-8:00", "8:00-12:00",
    "12:00-16:00", "16:00-20:00", "20:00-24:00",
]
SOC_TIMES = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"]
REPORT_DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]


def slot_label(t: int) -> str:
    sh, sm = divmod(t * 10, 60)
    eh, em = divmod((t + 1) * 10, 60)
    return f"{sh}:{sm:02d}-0:00+1" if eh == 24 else f"{sh}:{sm:02d}-{eh}:{em:02d}"


def time_columns() -> list:
    """Column labels for the 144 slots, taken verbatim from the result3 template.

    Note: the template labels start at "0:10-0:20" (shifted by one interval
    relative to the physical slots), which is the same convention as the
    result1/result2 deliverables, so the labels are copied as-is and the first
    value still corresponds to the 0:00-0:10 interval.
    """
    if FILE_TEMPLATE.exists():
        tmpl = pd.read_excel(FILE_TEMPLATE, sheet_name="计划购电量", nrows=1)
        cols = list(tmpl.columns)
        if len(cols) >= N_SLOTS + 1:
            return cols[1:N_SLOTS + 1]
    return [slot_label(t) for t in range(N_SLOTS)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-02-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--engine", default="commitment", choices=["commitment", "legacy"],
                        help="commitment=约1370万元的加权聚类承诺模型；legacy=原两阶段实现")
    parser.add_argument("--scenarios", type=int, default=5)
    parser.add_argument("--lookback", type=int, default=90)
    parser.add_argument("--nodes", default="0,1,2,3")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--cvar", type=float, default=0.0)
    parser.add_argument("--cvar-alpha", type=float, default=0.95)
    parser.add_argument("--pv-method", default="linear", choices=["linear", "step"])
    parser.add_argument("--battery", default="realtime", choices=["realtime", "committed"],
                        help="realtime=实际执行时由储能实时响应真实缺额（默认，参考解口径）；"
                             "committed=只执行 LP 在决策时刻定下的充放电量")
    parser.add_argument("--daily-reset", action="store_true",
                        help="每天 0:00 强制 SOC 回到 6000（默认跨日连续）")
    parser.add_argument("--settlement", default="sequential",
                        choices=["sequential", "final_net"],
                        help="逐次调整结算（正文口径）或只按最终净调整结算（敏感性）")
    parser.add_argument("--event-threshold", type=float, default=0.0)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    if args.engine == "commitment":
        import scipy
        import sklearn
        tested = ("1.17.1", "1.9.0")
        current = (scipy.__version__, sklearn.__version__)
        if current != tested:
            print("WARNING: 当前 scipy/scikit-learn=%s/%s，精确复现环境为 %s/%s；"
                  "多重最优与 K-means 局部最优可能造成约 0.1%% 的费用漂移。"
                  % (current[0], current[1], tested[0], tested[1]))

    nodes = tuple(int(x) for x in args.nodes.split(","))

    print("Loading attachments ...")
    dates, load, pv = load_actual()
    _, fc = load_pv_forecast()
    price = load_price()

    print("Building load forecast model and error pools ...")
    if args.engine == "commitment":
        model = CausalFourierLoadModel(dates, load)
        load_err = causal_load_error_pool(model, dates, load)
        runner = run_day_commitment
    else:
        model = load_forecast_model(dates, load)
        load_err = load_error_pool(model, dates, load)
        runner = run_day_legacy
    pv_err = tuple(pv_error_pool(pv, fc, i, method=args.pv_method) for i in range(4))

    start = day_index_of(dates, args.start)
    end = day_index_of(dates, args.end)
    days = list(range(start, end + 1))
    print(f"Running {len(days)} days ({dates[start].date()} .. {dates[end].date()}), "
          f"nodes={nodes}, scenarios={1 if args.deterministic else args.scenarios}")

    time_cols = time_columns()
    rows_plan, rows_adj, rows_block, rows_emg = [], [], [], []
    records = []
    t0 = time.perf_counter()
    soc_carry = 6000.0

    for i, day in enumerate(days):
        load_fc, _ = model.predict(str(dates[day])[:10])
        common = dict(
            day=day,
            dates=dates,
            load_actual=load,
            pv_actual=pv,
            pv_fc_day=fc[day],
            price=price,
            load_fc=load_fc,
            load_err=load_err,
            pv_err_pools=pv_err,
            n_scenarios=args.scenarios,
            lookback=args.lookback,
            nodes=nodes,
            stochastic=not args.deterministic,
            cvar_weight=args.cvar,
            cvar_alpha=args.cvar_alpha,
            pv_method=args.pv_method,
            battery_mode=args.battery,
            s_init=soc_carry,
        )
        if args.engine == "commitment":
            common.update(settlement=args.settlement,
                          event_threshold=args.event_threshold)
        res = runner(**common)
        if not args.daily_reset:
            soc_carry = float(res["soc"][-1])
        else:
            soc_carry = 6000.0
        records.append(res)

        date_str = res["date"]
        plan_row = {"日期": date_str}
        plan_row.update({c: v for c, v in zip(time_cols, res["plan"])})
        plan_row["全天购电量"] = res["plan_kwh"]
        plan_row["全天购电费"] = res["planned_cost"]
        rows_plan.append(plan_row)

        adj_row = {"日期": date_str}
        adj_row.update({c: v for c, v in zip(time_cols, res["adjusted"])})
        adj_row["全天购电量"] = res["adjusted_kwh"]
        adj_row["全天购电费"] = res["total_cost"]
        rows_adj.append(adj_row)

        for b in range(KEEP_DAYS):
            lo, hi = b * 24, (b + 1) * 24
            rows_block.append({
                "日期": date_str if b == 0 else np.nan,
                "时间段": TIME_BLOCKS[b],
                "充电量": float(res["charge"][lo:hi].sum()),
                "放电量": float(res["discharge"][lo:hi].sum()),
                "时刻": SOC_TIMES[b],
                "储电量": float(res["soc"][lo]),
            })
        rows_block.append({
            "日期": np.nan, "时间段": np.nan, "充电量": np.nan, "放电量": np.nan,
            "时刻": SOC_TIMES[-1], "储电量": float(res["soc"][N_SLOTS]),
        })

        for t in range(N_SLOTS):
            if res["emergency"][t] > 1e-7:
                rows_emg.append({
                    "日期": date_str, "购电时间段": slot_label(t),
                    "购电量": float(res["emergency"][t]),
                })

        if (i + 1) % 25 == 0:
            rate = (time.perf_counter() - t0) / (i + 1)
            print(f"  {i + 1}/{len(days)} days  ({rate:.2f}s/day, "
                  f"eta {rate * (len(days) - i - 1) / 60:.1f} min)")

    elapsed = time.perf_counter() - t0
    total_cost = sum(r["total_cost"] for r in records)
    total_plan = sum(r["plan_kwh"] for r in records)
    total_adj = sum(r["adjusted_kwh"] for r in records)
    total_emg = sum(r["emergency_kwh"] for r in records)
    print("\n" + "=" * 74)
    print(f"Rolling stochastic MPC [{args.engine}]: {len(days)} days in {elapsed / 60:.1f} min")
    print(f"计划购电量  {total_plan:14,.4f} kWh")
    print(f"最终调整量  {total_adj:14,.4f} kWh")
    print(f"紧急购电量  {total_emg:14,.4f} kWh")
    print(f"总费用      {total_cost:14,.2f} 元  ({total_cost / 1e4:.4f} 万元)")
    print("  = 计划购电费 %.2f + 调减违约金 %.2f + 调增购电费 %.2f + 紧急购电费 %.2f"
          % (sum(r["planned_cost"] for r in records),
             sum(r["reduce_penalty"] for r in records),
             sum(r["increase_cost"] for r in records),
             sum(r["emergency_cost"] for r in records)))
    print("=" * 74)

    print("\n指定日期（论文表 3 格式）")
    print(f"{'日期':<12}{'计划量/kWh':>14}{'最终调整量/kWh':>16}"
          f"{'紧急量/kWh':>13}{'总费用/元':>13}")
    for d in REPORT_DATES:
        hit = [r for r in records if r["date"] == d]
        if not hit:
            continue
        r = hit[0]
        print(f"{d:<12}{r['plan_kwh']:>14.4f}{r['adjusted_kwh']:>16.4f}"
              f"{r['emergency_kwh']:>13.4f}{r['total_cost']:>13.2f}")

    if args.no_write:
        print("\n--no-write: result3.xlsx not written")
        return

    df_plan = pd.DataFrame(rows_plan)
    df_adj = pd.DataFrame(rows_adj)
    df_block = pd.DataFrame(rows_block)
    df_emg = pd.DataFrame(rows_emg, columns=["日期", "购电时间段", "购电量"])

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        df_plan.to_excel(writer, sheet_name="计划购电量", index=False)
        df_adj.to_excel(writer, sheet_name="调整购电量", index=False)
        df_block.to_excel(writer, sheet_name="充放电量", index=False)
        df_emg.to_excel(writer, sheet_name="紧急购电量", index=False)
    print(f"\n结果已写入 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
