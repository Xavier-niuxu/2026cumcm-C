# -*- coding: utf-8 -*-
"""main_rolling implementation."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Implementation detail.
# Implementation detail.
# Implementation detail.
_Q3_DIR = Path(__file__).resolve().parents[1] / "question3"
sys.path.insert(0, str(_Q3_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from commitment import (  # noqa: E402
    CausalFourierLoadModel,
    causal_load_error_pool,
    run_day_commitment,
)
from data_loader import day_index_of, load_actual, load_pv_forecast  # noqa: E402
from pred.price_fourier import PriceFourierPredictor  # noqa: E402
from scenarios import pv_error_pool  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRICE_FILE = PROJECT_ROOT / "附件" / "附件4.xlsx"
OUTPUT_FILE = PROJECT_ROOT / "附件" / "附件5" / "result4-3.xlsx"

KEEP_DAYS = 6
TIME_BLOCKS = [
    "0:00-4:00", "4:00-8:00", "8:00-12:00",
    "12:00-16:00", "16:00-20:00", "20:00-24:00",
]
SOC_TIMES = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"]


def slot_label(t: int) -> str:
    sh, sm = divmod(t * 10, 60)
    eh, em = divmod((t + 1) * 10, 60)
    return f"{sh}:{sm:02d}-0:00+1" if eh == 24 else f"{sh}:{sm:02d}-{eh}:{em:02d}"


def load_price_series() -> tuple:
    """load_price_series implementation."""
    df = pd.read_excel(PRICE_FILE, sheet_name="Sheet1")
    dates = pd.DatetimeIndex(pd.to_datetime(df.iloc[:, 0]).dt.normalize())
    price = df.iloc[:, 1:].to_numpy(dtype=float)
    return dates, price


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-02-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--scenarios", type=int, default=5)
    parser.add_argument("--lookback", type=int, default=90)
    parser.add_argument("--nodes", default="0,1,2,3")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--pv-method", default="linear", choices=["linear", "step"])
    parser.add_argument("--settlement", default="sequential",
                        choices=["sequential", "final_net"],
                        help="逐次调整结算（正文口径）或只按最终净调整结算（敏感性）")
    parser.add_argument("--event-threshold", type=float, default=0.0)
    parser.add_argument("--daily-reset", action="store_true",
                        help="每天 0:00 强制 SOC 回到 6000（默认跨日连续）")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    nodes = tuple(int(x) for x in args.nodes.split(","))

    print("Loading attachments ...")
    dates, load, pv = load_actual()
    _, fc = load_pv_forecast()
    dates_price, price_data = load_price_series()

    print("Building causal load forecast model and error pool ...")
    model = CausalFourierLoadModel(dates, load)
    load_err = causal_load_error_pool(model, dates, load)
    pv_err = tuple(pv_error_pool(pv, fc, i, method=args.pv_method) for i in range(4))

    print("Building price forecast model (附件4) ...")
    price_model = PriceFourierPredictor(dates_price, price_data)
    price_row = {d: i for i, d in enumerate(dates_price)}

    start = day_index_of(dates, args.start)
    end = day_index_of(dates, args.end)
    days = list(range(start, end + 1))
    print(f"Running {len(days)} days ({dates[start].date()} .. {dates[end].date()}), "
          f"nodes={nodes}, scenarios={1 if args.deterministic else args.scenarios}")

    template = pd.read_excel(OUTPUT_FILE, sheet_name="计划购电量", nrows=1)
    date_col = template.columns[0]
    time_cols = list(template.columns[1:145])

    rows_plan, rows_adj, rows_block, rows_emg = [], [], [], []
    records = []
    t0 = time.perf_counter()
    soc_carry = 6000.0

    for i, day in enumerate(days):
        date_str = str(dates[day])[:10]
        load_fc, _ = model.predict(date_str)
        price_pred = price_model.predict(date_str)
        price_actual = price_data[price_row[pd.Timestamp(date_str)]]

        res = run_day_commitment(
            day=day,
            dates=dates,
            load_actual=load,
            pv_actual=pv,
            pv_fc_day=fc[day],
            price=price_pred,
            load_fc=load_fc,
            load_err=load_err,
            pv_err_pools=pv_err,
            n_scenarios=args.scenarios,
            lookback=args.lookback,
            nodes=nodes,
            stochastic=not args.deterministic,
            pv_method=args.pv_method,
            s_init=soc_carry,
            settlement=args.settlement,
            event_threshold=args.event_threshold,
            settlement_price=price_actual,
        )
        soc_carry = 6000.0 if args.daily_reset else float(res["soc"][-1])
        records.append(res)

        plan_row = {date_col: res["date"]}
        plan_row.update({c: v for c, v in zip(time_cols, res["plan"])})
        plan_row["全天购电量"] = res["plan_kwh"]
        plan_row["全天购电费"] = res["planned_cost"]
        rows_plan.append(plan_row)

        adj_row = {date_col: res["date"]}
        adj_row.update({c: v for c, v in zip(time_cols, res["adjusted"])})
        adj_row["全天购电量"] = res["adjusted_kwh"]
        adj_row["全天购电费"] = res["total_cost"]
        rows_adj.append(adj_row)

        for b in range(KEEP_DAYS):
            lo, hi = b * 24, (b + 1) * 24
            rows_block.append({
                "日期": res["date"] if b == 0 else np.nan,
                "时间段": TIME_BLOCKS[b],
                "充电量": float(res["charge"][lo:hi].sum()),
                "放电量": float(res["discharge"][lo:hi].sum()),
                "时刻": SOC_TIMES[b],
                "储电量": float(res["soc"][lo]),
            })
        rows_block.append({
            "日期": np.nan, "时间段": np.nan, "充电量": np.nan, "放电量": np.nan,
            "时刻": SOC_TIMES[-1], "储电量": float(res["soc"][-1]),
        })

        for t in range(144):
            if res["emergency"][t] > 1e-7:
                rows_emg.append({
                    "日期": res["date"],
                    "购电时间段": slot_label(t),
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
    print(f"Commitment rolling MPC (波动电价): {len(days)} days in {elapsed / 60:.1f} min")
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

    if args.no_write:
        print("\n--no-write: result4-3.xlsx not written")
        return

    df_plan = pd.DataFrame(rows_plan, columns=list(template.columns))
    df_adj = pd.DataFrame(rows_adj, columns=list(template.columns))
    df_block = pd.DataFrame(rows_block,
                            columns=["日期", "时间段", "充电量", "放电量", "时刻", "储电量"])
    df_emg = pd.DataFrame(rows_emg, columns=["日期", "购电时间段", "购电量"])

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        df_plan.to_excel(writer, sheet_name="计划购电量", index=False)
        df_adj.to_excel(writer, sheet_name="调整购电量", index=False)
        df_block.to_excel(writer, sheet_name="充放电量", index=False)
        df_emg.to_excel(writer, sheet_name="紧急购电量", index=False)
    print(f"\n结果已写入 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
