# -*- coding: utf-8 -*-
"""Question 4 (as Question 2) main entry point: time-varying price.

The model keeps the Question 2 two-stage structure and only swaps the price
for the 附件4 time-varying curve:

* the day-ahead LP plans the purchase on the *forecast* net load and the
  *forecast* price (``PriceFourierPredictor``);
* real-time settlement bills the planned quantity at the *actual* 附件4 price,
  flexes the battery against the realised net load and buys the residual
  shortfall as emergency power at 5x the actual price.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import (
    load_historical_load,
    load_historical_pv,
    load_time_varying_price,
    get_day_data,
    get_day_price,
    get_date_index,
)
from prediction import FourierLagQuantilePredictor, BasePredictor, PriceFourierPredictor
from cost import calculate_cost, S_INIT

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = PROJECT_ROOT / "附件" / "附件5" / "result4-2.xlsx"

# Same day-ahead planning band as Question 2 (see question2/main.py): the LP
# plans from a band narrower than the physical [1200, 10800] so the battery
# keeps headroom to absorb forecast error in real time, and it is asked to end
# the day with at least S_END_MIN so the next morning starts with a reserve.
S_MAX_PLAN = 10400.0
S_END_MIN = 2500.0

# Time period labels for 4h blocks
TIME_BLOCKS = ["0:00-4:00", "4:00-8:00", "8:00-12:00",
               "12:00-16:00", "16:00-20:00", "20:00-24:00"]
SOC_TIMES = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"]


def run_question4_single_date(
    target_date: str,
    predictor: BasePredictor,
    price_predictor: PriceFourierPredictor,
    dates_load: np.ndarray,
    load_data: np.ndarray,
    dates_pv: np.ndarray,
    pv_data: np.ndarray,
    dates_price: np.ndarray,
    price_data: np.ndarray,
    s_init: float = S_INIT,
    s_max_plan: float = S_MAX_PLAN,
    s_end_min: float = S_END_MIN,
):
    """Run question 4 for a single date. Returns LPResult.

    ``s_init`` is the battery SOC at 0:00 of ``target_date``; it is the previous
    day's closing SOC so that the storage state stays continuous across days.
    The forecast price drives the day-ahead plan, the actual 附件4 price settles
    both the normal and the emergency bill.
    """
    load_pred, pv_pred = predictor.predict(target_date)
    price_pred = price_predictor.predict(target_date)
    load_actual = get_day_data(dates_load, load_data, target_date)
    pv_actual = get_day_data(dates_pv, pv_data, target_date)
    price_actual = get_day_price(dates_price, price_data, target_date)
    return calculate_cost(
        price_pred, load_pred, pv_pred, load_actual, pv_actual,
        settlement_price=price_actual,
        s_init=s_init, s_end_min=s_end_min, s_max_plan=s_max_plan,
    )


def _aggregate_4h_blocks(arr_144):
    """Sum 144 x 10-min intervals into 6 x 4-hour blocks."""
    blocks = []
    for b in range(6):
        blocks.append(arr_144[b * 24:(b + 1) * 24].sum())
    return blocks


def run_question4_full_year(predictor: BasePredictor) -> None:
    """Run question 4 for full year and write results to result4-2.xlsx."""
    print(f"=== Question 4 Full Year Analysis (2025-02-01 to 2025-12-31) ===")
    print(f"Predictor: {predictor.__class__.__name__}\n")

    print("Loading historical data...")
    dates_load, load_data = load_historical_load()
    dates_pv, pv_data = load_historical_pv()
    dates_price, price_data = load_time_varying_price()

    price_predictor = PriceFourierPredictor(dates_price, price_data)

    start_date = "2025-02-01"
    end_date = "2025-12-31"
    start_idx = get_date_index(dates_load, start_date)
    end_idx = get_date_index(dates_load, end_date)
    n_days = end_idx - start_idx + 1

    print(f"Processing {n_days} days from {start_date} to {end_date}...\n")

    # Read template to get column structure.  The 144 slot labels of 附件1/附件4
    # ("0:10" ... "0:00+1") are identical to the template's, so the arrays are
    # written in their natural order -- no reordering is needed.
    template = pd.read_excel(OUTPUT_FILE, sheet_name="计划购电量")
    time_cols = list(template.columns[1:145])  # 144 time slot columns

    # Pre-allocate arrays
    grid_purchase_all = np.zeros((n_days, 144))
    emergency_purchase_all = np.zeros((n_days, 144))
    charge_all = np.zeros((n_days, 144))
    discharge_all = np.zeros((n_days, 144))
    soc_all = np.zeros((n_days, 145))  # 145 SOC values per day
    normal_cost_all = np.zeros(n_days)
    emergency_cost_all = np.zeros(n_days)

    # The battery SOC is continuous across days: only 2025-01-01 0:00 is fixed at
    # 6000 kWh by the problem statement, so each day inherits the previous day's
    # closing SOC (the 24:00 value) as its starting SOC.
    soc_carry = S_INIT
    for i, idx in enumerate(range(start_idx, end_idx + 1)):
        target_date = str(dates_load[idx])[:10]

        try:
            result = run_question4_single_date(
                target_date, predictor, price_predictor,
                dates_load, load_data, dates_pv, pv_data,
                dates_price, price_data, s_init=soc_carry,
            )

            grid_purchase_all[i] = result.grid_purchase
            emergency_purchase_all[i] = result.emergency_purchase
            charge_all[i] = result.charge
            discharge_all[i] = result.discharge
            soc_all[i, :144] = result.soc_start
            soc_all[i, 144] = result.soc_end[-1]
            normal_cost_all[i] = result.normal_cost
            emergency_cost_all[i] = result.emergency_cost

            soc_carry = float(result.soc_end[-1])

            if (i + 1) % 30 == 0:
                print(f"Processed {i + 1} days... (current: {target_date})")

        except ValueError as e:
            print(f"Warning: Skipping {target_date}: {e}")

    # === Write Sheet 1: 计划购电量 ===
    print("\nWriting results to Excel...")

    df_grid = pd.DataFrame(grid_purchase_all, columns=time_cols)
    df_grid.insert(0, template.columns[0],
                   [str(dates_load[idx])[:10] for idx in range(start_idx, end_idx + 1)])
    df_grid["全天购电量"] = grid_purchase_all.sum(axis=1)
    df_grid["全天购电费"] = normal_cost_all

    # === Write Sheet 2: 充放电量 ===
    rows_charge = []
    for i, idx in enumerate(range(start_idx, end_idx + 1)):
        date_str = str(dates_load[idx])[:10]
        charge_blocks = _aggregate_4h_blocks(charge_all[i])
        discharge_blocks = _aggregate_4h_blocks(discharge_all[i])

        for b in range(6):
            row = {
                "日期": date_str if b == 0 else np.nan,
                "时间段": TIME_BLOCKS[b],
                "充电量": charge_blocks[b],
                "放电量": discharge_blocks[b],
                "时刻": SOC_TIMES[b],
                "储电量": soc_all[i, b * 24],
            }
            rows_charge.append(row)

        # Add the 24:00 SOC row
        rows_charge.append({
            "日期": np.nan,
            "时间段": np.nan,
            "充电量": np.nan,
            "放电量": np.nan,
            "时刻": "24:00",
            "储电量": soc_all[i, 144],
        })

    df_charge = pd.DataFrame(rows_charge)

    # === Write Sheet 3: 紧急购电量 ===
    # Each 10-min interval is a separate row (only output slots with emergency > 0)
    emergency_rows = []
    for i, idx in enumerate(range(start_idx, end_idx + 1)):
        date_str = str(dates_load[idx])[:10]
        ep = emergency_purchase_all[i]
        for t in range(144):
            if ep[t] > 1e-7:
                sh, sm = divmod(t * 10, 60)
                eh, em = divmod((t + 1) * 10, 60)
                if eh == 24:
                    label = f"{sh}:{sm:02d}-0:00+1"
                else:
                    label = f"{sh}:{sm:02d}-{eh}:{em:02d}"
                emergency_rows.append({
                    "日期": date_str,
                    "购电时间段": label,
                    "购电量": ep[t],
                })

    df_emergency = pd.DataFrame(emergency_rows, columns=["日期", "购电时间段", "购电量"])

    # Write to Excel
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        df_grid.to_excel(writer, sheet_name="计划购电量", index=False)
        df_charge.to_excel(writer, sheet_name="充放电量", index=False)
        df_emergency.to_excel(writer, sheet_name="紧急购电量", index=False)

    # Print summary
    total_grid = grid_purchase_all.sum()
    total_emergency = emergency_purchase_all.sum()
    total_normal_cost = normal_cost_all.sum()
    total_emergency_cost = emergency_cost_all.sum()
    total_cost = total_normal_cost + total_emergency_cost

    print("\n" + "=" * 60)
    print("=== Full Year Results (2025-02-01 to 2025-12-31) ===")
    print("=" * 60)
    print(f"Total planned grid purchase: {total_grid:,.4f} kWh")
    print(f"Total emergency purchase: {total_emergency:,.4f} kWh")
    print(f"Total normal purchase cost: {total_normal_cost:,.4f} yuan")
    print(f"Total emergency purchase cost: {total_emergency_cost:,.4f} yuan")
    print(f"Total cost: {total_cost:,.4f} yuan")
    print("=" * 60)
    print(f"\nResults written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    # 加载历史数据
    dates_load, load_data = load_historical_load()
    dates_pv, pv_data = load_historical_pv()

    # 傅里叶 + 净负荷滞后项预测 + 新闻童分位安全边际（问题二口径）
    # （紧急购电价 = 5 倍正常电价，故日前计划量取净负荷的高分位数而非均值）
    predictor = FourierLagQuantilePredictor(
        dates_load, load_data, dates_pv, pv_data,
        lags=(1, 2, 3), quantile=0.75, residual_days=30,
    )
    run_question4_full_year(predictor)
