# -*- coding: utf-8 -*-
"""Question 4 main entry point: time-varying price with Fourier predictor."""

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
from prediction import FourierPredictor, BasePredictor
from cost import calculate_cost

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = PROJECT_ROOT / "附件" / "附件5" / "result4-2.xlsx"

# Time period labels for 4h blocks
TIME_BLOCKS = ["0:00-4:00", "4:00-8:00", "8:00-12:00",
               "12:00-16:00", "16:00-20:00", "20:00-24:00"]
SOC_TIMES = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"]


def run_question4_single_date(
    target_date: str,
    predictor: BasePredictor,
    dates_load: np.ndarray,
    load_data: np.ndarray,
    dates_pv: np.ndarray,
    pv_data: np.ndarray,
    dates_price: np.ndarray,
    price_data: np.ndarray,
):
    """Run question 4 for a single date. Returns LPResult and date string."""
    load_pred, pv_pred = predictor.predict(target_date)
    load_actual = get_day_data(dates_load, load_data, target_date)
    pv_actual = get_day_data(dates_pv, pv_data, target_date)
    price = get_day_price(dates_price, price_data, target_date)
    result = calculate_cost(price, load_pred, pv_pred, load_actual, pv_actual)
    return result


def _generate_time_columns():
    """Generate 144 time slot column labels matching the template order.
    
    Template order: 0:10-0:20, 0:20-0:30, ..., 23:50-0:00+1, 0:00-0:10+1
    This means slots 1-143 first, then slot 0 at the end.
    """
    cols = []
    for i in range(1, 144):  # slots 1 to 143
        start_min = i * 10
        end_min = (i + 1) * 10
        sh, sm = divmod(start_min, 60)
        eh, em = divmod(end_min, 60)
        if eh == 24:
            cols.append(f"{sh}:{sm:02d}-0:00+1")
        else:
            cols.append(f"{sh}:{sm:02d}-{eh}:{em:02d}")
    # Add slot 0 at the end
    cols.append("0:00-0:10+1")
    return cols


def _reorder_to_template(arr_144):
    """Reorder array from natural order (0-143) to template order (1-143, then 0)."""
    result = np.zeros_like(arr_144)
    result[:143] = arr_144[1:144]  # slots 1-143
    result[143] = arr_144[0]       # slot 0 at the end
    return result


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

    start_date = "2025-02-01"
    end_date = "2025-12-31"
    start_idx = get_date_index(dates_load, start_date)
    end_idx = get_date_index(dates_load, end_date)
    n_days = end_idx - start_idx + 1

    print(f"Processing {n_days} days from {start_date} to {end_date}...\n")

    # Read template to get column structure
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

    # Process each day
    for i, idx in enumerate(range(start_idx, end_idx + 1)):
        target_date = str(dates_load[idx])[:10]

        try:
            result = run_question4_single_date(
                target_date, predictor, dates_load, load_data, dates_pv, pv_data,
                dates_price, price_data
            )

            grid_purchase_all[i] = result.grid_purchase
            emergency_purchase_all[i] = result.emergency_purchase
            charge_all[i] = result.charge
            discharge_all[i] = result.discharge
            soc_all[i, :144] = result.soc_start
            soc_all[i, 144] = result.soc_end[-1]
            normal_cost_all[i] = result.normal_cost
            emergency_cost_all[i] = result.emergency_cost

            if (i + 1) % 30 == 0:
                print(f"Processed {i + 1} days... (current: {target_date})")

        except ValueError as e:
            print(f"Warning: Skipping {target_date}: {e}")

    # === Write Sheet 1: 计划购电量 ===
    print("\nWriting results to Excel...")

    # Reorder to match template column order
    grid_reordered = np.array([_reorder_to_template(grid_purchase_all[i]) for i in range(n_days)])
    df_grid = pd.DataFrame(grid_reordered, columns=time_cols)
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

    # 使用傅里叶预测方法
    predictor = FourierPredictor(dates_load, load_data, dates_pv, pv_data)
    run_question4_full_year(predictor)
