# -*- coding: utf-8 -*-
"""run_all implementation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SOLUTION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SOLUTION_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "附件" / "附件5"
STATIC_FILE = PROJECT_ROOT / "static.txt"

Q1_DIR = SOLUTION_DIR / "question1"
Q2_DIR = SOLUTION_DIR / "question2"
Q3_DIR = SOLUTION_DIR / "question3"
Q4_DIR = SOLUTION_DIR / "question4"


def _env() -> dict:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["OMP_NUM_THREADS"] = "1"
    env["LOKY_MAX_CPU_COUNT"] = "1"
    return env


def _run(script: str, cwd: Path) -> None:
    print(f"\n=== python {script} ===", flush=True)
    subprocess.run(
        [sys.executable, script],
        cwd=str(cwd),
        env=_env(),
        check=True,
    )


def _run_json(script: str, cwd: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(cwd),
        env=_env(),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    last_line = [line.strip() for line in proc.stdout.splitlines() if line.strip()][-1]
    return json.loads(last_line)


Q2_COMPARE_HELPER = r'''
import json
import numpy as np
import pandas as pd
import compare_predictors as cp
from data_loader import load_fixed_price, load_historical_load, load_historical_pv

dates_load, load = load_historical_load()
dates_pv, pv = load_historical_pv()
price = load_fixed_price()
stamps = pd.DatetimeIndex([pd.Timestamp(d) for d in dates_load])
mask = (stamps >= cp.START_DATE) & (stamps <= cp.END_DATE)
dates = [stamps[i].strftime("%Y-%m-%d") for i in np.where(mask)[0]]
preds = cp.build_predictors(dates_load, load, dates_pv, pv)
out = {}
for name, pred in preds.items():
    m = cp.evaluate(pred, dates, load[mask], pv[mask], price)
    out[name] = {
        "mae": m["mae"],
        "rmse": m["rmse"],
        "bias": m["bias"],
        "mape": m["mape"],
        "plan_kwh": m["plan_kwh"],
        "emergency_kwh": m["emergency_kwh"],
        "total_cost": m["total_cost"],
        "emergency_share": m["emergency_share"],
    }
print(json.dumps(out, ensure_ascii=False))
'''


Q3_EXTRA_HELPER = r'''
import json
import numpy as np
from data_loader import day_index_of, load_actual, load_price, load_pv_forecast
from commitment import CausalFourierLoadModel, causal_load_error_pool, run_day_commitment
from rolling import run_day as run_day_legacy
from scenarios import load_error_pool, load_forecast_model, pv_error_pool

dates, load, pv = load_actual()
_, fc = load_pv_forecast()
price = load_price()
start = day_index_of(dates, "2025-02-01")
end = day_index_of(dates, "2025-12-31")

commit_model = CausalFourierLoadModel(dates, load)
commit_err = causal_load_error_pool(commit_model, dates, load)
pv_err = tuple(pv_error_pool(pv, fc, i, method="linear") for i in range(4))

ablation = {}
for nodes in [(0,), (0, 1), (0, 1, 2), (0, 1, 2, 3)]:
    soc = 6000.0
    totals = {"total_cost": 0.0, "emergency_kwh": 0.0, "terminal_soc": 0.0}
    for day in range(start, end + 1):
        load_fc, _ = commit_model.predict(str(dates[day])[:10])
        res = run_day_commitment(
            day=day, dates=dates, load_actual=load, pv_actual=pv,
            pv_fc_day=fc[day], price=price, load_fc=load_fc,
            load_err=commit_err, pv_err_pools=pv_err,
            n_scenarios=5, lookback=90, nodes=nodes,
            stochastic=True, s_init=soc,
        )
        totals["total_cost"] += res["total_cost"]
        totals["emergency_kwh"] += res["emergency_kwh"]
        soc = float(res["soc"][-1])
    totals["terminal_soc"] = soc
    ablation[",".join(str(x) for x in nodes)] = totals

legacy_model = load_forecast_model(dates, load)
legacy_err = load_error_pool(legacy_model, dates, load)
soc = 6000.0
legacy = {"total_cost": 0.0, "emergency_kwh": 0.0, "plan_kwh": 0.0, "adjusted_kwh": 0.0}
for day in range(start, end + 1):
    load_fc, _ = legacy_model.predict(str(dates[day])[:10])
    res = run_day_legacy(
        day=day, dates=dates, load_actual=load, pv_actual=pv,
        pv_fc_day=fc[day], price=price, load_fc=load_fc,
        load_err=legacy_err, pv_err_pools=pv_err,
        n_scenarios=12, lookback=90, nodes=(0, 1, 2, 3),
        stochastic=True, s_init=soc,
    )
    legacy["total_cost"] += res["total_cost"]
    legacy["emergency_kwh"] += res["emergency_kwh"]
    legacy["plan_kwh"] += res["plan_kwh"]
    legacy["adjusted_kwh"] += res["adjusted_kwh"]
    soc = float(res["soc"][-1])
legacy["terminal_soc"] = soc

print(json.dumps({"ablation": ablation, "legacy": legacy}, ensure_ascii=False))
'''


def _read_result_sums(path: Path) -> dict:
    """_read_result_sums implementation."""
    xl = pd.ExcelFile(path)
    out = {}
    if "计划购电量" in xl.sheet_names:
        plan = pd.read_excel(xl, "计划购电量")
        out["plan_kwh"] = float(plan["全天购电量"].sum())
        out["planned_cost"] = float(plan["全天购电费"].sum())
    if "调整购电量" in xl.sheet_names:
        adj = pd.read_excel(xl, "调整购电量")
        out["adjusted_kwh"] = float(adj["全天购电量"].sum())
        out["total_cost"] = float(adj["全天购电费"].sum())
    if "紧急购电量" in xl.sheet_names:
        emg = pd.read_excel(xl, "紧急购电量")
        out["emergency_kwh"] = float(emg["购电量"].sum())
    if "充放电量" in xl.sheet_names:
        blk = pd.read_excel(xl, "充放电量")
        out["terminal_soc"] = float(blk["储电量"].iloc[-1])
    return out


def _q1_static() -> dict:
    path = OUTPUT_DIR / "result1.xlsx"
    full = pd.read_excel(path, sheet_name="完整调度")
    blocks = pd.read_excel(path, sheet_name="充放电量")
    dt = 1.0 / 6.0
    net_kwh = (full["负荷功率/kW"] - full["光伏预测功率/kW"]).to_numpy(float) * dt
    direct_cost = float(np.dot(full["电价/(元/kWh)"].to_numpy(float), net_kwh))
    grid = float(full["计划购电量/kWh"].sum())
    purchase_cost = float(np.dot(full["电价/(元/kWh)"].to_numpy(float), full["计划购电量/kWh"].to_numpy(float)))
    rows = []
    for _, row in blocks.iterrows():
        rows.append({
            "period": str(row["时间段"]),
            "charge_kwh": float(row["充电量/kWh"]),
            "discharge_kwh": float(row["放电量/kWh"]),
        })
    return {
        "grid_kwh": grid,
        "purchase_cost": purchase_cost,
        "net_load_kwh": float(net_kwh.sum()),
        "cost_without_storage": direct_cost,
        "saving": direct_cost - purchase_cost,
        "saving_pct": (direct_cost - purchase_cost) / direct_cost * 100.0,
        "storage_table": rows,
    }


def _q3_report_dates(path: Path) -> list:
    plan = pd.read_excel(path, sheet_name="计划购电量")
    adj = pd.read_excel(path, sheet_name="调整购电量")
    emg = pd.read_excel(path, sheet_name="紧急购电量")
    rows = []
    for date in ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]:
        p = plan.loc[plan["日期"].astype(str) == date]
        a = adj.loc[adj["日期"].astype(str) == date]
        e = emg.loc[emg["日期"].astype(str) == date, "购电量"].sum()
        rows.append({
            "date": date,
            "plan_kwh": float(p["全天购电量"].iloc[0]),
            "adjusted_kwh": float(a["全天购电量"].iloc[0]),
            "emergency_kwh": float(e),
            "total_cost": float(a["全天购电费"].iloc[0]),
        })
    return rows


def _q4_two_stage_static(path: Path) -> dict:
    """_q4_two_stage_static implementation."""
    plan = pd.read_excel(path, sheet_name="计划购电量")
    emg = pd.read_excel(path, sheet_name="紧急购电量")
    price_df = pd.read_excel(PROJECT_ROOT / "附件" / "附件4.xlsx", sheet_name="Sheet1")
    price = price_df.iloc[:, 1:].to_numpy(dtype=float)
    dates = price_df.iloc[:, 0].astype(str).tolist()

    normal_cost = float(plan["全天购电费"].sum())
    emergency_cost = 0.0
    for _, row in emg.iterrows():
        date = str(row["日期"])
        label = str(row["购电时间段"])
        start = label.split("-")[0]
        hour, minute = (int(x) for x in start.split(":"))
        slot = (hour * 60 + minute) // 10
        try:
            day_idx = dates.index(date)
        except ValueError:
            # The workbook may store dates as Timestamps; try the normalized form.
            norm = [str(pd.Timestamp(d).date()) for d in dates]
            day_idx = norm.index(str(pd.Timestamp(date).date()))
        emergency_cost += 5.0 * price[day_idx, slot] * float(row["购电量"])
    return {
        "plan_kwh": float(plan["全天购电量"].sum()),
        "emergency_kwh": float(emg["购电量"].sum()),
        "normal_cost": normal_cost,
        "emergency_cost": emergency_cost,
        "total_cost": normal_cost + emergency_cost,
    }


def _format_static(q1: dict, q2: dict, compare: dict, q3: dict, extra: dict, q4_2: dict, q4_3: dict) -> str:
    lines: list[str] = []
    lines.append("Question 1")
    lines.append(f"grid_purchase_kwh={q1['grid_kwh']:.4f}")
    lines.append(f"purchase_cost_yuan={q1['purchase_cost']:.4f}")
    lines.append(f"net_load_kwh={q1['net_load_kwh']:.4f}")
    lines.append(f"cost_without_storage_yuan={q1['cost_without_storage']:.4f}")
    lines.append(f"cost_saving_yuan={q1['saving']:.4f}")
    lines.append(f"cost_saving_pct={q1['saving_pct']:.4f}")
    lines.append("storage_4h")
    for row in q1["storage_table"]:
        lines.append(f"{row['period']}\t{row['charge_kwh']:.4f}\t{row['discharge_kwh']:.4f}")

    lines.append("")
    lines.append("Question 2 predictor comparison")
    lines.append("name\tMAE_kW\tRMSE_kW\tMAPE_pct\tplan_10kWh\temergency_10kWh\ttotal_10kyuan\temergency_pct")
    for name, m in compare.items():
        lines.append(
            f"{name}\t{m['mae']:.4f}\t{m['rmse']:.4f}\t{m['mape']:.4f}\t"
            f"{m['plan_kwh'] / 1e4:.4f}\t{m['emergency_kwh'] / 1e4:.4f}\t"
            f"{m['total_cost'] / 1e4:.4f}\t{m['emergency_share']:.4f}"
        )
    lines.append("")
    lines.append("Question 2 final")
    lines.append(f"grid_purchase_kwh={q2['plan_kwh']:.4f}")
    lines.append(f"emergency_kwh={q2['emergency_kwh']:.4f}")
    lines.append(f"normal_cost_yuan={q2['planned_cost']:.4f}")
    q2_final = compare["Fourier+Lag+q0.75 (final)"]
    lines.append(f"emergency_cost_yuan={q2_final['total_cost'] - q2['planned_cost']:.4f}")
    lines.append(f"total_cost_yuan={q2_final['total_cost']:.4f}")

    lines.append("")
    lines.append("Question 3 main")
    lines.append(f"planned_cost_yuan={q3['planned_cost']:.4f}")
    lines.append(f"total_cost_yuan={q3['total_cost']:.4f}")
    lines.append(f"plan_kwh={q3['plan_kwh']:.4f}")
    lines.append(f"adjusted_kwh={q3['adjusted_kwh']:.4f}")
    lines.append(f"emergency_kwh={q3['emergency_kwh']:.4f}")
    lines.append(f"terminal_soc_kwh={q3['terminal_soc']:.4f}")
    lines.append("report_dates\tplan_kwh\tadjusted_kwh\temergency_kwh\ttotal_cost_yuan")
    for row in _q3_report_dates(OUTPUT_DIR / "result3.xlsx"):
        lines.append(
            f"{row['date']}\t{row['plan_kwh']:.4f}\t{row['adjusted_kwh']:.4f}\t"
            f"{row['emergency_kwh']:.4f}\t{row['total_cost']:.4f}"
        )

    lines.append("")
    lines.append("Question 3 ablation")
    for key, value in extra["ablation"].items():
        lines.append(
            f"nodes={key}\ttotal_cost={value['total_cost']:.4f}\t"
            f"emergency_kwh={value['emergency_kwh']:.4f}\tterminal_soc={value['terminal_soc']:.4f}"
        )
    lines.append("Question 3 legacy")
    legacy = extra["legacy"]
    lines.append(f"total_cost_yuan={legacy['total_cost']:.4f}")
    lines.append(f"emergency_kwh={legacy['emergency_kwh']:.4f}")

    lines.append("")
    lines.append("Question 4 two-stage")
    lines.append(f"grid_purchase_kwh={q4_2['plan_kwh']:.4f}")
    lines.append(f"emergency_kwh={q4_2['emergency_kwh']:.4f}")
    lines.append(f"normal_cost_yuan={q4_2['normal_cost']:.4f}")
    lines.append(f"emergency_cost_yuan={q4_2['emergency_cost']:.4f}")
    lines.append(f"total_cost_yuan={q4_2['total_cost']:.4f}")
    lines.append("Question 4 commitment")
    lines.append(f"planned_cost_yuan={q4_3['planned_cost']:.4f}")
    lines.append(f"total_cost_yuan={q4_3['total_cost']:.4f}")
    lines.append(f"plan_kwh={q4_3['plan_kwh']:.4f}")
    lines.append(f"adjusted_kwh={q4_3['adjusted_kwh']:.4f}")
    lines.append(f"emergency_kwh={q4_3['emergency_kwh']:.4f}")
    lines.append(f"terminal_soc_kwh={q4_3['terminal_soc']:.4f}")
    return "\n".join(lines)


def main() -> None:
    print(f"Project root: {PROJECT_ROOT}", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _run("main.py", Q1_DIR)
    _run("main.py", Q2_DIR)
    compare = _run_json(Q2_COMPARE_HELPER, Q2_DIR)
    _run("main.py", Q3_DIR)
    extra = _run_json(Q3_EXTRA_HELPER, Q3_DIR)
    _run("main.py", Q4_DIR)
    _run("main_rolling.py", Q4_DIR)

    q1 = _q1_static()
    q2 = _read_result_sums(OUTPUT_DIR / "result2.xlsx")
    q3 = _read_result_sums(OUTPUT_DIR / "result3.xlsx")
    q4_2 = _q4_two_stage_static(OUTPUT_DIR / "result4-2.xlsx")
    q4_3 = _read_result_sums(OUTPUT_DIR / "result4-3.xlsx")

    static_text = _format_static(q1, q2, compare, q3, extra, q4_2, q4_3)
    STATIC_FILE.write_text(static_text, encoding="utf-8")
    print(f"\nAll result workbooks are ready in {OUTPUT_DIR}")
    print(f"Static results written to {STATIC_FILE}")


if __name__ == "__main__":
    main()
