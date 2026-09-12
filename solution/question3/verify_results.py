# -*- coding: utf-8 -*-
"""结果复核：校验 附件5/result3.xlsx 与模型、与物理约束是否一致。

检查项：
  A 文件结构：四张表、行数、列数、标签；
  B 计划/调整表：144 列之和 == 全天购电量、无缺失、非负；
  C 充放电表：每日 7 行、SOC ∈ [1200, 10800]、单时段充放电 ≤ 833.3333、
    由充放电反推的 SOC 与记录一致、日末 SOC 已记录；
  D 费用恒等式：Σ(计划费用) + 调减 + 调增 + 紧急 == 调整表 Σ(全天购电费)；
  E 与模型重算一致：抽样若干天用 run_day 重算，与表内数值逐项比对。

用法：``python verify_results.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import (  # noqa: E402
    ETA,
    FILE_TEMPLATE,
    ISSUE_SLOTS,
    N_SLOTS,
    P_MAX,
    S_MAX,
    S_MIN,
    day_index_of,
    load_actual,
    load_price,
    load_pv_forecast,
)
from rolling import run_day  # noqa: E402
from scenarios import load_error_pool, load_forecast_model, pv_error_pool  # noqa: E402

DELTA_T = 1 / 6
P_MAX_SLOT = P_MAX * DELTA_T
PASS, FAIL = "PASS", "FAIL"
results = []


def check(name: str, ok: bool, detail: str = ""):
    results.append((name, ok, detail))
    print(f"[{PASS if ok else FAIL}] {name}" + (f"   {detail}" if detail else ""))


def main() -> None:
    xl = pd.ExcelFile(FILE_TEMPLATE)
    plan = pd.read_excel(xl, "计划购电量")
    adj = pd.read_excel(xl, "调整购电量")
    blk = pd.read_excel(xl, "充放电量")
    emg = pd.read_excel(xl, "紧急购电量")
    n_days = len(plan)

    # ---------------- A 文件结构 ----------------
    check("A1 四张工作表齐全", set(xl.sheet_names) == {"计划购电量", "调整购电量", "充放电量", "紧急购电量"},
          str(xl.sheet_names))
    check("A2 计划/调整表行数一致且覆盖 334 天", len(plan) == len(adj) == n_days == 334,
          f"{n_days} 天")
    check("A3 计划/调整表列数 = 1+144+2", plan.shape[1] == adj.shape[1] == N_SLOTS + 3,
          f"{plan.shape[1]} 列")
    check("A4 充放电表 = 334×7 行", len(blk) == 334 * 7, f"{len(blk)} 行")

    time_cols = list(plan.columns[1:N_SLOTS + 1])
    e_tbl = (emg.groupby(emg["日期"].astype(str))["购电量"].sum()
             .reindex(adj["日期"].astype(str)).fillna(0.0).to_numpy(dtype=float))

    # ---------------- B 计划/调整表 ----------------
    for tag, df in (("计划", plan), ("调整", adj)):
        vals = df[time_cols].to_numpy(dtype=float)
        check(f"B1 {tag}表无缺失值", not np.isnan(vals).any())
        check(f"B2 {tag}表所有量非负", bool((vals >= -1e-9).all()),
              f"min = {vals.min():.4f}")
        gap = np.abs(vals.sum(axis=1) - df["全天购电量"].to_numpy(dtype=float)).max()
        check(f"B3 {tag}表 144 列之和 == 全天购电量", gap < 1e-6, f"max|Δ| = {gap:.2e}")

    # ---------------- C 充放电表 ----------------
    days = adj["日期"].astype(str).tolist()
    soc_all, charge_all, discharge_all = [], [], []
    ok_rows, ok_power, ok_soc, ok_rec = True, True, True, True
    max_rec_err = 0.0
    for i, date in enumerate(days):
        block = blk.iloc[i * 7:(i + 1) * 7]
        if len(block) != 7 or block["时间段"].iloc[0] != "0:00-4:00" or block["时刻"].iloc[6] != "24:00":
            ok_rows = False
        c = block["充电量"].to_numpy(dtype=float)[:6]
        d = block["放电量"].to_numpy(dtype=float)[:6]
        soc = np.concatenate([block["储电量"].to_numpy(dtype=float)[:6],
                              [block["储电量"].iloc[6]]])
        if np.isnan(c).any() or np.isnan(d).any() or np.isnan(soc).any():
            ok_rows = False
            continue
        if (c > P_MAX_SLOT * 24 + 1e-6).any() or (d > P_MAX_SLOT * 24 + 1e-6).any():
            ok_power = False
        if (soc < S_MIN - 1e-6).any() or (soc > S_MAX + 1e-6).any():
            ok_soc = False
        # 由 4 小时段的充放电量反推下一个边界的 SOC
        for b in range(6):
            soc_next = soc[b] + ETA * c[b] - d[b] / ETA
            max_rec_err = max(max_rec_err, abs(soc_next - soc[b + 1]))
        charge_all.append(c.sum())
        discharge_all.append(d.sum())
        soc_all.append(soc)

    check("C1 每天 7 行（6 段 + 24:00）", ok_rows)
    check("C2 单时段充放电 ≤ 833.3333 kWh", ok_power)
    check("C3 SOC 始终在 [1200, 10800]", ok_soc)
    check("C4 由充放电反推的 SOC 与记录一致", max_rec_err < 1e-3, f"max|Δ| = {max_rec_err:.2e}")
    year_end_soc = soc_all[-1][-1]
    check("C5 年末 SOC 在可行区间内", S_MIN <= year_end_soc <= S_MAX, f"{year_end_soc:.2f} kWh")

    # ---------------- D 费用恒等式 ----------------
    # 只用表格自身的内容按题目结算规则独立重算，避免与具体模型绑死
    from data_loader import load_price

    price = load_price()

    def label_to_slot(label: str) -> int:
        start = str(label).split("-")[0]
        h, m = start.split(":")
        return (int(h) * 60 + int(m)) // 10

    emg_slot = np.zeros((n_days, N_SLOTS))
    dup = 0
    for _, row in emg.iterrows():
        date = str(row["日期"])
        if date not in days:
            continue
        i = days.index(date)
        s = label_to_slot(row["购电时间段"])
        if 0 <= s < N_SLOTS:
            if emg_slot[i, s] != 0:
                dup += 1
            emg_slot[i, s] += float(row["购电量"])

    planned_cost = 0.0
    total_cost = 0.0
    recomputed_plan = 0.0
    recomputed_total = 0.0
    for i in range(n_days):
        g0 = plan.iloc[i][time_cols].to_numpy(dtype=float)
        ga = adj.iloc[i][time_cols].to_numpy(dtype=float)
        u = np.maximum(0.0, g0 - ga)
        v = np.maximum(0.0, ga - g0)
        planned_cost += float(g0 @ price)
        # 结算规则：计划按原价支付；调减退回原价并承担 50% 违约成本（即 -0.5p·u）；
        # 调增超出部分按 1.5 倍；缺额按 5 倍紧急购电
        total_cost += float(
            g0 @ price
            - 0.5 * (u @ price)
            + 1.5 * (v @ price)
            + 5 * (emg_slot[i] @ price)
        )
        recomputed_plan += g0.sum()
        recomputed_total += ga.sum()

    sheet_plan_cost = float(plan["全天购电费"].sum())
    sheet_total_cost = float(adj["全天购电费"].sum())
    emg_kwh = float(emg["购电量"].sum())
    check("D1 由表内计划量独立重算的计划费用 == 计划表费用",
          abs(planned_cost - sheet_plan_cost) < 1.0,
          f"重算 {planned_cost:,.2f} vs 表 {sheet_plan_cost:,.2f} 元")
    check("D2 由结算规则独立重算的总费用 == 调整表费用",
          abs(total_cost - sheet_total_cost) < 1.0,
          f"重算 {total_cost:,.2f} vs 表 {sheet_total_cost:,.2f} 元")
    check("D3 计划量列和 == 计划表全天购电量", abs(recomputed_plan - plan["全天购电量"].sum()) < 1e-6)
    check("D4 调整量列和 == 调整表全天购电量", abs(recomputed_total - adj["全天购电量"].sum()) < 1e-6)
    check("D5 紧急购电记录均为正值且每天同一时段不重复", bool((emg["购电量"] > 0).all()) and dup == 0)
    check("D6 紧急购电合计为有限正值", np.isfinite(emg_kwh) and emg_kwh > 0,
          f"{emg_kwh / 1e4:.4f} 万 kWh")

    # ---------------- E 与模型重算一致 ----------------
    dates, load, pv = load_actual()
    _, fc = load_pv_forecast()
    price = load_price()
    model = load_forecast_model(dates, load)
    load_err = load_error_pool(model, dates, load)
    pv_err = tuple(pv_error_pool(pv, fc, i, method="linear") for i in range(4))

    sample = ["2025-02-01", "2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]
    max_plan_dev = max_adj_dev = max_cost_dev = max_soc_dev = max_emg_dev = 0.0
    for date in sample:
        row = days.index(date)
        day = day_index_of(dates, date)
        s_init = soc_all[row][0]
        load_fc, _ = model.predict(date)
        res = run_day(
            day=day, dates=dates, load_actual=load, pv_actual=pv, pv_fc_day=fc[day],
            price=price, load_fc=load_fc, load_err=load_err, pv_err_pools=pv_err,
            n_scenarios=12, lookback=90, s_init=s_init, battery_mode="realtime",
        )
        max_plan_dev = max(max_plan_dev,
                           np.abs(res["plan"] - plan.iloc[row][time_cols].to_numpy(float)).max())
        max_adj_dev = max(max_adj_dev,
                          np.abs(res["adjusted"] - adj.iloc[row][time_cols].to_numpy(float)).max())
        max_cost_dev = max(max_cost_dev,
                           abs(res["total_cost"] - adj.iloc[row]["全天购电费"]))
        soc_tbl = np.concatenate([blk.iloc[row * 7:row * 7 + 6]["储电量"].to_numpy(float),
                                  [blk.iloc[row * 7 + 6]["储电量"]]])
        max_soc_dev = max(max_soc_dev, np.abs(res["soc"][::24] - soc_tbl).max())
        max_emg_dev = max(max_emg_dev, abs(res["emergency_kwh"] - e_tbl[row]))
    check("E1 抽样 5 天重算：计划量逐时段一致", max_plan_dev < 1e-4, f"max|Δ| = {max_plan_dev:.2e}")
    check("E2 抽样 5 天重算：调整量逐时段一致", max_adj_dev < 1e-4, f"max|Δ| = {max_adj_dev:.2e}")
    check("E3 抽样 5 天重算：总费用一致", max_cost_dev < 0.01, f"max|Δ| = {max_cost_dev:.2e} 元")
    check("E4 抽样 5 天重算：SOC 轨迹一致", max_soc_dev < 1e-3, f"max|Δ| = {max_soc_dev:.2e} kWh")
    check("E5 抽样 5 天重算：紧急购电量一致", max_emg_dev < 1e-4, f"max|Δ| = {max_emg_dev:.2e} kWh")

    # ---------------- 汇总 ----------------
    bad = [r for r in results if not r[1]]
    print("\n" + "=" * 62)
    print(f"共 {len(results)} 项检查，通过 {len(results) - len(bad)} 项，失败 {len(bad)} 项")
    if bad:
        for name, _, detail in bad:
            print(f"  FAIL: {name} {detail}")
    print("=" * 62)


if __name__ == "__main__":
    main()
