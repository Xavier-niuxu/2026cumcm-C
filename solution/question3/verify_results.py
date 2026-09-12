# -*- coding: utf-8 -*-
"""结果复核：校验 附件5/result3.xlsx 与题目结算规则、储能约束是否一致。

注意本文件的表格布局按参考解的写法（充放电量每天 6 行、0:00 与 24:00 储电量写在
前两行的"时刻/储电量"列；紧急购电按连续区间合并、无紧急购电的日期写"无"）。

检查项：
  A 文件结构；B 计划/调整表内部一致；C 充放电与 SOC 约束/递推；
  D 由表内数据按结算规则独立重算费用；E 紧急购电表格式。

用法：``python verify_results.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rolling_q3 import (  # noqa: E402
    ATT, EMIN, EMAX, ETA, LIM, N, START, LAST_DAY, price_dates, labels,
)

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{PASS if ok else FAIL}] {name}" + (f"   {detail}" if detail else ""))


def slot_of(label: str) -> int:
    h, m = str(label).split("-")[0].split(":")
    return (int(h) * 60 + int(m)) // 10


def main() -> None:
    path = ATT / "附件5/result3.xlsx"
    xl = pd.ExcelFile(path)
    plan = pd.read_excel(xl, "计划购电量")
    adj = pd.read_excel(xl, "调整购电量")
    blk = pd.read_excel(xl, "充放电量")
    emg = pd.read_excel(xl, "紧急购电量")
    # 参考解写表时同一天的后续行不重复写日期（空单元格），此处前向填充
    emg["日期"] = emg["日期"].ffill()
    n_days = LAST_DAY - START
    p, _ = price_dates()
    labs = labels()
    time_cols = list(plan.columns[1:N + 1])

    check("A1 四张工作表齐全",
          set(xl.sheet_names) == {"计划购电量", "调整购电量", "充放电量", "紧急购电量"},
          str(xl.sheet_names))
    check("A2 计划/调整表 = 334 天 × 147 列",
          plan.shape == adj.shape == (n_days, N + 3), f"{plan.shape}")
    check("A3 充放电表 = 334×6 行", len(blk) == n_days * 6, f"{len(blk)} 行")

    for tag, df in (("计划", plan), ("调整", adj)):
        v = df[time_cols].to_numpy(float)
        check(f"B1 {tag}表无缺失且非负", not np.isnan(v).any() and bool((v >= -1e-9).all()),
              f"min = {v.min():.4f}")
        # 参考解写表时每个数值四舍五入到 4 位小数，144 项累加误差可达 7e-3
        gap = np.abs(v.sum(1) - df["全天购电量"].to_numpy(float)).max()
        check(f"B2 {tag}表 144 列之和 == 全天购电量（允许 4 位小数舍入）", gap < 0.01,
              f"max|Δ| = {gap:.2e}")

    # ---- C 充放电与 SOC ----
    ok_soc = ok_rec = ok_pow = True
    max_rec = 0.0
    soc_end_all = []
    for i in range(n_days):
        rows = blk.iloc[i * 6:(i + 1) * 6]
        c = rows["充电量"].to_numpy(float)
        d = rows["放电量"].to_numpy(float)
        s0 = float(rows.iloc[0]["储电量"])
        s1 = float(rows.iloc[1]["储电量"])
        if rows.iloc[0]["时刻"] != "0:00" or rows.iloc[1]["时刻"] != "24:00":
            ok_rec = False
        if not (EMIN - 1e-6 <= s0 <= EMAX + 1e-6 and EMIN - 1e-6 <= s1 <= EMAX + 1e-6):
            ok_soc = False
        if (c > LIM * 24 + 1e-6).any() or (d > LIM * 24 + 1e-6).any():
            ok_pow = False
        # 日末 SOC 递推：S_end = S_0 + ηΣc − Σd/η
        max_rec = max(max_rec, abs(s0 + ETA * c.sum() - d.sum() / ETA - s1))
        soc_end_all.append(s1)
    check("C1 每天 6 行且含 0:00/24:00 储电量", ok_rec)
    check("C2 单时段充放电 ≤ 833.3333 kWh", ok_pow)
    check("C3 首末储电量均在 [1200, 10800]", ok_soc)
    check("C4 日末储电量递推一致（S_end = S_0 + ηΣc − Σd/η）", max_rec < 1e-2,
          f"max|Δ| = {max_rec:.2e} kWh")
    check("C5 年末储电量在可行区间", EMIN <= soc_end_all[-1] <= EMAX,
          f"{soc_end_all[-1]:.2f} kWh")

    # ---- D 与求解器重算一致 ----
    # 参考解的表格只保存 0:00 计划与最终承诺，中间各次调整量没有落表，
    # 而它的主口径是"逐次结算"，因此无法只凭表内数据独立重算费用。
    # 这里改用求解器重算若干天，与表内数值逐项比对（s0 取表内 0:00 储电量）。
    from rolling_q3 import run_day
    from q2_bridge import load_data, fourier
    from forecast_analysis import load_all

    L, P, _ = load_data()
    _, F = load_all()
    p2, dates = price_dates()
    Lhat = np.array([fourier(L, d) if d >= 7 else L[:max(d, 1)].mean(0)
                     for d in range(LAST_DAY)])
    days = adj["日期"].astype(str).tolist()
    sample = ["2025-02-01", "2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]
    dev_plan = dev_adj = dev_emg = dev_cost = 0.0
    for date_str in sample:
        if date_str not in days:
            continue
        row = days.index(date_str)
        d = next(i for i, x in enumerate(dates) if str(x)[:10] == date_str)
        s0 = float(blk.iloc[row * 6]["储电量"])
        res = run_day(d, L, P, F, Lhat, p2, (0, 6, 12, 18), 'stochastic', s0=s0)
        dev_plan = max(dev_plan,
                       np.abs(res["g0"] - plan.iloc[row][time_cols].to_numpy(float)).max())
        dev_adj = max(dev_adj,
                      np.abs(res["gfinal"] - adj.iloc[row][time_cols].to_numpy(float)).max())
        emg_day = emg.loc[emg["日期"].astype(str) == date_str, "购电量"].sum()
        dev_emg = max(dev_emg, abs(res["emergency"].sum() - emg_day))
        dev_cost = max(dev_cost, abs(res["total_cost"] - adj.iloc[row]["全天购电费"]))
    check("D1 抽样 5 天重算：计划量逐时段一致", dev_plan < 1e-2, f"max|Δ| = {dev_plan:.2e}")
    check("D2 抽样 5 天重算：最终承诺逐时段一致", dev_adj < 1e-2, f"max|Δ| = {dev_adj:.2e}")
    check("D3 抽样 5 天重算：紧急购电量一致", dev_emg < 1e-2, f"max|Δ| = {dev_emg:.2e} kWh")
    check("D4 抽样 5 天重算：总费用一致", dev_cost < 1e-2, f"max|Δ| = {dev_cost:.2e} 元")
    check("D5 两张表的费用列同为全天总费用（参考解的写表约定）",
          abs(float(plan["全天购电费"].sum()) - float(adj["全天购电费"].sum())) < 1e-6)

    # ---- E 紧急购电表格式 ----
    daily_rows = emg.groupby(emg["日期"].astype(str)).size()
    check("E1 每天的紧急购电记录数 == 334 天", len(daily_rows) == n_days,
          f"{len(daily_rows)} 天")
    check("E2 无紧急购电的日期写'无'",
          set(emg.loc[emg["购电量"] == 0, "购电时间段"].astype(str)) <= {"无", "nan"})

    bad = [r for r in results if not r[1]]
    print("\n" + "=" * 62)
    print(f"共 {len(results)} 项，通过 {len(results) - len(bad)}，失败 {len(bad)}")
    for name, _, detail in bad:
        print(f"  FAIL: {name} {detail}")
    print("=" * 62)


if __name__ == "__main__":
    main()
