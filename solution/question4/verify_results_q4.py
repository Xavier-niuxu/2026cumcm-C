# -*- coding: utf-8 -*-
"""问题 4 结果复核：校验 output/result4-2.xlsx 与 result4-3.xlsx。

检查项：
  A 结构（工作表、行数、参考解表格布局）；
  B 计划/调整表 144 列之和 == 全天购电量、无缺失、非负；
  C 充放电与储能约束（SOC 区间、单时段功率、日末 SOC 递推一致）；
  D 用求解器重跑两个小问的最优方案，与表内合计对照（计划量、调整量、紧急量、总费用）。

用法：``python verify_results_q4.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import q4_models as q  # noqa: E402
import q42_q2_inheritance as q42i  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{PASS if ok else FAIL}] {name}" + (f"   {detail}" if detail else ""))


def load_sheets(path: Path):
    xl = pd.ExcelFile(path)
    return {s: pd.read_excel(xl, s) for s in xl.sheet_names}, xl.sheet_names


def check_common(tag, sheets, n_days=334):
    plan = sheets["计划购电量"]
    time_cols = list(plan.columns[1:q.N + 1])
    check(f"{tag}-A1 计划表行数/列数", plan.shape == (n_days, q.N + 3), f"{plan.shape}")
    v = plan[time_cols].to_numpy(float)
    check(f"{tag}-B1 计划表无缺失且非负", not np.isnan(v).any() and bool((v >= -1e-9).all()))
    gap = np.abs(v.sum(1) - plan["全天购电量"].to_numpy(float)).max()
    check(f"{tag}-B2 计划 144 列之和 == 全天购电量", gap < 0.01, f"max|Δ| = {gap:.2e}")
    if "调整购电量" in sheets:
        adj = sheets["调整购电量"]
        va = adj[time_cols].to_numpy(float)
        check(f"{tag}-B3 调整表无缺失且非负", not np.isnan(va).any() and bool((va >= -1e-9).all()))
        gap = np.abs(va.sum(1) - adj["全天购电量"].to_numpy(float)).max()
        check(f"{tag}-B4 调整 144 列之和 == 全天购电量", gap < 0.01, f"max|Δ| = {gap:.2e}")

    blk = sheets["充放电量"]
    check(f"{tag}-A2 充放电表 = {n_days}×6 行", len(blk) == n_days * 6, f"{len(blk)} 行")
    ok_soc = ok_pow = ok_rec = True
    max_rec = 0.0
    end_soc = []
    for i in range(n_days):
        rows = blk.iloc[i * 6:(i + 1) * 6]
        c = rows["充电量"].to_numpy(float)
        d = rows["放电量"].to_numpy(float)
        s0 = float(rows.iloc[0]["储电量"])
        s1 = float(rows.iloc[1]["储电量"])
        if rows.iloc[0]["时刻"] != "0:00" or rows.iloc[1]["时刻"] != "24:00":
            ok_rec = False
        if not (q.EMIN - 1e-6 <= s0 <= q.EMAX + 1e-6 and q.EMIN - 1e-6 <= s1 <= q.EMAX + 1e-6):
            ok_soc = False
        if (c > q.LIM * 24 + 1e-6).any() or (d > q.LIM * 24 + 1e-6).any():
            ok_pow = False
        max_rec = max(max_rec, abs(s0 + q.ETA * c.sum() - d.sum() / q.ETA - s1))
        end_soc.append(s1)
    check(f"{tag}-C1 每天 6 行且含 0:00/24:00 储电量", ok_rec)
    check(f"{tag}-C2 SOC 均在 [1200, 10800]", ok_soc)
    check(f"{tag}-C3 单时段充放电 ≤ 833.3333 kWh", ok_pow)
    check(f"{tag}-C4 日末 SOC 递推一致（S_end = S_0 + ηΣc − Σd/η）", max_rec < 1e-2,
          f"max|Δ| = {max_rec:.2e} kWh")
    check(f"{tag}-C5 年末 SOC 在可行区间", q.EMIN <= end_soc[-1] <= q.EMAX,
          f"{end_soc[-1]:.2f} kWh")
    return sheets.get("调整购电量", plan), sheets["计划购电量"]


def main() -> None:
    out = HERE / "output"
    sheets42, names42 = load_sheets(out / "result4-2.xlsx")
    sheets43, names43 = load_sheets(out / "result4-3.xlsx")
    check("A0 两个结果文件的工作表齐全",
          set(names42) == {"计划购电量", "充放电量", "紧急购电量"}
          and set(names43) == {"计划购电量", "调整购电量", "充放电量", "紧急购电量"},
          f"{names42} | {names43}")
    plan42, _ = check_common("Q4-2", sheets42)
    plan43, _ = check_common("Q4-3", sheets43)

    # ---- D 用求解器重跑最优方案并对照表内合计 ----
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    L, P, netall = q.load_data()
    _, F = q.load_all()
    prices_all, dates = q.load_prices_dates()
    price_hat = np.load(out / "price_forecasts.npz")["HistGB_XGB_fallback"]
    phat = np.vstack([prices_all[:q.START], price_hat])
    Lhat = np.array([q.fourier(L, d) if d >= 7 else L[:max(d, 1)].mean(0) for d in range(365)])
    net = netall[q.START:]
    net_hat = np.array([q.fourier(netall, d) for d in range(q.START, 365)])
    prices = prices_all[q.START:]

    w42 = summary["q42_winner"]
    if w42.get("name") == "Q2_empirical_quantile":
        r42, days42 = q42i.run(w42["quantile"], net, net_hat, prices, price_hat)
        dev_plan = abs(r42["plan_cost"] + r42["emergency_cost"]
                       - float(sheets42["计划购电量"]["全天购电费"].sum()))
        check("D1 重跑 Q4-2 最优方案 -> 总费用与表内一致", dev_plan < 1.0,
              f"重跑 {r42['total_cost']:,.2f} vs 表 "
              f"{float(sheets42['计划购电量']['全天购电费'].sum()):,.2f} 元")
        plan_kwh = float(sum(x["g"].sum() for x in days42))
        check("D2 重跑 Q4-2 -> 计划量与紧急量与表内一致",
              abs(plan_kwh - float(sheets42["计划购电量"]["全天购电量"].sum())) < 1e-2
              and abs(r42["emergency_kwh"]
                      - float(sheets42["紧急购电量"]["购电量"].sum())) < 1e-2,
              f"计划 {plan_kwh:,.1f} kWh｜紧急 {r42['emergency_kwh']:.2f} kWh")
    else:
        print("（Q4-2 最优方案不是 Q2 经验分位，跳过 D1/D2 重跑对照）")

    w43 = summary["q43_winner"]
    # Q4-3 结果表由 q4_models 写出；用同一配置重跑并对照合计
    try:
        r43, _ = q.q43_backtest(
            w43.get("name", "stochastic_dynamic_price"), L, P, F, Lhat, prices_all, phat,
            mode=w43.get("scenario", "joint_block"),
            risk_lambda=float(w43.get("risk_lambda", 0)),
            alpha=float(w43.get("alpha", 0.95)),
            deterministic=(w43.get("name") == "deterministic_dynamic_price"),
            pv_updates=bool(w43.get("pv_updates", True)),
            price_updates=bool(w43.get("price_updates", True)),
        )
        sheet_total = float(sheets43["调整购电量"]["全天购电费"].sum())
        check("D3 重跑 Q4-3 最优方案 -> 总费用与表内一致", abs(r43["total_cost"] - sheet_total) < 1.0,
              f"重跑 {r43['total_cost']:,.2f} vs 表 {sheet_total:,.2f} 元")
        check("D4 重跑 Q4-3 -> 紧急购电量与表内一致",
              abs(r43["emergency_kwh"] - float(sheets43["紧急购电量"]["购电量"].sum())) < 1e-2,
              f"{r43['emergency_kwh']:.2f} kWh")
    except Exception as exc:  # pragma: no cover
        check("D3 重跑 Q4-3 最优方案", False, f"{type(exc).__name__}: {exc}")

    bad = [r for r in results if not r[1]]
    print("\n" + "=" * 62)
    print(f"共 {len(results)} 项，通过 {len(results) - len(bad)}，失败 {len(bad)}")
    for name, _, detail in bad:
        print(f"  FAIL: {name} {detail}")
    print("=" * 62)


if __name__ == "__main__":
    main()
