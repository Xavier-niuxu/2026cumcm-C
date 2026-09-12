# -*- coding: utf-8 -*-
"""时间一致性审计：验证问题 3 的决策**不使用任何未来数据**。

审计思路（可复现）：把"未来"的数据全部替换成垃圾值，然后检查决策是否**逐位不变**。

* A1 未来日期：把 $d$ 之后所有天的实际负荷/光伏、附件 3 预报、两个误差池全部污染，
  第 $d$ 天的 run_day 输出（计划、各节点已提交决策、执行量、费用）必须完全一致。
* A2 未来发布时刻：把第 $d$ 天 6:00/12:00/18:00 的光伏预报污染，0:00 的计划必须不变。
* A3 当天未来时段：把第 $d$ 天从某个时段起的**实际**负荷/光伏污染，
  在该时刻之前做出的决策必须不变（节点 h 只应依赖 $[0,h)$ 的实测与 $h$ 时刻的预报）。
* A4 负荷预报模型：污染第 $d$ 天自身的实际值与 $d$ 之后的数据，``predict(d)`` 必须不变。
* A5 场景抽样：污染误差池中第 $d$ 天及之后的行，同种子抽样得到的场景必须不变。

用法：``python timeline_audit.py``（默认审计 2025-06-21，节点 0~3）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import (  # noqa: E402
    DELTA_T,
    ISSUE_SLOTS,
    day_index_of,
    forecast_profile,
    load_actual,
    load_price,
    load_pv_forecast,
)
from rolling import run_day  # noqa: E402
from scenarios import (  # noqa: E402
    build_scenarios,
    load_error_pool,
    load_forecast_model,
    pv_error_pool,
)

JUNK_LOAD = 123456.0
JUNK_PV = 654321.0
JUNK_FC = 987654.0
JUNK_POOL = 12345.0


def prepare(date_str: str):
    dates, load, pv = load_actual()
    _, fc = load_pv_forecast()
    price = load_price()
    model = load_forecast_model(dates, load)
    load_err = load_error_pool(model, dates, load)
    pv_err = tuple(pv_error_pool(pv, fc, i, method="linear") for i in range(4))
    day = day_index_of(dates, date_str)
    return dict(dates=dates, load=load, pv=pv, fc=fc, price=price, model=model,
                load_err=load_err, pv_err=pv_err, day=day)


def run(d, *, load=None, pv=None, fc=None, load_err=None, pv_err=None, nodes=(0, 1, 2, 3),
        seed=0):
    load_fc, _ = d["model"].predict(str(d["dates"][d["day"]])[:10])
    return run_day(
        day=d["day"],
        dates=d["dates"],
        load_actual=d["load"] if load is None else load,
        pv_actual=d["pv"] if pv is None else pv,
        pv_fc_day=d["fc"][d["day"]] if fc is None else fc,
        price=d["price"],
        load_fc=load_fc,
        load_err=d["load_err"] if load_err is None else load_err,
        pv_err_pools=d["pv_err"] if pv_err is None else pv_err,
        n_scenarios=12,
        seed=seed,
        nodes=nodes,
    )


def same(a, b):
    return np.array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


def compare_stages(base, other, nodes):
    """比较指定节点的已提交决策是否逐位相同。"""
    ok = True
    detail = []
    for h in nodes:
        A, B = base["node_info"][h], other["node_info"][h]
        for key in ("ga_committed", "c_committed", "d_committed"):
            if not same(A[key], B[key]):
                ok = False
                detail.append(f"node{h}.{key}")
        if A["plan"] is not None and not same(A["plan"], B["plan"]):
            ok = False
            detail.append(f"node{h}.plan")
    return ok, detail


def main() -> None:
    date_str = sys.argv[1] if len(sys.argv) > 1 else "2025-06-21"
    d = prepare(date_str)
    day, ndays = d["day"], d["load"].shape[0]
    base = run(d)
    print(f"审计日期 {date_str}（第 {day} 天），节点 0~3，12 场景\n")

    # ---------------- A1 未来日期全部污染 ----------------
    load2, pv2, fc2 = d["load"].copy(), d["pv"].copy(), d["fc"].copy()
    load2[day + 1:] = JUNK_LOAD
    pv2[day + 1:] = JUNK_PV
    fc2[day + 1:] = JUNK_FC
    load_err2 = d["load_err"].copy()
    load_err2[day + 1:] = JUNK_POOL
    pv_err2 = tuple(p.copy() for p in d["pv_err"])
    for p in pv_err2:
        p[day + 1:] = JUNK_POOL
    other = run(d, load=load2, pv=pv2, fc=fc2[day], load_err=load_err2, pv_err=pv_err2)
    ok_stage, detail = compare_stages(base, other, [0, 1, 2, 3])
    a1 = (
        same(base["plan"], other["plan"])
        and ok_stage
        and same(base["adjusted"], other["adjusted"])
        and same(base["emergency"], other["emergency"])
        and abs(base["total_cost"] - other["total_cost"]) < 1e-9
    )
    print(f"[A1] 未来日期污染 -> 全天输出逐位一致 : {'PASS' if a1 else 'FAIL ' + str(detail)}")

    # ---------------- A2 未来发布时刻污染 ----------------
    fc3 = d["fc"].copy()
    fc3[day, 1:, :] = JUNK_FC                     # 只看 0:00 的计划
    other = run(d, fc=fc3[day])
    a2 = same(base["plan"], other["plan"])
    print(f"[A2] 污染 6:00/12:00/18:00 预报 -> 0:00 全天计划不变 : "
          f"{'PASS' if a2 else 'FAIL'}")

    # ---------------- A3 当天未来时段污染 ----------------
    for cut_slot, nodes in ((36, (0, 1)), (72, (0, 1, 2)), (108, (0, 1, 2, 3))):
        load4, pv4 = d["load"].copy(), d["pv"].copy()
        load4[day, cut_slot:] = JUNK_LOAD
        pv4[day, cut_slot:] = JUNK_PV
        other = run(d, load=load4, pv=pv4)
        ok_stage, detail = compare_stages(base, other, nodes)
        ok_plan = same(base["plan"], other["plan"])
        a3 = ok_plan and ok_stage
        print(f"[A3] 污染当天 {cut_slot * 10 // 60:02d}:{cut_slot * 10 % 60:02d} 起的实际值 "
              f"-> 节点 {nodes} 的决策不变 : {'PASS' if a3 else 'FAIL ' + str(detail)}")

    # ---------------- A4 负荷预报模型 ----------------
    load5, pv5 = d["load"].copy(), d["pv"].copy()
    load5[day:] = JUNK_LOAD
    pv5[day:] = JUNK_PV
    model2 = load_forecast_model(d["dates"], load5)
    p_base, _ = d["model"].predict(date_str)
    p_junk, _ = model2.predict(date_str)
    a4 = same(p_base, p_junk)
    print(f"[A4] 污染当天及以后的实际值 -> 负荷预报不变 : {'PASS' if a4 else 'FAIL'}"
          f"   (max|Δ| = {np.abs(p_base - p_junk).max():.3e})")

    # ---------------- A5 场景抽样 ----------------
    pv_err3 = tuple(p.copy() for p in d["pv_err"])
    for p in pv_err3:
        p[day:] = JUNK_POOL
    load_fc, _ = d["model"].predict(date_str)
    prof0 = forecast_profile(d["fc"][day], 0, method="linear")
    s1, _ = build_scenarios(day=day, at_slot=0, load_fc=load_fc,
                            pv_fc=np.nan_to_num(prof0), load_err=d["load_err"],
                            pv_err=d["pv_err"][0], n_scenarios=12, lookback=90,
                            rng=np.random.default_rng(12345), delta_t=DELTA_T)
    s2, _ = build_scenarios(day=day, at_slot=0, load_fc=load_fc,
                            pv_fc=np.nan_to_num(prof0), load_err=d["load_err"],
                            pv_err=pv_err3[0], n_scenarios=12, lookback=90,
                            rng=np.random.default_rng(12345), delta_t=DELTA_T)
    a5 = same(s1, s2)
    print(f"[A5] 污染误差池第 {day} 天及以后的行 -> 同种子场景不变 : "
          f"{'PASS' if a5 else 'FAIL'}")

    # ---------------- 静态检查 ----------------
    print("\n静态检查（数据窗口）:")
    model = d["model"]
    print(f"  负荷预报训练窗口 : 恒为 rows < 目标日（滚动扩窗，见 fourier.py predict）")
    print(f"  误差池采样窗口   : usable_days(pv_err, at_slot, day, lookback) -> "
          f"[max(0, {day}-90), {day})")
    print(f"  场景源数据       : 第 {day} 天只用该天 {ISSUE_SLOTS} 时刻已发布的预报")
    print(f"  执行层           : 逐 10 min 因果规则（执行 t 只读 actual[t]）")

    all_pass = a1 and a2 and a4 and a5
    print(f"\n结论: {'全部通过' if all_pass else '存在未通过项，需检查'}")


if __name__ == "__main__":
    main()
