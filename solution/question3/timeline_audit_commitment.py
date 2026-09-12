# -*- coding: utf-8 -*-
"""时间一致性审计（针对 commitment 引擎，即 1370.6868 万元的主模型）。

``timeline_audit.py`` 审计的是 legacy 两阶段引擎（``rolling.run_day``）；
本脚本用同样的"污染未来数据"思路审计 ``commitment.run_day_commitment``：

  A1 污染第 d 天之后所有天的实测负荷/光伏、附件 3 预报、负荷误差池与光伏误差池
     -> 第 d 天全部输出（全天计划、各节点承诺、执行量、SOC、紧急购电、费用）逐位不变
  A2 污染第 d 天 6:00/12:00/18:00 的光伏预报 -> 0:00 全天计划不变
  A3 污染第 d 天某时刻起的实测值 -> 该时刻之前各节点的购电承诺不变
  A4 污染第 d 天及以后的实际值 -> 因果负荷预报 CausalFourierLoadModel.predict(d) 不变
  A5 污染误差池第 d 天及以后的行 -> combined_error_history 给出的历史样本不变

用法：``python timeline_audit_commitment.py [日期]``（默认 2025-06-21）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data_loader import (  # noqa: E402
    day_index_of,
    load_actual,
    load_price,
    load_pv_forecast,
)
from commitment import (  # noqa: E402
    CausalFourierLoadModel,
    causal_load_error_pool,
    combined_error_history,
    run_day_commitment,
)
from scenarios import pv_error_pool  # noqa: E402

JUNK = 987654.0
PASS, FAIL = "PASS", "FAIL"
S_INIT = 6000.0


def prepare():
    dates, load, pv = load_actual()
    _, fc = load_pv_forecast()
    price = load_price()
    model = CausalFourierLoadModel(dates, load)
    load_err = causal_load_error_pool(model, dates, load)
    pv_err = tuple(pv_error_pool(pv, fc, i, method="linear") for i in range(4))
    return dates, load, pv, fc, price, model, load_err, pv_err


def run(d, dates, load, pv, fc, price, model, load_err, pv_err,
        nodes=(0, 1, 2, 3), s_init=S_INIT):
    load_fc, _ = model.predict(str(dates[d])[:10])
    return run_day_commitment(
        day=d, dates=dates, load_actual=load, pv_actual=pv, pv_fc_day=fc[d],
        price=price, load_fc=load_fc, load_err=load_err, pv_err_pools=pv_err,
        n_scenarios=5, lookback=90, nodes=nodes, stochastic=True, s_init=s_init,
    )


def same(a, b):
    return np.array_equal(np.asarray(a, float), np.asarray(b, float))


def main():
    date_str = sys.argv[1] if len(sys.argv) > 1 else "2025-06-21"
    dates, load, pv, fc, price, model, load_err, pv_err = prepare()
    d = day_index_of(dates, date_str)
    print(f"审计日期 {date_str}（第 {d} 天），引擎 commitment，5 场景\n")

    base = run(d, dates, load, pv, fc, price, model, load_err, pv_err)

    # A1 未来日期
    load2, pv2, fc2 = load.copy(), pv.copy(), fc.copy()
    load2[d + 1:] = JUNK
    pv2[d + 1:] = JUNK
    fc2[d + 1:] = JUNK
    le2 = load_err.copy()
    le2[d + 1:] = JUNK
    pe2 = tuple(p.copy() for p in pv_err)
    for p in pe2:
        p[d + 1:] = JUNK
    other = run(d, dates, load2, pv2, fc2, price, model, le2, pe2)
    a1 = all(same(base[k], other[k]) for k in
             ("plan", "adjusted", "charge", "discharge", "soc", "emergency")) \
        and abs(base["total_cost"] - other["total_cost"]) < 1e-9
    print(f"[A1] 污染未来日期 -> 当天输出逐位一致 : {PASS if a1 else FAIL}")

    # A2 当天更晚的发布时刻
    fc3 = fc.copy()
    fc3[d, 1:] = JUNK
    other = run(d, dates, load, pv, fc3, price, model, load_err, pv_err)
    a2 = same(base["plan"], other["plan"])
    print(f"[A2] 污染 6:00/12:00/18:00 预报 -> 0:00 全天计划不变 : {PASS if a2 else FAIL}")

    # A3 当天未来时段
    for cut, n_nodes in ((36, 2), (72, 3), (108, 4)):
        load4, pv4 = load.copy(), pv.copy()
        load4[d, cut:] = JUNK
        pv4[d, cut:] = JUNK
        other = run(d, dates, load4, pv4, fc, price, model, load_err, pv_err)
        ok = same(base["plan"], other["plan"]) and all(
            same(base["node_info"][i]["purchase_remaining"],
                 other["node_info"][i]["purchase_remaining"])
            for i in range(min(n_nodes, len(base["node_info"])))
        )
        print(f"[A3] 污染当天 {cut * 10 // 60:02d}:00 起实际值 -> 该时刻前 {n_nodes} "
              f"个节点承诺不变 : {PASS if ok else FAIL}")

    # A4 因果负荷预报
    load5 = load.copy()
    load5[d:] = JUNK
    m2 = CausalFourierLoadModel(dates, load5)
    p1, _ = model.predict(date_str)
    p2, _ = m2.predict(date_str)
    a4 = same(p1, p2)
    print(f"[A4] 污染当天及以后实际值 -> 负荷预报不变 : {PASS if a4 else FAIL}"
          f"   (max|Δ| = {np.abs(p1 - p2).max():.3e})")

    # A5 历史误差样本
    le6 = load_err.copy()
    le6[d:] = JUNK
    pe6 = tuple(p.copy() for p in pv_err)
    for p in pe6:
        p[d:] = JUNK
    h1 = combined_error_history(day=d, at_slot=36, load_err=load_err,
                                pv_err=pv_err[1], lookback=90)
    h2 = combined_error_history(day=d, at_slot=36, load_err=le6, pv_err=pe6[1],
                                lookback=90)
    a5 = same(h1, h2)
    print(f"[A5] 误差样本只用 d 之前的天（样本形状 {h1.shape}）: {PASS if a5 else FAIL}")

    print(f"\n结论: {'全部通过' if (a1 and a2 and a4 and a5) else '存在未通过项'}")


if __name__ == "__main__":
    main()
