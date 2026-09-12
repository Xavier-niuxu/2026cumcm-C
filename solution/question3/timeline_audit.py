# -*- coding: utf-8 -*-
"""时间一致性审计：验证问题 3 的决策**不使用任何未来数据**。

做法是把"未来"的数据替换成垃圾值，检查决策是否逐位不变：

  A1 污染第 d 天之后所有天的实测负荷/光伏、附件 3 预报与日前负荷预报矩阵
     -> 第 d 天的 run_day 输出（计划、调整量、充放电、SOC、紧急购电、费用）必须完全一致
  A2 污染第 d 天 6:00/12:00/18:00 的光伏预报 -> 0:00 制定的全天计划必须不变
  A3 污染第 d 天某时刻起的实测负荷/光伏 -> 在该时刻之前求解的节点决策必须不变
  A4 污染第 d 天自身及之后所有天的实际值 -> 日前负荷预报 Lhat[d] 必须不变
  A5 污染误差池中第 d 天及之后的行 -> 同一时刻的历史误差样本必须不变

用法：``python timeline_audit.py [日期]``（默认 2025-06-21）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rolling_q3 as R  # noqa: E402
from q2_bridge import load_data, fourier  # noqa: E402
from forecast_analysis import load_all  # noqa: E402

JUNK = 987654.0
PASS, FAIL = "PASS", "FAIL"


def prepare():
    L, P, _ = load_data()
    P2, F = load_all()
    p, dates = R.price_dates()
    Lhat = np.array([fourier(L, d) if d >= 7 else L[:max(d, 1)].mean(0)
                     for d in range(R.LAST_DAY)])
    return L, P, F, Lhat, p, dates


def day_index(dates, date_str) -> int:
    for i, d in enumerate(dates):
        if str(d)[:10] == date_str:
            return i
    raise ValueError(date_str)


def run_day_spy(d, L, P, F, Lhat, p, **kw):
    """跑一天并记录每个节点的购电承诺 q（通过临时替换 optimize_commit）。"""
    records = []
    orig = R.optimize_commit

    def spy(point, pp, s0, prev=None, scenarios=None, weights=None,
            settlement='refund50'):
        q, obj = orig(point, pp, s0, prev, scenarios, weights, settlement)
        records.append(q.copy())
        return q, obj

    R.optimize_commit = spy
    try:
        res = R.run_day(d, L, P, F, Lhat, p, **kw)
    finally:
        R.optimize_commit = orig
    return res, records


def same(a, b):
    return np.array_equal(np.asarray(a, float), np.asarray(b, float))


def main():
    date_str = sys.argv[1] if len(sys.argv) > 1 else "2025-06-21"
    L, P, F, Lhat, p, dates = prepare()
    d = day_index(dates, date_str)
    print(f"审计日期 {date_str}（第 {d} 天）\n")

    base, base_q = run_day_spy(d, L, P, F, Lhat, p)

    # A1 未来日期
    L2, P2, F2, Lhat2 = L.copy(), P.copy(), F.copy(), Lhat.copy()
    L2[d + 1:] = JUNK
    P2[d + 1:] = JUNK
    F2[d + 1:] = JUNK
    Lhat2[d + 1:] = JUNK
    other, other_q = run_day_spy(d, L2, P2, F2, Lhat2, p)
    a1 = all(same(base[k], other[k]) for k in ("g0", "gfinal", "c", "d", "soc", "emergency")) \
        and abs(base["total_cost"] - other["total_cost"]) < 1e-9
    print(f"[A1] 污染未来日期 -> 第 d 天输出逐位一致 : {PASS if a1 else FAIL}")

    # A2 当天更晚的发布时刻的预报
    F3 = F.copy()
    F3[d, 1:] = JUNK
    other, _ = run_day_spy(d, L, P, F3, Lhat, p)
    a2 = same(base["g0"], other["g0"])
    print(f"[A2] 污染 6:00/12:00/18:00 预报 -> 0:00 全天计划不变 : {PASS if a2 else FAIL}")

    # A3 当天未来时段
    for cut, nodes in ((36, 2), (72, 3), (108, 4)):
        L4, P4 = L.copy(), P.copy()
        L4[d, cut:] = JUNK
        P4[d, cut:] = JUNK
        other, other_q = run_day_spy(d, L4, P4, F, Lhat, p)
        ok = len(other_q) == len(base_q) and all(
            same(base_q[i], other_q[i]) for i in range(min(nodes, len(base_q)))
        ) and same(base["g0"], other["g0"])
        print(f"[A3] 污染当天 {cut * 10 // 60:02d}:00 起实际值 -> 该时刻前 "
              f"{nodes} 个节点决策不变 : {PASS if ok else FAIL}")

    # A4 日前负荷预报的因果性
    L5, P5 = L.copy(), P.copy()
    L5[d:] = JUNK
    P5[d:] = JUNK
    from pred.fourier import FourierPredictor
    date = str(dates[d])[:10]
    m1 = FourierPredictor(dates, L, dates, np.zeros_like(L))
    m2 = FourierPredictor(dates, L5, dates, np.zeros_like(L5))
    q1, _ = m1.predict(date)
    q2, _ = m2.predict(date)
    a4 = same(q1, q2)
    print(f"[A4] 污染当天及以后实际值 -> 日前负荷预报不变 : {PASS if a4 else FAIL}"
          f"   (max|Δ| = {np.abs(q1 - q2).max():.3e})")

    # A5 历史误差样本只取 d 之前的天（把 d 及之后的行全部污染后样本应不变）
    L6, P6, F6, Lhat6 = L.copy(), P.copy(), F.copy(), Lhat.copy()
    L6[d:] = JUNK
    P6[d:] = JUNK
    F6[d:] = JUNK
    Lhat6[d:] = JUNK
    j = 1                                        # 6:00 发布时刻
    hist_base = R.histories(L, P, F, Lhat, d, j)
    hist_junk = R.histories(L6, P6, F6, Lhat6, d, j)
    a5 = same(hist_base, hist_junk)
    print(f"[A5] 误差样本只用 d 之前的天（每时刻样本 {hist_base.shape}）: "
          f"{PASS if a5 else FAIL}")

    ok = a1 and a2 and a4 and a5
    print(f"\n结论: {'全部通过' if ok else '存在未通过项'}")


if __name__ == "__main__":
    main()
