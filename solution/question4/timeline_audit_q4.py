# -*- coding: utf-8 -*-
"""问题 4 的时间一致性审计：重点验证**不泄漏未来价格**。

Q4 相比 Q2/Q3 新增了随机实时电价，最需要检查的是"决策是否偷看了未来价格"。
思路同 Q3：把未来数据替换成垃圾值，检查当前决策是否逐位不变。

  A1 污染第 d 天之后所有天的真实价格 / 价格预报 / 负荷 / 光伏 -> 第 d 天决策不变
  A2 污染第 d 天某时刻起的**真实价格** -> 该时刻之前做出的承诺不变（价格不泄漏的核心）
  A3 污染第 d 天更晚发布时刻的 PV 预报 -> 0:00 全天计划不变
  A4 污染第 d 天自身的真实价格 -> Q4-2（0:00 一次性决策）的全天计划不变

用法：``python timeline_audit_q4.py [日期]``（默认 2025-06-21）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import q4_models as q  # noqa: E402

JUNK = 999.0
PASS, FAIL = "PASS", "FAIL"


def same(a, b):
    return np.array_equal(np.asarray(a, float), np.asarray(b, float))


def prepare():
    L, P, netall = q.load_data()
    _, F = q.load_all()
    prices_all, dates = q.load_prices_dates()
    z = np.load(q.OUT / "price_forecasts.npz")
    price_hat = z["HistGB_XGB_fallback"]
    Lhat = np.array([q.fourier(L, d) if d >= 7 else L[:max(d, 1)].mean(0)
                     for d in range(365)])
    phat = np.vstack([prices_all[:q.START], price_hat])
    day = next(i for i, x in enumerate(dates) if str(x)[:10] == DATE)
    return L, P, netall, F, prices_all, phat, Lhat, day, dates


DATE = sys.argv[1] if len(sys.argv) > 1 else "2025-06-21"


def main() -> None:
    L, P, netall, F, prices_all, phat, Lhat, d, dates = prepare()
    print(f"审计日期 {DATE}（第 {d} 天）｜引擎 q4_models（Q4-3 联合块场景 + Q4-2 分位）\n")
    s0 = q.TARGET

    def q43(prices, phat_use):
        return q.q43_day(d, L, P, F, Lhat, prices, phat_use, s0, mode="joint_block",
                         risk_lambda=0, alpha=0.95)

    base = q43(prices_all, phat)

    # A1 未来日期
    pr2, ph2, L2, P2 = prices_all.copy(), phat.copy(), L.copy(), P.copy()
    pr2[d + 1:] = JUNK
    ph2[d + 1:] = JUNK
    L2[d + 1:] = JUNK
    P2[d + 1:] = JUNK
    other = q.q43_day(d, L2, P2, F, Lhat, pr2, ph2, s0, mode="joint_block")
    a1 = same(base["g0"], other["g0"]) and same(base["gfinal"], other["gfinal"]) \
        and same(base["soc"], other["soc"]) and same(base["e"], other["e"])
    print(f"[A1] 污染未来日期的价格/预报/负荷/光伏 -> 当天决策与执行逐位一致 : "
          f"{PASS if a1 else FAIL}")

    # A2 当天未来时段的真实价格（价格不泄漏）
    for cut, name in ((36, "06:00"), (72, "12:00"), (108, "18:00")):
        pr3 = prices_all.copy()
        pr3[d, cut:] = JUNK
        other = q43(pr3, phat)
        ok = same(base["g0"], other["g0"]) and same(base["gfinal"][:cut], other["gfinal"][:cut])
        print(f"[A2] 污染当天 {name} 起的真实价格 -> 该时刻之前的承诺与执行不变 : "
              f"{PASS if ok else FAIL}")

    # A3 更晚发布时刻的 PV 预报
    F3 = F.copy()
    F3[d, 1:] = JUNK
    other = q.q43_day(d, L, P, F3, Lhat, prices_all, phat, s0, mode="joint_block")
    a3 = same(base["g0"], other["g0"])
    print(f"[A3] 污染 6:00/12:00/18:00 的 PV 预报 -> 0:00 全天计划不变 : {PASS if a3 else FAIL}")

    # A4 Q4-2：只有"更晚的天"的真实价格被污染时，较早当天的 0:00 计划不变
    #     （q42_backtest 只用 prices[:i] 构造价格误差历史，结算价不进决策）
    i0 = d - q.START
    sl = slice(i0, i0 + 3)
    net = netall[q.START:][sl]
    net_hat = np.array([q.fourier(netall, x) for x in range(d, d + 3)])
    prices = prices_all[q.START:][sl].copy()
    pz = np.load(q.OUT / "price_forecasts.npz")["HistGB_XGB_fallback"][sl]
    r_base, d_base = q.q42_backtest("audit", net, net_hat, prices, pz, mode="joint_block")
    prices_junk = prices.copy()
    prices_junk[1:] = JUNK                      # 只污染后两天的真实价格
    r_junk, d_junk = q.q42_backtest("audit", net, net_hat, prices_junk, pz, mode="joint_block")
    a4 = same(d_base[0]["g"], d_junk[0]["g"])
    print(f"[A4] Q4-2：污染后续日的真实价格 -> 首日 0:00 计划不变（只影响结算）: "
          f"{PASS if a4 else FAIL}")

    ok = a1 and a3 and a4
    print(f"\n结论: {'全部通过' if ok else '存在未通过项'}")


if __name__ == "__main__":
    main()
