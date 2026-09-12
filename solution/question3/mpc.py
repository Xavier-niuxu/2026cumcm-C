# -*- coding: utf-8 -*-
"""Rolling stochastic MPC: one decision node of Question 3.

At each issue time (0:00 / 6:00 / 12:00 / 18:00) the microgrid solves a
scenario-based stochastic program over the remaining slots of the day:

    第一时段（到下一个发布时刻为止的 36 个 10 min 周期）
        = 一阶段决策 g, c, d, SOC，对所有场景相同（非预期性约束）；
    其余时段
        = 二阶段追索变量，逐场景可以不同（该场景下的最优调整与充放电）。

经济结构（问题 3，逐时段，等价形式）：

    p*ga + 0.5*p*|g0 - ga| + 5*p*e

它与题目原文的结算口径一致：实际调整购电按原价支付，调减的取消部分再付原价
50% 的违约金（相对原计划节省 50%），调增的超出部分按原价 1.5 倍支付。

    ga <= g0 :  p*ga + 0.5*p*(g0-ga) = p*g0 - 0.5*p*(g0-ga)
    ga >= g0 :  p*g0 + 1.5*p*(ga-g0) = p*ga + 0.5*p*(ga-g0)

``ga`` 是最终执行的购电量，紧急购电由实际净负荷与 ga 的缺口决定：

    e >= net_load + c - d - ga,   e >= 0

目标为情景加权期望费用；``cvar_weight>0`` 时加入 CVaR 风险项
（Rockafellar-Uryasev 线性化）。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, lil_matrix

from data_loader import (
    DELTA_T,
    EMERGENCY_MULTIPLIER,
    ETA,
    N_SLOTS,
    P_MAX,
    PLAN_DEVIATION_RATE,
    S_MAX,
    S_MIN,
)

P_MAX_SLOT = P_MAX * DELTA_T          # 833.3333 kWh per 10-min slot
BLOCK_SLOTS = 36                      # 6 h between two forecast issues


class _Rows:
    """Accumulate sparse constraint rows as dicts, then freeze to CSR."""

    def __init__(self):
        self.rows = []
        self.rhs = []

    def add(self, coeffs: dict, b: float = 0.0):
        self.rows.append({k: float(v) for k, v in coeffs.items() if v != 0.0})
        self.rhs.append(float(b))

    def freeze(self, nv: int):
        if not self.rows:
            return csr_matrix((0, nv)), np.zeros(0)
        a = lil_matrix((len(self.rows), nv))
        for i, row in enumerate(self.rows):
            for j, v in row.items():
                a[i, j] = v
        return a.tocsr(), np.asarray(self.rhs)


def solve_node(
    *,
    price: np.ndarray,
    net_scen: np.ndarray,
    weights: np.ndarray,
    at_slot: int,
    s_init: float,
    s_end: float,
    g0: np.ndarray | None = None,
    block: int = BLOCK_SLOTS,
    cvar_weight: float = 0.0,
    cvar_alpha: float = 0.95,
    future_adjust: bool = True,
) -> dict:
    """Solve one MPC node and return the first-stage (committed) decisions.

    Args:
        price:     (144,) 正常电价，元/kWh。
        net_scen:  (N,144) 净负荷场景，单位 kWh/时段；``at_slot`` 之前填 0。
        weights:   (N,) 场景权重（和为 1）。
        at_slot:   本节点起始时段（0/36/72/108）。
        s_init:    本节点起始储电量，kWh。
        s_end:     日末储电量目标，kWh。
        g0:        计划购电量（144,）。``None`` 表示本节点同时制定计划（0:00）。
        block:     一阶段覆盖的时段数。
        cvar_weight: CVaR 权重 λ∈[0,1)。
        cvar_alpha:  CVaR 置信水平。
        future_adjust: 0:00 节点是否把"未来的调整"建成追索变量。``True``
            （默认）把计划当作罚金基准、后续时段可逐场景调整；``False`` 把
            0:00 计划直接当作随后执行的购电量，仅储能与紧急购电是追索，
            与参考解 node-0 LP 的口径一致。

    Returns:
        含一阶段决策与诊断量的字典。
    """
    price = np.asarray(price, dtype=float)
    net_scen = np.asarray(net_scen, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n_scen = net_scen.shape[0]

    slots = np.arange(at_slot, N_SLOTS)
    comm = np.arange(at_slot, min(at_slot + block, N_SLOTS))
    rec = (
        np.arange(comm[-1] + 1, N_SLOTS)
        if comm[-1] + 1 < N_SLOTS
        else np.array([], dtype=int)
    )
    nc, nr, ns = comm.size, rec.size, slots.size
    plan_decided = g0 is None
    comm_set = set(comm.tolist())
    # 0:00 节点是否把"未来调整"建成追索；其它节点本来就是调整节点。
    model_adjust = future_adjust or not plan_decided

    # ---------------- 变量分配 ----------------
    layout = {}
    bounds = []
    nv = 0

    def new(name, size, lo, hi):
        nonlocal nv
        layout[name] = (nv, size)
        bounds.extend([(lo, hi)] * size)
        nv += size
        return layout[name][0]

    def idx(name, k):
        return layout[name][0] + k

    if plan_decided:
        new("plan", N_SLOTS, 0.0, None)
    else:
        new("ga_c", nc, 0.0, None)
    new("c_c", nc, 0.0, P_MAX_SLOT)
    new("d_c", nc, 0.0, P_MAX_SLOT)
    new("s_c", nc + 1, S_MIN, S_MAX)
    if nr:
        if model_adjust:
            new("ga_r", n_scen * nr, 0.0, None)
        new("c_r", n_scen * nr, 0.0, P_MAX_SLOT)
        new("d_r", n_scen * nr, 0.0, P_MAX_SLOT)
        if nr > 1:
            new("s_r", n_scen * (nr - 1), S_MIN, S_MAX)
        if model_adjust:
            new("u_r", n_scen * nr, 0.0, None)
            new("v_r", n_scen * nr, 0.0, None)
    new("e", n_scen * ns, 0.0, None)
    if not plan_decided:
        new("u_c", nc, 0.0, None)
        new("v_c", nc, 0.0, None)

    obj = np.zeros(nv)
    eq_rows, ub_rows = _Rows(), _Rows()
    scen_cost = lil_matrix((n_scen, nv))

    def ga_var(t, s):
        """执行购电量 ga 的变量下标；committed 块为共享的一阶段变量。"""
        if t in comm_set:
            i = t - at_slot
            return idx("plan", t) if plan_decided else idx("ga_c", i)
        if not model_adjust:            # 0:00 的计划即最终购电量
            return idx("plan", t)
        return idx("ga_r", s * nr + (t - rec[0]))

    # ---------------- 购电能量费 p*ga（按实际执行量结算） ----------------
    if plan_decided:
        # 一阶段块：ga = 计划量；之后的时段由追索变量 ga_r 决定
        for t in comm:
            obj[idx("plan", t)] += price[t]
    else:
        for i, t in enumerate(comm):
            obj[idx("ga_c", i)] += price[t]
    if model_adjust:
        for s in range(n_scen):
            for j, t in enumerate(rec):
                scen_cost[s, idx("ga_r", s * nr + j)] += price[t]

    # ---------------- 一阶段 SOC 递推 ----------------
    eq_rows.add({idx("s_c", 0): 1.0}, s_init)
    for i in range(nc):
        eq_rows.add(
            {
                idx("s_c", i + 1): 1.0,
                idx("s_c", i): -1.0,
                idx("c_c", i): -ETA,
                idx("d_c", i): 1.0 / ETA,
            },
            0.0,
        )
    if nr == 0:                                   # 最后一个节点：直接约束日末 SOC
        eq_rows.add({idx("s_c", nc): 1.0}, s_end)

    # ---------------- 追索 SOC 递推（逐场景） ----------------
    for s in range(n_scen):
        for j in range(nr):
            prev = idx("s_c", nc) if j == 0 else idx("s_r", s * (nr - 1) + j - 1)
            row = {
                prev: -1.0,
                idx("c_r", s * nr + j): -ETA,
                idx("d_r", s * nr + j): 1.0 / ETA,
            }
            if j == nr - 1:                       # 日末 SOC = s_end
                eq_rows.add(row, -s_end)
            else:
                row[idx("s_r", s * (nr - 1) + j)] = 1.0
                eq_rows.add(row, 0.0)

    # ---------------- 紧急购电 & 场景费用 ----------------
    for s in range(n_scen):
        for pos, t in enumerate(slots):
            e_var = idx("e", s * ns + pos)
            # e >= net_load + c - d - ga
            # <=>  -e + c - d - ga <= -net_load
            row = {e_var: -1.0, ga_var(t, s): -1.0}
            if t in comm_set:
                i = t - at_slot
                row[idx("c_c", i)] = 1.0
                row[idx("d_c", i)] = -1.0
            else:
                j = t - rec[0]
                row[idx("c_r", s * nr + j)] = 1.0
                row[idx("d_r", s * nr + j)] = -1.0
            ub_rows.add(row, -net_scen[s, t])
            scen_cost[s, e_var] += EMERGENCY_MULTIPLIER * price[t]

    # ---------------- 调减违约金 / 调增电价 ----------------
    def add_penalty_rows(t, ga_col, u_col, v_col):
        """u >= max(0, plan - ga), v >= max(0, ga - plan)."""
        if plan_decided:      # 计划本身是本节点的决策变量
            ub_rows.add({idx("plan", t): 1.0, ga_col: -1.0, u_col: -1.0}, 0.0)
            ub_rows.add({ga_col: 1.0, idx("plan", t): -1.0, v_col: -1.0}, 0.0)
        else:                 # 计划已在 0:00 固定
            ub_rows.add({ga_col: -1.0, u_col: -1.0}, -g0[t])
            ub_rows.add({ga_col: 1.0, v_col: -1.0}, g0[t])

    if not plan_decided:
        for i, t in enumerate(comm):
            add_penalty_rows(t, idx("ga_c", i), idx("u_c", i), idx("v_c", i))
            obj[idx("u_c", i)] += PLAN_DEVIATION_RATE * price[t]
            obj[idx("v_c", i)] += PLAN_DEVIATION_RATE * price[t]

    if model_adjust:
        for s in range(n_scen):
            for j, t in enumerate(rec):
                col = s * nr + j
                add_penalty_rows(t, idx("ga_r", col), idx("u_r", col), idx("v_r", col))
                scen_cost[s, idx("u_r", col)] += PLAN_DEVIATION_RATE * price[t]
                scen_cost[s, idx("v_r", col)] += PLAN_DEVIATION_RATE * price[t]

    # ---------------- 组装目标（含可选 CVaR） ----------------
    scen_cost = scen_cost.tocsr()
    exp_cost_vec = np.asarray(weights @ scen_cost).ravel()
    obj = obj + (1.0 - cvar_weight) * exp_cost_vec

    if cvar_weight > 0:
        eta_var = nv
        z0 = nv + 1
        nv += 1 + n_scen
        bounds.extend([(None, None)])
        bounds.extend([(0.0, None)] * n_scen)
        obj = np.concatenate([obj, np.zeros(1 + n_scen)])
        obj[eta_var] += cvar_weight
        obj[z0:z0 + n_scen] += cvar_weight / ((1.0 - cvar_alpha) * n_scen)
        # C_s x - eta - z_s <= 0
        for s in range(n_scen):
            coo = scen_cost.getrow(s).tocoo()
            row = {int(j): float(v) for j, v in zip(coo.col, coo.data)}
            row[eta_var] = -1.0
            row[z0 + s] = -1.0
            ub_rows.add(row, 0.0)

    a_ub, b_ub = ub_rows.freeze(nv)
    a_eq, b_eq = eq_rows.freeze(nv)

    res = linprog(
        c=obj,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(
            f"stochastic node LP failed at slot {at_slot}: "
            f"status={res.status}, message={res.message}"
        )

    x = res.x
    plan_out = x[idx("plan", 0):idx("plan", 0) + N_SLOTS] if plan_decided else None
    ga_comm = (
        plan_out[comm] if plan_decided else x[idx("ga_c", 0):idx("ga_c", 0) + nc]
    )
    c_comm = x[idx("c_c", 0):idx("c_c", 0) + nc]
    d_comm = x[idx("d_c", 0):idx("d_c", 0) + nc]
    s_comm = x[idx("s_c", 0):idx("s_c", 0) + nc + 1]

    # 诊断：本节点优化得到的期望紧急购电与期望调整费用
    e_mat = x[idx("e", 0):idx("e", 0) + n_scen * ns].reshape(n_scen, ns)
    exp_emergency_kwh = float((weights[:, None] * e_mat).sum())

    return {
        "at_slot": at_slot,
        "plan": plan_out,
        "ga_committed": ga_comm,
        "c_committed": c_comm,
        "d_committed": d_comm,
        "soc": s_comm,
        "objective": float(res.fun),
        "expected_emergency_kwh": exp_emergency_kwh,
        "n_vars": nv,
        "n_scen": n_scen,
    }
