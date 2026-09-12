# -*- coding: utf-8 -*-
"""问题 3 的承诺型滚动随机优化实现。

本模块复现论文中费用约 1,370 万元的主方案。与 ``rolling.py`` 中保留的
两阶段 ``legacy`` 实现不同，本模型在每个预报发布时刻共同决定余下时段的
购电承诺，并用 5 个加权 K-means 日场景评价未来紧急购电风险。只执行到下一
发布时刻；实际净负荷到达后，储能按当前缺额因果响应。

所有预测、误差样本和聚类数据都严格来自目标日之前。Q1/Q2 的源文件不会被
修改；这里的负荷预测器是 Q3 内部的只读兼容实现，用来复现 Q2 傅里叶模型在
最初 7 天暖启动时的历史误差口径。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from data_loader import (
    DELTA_T,
    EMERGENCY_MULTIPLIER,
    ETA,
    ISSUE_SLOTS,
    N_SLOTS,
    P_MAX,
    PLAN_INCREASE_RATE,
    PLAN_REDUCE_REFUND,
    S_INIT,
    S_MAX,
    S_MIN,
    forecast_profile,
)

try:
    from sklearn.cluster import KMeans
except ImportError as exc:  # pragma: no cover - 给出可操作的依赖提示
    raise ImportError(
        "承诺型 Q3 引擎需要 scikit-learn；请安装 solution/question3/requirements.txt"
    ) from exc


P_MAX_SLOT = P_MAX * DELTA_T


def _calendar(day: int) -> np.ndarray:
    """与 Q2 傅里叶回归等价的日历特征（含冗余截距以复现暖启动）。"""
    return np.r_[
        1.0,
        np.eye(7)[day % 7],
        np.sin(2 * np.pi * day / 365.0),
        np.cos(2 * np.pi * day / 365.0),
        np.sin(4 * np.pi * day / 365.0),
        np.cos(4 * np.pi * day / 365.0),
    ]


class CausalFourierLoadModel:
    """Q3 内部的严格因果傅里叶负荷预测器。

    第 ``d`` 天只拟合 ``[0,d)``。前 7 天用历史均值暖启动；从第 8 天起使用
    weekday + annual/semi-annual harmonic OLS。对正式回测区间（2 月 1 日起）
    它与 Q2 傅里叶模型的预测逐点相同。
    """

    def __init__(self, dates, load: np.ndarray):
        self.dates = pd.DatetimeIndex(dates)
        self.load = np.asarray(load, dtype=float)
        self._row = {pd.Timestamp(x).normalize(): i for i, x in enumerate(self.dates)}

    def predict(self, target_date: str) -> tuple[np.ndarray, np.ndarray]:
        d = self._row[pd.Timestamp(target_date).normalize()]
        if d == 0:
            pred = self.load[0].copy()
        elif d < 7:
            pred = self.load[:d].mean(axis=0)
        else:
            x = np.array([_calendar(i) for i in range(d)])
            beta, *_ = np.linalg.lstsq(x, self.load[:d], rcond=None)
            pred = _calendar(d) @ beta
        return np.asarray(pred, dtype=float), np.zeros(N_SLOTS)


def causal_load_error_pool(model: CausalFourierLoadModel, dates, load) -> np.ndarray:
    """生成逐日样本外负荷误差；第 d 行也只由 d 日以前的数据拟合。"""
    out = np.empty_like(np.asarray(load, dtype=float))
    for d, date in enumerate(dates):
        pred, _ = model.predict(str(date)[:10])
        out[d] = load[d] - pred
    return out


def combined_error_history(
    *, day: int, at_slot: int, load_err: np.ndarray, pv_err: np.ndarray,
    lookback: int,
) -> np.ndarray:
    """目标节点可见的历史净负荷预测误差日型，单位 kW。"""
    start = max(0, day - lookback)
    hist = load_err[start:day, at_slot:] - pv_err[start:day, at_slot:]
    return hist[~np.isnan(hist).any(axis=1)]


def reduce_scenarios(hist: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """用加权 K-means 压缩完整日误差轨迹，保留 10 分钟时序相关性。"""
    hist = np.asarray(hist, dtype=float)
    if hist.shape[0] == 0:
        return np.zeros((1, hist.shape[1])), np.ones(1)
    if hist.shape[0] <= k:
        return hist.copy(), np.full(hist.shape[0], 1.0 / hist.shape[0])
    estimator = KMeans(n_clusters=k, random_state=2026, n_init=10)
    try:
        km = estimator.fit(hist)
    except AttributeError:
        # 部分 macOS/Anaconda 组合中 threadpoolctl 无法读取 Accelerate 的
        # version string；这只影响 sklearn 的并行库探测，不影响 K-means 数值。
        import sklearn.cluster._kmeans as _sk_kmeans
        from contextlib import nullcontext
        _sk_kmeans.threadpool_info = lambda: []
        _sk_kmeans.threadpool_limits = lambda **_kwargs: nullcontext()
        km = estimator.fit(hist)
    weights = np.bincount(km.labels_, minlength=k).astype(float) / hist.shape[0]
    return km.cluster_centers_, weights


def solve_commitment(
    *, point_kw: np.ndarray, price: np.ndarray, s_init: float,
    previous: np.ndarray | None, scenarios_kw: np.ndarray,
    weights: np.ndarray, s_target: float = S_INIT,
) -> tuple[np.ndarray, float]:
    """求一个发布节点的余下时段购电承诺。

    ``previous is None`` 表示 0:00 首次计划，目标中计入正常购电费；否则用
    ``q = previous + u - v`` 线性化增购/减购，逐次调整费用为
    ``1.5*p*u - 0.5*p*v``。每个误差场景拥有独立的充放电、SOC 与紧急购电
    追索变量，购电承诺 q 在所有场景间共享。
    """
    point_kw = np.asarray(point_kw, dtype=float)
    price = np.asarray(price, dtype=float)
    scenarios_kw = np.asarray(scenarios_kw, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = point_kw.size
    n_scen = scenarios_kw.shape[0]
    adjust = previous is not None
    base = n + (2 * n if adjust else 0)
    nv = base + n_scen * 4 * n
    objective = np.zeros(nv)

    if adjust:
        objective[n:2 * n] = PLAN_INCREASE_RATE * price
        objective[2 * n:3 * n] = -PLAN_REDUCE_REFUND * price
    else:
        objective[:n] = price
    for s in range(n_scen):
        off = base + s * 4 * n
        objective[off + 3 * n:off + 4 * n] = (
            weights[s] * EMERGENCY_MULTIPLIER * price
        )

    a_eq, b_eq, a_ub, b_ub = [], [], [], []
    if adjust:
        previous = np.asarray(previous, dtype=float)
        for t in range(n):
            row = np.zeros(nv)
            row[t], row[n + t], row[2 * n + t] = 1.0, -1.0, 1.0
            a_eq.append(row)
            b_eq.append(previous[t])

    for s in range(n_scen):
        off = base + s * 4 * n
        net_kw = point_kw + scenarios_kw[s]
        for t in range(n):
            # SOC_t = SOC_{t-1} + eta*c_t - d_t/eta
            row = np.zeros(nv)
            row[off + t] = -ETA
            row[off + n + t] = 1.0 / ETA
            row[off + 2 * n + t] = 1.0
            if t:
                row[off + 2 * n + t - 1] = -1.0
                rhs = 0.0
            else:
                rhs = s_init
            a_eq.append(row)
            b_eq.append(rhs)

            # e_t >= (net_t*dt + c_t - d_t) - q_t
            row = np.zeros(nv)
            row[t] = -1.0
            row[off + t] = 1.0
            row[off + n + t] = -1.0
            row[off + 3 * n + t] = -1.0
            a_ub.append(row)
            b_ub.append(-net_kw[t] * DELTA_T)

        row = np.zeros(nv)
        row[off + 3 * n - 1] = 1.0
        a_eq.append(row)
        b_eq.append(s_target)

    bounds = [(0.0, None)] * n
    if adjust:
        bounds += [(0.0, None)] * (2 * n)
    for _ in range(n_scen):
        bounds += (
            [(0.0, P_MAX_SLOT)] * (2 * n)
            + [(S_MIN, S_MAX)] * n
            + [(0.0, None)] * n
        )

    result = linprog(
        objective,
        A_ub=np.asarray(a_ub), b_ub=np.asarray(b_ub),
        A_eq=np.asarray(a_eq), b_eq=np.asarray(b_eq),
        bounds=bounds, method="highs",
    )
    if not result.success:
        raise RuntimeError(f"commitment LP failed: {result.message}")
    return result.x[:n], float(result.fun)


def execute_stage(
    purchase: np.ndarray, actual_net_kw: np.ndarray,
    start: int, end: int, s_init: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """只用当前真实净负荷的因果储能执行器。"""
    charge = np.zeros(end - start)
    discharge = np.zeros(end - start)
    emergency = np.zeros(end - start)
    soc = np.empty(end - start + 1)
    soc[0] = s_init
    for j, t in enumerate(range(start, end)):
        gap = actual_net_kw[t] * DELTA_T - purchase[t]
        if gap > 0:
            discharge[j] = min(gap, P_MAX_SLOT, (soc[j] - S_MIN) * ETA)
        else:
            charge[j] = min(-gap, P_MAX_SLOT, (S_MAX - soc[j]) / ETA)
        soc[j + 1] = soc[j] + ETA * charge[j] - discharge[j] / ETA
        emergency[j] = max(0.0, gap + charge[j] - discharge[j])
    return charge, discharge, emergency, soc


def _predicted_keep_cost(previous, point_kw, price, s_init) -> float:
    _, _, emergency, _ = execute_stage(
        previous, point_kw, 0, len(point_kw), s_init
    )
    return float(np.dot(emergency, EMERGENCY_MULTIPLIER * price))


def run_day_commitment(
    *, day: int, dates, load_actual: np.ndarray, pv_actual: np.ndarray,
    pv_fc_day: np.ndarray, price: np.ndarray, load_fc: np.ndarray,
    load_err: np.ndarray, pv_err_pools: tuple, n_scenarios: int = 5,
    lookback: int = 90, nodes: tuple = (0, 1, 2, 3),
    stochastic: bool = True, pv_method: str = "linear",
    s_init: float = S_INIT, event_threshold: float = 0.0,
    settlement: str = "sequential",
    **_ignored,
) -> dict:
    """完成一天的滚动决策、因果执行和真实结算。"""
    nodes = tuple(sorted(set(nodes)))
    if not nodes or nodes[0] != 0:
        raise ValueError("node 0 (0:00) must always be included")
    if settlement not in {"sequential", "final_net"}:
        raise ValueError("settlement must be 'sequential' or 'final_net'")

    actual_net_kw = load_actual[day] - pv_actual[day]
    plan = None
    current = None
    adjusted = np.zeros(N_SLOTS)
    charge = np.zeros(N_SLOTS)
    discharge = np.zeros(N_SLOTS)
    emergency = np.zeros(N_SLOTS)
    soc = np.empty(N_SLOTS + 1)
    soc[0] = s_init
    reduce_cost = 0.0
    increase_cost = 0.0
    adjustment_kwh = 0.0
    node_info = []

    for pos, node in enumerate(nodes):
        start = ISSUE_SLOTS[node]
        end = ISSUE_SLOTS[nodes[pos + 1]] if pos + 1 < len(nodes) else N_SLOTS
        pv_fc = np.maximum(
            forecast_profile(pv_fc_day, node, method=pv_method)[start:], 0.0
        )
        point_kw = load_fc[start:] - pv_fc
        hist = combined_error_history(
            day=day, at_slot=start, load_err=load_err,
            pv_err=pv_err_pools[node], lookback=lookback,
        )
        if stochastic:
            scenarios, weights = reduce_scenarios(hist, k=n_scenarios)
        else:
            scenarios = np.zeros((1, point_kw.size))
            weights = np.ones(1)

        previous = None if node == 0 else current[start:].copy()
        q, objective = solve_commitment(
            point_kw=point_kw, price=price[start:], s_init=soc[start],
            previous=previous, scenarios_kw=scenarios, weights=weights,
        )

        if node == 0:
            plan = np.r_[np.zeros(start), q]
            current = plan.copy()
            accepted = True
        else:
            up = np.maximum(q - previous, 0.0)
            down = np.maximum(previous - q, 0.0)
            proposed_adjustment = float(
                np.dot(PLAN_INCREASE_RATE * price[start:], up)
                - np.dot(PLAN_REDUCE_REFUND * price[start:], down)
            )
            keep_cost = _predicted_keep_cost(
                previous, point_kw, price[start:], soc[start]
            )
            accepted = (keep_cost - objective) > event_threshold
            if accepted:
                current[start:] = q
                increase_cost += float(np.dot(PLAN_INCREASE_RATE * price[start:], up))
                reduce_cost -= float(np.dot(PLAN_REDUCE_REFUND * price[start:], down))
                adjustment_kwh += float(up.sum() + down.sum())

        adjusted[start:end] = current[start:end]
        c, d, e, stage_soc = execute_stage(
            current, actual_net_kw, start, end, soc[start]
        )
        charge[start:end], discharge[start:end], emergency[start:end] = c, d, e
        soc[start:end + 1] = stage_soc
        node_info.append({
            "node": node,
            "at_slot": start,
            "accepted": bool(accepted),
            "objective": objective,
            "n_scen": int(len(weights)),
            "weights": weights.copy(),
            "purchase_remaining": current[start:].copy(),
        })

    planned_cost = float(np.dot(price, plan))
    emergency_cost = float(np.dot(price, EMERGENCY_MULTIPLIER * emergency))

    final_reduce = -float(
        np.dot(price, PLAN_REDUCE_REFUND * np.maximum(plan - adjusted, 0.0))
    )
    final_increase = float(
        np.dot(price, PLAN_INCREASE_RATE * np.maximum(adjusted - plan, 0.0))
    )
    sequential_total = planned_cost + reduce_cost + increase_cost + emergency_cost
    final_net_total = planned_cost + final_reduce + final_increase + emergency_cost
    if settlement == "sequential":
        reported_reduce, reported_increase, total = reduce_cost, increase_cost, sequential_total
    else:
        reported_reduce, reported_increase, total = final_reduce, final_increase, final_net_total

    return {
        "day": day,
        "date": str(dates[day])[:10],
        "plan": plan,
        "adjusted": adjusted,
        "charge": charge,
        "discharge": discharge,
        "soc": soc,
        "emergency": emergency,
        "planned_cost": planned_cost,
        "reduce_penalty": reported_reduce,
        "increase_cost": reported_increase,
        "emergency_cost": emergency_cost,
        "total_cost": total,
        "sequential_total_cost": sequential_total,
        "final_net_total_cost": final_net_total,
        "sequential_reduce_cost": reduce_cost,
        "sequential_increase_cost": increase_cost,
        "final_net_reduce_cost": final_reduce,
        "final_net_increase_cost": final_increase,
        "plan_kwh": float(plan.sum()),
        "adjusted_kwh": float(adjusted.sum()),
        "emergency_kwh": float(emergency.sum()),
        "adjustment_kwh": adjustment_kwh,
        "node_info": node_info,
    }
