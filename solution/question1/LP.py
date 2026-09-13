# -*- coding: utf-8 -*-
"""LP implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from data_loader import DELTA_T, load_arrays


ETA = 0.90
S_MIN = 1200.0
S_MAX = 10800.0
S_INIT = 6000.0
P_MAX = 5000.0
EPS = 1e-7


@dataclass
class LPResult:
    step: np.ndarray
    price: np.ndarray
    load: np.ndarray
    pv_forecast: np.ndarray
    grid_purchase: np.ndarray       # kWh / interval
    charge: np.ndarray               # kWh / interval
    discharge: np.ndarray            # kWh / interval
    soc_start: np.ndarray            # kWh, S_t
    soc_end: np.ndarray              # kWh, S_{t+1}
    objective: float
    purchase_cost: float
    result: object


def _indices(n: int):
    """_indices implementation."""
    g = slice(0, n)
    c = slice(n, 2 * n)
    d = slice(2 * n, 3 * n)
    s = slice(3 * n, 4 * n + 1)  # S_1 ... S_{n+1}
    return g, c, d, s


def solve_question1(
    step: np.ndarray,
    price: np.ndarray,
    load: np.ndarray,
    pv_forecast: np.ndarray,
    *,
    delta_t: float = DELTA_T,
    eta: float = ETA,
    s_min: float = S_MIN,
    s_max: float = S_MAX,
    s_init: float = S_INIT,
    p_max: float = P_MAX,
    eps: float = EPS,
    method: str = "highs",
) -> LPResult:
    """solve_question1 implementation."""
    step = np.asarray(step)
    price = np.asarray(price, dtype=float)
    load = np.asarray(load, dtype=float)
    pv_forecast = np.asarray(pv_forecast, dtype=float)

    n = len(price)
    if not (len(step) == len(load) == len(pv_forecast) == n):
        raise ValueError("step/price/load/pv_forecast 长度必须一致")
    if n == 0:
        raise ValueError("输入数据为空")

    g, c, d, s = _indices(n)
    nv = 4 * n + 1

    # Objective:
    # min Σ p_t g_t + eps Σ(c_t+d_t)
    objective = np.zeros(nv)
    objective[g] = price
    objective[c] = eps
    objective[d] = eps

    # Equality constraints:
    # S_{t+1} - S_t - eta*c_t + d_t/eta = 0
    # plus S_1 = s_init and S_{n+1} = s_init.
    Aeq = lil_matrix((n + 2, nv), dtype=float)
    beq = np.zeros(n + 2)

    for t in range(n):
        Aeq[t, s.start + t] = -1.0
        Aeq[t, s.start + t + 1] = 1.0
        Aeq[t, c.start + t] = -eta
        Aeq[t, d.start + t] = 1.0 / eta

    Aeq[n, s.start] = 1.0
    beq[n] = s_init
    Aeq[n + 1, s.start + n] = 1.0
    beq[n + 1] = s_init

    # Inequality:
    # g_t + PV_t*dt + d_t >= load_t*dt + c_t
    # -> -g_t + c_t - d_t <= (PV_t-load_t)*dt
    Aub = lil_matrix((n, nv), dtype=float)
    bub = np.empty(n, dtype=float)
    net_load = (load - pv_forecast) * delta_t
    for t in range(n):
        Aub[t, g.start + t] = -1.0
        Aub[t, c.start + t] = 1.0
        Aub[t, d.start + t] = -1.0
        bub[t] = -net_load[t]

    # Bounds:
    # g >= 0
    # c,d <= P_MAX*dt
    # S in [S_MIN,S_MAX]
    bounds = (
        [(0.0, None)] * n
        + [(0.0, p_max * delta_t)] * n
        + [(0.0, p_max * delta_t)] * n
        + [(s_min, s_max)] * (n + 1)
    )

    res = linprog(
        c=objective,
        A_ub=Aub.tocsr(),
        b_ub=bub,
        A_eq=Aeq.tocsr(),
        b_eq=beq,
        bounds=bounds,
        method=method,
    )

    if not res.success:
        raise RuntimeError(
            "问题1线性规划求解失败："
            f"status={res.status}, message={res.message}"
        )

    x = res.x
    grid = x[g]
    charge = x[c]
    discharge = x[d]
    soc = x[s]

    purchase_cost = float(np.dot(price, grid))

    return LPResult(
        step=step,
        price=price,
        load=load,
        pv_forecast=pv_forecast,
        grid_purchase=grid.copy(),
        charge=charge.copy(),
        discharge=discharge.copy(),
        soc_start=soc[:-1].copy(),
        soc_end=soc[1:].copy(),
        objective=float(res.fun),
        purchase_cost=purchase_cost,
        result=res,
    )


def solve_from_file(file_path=None) -> LPResult:
    step, price, load, pv = load_arrays(file_path) if file_path else load_arrays()
    return solve_question1(step, price, load, pv)


if __name__ == "__main__":
    ans = solve_from_file()
    print(f"全天购电量: {ans.grid_purchase.sum():.4f} kWh")
    print(f"全天购电费: {ans.purchase_cost:.4f} 元")
