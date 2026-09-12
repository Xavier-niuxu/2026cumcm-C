# -*- coding: utf-8 -*-
"""Cost module: LP optimization and electricity cost calculation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from data_loader import DELTA_T

# LP parameters (same as question1)
ETA = 0.90
S_MIN = 1200.0
S_MAX = 10800.0
S_INIT = 6000.0
P_MAX = 5000.0
EPS = 1e-7

# Emergency purchase parameters
EMERGENCY_PRICE_MULTIPLIER = 5.0  # Emergency price = 5 * normal price
EMERGENCY_MAX = 1000.0  # Max emergency purchase per interval (kWh)


@dataclass
class LPResult:
    """LP optimization result."""
    grid_purchase: np.ndarray       # kWh / interval (normal purchase)
    emergency_purchase: np.ndarray  # kWh / interval (emergency purchase)
    charge: np.ndarray              # kWh / interval
    discharge: np.ndarray           # kWh / interval
    soc_start: np.ndarray           # kWh, S_t
    soc_end: np.ndarray             # kWh, S_{t+1}
    objective: float
    normal_cost: float              # Normal purchase cost
    emergency_cost: float           # Emergency purchase cost
    total_cost: float               # Total cost


def solve_lp_day_ahead(
    price: np.ndarray,
    load_forecast: np.ndarray,
    pv_forecast: np.ndarray,
    *,
    delta_t: float = DELTA_T,
    eta: float = ETA,
    s_min: float = S_MIN,
    s_max: float = S_MAX,
    s_init: float = S_INIT,
    s_end_min: float | None = None,
    p_max: float = P_MAX,
    eps: float = EPS,
) -> dict:
    """Solve day-ahead LP optimization (only normal grid purchase).
    
    Variables:
        g_t: normal grid purchase (kWh/interval)
        c_t: charge (kWh/interval)
        d_t: discharge (kWh/interval)
        S_t: SOC at start of interval t (kWh)
    
    Objective:
        min sum_t (p_t * g_t + eps * (c_t + d_t))
    
    Constraints:
        Energy balance: g_t + PV_t*dt + d_t >= L_t*dt + c_t
        SOC dynamics: S_{t+1} = S_t + eta*c_t - d_t/eta
        Bounds: 0 <= g_t
                0 <= c_t, d_t <= P_MAX*dt
                S_min <= S_t <= S_max
                S_1 = s_init
    
    The terminal SOC S_{n+1} is free by default: the objective has no value on
    stored energy, so the LP empties the battery down to ``s_min`` by the end of
    the day.  Passing ``s_end_min`` imposes S_{n+1} >= s_end_min, which keeps a
    real-time reserve for the next day's forecast errors.
    
    Returns:
        dict with keys: grid, charge, discharge, soc
    """
    price = np.asarray(price, dtype=float)
    load_forecast = np.asarray(load_forecast, dtype=float)
    pv_forecast = np.asarray(pv_forecast, dtype=float)
    
    n = len(price)
    if not (len(load_forecast) == len(pv_forecast) == n):
        raise ValueError("price/load/pv_forecast length mismatch")
    
    # Variable indices:
    # g: [0, n)
    # c: [n, 2n)
    # d: [2n, 3n)
    # s: [3n, 4n+1)
    g = slice(0, n)
    c = slice(n, 2 * n)
    d = slice(2 * n, 3 * n)
    s = slice(3 * n, 4 * n + 1)
    nv = 4 * n + 1
    
    # Objective
    objective = np.zeros(nv)
    objective[g] = price
    objective[c] = eps
    objective[d] = eps
    
    # Equality constraints:
    # S_{t+1} - S_t - eta*c_t + d_t/eta = 0
    # S_1 = s_init (only initial SOC is fixed, final SOC is free)
    Aeq = lil_matrix((n + 1, nv), dtype=float)
    beq = np.zeros(n + 1)

    for t in range(n):
        Aeq[t, s.start + t] = -1.0
        Aeq[t, s.start + t + 1] = 1.0
        Aeq[t, c.start + t] = -eta
        Aeq[t, d.start + t] = 1.0 / eta

    Aeq[n, s.start] = 1.0
    beq[n] = s_init
    
    # Inequality constraints:
    # g_t + PV_t*dt + d_t >= L_t*dt + c_t
    # -> -g_t + c_t - d_t <= (PV_t - L_t)*dt
    Aub = lil_matrix((n, nv), dtype=float)
    bub = np.empty(n, dtype=float)
    net_load = (load_forecast - pv_forecast) * delta_t
    
    for t in range(n):
        Aub[t, g.start + t] = -1.0
        Aub[t, c.start + t] = 1.0
        Aub[t, d.start + t] = -1.0
        bub[t] = -net_load[t]
    
    # Bounds
    soc_upper = [(s_min, s_max)] * (n + 1)
    if s_end_min is not None:
        soc_upper[n] = (max(s_min, s_end_min), s_max)
    bounds = (
        [(0.0, None)] * n                           # g >= 0
        + [(0.0, p_max * delta_t)] * n              # 0 <= c <= P_MAX*dt
        + [(0.0, p_max * delta_t)] * n              # 0 <= d <= P_MAX*dt
        + soc_upper                                 # S_min <= S <= S_max
    )
    
    res = linprog(
        c=objective,
        A_ub=Aub.tocsr(),
        b_ub=bub,
        A_eq=Aeq.tocsr(),
        b_eq=beq,
        bounds=bounds,
        method="highs",
    )
    
    if not res.success:
        raise RuntimeError(
            f"LP solve failed: status={res.status}, message={res.message}"
        )
    
    x = res.x
    return {
        "grid": x[g].copy(),
        "charge": x[c].copy(),
        "discharge": x[d].copy(),
        "soc": x[s].copy(),
    }


def solve_lp_stochastic(
    price: np.ndarray,
    scenarios: np.ndarray,
    *,
    delta_t: float = DELTA_T,
    eta: float = ETA,
    s_min: float = S_MIN,
    s_max: float = S_MAX,
    s_init: float = S_INIT,
    p_max: float = P_MAX,
    emergency_price_multiplier: float = EMERGENCY_PRICE_MULTIPLIER,
    weights: np.ndarray | None = None,
    eps: float = 1e-7,
) -> dict:
    """Two-stage stochastic day-ahead LP (sample average approximation).

    The day-ahead purchase ``g_t`` is a here-and-now decision shared by every
    scenario, while the battery and the emergency purchase are recourse
    decisions that adapt to each realised net-load path:

        min_g  sum_t p_t g_t
               + sum_k w_k * sum_t emergency_multiplier * p_t * e^k_t
               + eps * sum_k w_k * sum_t (c^k_t + d^k_t)

    subject, for every scenario k, to

        S^k_{t+1} = S^k_t + eta * c^k_t - d^k_t / eta,   S^k_1 = s_init
        g_t + d^k_t + e^k_t >= net^k_t + c^k_t
        0 <= c^k_t, d^k_t <= P_MAX * delta_t,  e^k_t >= 0
        S_MIN <= S^k_t <= S_MAX

    The recourse here is a *non-causal* within-day LP: it sees the whole
    realised path before choosing when to discharge.  It is therefore a lower
    bound for any causal real-time rule, and it is **not** reproduced by
    ``execute_real_time_dispatch``.  The greedy rule there discharges into
    every shortfall regardless of price, which wastes stored energy on cheap
    slots that a foresighted scheduler would save for the evening peak; the
    two only coincide when the forecast is exact.  Measured on the full year
    the greedy rule costs about 19 万元 more than this recourse bound.

    Args:
        price: (n,) electricity price per interval.
        scenarios: (K, n) net load (L - P) * delta_t per interval, kWh, one row
            per sampled residual path.  Building them from the model's own past
            out-of-sample residuals keeps the decision causal.

    Returns:
        dict with key ``grid`` (the committed purchase, kWh per interval) plus
        the expected cost split into ``normal_cost`` / ``emergency_cost``.
    """
    price = np.asarray(price, dtype=float)
    scenarios = np.atleast_2d(np.asarray(scenarios, dtype=float))
    K, n = scenarios.shape
    if price.size != n:
        raise ValueError("price/scenario length mismatch")
    if weights is None:
        w = np.full(K, 1.0 / K)
    else:
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()

    cap = p_max * delta_t
    per = 4 * n + 1               # c, d, e, S per scenario
    nv = n + K * per

    objective = np.zeros(nv)
    objective[0:n] = price

    Aeq = lil_matrix((K * (n + 1), nv), dtype=float)
    beq = np.zeros(K * (n + 1))
    Aub = lil_matrix((K * n, nv), dtype=float)
    bub = np.empty(K * n)

    bounds = [(0.0, None)] * n    # committed purchase g >= 0
    for k in range(K):
        base = n + k * per
        c0, d0, e0, s0 = base, base + n, base + 2 * n, base + 3 * n
        objective[e0:e0 + n] = w[k] * emergency_price_multiplier * price
        objective[c0:c0 + n] = eps
        objective[d0:d0 + n] = eps

        for t in range(n):
            r = k * (n + 1) + t
            Aeq[r, s0 + t] = -1.0
            Aeq[r, s0 + t + 1] = 1.0
            Aeq[r, c0 + t] = -eta
            Aeq[r, d0 + t] = 1.0 / eta
        r = k * (n + 1) + n
        Aeq[r, s0] = 1.0
        beq[r] = s_init

        for t in range(n):
            r = k * n + t
            Aub[r, t] = -1.0        # g_t shared across scenarios
            Aub[r, d0 + t] = -1.0
            Aub[r, e0 + t] = -1.0
            Aub[r, c0 + t] = 1.0
            bub[r] = -scenarios[k, t]

        bounds += [(0.0, cap)] * n
        bounds += [(0.0, cap)] * n
        bounds += [(0.0, None)] * n
        bounds += [(s_min, s_max)] * (n + 1)

    res = linprog(
        c=objective,
        A_ub=Aub.tocsr(),
        b_ub=bub,
        A_eq=Aeq.tocsr(),
        b_eq=beq,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(
            f"stochastic LP failed: status={res.status}, message={res.message}"
        )
    g = res.x[0:n].copy()
    return {
        "grid": g,
        "normal_cost": float(np.dot(price, g)),
        "emergency_cost": float(res.fun - np.dot(price, g)),
        "objective": float(res.fun),
    }


def calculate_emergency_purchase(
    price: np.ndarray,
    load_actual: np.ndarray,
    pv_actual: np.ndarray,
    grid_planned: np.ndarray,
    charge_planned: np.ndarray,
    discharge_planned: np.ndarray,
    *,
    delta_t: float = DELTA_T,
    emergency_price_multiplier: float = EMERGENCY_PRICE_MULTIPLIER,
) -> tuple:
    """Calculate emergency purchase based on actual vs planned.
    
    Emergency purchase at time t:
        e_t = max(0, (L_t - P_t)*dt + c_t - d_t - g_t)
    
    Args:
        price: electricity price at each time slot
        load_actual: actual load power (kW)
        pv_actual: actual PV power (kW)
        grid_planned: planned grid purchase from day-ahead optimization
        charge_planned: planned charge from day-ahead optimization
        discharge_planned: planned discharge from day-ahead optimization
    
    Returns:
        emergency_purchase: array of emergency purchase (kWh)
        emergency_cost: total emergency purchase cost
    """
    price = np.asarray(price, dtype=float)
    load_actual = np.asarray(load_actual, dtype=float)
    pv_actual = np.asarray(pv_actual, dtype=float)
    grid_planned = np.asarray(grid_planned, dtype=float)
    charge_planned = np.asarray(charge_planned, dtype=float)
    discharge_planned = np.asarray(discharge_planned, dtype=float)
    
    # Energy deficit: (L - P)*dt + c - d - g
    energy_deficit = (load_actual - pv_actual) * delta_t + charge_planned - discharge_planned - grid_planned
    
    # Emergency purchase is the positive part of deficit
    emergency_purchase = np.maximum(0, energy_deficit)
    
    # Emergency cost: 5 * price * emergency_purchase
    emergency_cost = float(np.sum(price * emergency_price_multiplier * emergency_purchase))
    
    return emergency_purchase, emergency_cost


def execute_real_time_dispatch(
    price: np.ndarray,
    load_actual: np.ndarray,
    pv_actual: np.ndarray,
    grid_planned: np.ndarray,
    *,
    delta_t: float = DELTA_T,
    eta: float = ETA,
    s_min: float = S_MIN,
    s_max: float = S_MAX,
    s_init: float = S_INIT,
    p_max: float = P_MAX,
    emergency_price_multiplier: float = EMERGENCY_PRICE_MULTIPLIER,
) -> dict:
    """Real-time battery dispatch against the committed day-ahead purchase.

    The day-ahead plan fixes the grid purchase ``g_t``; the normal purchase
    bill is settled on that planned quantity, so ``g_t`` cannot be revised in
    real time.  The battery, however, is operated in real time: it charges
    from any surplus (planned purchase above the realised net load) and
    discharges into any shortfall, subject to its power and energy limits.
    Only the shortfall the battery cannot cover is bought as emergency power
    at ``emergency_price_multiplier`` times the normal price.

    With a perfect forecast this reproduces the planned schedule exactly and
    incurs no emergency purchase, so it is a strict generalisation of the
    "actual charge/discharge = planned charge/discharge" settlement.

    Returns:
        dict with keys: emergency, charge, discharge, soc_start, soc_end,
        emergency_cost
    """
    price = np.asarray(price, dtype=float)
    load_actual = np.asarray(load_actual, dtype=float)
    pv_actual = np.asarray(pv_actual, dtype=float)
    grid_planned = np.asarray(grid_planned, dtype=float)

    n = len(price)
    net_actual = (load_actual - pv_actual) * delta_t  # kWh needed per interval
    cap = p_max * delta_t

    emergency = np.zeros(n)
    charge = np.zeros(n)
    discharge = np.zeros(n)
    soc_start = np.empty(n)
    soc_end = np.empty(n)

    soc = float(s_init)
    for t in range(n):
        soc_start[t] = soc
        surplus = grid_planned[t] - net_actual[t]
        if surplus >= 0.0:
            # Excess purchase is stored, not wasted (up to power/SOC headroom).
            c = min(surplus, cap, max(0.0, (s_max - soc) / eta))
            charge[t] = c
            soc += eta * c
        else:
            # Shortfall is covered by the battery first, then emergency power.
            d = min(-surplus, cap, max(0.0, (soc - s_min) * eta))
            discharge[t] = d
            soc -= d / eta
            emergency[t] = -surplus - d
        soc_end[t] = soc

    emergency_cost = float(
        np.sum(price * emergency_price_multiplier * emergency)
    )
    return {
        "emergency": emergency,
        "charge": charge,
        "discharge": discharge,
        "soc_start": soc_start,
        "soc_end": soc_end,
        "emergency_cost": emergency_cost,
    }


def calculate_cost(
    price: np.ndarray,
    load_forecast: np.ndarray,
    pv_forecast: np.ndarray,
    load_actual: np.ndarray = None,
    pv_actual: np.ndarray = None,
    *,
    real_time_dispatch: bool = True,
    s_init: float = S_INIT,
    s_end_min: float | None = None,
    s_min_plan: float | None = None,
    s_max_plan: float | None = None,
) -> LPResult:
    """Calculate optimal purchase strategy and cost.
    
    Two-stage approach:
    1. Day-ahead optimization: optimize normal grid purchase using forecasts
    2. Real-time stage: settle normal cost on the planned purchase, dispatch the
       battery against the realised net load, and buy the residual shortfall as
       emergency power at 5x the normal price
    
    If load_actual and pv_actual are not provided, assumes perfect forecast
    (no emergency) and reports the planned battery schedule.
    
    Args:
        price: electricity price at each time slot
        load_forecast: predicted load power (kW)
        pv_forecast: predicted PV power (kW)
        load_actual: actual load power (kW), optional
        pv_actual: actual PV power (kW), optional
        real_time_dispatch: if True (default) the battery flexes in real time to
            absorb forecast errors; if False the actual charge/discharge is
            frozen at the planned values (the original conservative model).
        s_init: SOC at 0:00 of the current day (kWh). Only 2025-01-01 0:00 is
            fixed at 6000 kWh by the problem statement; for every later day the
            caller passes the previous day's closing SOC so the battery state is
            continuous across days.
        s_min_plan, s_max_plan: planning band for the day-ahead LP only.  The
            plan is drawn from a band narrower than the physical one
            ``[S_MIN, S_MAX]``, which reserves battery headroom for the next
            day's forecast error; the real-time dispatch still respects the
            physical limits.  The band is widened as needed to contain
            ``s_init`` so the LP stays feasible.  ``None`` keeps the physical
            band.
    
    Returns:
        LPResult with optimization and emergency purchase results; ``soc_end[-1]``
        is the SOC at 24:00 and is carried into the next day.
    """
    # Stage 1: Day-ahead optimization
    # The plan is drawn from a band narrower than [S_MIN, S_MAX], leaving the
    # battery headroom to absorb the next day's forecast error in real time.
    plan_lo = S_MIN if s_min_plan is None else min(s_min_plan, s_init)
    plan_hi = S_MAX if s_max_plan is None else max(s_max_plan, s_init)
    plan = solve_lp_day_ahead(
        price, load_forecast, pv_forecast, s_init=s_init,
        s_end_min=s_end_min, s_min=plan_lo, s_max=plan_hi,
    )
    
    grid = plan["grid"]
    normal_cost = float(np.dot(price, grid))
    
    # Stage 2: Real-time settlement
    if load_actual is not None and pv_actual is not None:
        if real_time_dispatch:
            rt = execute_real_time_dispatch(
                price, load_actual, pv_actual, grid, s_init=s_init
            )
            emergency = rt["emergency"]
            emergency_cost = rt["emergency_cost"]
            charge = rt["charge"]
            discharge = rt["discharge"]
            soc_start = rt["soc_start"]
            soc_end = rt["soc_end"]
        else:
            emergency, emergency_cost = calculate_emergency_purchase(
                price, load_actual, pv_actual, grid, plan["charge"], plan["discharge"]
            )
            charge = plan["charge"]
            discharge = plan["discharge"]
            soc_start = plan["soc"][:-1].copy()
            soc_end = plan["soc"][1:].copy()
    else:
        # No actual data provided, assume perfect forecast
        emergency = np.zeros(len(price))
        emergency_cost = 0.0
        charge = plan["charge"]
        discharge = plan["discharge"]
        soc_start = plan["soc"][:-1].copy()
        soc_end = plan["soc"][1:].copy()

    return LPResult(
        grid_purchase=grid,
        emergency_purchase=emergency,
        charge=charge,
        discharge=discharge,
        soc_start=soc_start,
        soc_end=soc_end,
        objective=normal_cost,
        normal_cost=normal_cost,
        emergency_cost=emergency_cost,
        total_cost=normal_cost + emergency_cost,
    )


if __name__ == "__main__":
    # Test with simple data
    n = 144
    price = np.ones(n) * 0.5
    load = np.ones(n) * 3000
    pv = np.zeros(n)
    
    result = calculate_cost(price, load, pv)
    print(f"Normal cost: {result.normal_cost:.2f} yuan")
    print(f"Emergency cost: {result.emergency_cost:.2f} yuan")
    print(f"Total cost: {result.total_cost:.2f} yuan")
    print(f"Total grid purchase: {result.grid_purchase.sum():.2f} kWh")
    print(f"Total emergency purchase: {result.emergency_purchase.sum():.2f} kWh")
