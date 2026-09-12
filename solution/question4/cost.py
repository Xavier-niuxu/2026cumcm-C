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
                S_1 = S_{n+1} = s_init
    
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
    bounds = (
        [(0.0, None)] * n                           # g >= 0
        + [(0.0, p_max * delta_t)] * n              # 0 <= c <= P_MAX*dt
        + [(0.0, p_max * delta_t)] * n              # 0 <= d <= P_MAX*dt
        + [(s_min, s_max)] * (n + 1)                # S_min <= S <= S_max
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


def calculate_cost(
    price: np.ndarray,
    load_forecast: np.ndarray,
    pv_forecast: np.ndarray,
    load_actual: np.ndarray = None,
    pv_actual: np.ndarray = None,
) -> LPResult:
    """Calculate optimal purchase strategy and cost.
    
    Two-stage approach per reference paper:
    1. Day-ahead optimization: optimize normal grid purchase using forecasts
    2. Emergency purchase calculation: use actual values to compute emergency purchase
    
    If load_actual and pv_actual are not provided, assumes perfect forecast (no emergency).
    
    Args:
        price: electricity price at each time slot
        load_forecast: predicted load power (kW)
        pv_forecast: predicted PV power (kW)
        load_actual: actual load power (kW), optional
        pv_actual: actual PV power (kW), optional
    
    Returns:
        LPResult with optimization and emergency purchase results
    """
    # Stage 1: Day-ahead optimization
    plan = solve_lp_day_ahead(price, load_forecast, pv_forecast)
    
    grid = plan["grid"]
    charge = plan["charge"]
    discharge = plan["discharge"]
    soc = plan["soc"]
    
    normal_cost = float(np.dot(price, grid))
    
    # Stage 2: Calculate emergency purchase
    if load_actual is not None and pv_actual is not None:
        emergency, emergency_cost = calculate_emergency_purchase(
            price, load_actual, pv_actual, grid, charge, discharge
        )
    else:
        # No actual data provided, assume perfect forecast
        emergency = np.zeros(len(price))
        emergency_cost = 0.0
    
    return LPResult(
        grid_purchase=grid,
        emergency_purchase=emergency,
        charge=charge,
        discharge=discharge,
        soc_start=soc[:-1].copy(),
        soc_end=soc[1:].copy(),
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
