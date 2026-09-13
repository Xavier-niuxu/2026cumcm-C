# -*- coding: utf-8 -*-
"""rolling implementation."""

from __future__ import annotations

import numpy as np

from data_loader import (
    DELTA_T,
    EMERGENCY_MULTIPLIER,
    ETA,
    ISSUE_SLOTS,
    N_SLOTS,
    PLAN_INCREASE_RATE,
    PLAN_REDUCE_REFUND,
    forecast_profile,
)
from mpc import solve_node
from scenarios import build_scenarios


def run_day(
    *,
    day: int,
    dates,
    load_actual: np.ndarray,
    pv_actual: np.ndarray,
    pv_fc_day: np.ndarray,
    price: np.ndarray,
    load_fc: np.ndarray,
    load_err: np.ndarray,
    pv_err_pools: tuple,
    n_scenarios: int = 12,
    lookback: int = 90,
    seed: int = 0,
    nodes: tuple = (0, 1, 2, 3),
    stochastic: bool = True,
    cvar_weight: float = 0.0,
    cvar_alpha: float = 0.95,
    pv_method: str = "linear",
    s_init: float = 6000.0,
    s_end: float = 6000.0,
    battery_mode: str = "realtime",
    future_adjust: bool = True,
    scenario_mode: str = "bootstrap",
    soc_floor: float = 1200.0,
    soc_ceil: float = 10800.0,
    p_max_slot: float = 833.3333333333334,
    settlement_price: np.ndarray | None = None,
) -> dict:
    """run_day implementation."""
    nodes = tuple(sorted(set(nodes)))
    if 0 not in nodes:
        raise ValueError("node 0 (0:00) must always be included")

    rng = np.random.default_rng(seed + day)
    plan = np.zeros(N_SLOTS)
    ga = np.zeros(N_SLOTS)
    charge = np.zeros(N_SLOTS)
    discharge = np.zeros(N_SLOTS)
    soc_path = np.zeros(N_SLOTS + 1)
    soc_path[0] = s_init
    node_info = []

    for k, node in enumerate(nodes):
        at_slot = ISSUE_SLOTS[node]
        next_slot = ISSUE_SLOTS[nodes[k + 1]] if k + 1 < len(nodes) else N_SLOTS
        block = next_slot - at_slot
        s_now = soc_path[at_slot]

        pv_fc = forecast_profile(pv_fc_day, node, method=pv_method)
        net_scen, _ = build_scenarios(
            day=day,
            at_slot=at_slot,
            load_fc=load_fc,
            pv_fc=np.nan_to_num(pv_fc),
            load_err=load_err,
            pv_err=pv_err_pools[node],
            n_scenarios=n_scenarios if stochastic else 1,
            lookback=lookback,
            rng=rng,
            delta_t=DELTA_T,
            mode=scenario_mode,
        )
        weights = np.full(net_scen.shape[0], 1.0 / net_scen.shape[0])

        res = solve_node(
            price=price,
            net_scen=net_scen,
            weights=weights,
            at_slot=at_slot,
            s_init=s_now,
            s_end=s_end,
            g0=None if node == 0 else plan,
            block=block,
            cvar_weight=cvar_weight,
            cvar_alpha=cvar_alpha,
            future_adjust=future_adjust,
        )

        if node == 0:
            plan = res["plan"].copy()
        ga[at_slot:next_slot] = res["ga_committed"]
        charge[at_slot:next_slot] = res["c_committed"]
        discharge[at_slot:next_slot] = res["d_committed"]
        soc_path[at_slot:next_slot + 1] = res["soc"]
        node_info.append(
            {
                "node": node,
                "at_slot": at_slot,
                "soc_start": float(s_now),
                "plan": res["plan"].copy() if node == 0 else None,
                "ga_committed": res["ga_committed"].copy(),
                "c_committed": res["c_committed"].copy(),
                "d_committed": res["d_committed"].copy(),
                "objective": res["objective"],
                "expected_emergency_kwh": res["expected_emergency_kwh"],
                "n_scen": res["n_scen"],
            }
        )

    # Implementation detail.
    net_act = (load_actual[day] - pv_actual[day]) * DELTA_T          # Implementation detail.
    if battery_mode == "realtime":
        # Implementation detail.
        charge = np.zeros(N_SLOTS)
        discharge = np.zeros(N_SLOTS)
        soc_path = np.zeros(N_SLOTS + 1)
        soc_path[0] = s_init
        for t in range(N_SLOTS):
            gap = net_act[t] - ga[t]
            s = soc_path[t]
            if gap > 0:                       # Implementation detail.
                discharge[t] = min(gap, p_max_slot, max(0.0, (s - soc_floor)) * ETA)
                charge[t] = 0.0
            else:                             # Implementation detail.
                charge[t] = min(-gap, p_max_slot, max(0.0, (soc_ceil - s)) / ETA)
                discharge[t] = 0.0
            soc_path[t + 1] = s + ETA * charge[t] - discharge[t] / ETA
        soc_path = np.clip(soc_path, soc_floor, soc_ceil)
    elif battery_mode != "committed":
        raise ValueError(f"unknown battery_mode: {battery_mode}")
    need = net_act + charge - discharge
    emergency = np.maximum(0.0, need - ga)
    short = np.maximum(0.0, ga - need)          # Implementation detail.

    # Implementation detail.
    # Implementation detail.
    # Implementation detail.
    # Implementation detail.
    # Implementation detail.
    # Implementation detail.
    settle = price if settlement_price is None else np.asarray(settlement_price)
    planned_cost = float(np.dot(settle, plan))
    reduce_penalty = -float(
        np.dot(settle, PLAN_REDUCE_REFUND * np.maximum(0.0, plan - ga))
    )
    increase_cost = float(
        np.dot(settle, PLAN_INCREASE_RATE * np.maximum(0.0, ga - plan))
    )
    emergency_cost = float(np.dot(settle, EMERGENCY_MULTIPLIER * emergency))
    total = planned_cost + reduce_penalty + increase_cost + emergency_cost

    return {
        "day": day,
        "date": str(dates[day])[:10],
        "plan": plan,
        "adjusted": ga,
        "charge": charge,
        "discharge": discharge,
        "soc": soc_path,
        "emergency": emergency,
        "planned_cost": planned_cost,
        "reduce_penalty": reduce_penalty,
        "increase_cost": increase_cost,
        "emergency_cost": emergency_cost,
        "total_cost": total,
        "plan_kwh": float(plan.sum()),
        "adjusted_kwh": float(ga.sum()),
        "emergency_kwh": float(emergency.sum()),
        "node_info": node_info,
    }


def check_feasibility(result: dict, *, s_end: float = 6000.0, tol: float = 1e-6) -> dict:
    """check_feasibility implementation."""
    charge, discharge, soc = result["charge"], result["discharge"], result["soc"]
    rebuilt = np.empty(N_SLOTS + 1)
    rebuilt[0] = soc[0]
    for t in range(N_SLOTS):
        rebuilt[t + 1] = rebuilt[t] + ETA * charge[t] - discharge[t] / ETA
    return {
        "soc_consistency": float(np.abs(rebuilt - soc).max()),
        "soc_min": float(soc.min()),
        "soc_max": float(soc.max()),
        "terminal_soc": float(soc[-1]),
        "terminal_gap": float(abs(soc[-1] - s_end)),
        "max_charge_step": float(charge.max()),
        "max_discharge_step": float(discharge.max()),
    }
