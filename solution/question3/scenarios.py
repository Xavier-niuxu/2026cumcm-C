# -*- coding: utf-8 -*-
"""scenarios implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Implementation detail.
# Implementation detail.
sys.path.append(str(Path(__file__).resolve().parents[1] / "question2"))

from pred.fourier import FourierPredictor  # noqa: E402

from data_loader import N_SLOTS, forecast_profile  # noqa: E402


def load_forecast_model(dates, load):
    """load_forecast_model implementation."""
    return FourierPredictor(dates, load, dates, np.zeros_like(load))


def load_error_pool(model, dates, load) -> np.ndarray:
    """load_error_pool implementation."""
    n_days = len(dates)
    pool = np.full((n_days, N_SLOTS), np.nan)
    for i in range(n_days):
        try:
            load_pred, _ = model.predict(str(dates[i])[:10])
        except ValueError:
            continue
        pool[i] = load[i] - load_pred
    return pool


def pv_error_pool(pv_actual, pv_fc, issue_idx: int, method: str = "linear") -> np.ndarray:
    """pv_error_pool implementation."""
    n_days = pv_actual.shape[0]
    pool = np.full((n_days, N_SLOTS), np.nan)
    for d in range(n_days):
        pool[d] = pv_actual[d] - forecast_profile(pv_fc[d], issue_idx, method=method)
    return pool


def usable_days(pv_err: np.ndarray, at_slot: int, day: int, lookback: int) -> np.ndarray:
    """usable_days implementation."""
    start = max(0, day - lookback)
    cand = np.arange(start, day)
    return np.array([i for i in cand if not np.isnan(pv_err[i, at_slot:]).all()])


def build_scenarios(
    *,
    day: int,
    at_slot: int,
    load_fc: np.ndarray,
    pv_fc: np.ndarray,
    load_err: np.ndarray,
    pv_err: np.ndarray,
    n_scenarios: int,
    lookback: int,
    rng: np.random.Generator,
    delta_t: float,
    use_load_error: bool = True,
    mode: str = "bootstrap",
) -> tuple:
    """build_scenarios implementation."""
    cand = usable_days(pv_err, at_slot, day, lookback)
    if cand.size == 0:
        raise ValueError(f"no usable PV scenario history before day {day}")

    if mode == "bootstrap":
        picks = rng.choice(cand, size=n_scenarios, replace=True)
    elif mode == "reduced":
        # Implementation detail.
        # Implementation detail.
        severity = np.array(
            [np.maximum(pv_fc[at_slot:] - (pv_fc[at_slot:] + pv_err[j, at_slot:]), 0).sum()
             for j in cand]
        )
        order = np.argsort(severity)
        idx = np.linspace(0, order.size - 1, n_scenarios).round().astype(int)
        picks = cand[order[idx]]
    else:
        raise ValueError(f"unknown scenario mode: {mode}")

    pv_scen = np.zeros((n_scenarios, N_SLOTS))
    pv_scen[:, at_slot:] = pv_fc[at_slot:] + pv_err[picks][:, at_slot:]
    pv_scen = np.maximum(pv_scen, 0.0)

    load_scen = np.repeat(load_fc[None, :], n_scenarios, axis=0)
    if use_load_error:
        le = load_err[picks]
        load_scen = load_scen + np.nan_to_num(le, nan=0.0)

    net_scen = (load_scen - pv_scen) * delta_t
    net_scen[:, :at_slot] = 0.0
    return net_scen, pv_scen
