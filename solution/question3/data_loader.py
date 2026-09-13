# -*- coding: utf-8 -*-
"""data_loader implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FILE_PRICE = PROJECT_ROOT / "附件" / "附件1.xlsx"
FILE_ACTUAL = PROJECT_ROOT / "附件" / "附件2.xlsx"
FILE_PV_FC = PROJECT_ROOT / "附件" / "附件3.xlsx"
FILE_TEMPLATE = PROJECT_ROOT / "附件" / "附件5" / "result3.xlsx"

DELTA_T = 10 / 60          # hours per 10-minute slot
N_SLOTS = 144
SLOTS_PER_HOUR = 6
ISSUE_SLOTS = (0, 36, 72, 108)          # 0:00, 6:00, 12:00, 18:00
ISSUE_LABELS = ("0:00", "6:00", "12:00", "18:00")

# Implementation detail.
ETA = 0.90
S_MIN = 1200.0
S_MAX = 10800.0
S_INIT = 6000.0
P_MAX = 5000.0             # kW -> P_MAX * DELTA_T = 833.3333 kWh per slot

# Implementation detail.
EMERGENCY_MULTIPLIER = 5.0   # Implementation detail.
# Implementation detail.
# Implementation detail.
# Implementation detail.
PLAN_DEVIATION_RATE = 0.5
PLAN_REDUCE_REFUND = 0.5     # Implementation detail.
PLAN_INCREASE_RATE = 1.5     # Implementation detail.


def load_price(file_path=FILE_PRICE) -> np.ndarray:
    """load_price implementation."""
    df = pd.read_excel(file_path, sheet_name="Sheet1")
    df = df.rename(columns={"电价": "price"})
    return df["price"].to_numpy(dtype=float)[:N_SLOTS]


def load_actual(file_path=FILE_ACTUAL) -> tuple:
    """load_actual implementation."""
    load = pd.read_excel(file_path, sheet_name="小区负载")
    pv = pd.read_excel(file_path, sheet_name="光伏发电实际功率")
    dates = pd.DatetimeIndex(pd.to_datetime(load.iloc[:, 0]))
    load_data = load.iloc[:, 1:].to_numpy(dtype=float)
    pv_data = pv.iloc[:, 1:].to_numpy(dtype=float)
    if load_data.shape != pv_data.shape:
        raise ValueError("load and PV sheets have different shapes")
    return dates, load_data, pv_data


def load_pv_forecast(file_path=FILE_PV_FC) -> tuple:
    """load_pv_forecast implementation."""
    df = pd.read_excel(file_path, sheet_name="Sheet1")
    if df.shape[0] % 4 != 0:
        raise ValueError(f"expected 4 rows per day, got {df.shape[0]} rows")

    n_days = df.shape[0] // 4
    hour_cols = list(df.columns[2:26])            # Implementation detail.
    date_col = df.columns[0]
    fc = np.empty((n_days, 4, 24), dtype=float)
    dates = []

    for d in range(n_days):
        block = df.iloc[d * 4:(d + 1) * 4]
        raw_date = block[date_col].iloc[0]
        if pd.isna(raw_date):
            raw_date = dates[-1]
        dates.append(pd.Timestamp(raw_date).normalize())
        fc[d] = block[hour_cols].to_numpy(dtype=float)

    return pd.DatetimeIndex(dates), fc


def hourly_to_slots(hourly: np.ndarray, method: str = "step") -> np.ndarray:
    """hourly_to_slots implementation."""
    hourly = np.asarray(hourly, dtype=float)
    if hourly.size != 24:
        raise ValueError("expected 24 hourly values")

    if method == "step":
        return np.repeat(hourly, SLOTS_PER_HOUR)

    if method == "linear":
        centres = np.arange(24) * SLOTS_PER_HOUR + (SLOTS_PER_HOUR - 1) / 2.0
        slots = np.arange(N_SLOTS, dtype=float)
        return np.interp(slots, centres, hourly, left=hourly[0], right=hourly[-1])

    raise ValueError(f"unknown method: {method}")


def forecast_profile(fc_day: np.ndarray, issue_idx: int, method: str = "step") -> np.ndarray:
    """forecast_profile implementation."""
    # hourly_to_slots returns 144 values *starting at the issue time*, so only
    # the first (144 - issue slot) values belong to the remaining part of today.
    full = hourly_to_slots(fc_day[issue_idx], method=method)
    profile = np.full(N_SLOTS, np.nan)
    profile[ISSUE_SLOTS[issue_idx]:] = full[: N_SLOTS - ISSUE_SLOTS[issue_idx]]
    return profile


def day_index_of(dates: pd.DatetimeIndex, date_str: str) -> int:
    """day_index_of implementation."""
    target = pd.Timestamp(date_str).normalize()
    matches = np.where(dates == target)[0]
    if matches.size == 0:
        raise ValueError(f"{date_str} not found in attachment 2/3")
    return int(matches[0])
