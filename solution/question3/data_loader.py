# -*- coding: utf-8 -*-
"""Data access for Question 3.

* 附件1 : 电价 (元/kWh)。问题 3 每天电价相同，取附件 1 的 144 个时段价格。
* 附件2 : 2025.1.1-12.31 实际小区负载 / 实际光伏出力 (365 x 144)。
* 附件3 : 2025.1.1-12.31 每天 0:00、6:00、12:00、18:00 发布的未来 24 小时
          整点光伏预报 (365 x 4 x 24)。

预报的整点值按"该小时内 6 个 10 min 周期的平均功率"折算成 10 min 曲线
（与参考论文一致）；也可选用线性插值，见 ``hourly_to_slots``。
"""

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

# 储能参数（附录 1）
ETA = 0.90
S_MIN = 1200.0
S_MAX = 10800.0
S_INIT = 6000.0
P_MAX = 5000.0             # kW -> P_MAX * DELTA_T = 833.3333 kWh per slot

# 结算参数（问题 3）
EMERGENCY_MULTIPLIER = 5.0   # 紧急购电 = 5 倍交易时刻电价
# 调减：计划量高于调整量的部分，按原价 50% 计（相对原计划节省 50%）
# 调增：调整量高于计划量的部分，按原价 1.5 倍计
# 两种情形合并后，逐时段费用为  p*ga + 0.5*p*|g0-ga|  (+ 紧急购电)
PLAN_DEVIATION_RATE = 0.5
PLAN_REDUCE_REFUND = 0.5     # 调减时的冲减比例（用于费用分解报告）
PLAN_INCREASE_RATE = 1.5     # 调增时超出部分的电价倍数（用于费用分解报告）


def load_price(file_path=FILE_PRICE) -> np.ndarray:
    """144 normal electricity prices (元/kWh), identical for every day."""
    df = pd.read_excel(file_path, sheet_name="Sheet1")
    df = df.rename(columns={"电价": "price"})
    return df["price"].to_numpy(dtype=float)[:N_SLOTS]


def load_actual(file_path=FILE_ACTUAL) -> tuple:
    """Actual load / PV: returns (dates, load (365,144), pv (365,144))."""
    load = pd.read_excel(file_path, sheet_name="小区负载")
    pv = pd.read_excel(file_path, sheet_name="光伏发电实际功率")
    dates = pd.DatetimeIndex(pd.to_datetime(load.iloc[:, 0]))
    load_data = load.iloc[:, 1:].to_numpy(dtype=float)
    pv_data = pv.iloc[:, 1:].to_numpy(dtype=float)
    if load_data.shape != pv_data.shape:
        raise ValueError("load and PV sheets have different shapes")
    return dates, load_data, pv_data


def load_pv_forecast(file_path=FILE_PV_FC) -> tuple:
    """Hourly PV forecasts: returns (dates (365,), fc (365,4,24) kW).

    The sheet lists 4 rows per day (0:00 / 6:00 / 12:00 / 18:00), the date
    being written only on the first row of each group.
    """
    df = pd.read_excel(file_path, sheet_name="Sheet1")
    if df.shape[0] % 4 != 0:
        raise ValueError(f"expected 4 rows per day, got {df.shape[0]} rows")

    n_days = df.shape[0] // 4
    hour_cols = list(df.columns[2:26])            # 预报1小时 ... 预报24小时
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
    """Expand hourly values to the 144-slot (10 min) profile.

    ``hourly`` has 24 values covering hours 1..24 from the issue time.
    ``method='step'``   : each hourly value is the average power of its 6 slots.
    ``method='linear'`` : piecewise linear interpolation between hour centres,
                          anchored at the first/last hour value.
    """
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
    """144-slot PV forecast issued at ``issue_idx`` for the rest of that day.

    Slots before the issue time are NaN (already observed in the rolling
    scheme), because the 24 h forecast window starts at the issue time.
    """
    # hourly_to_slots returns 144 values *starting at the issue time*, so only
    # the first (144 - issue slot) values belong to the remaining part of today.
    full = hourly_to_slots(fc_day[issue_idx], method=method)
    profile = np.full(N_SLOTS, np.nan)
    profile[ISSUE_SLOTS[issue_idx]:] = full[: N_SLOTS - ISSUE_SLOTS[issue_idx]]
    return profile


def day_index_of(dates: pd.DatetimeIndex, date_str: str) -> int:
    """Row index of a date in the 365-day series."""
    target = pd.Timestamp(date_str).normalize()
    matches = np.where(dates == target)[0]
    if matches.size == 0:
        raise ValueError(f"{date_str} not found in attachment 2/3")
    return int(matches[0])
