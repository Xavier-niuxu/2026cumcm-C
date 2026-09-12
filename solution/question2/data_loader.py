# -*- coding: utf-8 -*-
"""Load historical data from 附件2.xlsx (load/PV) and 附件1.xlsx (fixed price)."""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE_LOAD = PROJECT_ROOT / "附件" / "附件2.xlsx"
DATA_FILE_PRICE = PROJECT_ROOT / "附件" / "附件1.xlsx"

DELTA_T = 10 / 60  # 10 min in hours


def _parse_time_columns(df: pd.DataFrame) -> tuple:
    """Extract date column and time slot columns from wide-format DataFrame."""
    date_col = df.columns[0]
    time_cols = df.columns[1:]  # 144 time slots
    dates = df[date_col].values
    data = df[time_cols].values.astype(float)
    return dates, data


def load_historical_load(file_path: Path | str = DATA_FILE_LOAD) -> tuple:
    """Load historical load data from 附件2.xlsx.
    
    Returns:
        dates: array of shape (365,) with date strings
        data: array of shape (365, 144) with load values in kW
    """
    df = pd.read_excel(file_path, sheet_name="小区负载")
    return _parse_time_columns(df)


def load_historical_pv(file_path: Path | str = DATA_FILE_LOAD) -> tuple:
    """Load historical PV actual power from 附件2.xlsx.
    
    Returns:
        dates: array of shape (365,) with date strings
        data: array of shape (365, 144) with PV power values in kW
    """
    df = pd.read_excel(file_path, sheet_name="光伏发电实际功率")
    return _parse_time_columns(df)


def load_fixed_price(file_path: Path | str = DATA_FILE_PRICE) -> np.ndarray:
    """Load fixed electricity price from 附件1.xlsx.
    
    Returns:
        price: array of shape (144,) with price values in yuan/kWh
    """
    df = pd.read_excel(file_path, sheet_name="Sheet1")
    df = df.rename(columns={"电价": "price"})
    return df["price"].to_numpy(dtype=float)


def get_day_data(dates: np.ndarray, data: np.ndarray, date_str: str) -> np.ndarray:
    """Get data for a specific date.
    
    Args:
        dates: array of date strings
        data: array of shape (365, 144)
        date_str: target date string (e.g., "2025-01-08")
    
    Returns:
        array of shape (144,) with data for that date
    """
    # Try exact match first
    mask = dates == date_str
    if mask.any():
        idx = np.where(mask)[0][0]
        return data[idx]
    
    # Try parsing dates more flexibly
    target = pd.Timestamp(date_str)
    for i, d in enumerate(dates):
        try:
            if pd.Timestamp(d) == target:
                return data[i]
        except:
            continue
    
    raise ValueError(f"Date {date_str} not found in data")


def get_date_index(dates: np.ndarray, date_str: str) -> int:
    """Get the index of a date in the dates array."""
    mask = dates == date_str
    if mask.any():
        return int(np.where(mask)[0][0])
    
    target = pd.Timestamp(date_str)
    for i, d in enumerate(dates):
        try:
            if pd.Timestamp(d) == target:
                return i
        except:
            continue
    
    raise ValueError(f"Date {date_str} not found in data")


if __name__ == "__main__":
    # Test loading
    dates_load, load_data = load_historical_load()
    dates_pv, pv_data = load_historical_pv()
    dates_price, price_data = load_historical_price()
    
    print(f"Load data shape: {load_data.shape}")
    print(f"PV data shape: {pv_data.shape}")
    print(f"Price data shape: {price_data.shape}")
    print(f"First date: {dates_load[0]}")
    print(f"Last date: {dates_load[-1]}")
    
    # Test getting a specific day
    day_load = get_day_data(dates_load, load_data, "2025-01-08")
    print(f"\n2025-01-08 load (first 10 slots): {day_load[:10]}")
