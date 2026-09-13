# -*- coding: utf-8 -*-
"""data_loader implementation."""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE_LOAD = PROJECT_ROOT / "附件" / "附件2.xlsx"
DATA_FILE_PRICE = PROJECT_ROOT / "附件" / "附件4.xlsx"

DELTA_T = 10 / 60  # 10 min in hours


def _parse_time_columns(df: pd.DataFrame) -> tuple:
    """_parse_time_columns implementation."""
    date_col = df.columns[0]
    time_cols = df.columns[1:]  # 144 time slots
    dates = df[date_col].values
    data = df[time_cols].values.astype(float)
    return dates, data


def load_historical_load(file_path: Path | str = DATA_FILE_LOAD) -> tuple:
    """load_historical_load implementation."""
    df = pd.read_excel(file_path, sheet_name="小区负载")
    return _parse_time_columns(df)


def load_historical_pv(file_path: Path | str = DATA_FILE_LOAD) -> tuple:
    """load_historical_pv implementation."""
    df = pd.read_excel(file_path, sheet_name="光伏发电实际功率")
    return _parse_time_columns(df)


def load_time_varying_price(file_path: Path | str = DATA_FILE_PRICE) -> tuple:
    """load_time_varying_price implementation."""
    df = pd.read_excel(file_path, sheet_name="Sheet1")
    return _parse_time_columns(df)


def get_day_price(dates: np.ndarray, price: np.ndarray, date_str: str) -> np.ndarray:
    """get_day_price implementation."""
    # Try exact match first
    mask = dates == date_str
    if mask.any():
        idx = np.where(mask)[0][0]
        return price[idx]
    
    # Try parsing dates more flexibly
    target = pd.Timestamp(date_str)
    for i, d in enumerate(dates):
        try:
            if pd.Timestamp(d) == target:
                return price[i]
        except:
            continue
    
    raise ValueError(f"Date {date_str} not found in price data")


def get_day_data(dates: np.ndarray, data: np.ndarray, date_str: str) -> np.ndarray:
    """get_day_data implementation."""
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
    """get_date_index implementation."""
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
    dates_price, price_data = load_time_varying_price()
    
    print(f"Load data shape: {load_data.shape}")
    print(f"PV data shape: {pv_data.shape}")
    print(f"Price data shape: {price_data.shape}")
    print(f"First date: {dates_load[0]}")
    print(f"Last date: {dates_load[-1]}")
    
    # Test getting a specific day
    day_price = get_day_price(dates_price, price_data, "2025-02-01")
    print(f"\n2025-02-01 price (first 10 slots): {day_price[:10]}")
    print(f"2025-02-01 price min: {day_price.min():.4f}, max: {day_price.max():.4f}, mean: {day_price.mean():.4f}")
