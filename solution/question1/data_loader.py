# -*- coding: utf-8 -*-
"""data_loader implementation."""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "附件" / "附件1.xlsx"

DELTA_T = 10 / 60  # 10 min in hours


def _parse_time(raw: str) -> pd.Timestamp:
    """_parse_time implementation."""
    s = str(raw).strip()
    day_offset = 0
    if s.endswith("+1"):
        day_offset = 1
        s = s[:-2]

    parts = s.split(":")
    hour = int(parts[0])
    minute = int(parts[1]) if len(parts) > 1 else 0
    # Ignore seconds if present (parts[2])

    return pd.Timestamp(2025, 1, 1 + day_offset, hour, minute)


def load_data(file_path: Path | str = DATA_FILE) -> pd.DataFrame:
    """load_data implementation."""
    file_path = Path(file_path)
    df = pd.read_excel(file_path, sheet_name="Sheet1")

    df = df.rename(columns={
        "时间": "time",
        "电价": "price",
        "小区负载": "load",
        "光伏发电预测功率": "pv_forecast",
    })

    df["time"] = df["time"].map(_parse_time)
    midnight = df["time"].iloc[0].normalize()
    df["minute_of_day"] = ((df["time"] - midnight).dt.total_seconds() // 60).astype(int)
    df["step"] = np.arange(len(df))

    df = df[["step", "time", "minute_of_day", "price", "load", "pv_forecast"]]
    return df


def load_arrays(file_path: Path | str = DATA_FILE):
    """load_arrays implementation."""
    df = load_data(file_path)
    return (
        df["step"].to_numpy(),
        df["price"].to_numpy(dtype=float),
        df["load"].to_numpy(dtype=float),
        df["pv_forecast"].to_numpy(dtype=float),
    )