# -*- coding: utf-8 -*-
"""output implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from LP import LPResult


def _time_label(t: int) -> str:
    """_time_label implementation."""
    h1 = (t * 10) // 60
    m1 = (t * 10) % 60
    h2 = ((t + 1) * 10) // 60
    m2 = ((t + 1) * 10) % 60
    return f"{h1}:{m1:02d}-{h2}:{m2:02d}"


def _four_hour_label(k: int) -> str:
    return f"{k * 4}:00-{(k + 1) * 4}:00"


def build_purchase_sheet(ans: LPResult) -> pd.DataFrame:
    n = len(ans.grid_purchase)
    if n != 144:
        raise ValueError(f"问题1应有144个10分钟时段，当前为 {n}")

    rows = []
    for t in range(n):
        rows.append({
            "序号": t + 1,
            "时间段": _time_label(t),
            "购电量/kWh": ans.grid_purchase[t],
        })

    rows.append({
        "序号": "",
        "时间段": "全天",
        "购电量/kWh": ans.grid_purchase.sum(),
    })
    rows.append({
        "序号": "",
        "时间段": "全天购电费",
        "购电量/kWh": ans.purchase_cost,
    })
    return pd.DataFrame(rows)


def build_storage_sheet(ans: LPResult) -> pd.DataFrame:
    # Implementation detail.
    rows = []
    for k in range(6):
        lo = k * 24
        hi = (k + 1) * 24
        rows.append({
            "时间段": _four_hour_label(k),
            "充电量/kWh": ans.charge[lo:hi].sum(),
            "放电量/kWh": ans.discharge[lo:hi].sum(),
        })

    rows.append({
        "时间段": "0:00储电量",
        "充电量/kWh": ans.soc_start[0],
        "放电量/kWh": np.nan,
    })
    rows.append({
        "时间段": "24:00储电量",
        "充电量/kWh": ans.soc_end[-1],
        "放电量/kWh": np.nan,
    })
    return pd.DataFrame(rows)


def build_full_schedule_sheet(ans: LPResult) -> pd.DataFrame:
    """build_full_schedule_sheet implementation."""
    n = len(ans.grid_purchase)
    dt = 1 / 6
    return pd.DataFrame({
        "step": np.arange(1, n + 1),
        "时间段": [_time_label(t) for t in range(n)],
        "电价/(元/kWh)": ans.price,
        "负荷功率/kW": ans.load,
        "光伏预测功率/kW": ans.pv_forecast,
        "计划购电量/kWh": ans.grid_purchase,
        "充电量/kWh": ans.charge,
        "放电量/kWh": ans.discharge,
        "0.1小时负荷电量/kWh": ans.load * dt,
        "0.1小时光伏电量/kWh": ans.pv_forecast * dt,
        "时段开始储电量/kWh": ans.soc_start,
        "时段结束储电量/kWh": ans.soc_end,
    })


def save_result1(ans: LPResult, output_path: Path | str) -> Path:
    """save_result1 implementation."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    purchase = build_purchase_sheet(ans)
    storage = build_storage_sheet(ans)
    full = build_full_schedule_sheet(ans)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        purchase.to_excel(writer, sheet_name="计划购电量", index=False)
        storage.to_excel(writer, sheet_name="充放电量", index=False)
        full.to_excel(writer, sheet_name="完整调度", index=False)

    return output_path


def print_summary(ans: LPResult) -> None:
    print("=" * 60)
    print("问题1：确定性日前调度 LP 求解结果")
    print("=" * 60)
    print(f"全天购电量：{ans.grid_purchase.sum():.4f} kWh")
    print(f"全天购电费：{ans.purchase_cost:.4f} 元")
    print()
    print("储能设备分时段充放电量：")
    print(f"{'时间段':<15}{'充电量/kWh':>16}{'放电量/kWh':>16}")
    for k in range(6):
        lo, hi = k * 24, (k + 1) * 24
        print(
            f"{_four_hour_label(k):<15}"
            f"{ans.charge[lo:hi].sum():>16.4f}"
            f"{ans.discharge[lo:hi].sum():>16.4f}"
        )
    print()
    print(f"0:00 储电量：{ans.soc_start[0]:.4f} kWh")
    print(f"24:00储电量：{ans.soc_end[-1]:.4f} kWh")
    print("=" * 60)
