# -*- coding: utf-8 -*-
"""Scenario generation for the rolling stochastic MPC of Question 3.

Two uncertainty sources are modelled:

* 光伏预报误差 : 附件2 实际光伏  -  附件3 同一发布时刻的预报 (整日 144 维)
* 负荷预报误差 : 附件2 实际负荷  -  问题 2 傅里叶模型的滚动样本外预测

Scenarios are sampled as **whole days** (整日 bootstrap): 第 j 天的 144 维误差
向量整体作为一个样本。这样场景内部保留时段间的时序相关性（不会出现"上午
阴天、下午暴晒"这类物理上不可能的组合），也保留了负荷与光伏误差的同期相关；
两个误差池共用同一个抽取的日期索引，因此相关性得以保持。

第 d 天只允许使用 d 之前 lookback 窗口内的历史误差样本，避免未来信息泄漏。
光伏误差按**发布时刻**分别建池：6:00 发布的预报与 0:00 发布的预报，其误差分布
（随预报提前期缩短而变小）本来就不同。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# 追加到 sys.path 末尾（而不是 insert(0)）：问题 2 目录下有同名的 data_loader.py
# 与 main.py，插到最前面会遮蔽问题 3 自己的模块。
sys.path.append(str(Path(__file__).resolve().parents[1] / "question2"))

from pred.fourier import FourierPredictor  # noqa: E402

from data_loader import N_SLOTS, forecast_profile  # noqa: E402


def load_forecast_model(dates, load):
    """傅里叶负荷预测模型（问题 2 的模型，target 改为负荷）。

    把光伏序列置 0 即可：模型建模的是 ``load - pv``，于是得到的就是负荷本身，
    无需改动问题 2 的任何代码。``predict`` 返回 ``(load_hat, zeros)``。
    """
    return FourierPredictor(dates, load, dates, np.zeros_like(load))


def load_error_pool(model, dates, load) -> np.ndarray:
    """(n_days, 144) 滚动样本外负荷预测误差；不可得的行为 NaN。"""
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
    """(n_days, 144) 某发布时刻的光伏预报误差；发布时刻之前的时段为 NaN。"""
    n_days = pv_actual.shape[0]
    pool = np.full((n_days, N_SLOTS), np.nan)
    for d in range(n_days):
        pool[d] = pv_actual[d] - forecast_profile(pv_fc[d], issue_idx, method=method)
    return pool


def usable_days(pv_err: np.ndarray, at_slot: int, day: int, lookback: int) -> np.ndarray:
    """可用作场景的历史日期（严格早于 day，且剩余时段误差非全 NaN）。"""
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
    """生成净负荷场景（kWh/时段），返回 (net_scen (N,144), pv_scen (N,144))。

    仅 ``at_slot`` 之后（含）的时段有效，之前的时段填 0。
    """
    cand = usable_days(pv_err, at_slot, day, lookback)
    if cand.size == 0:
        raise ValueError(f"no usable PV scenario history before day {day}")

    if mode == "bootstrap":
        picks = rng.choice(cand, size=n_scenarios, replace=True)
    elif mode == "reduced":
        # 场景缩减：按"缺额严重度"排序后取 n_scenarios 个分位代表（等权）。
        # 对应参考解用 K-means 把 90 天误差向量压成若干代表场景的做法。
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
