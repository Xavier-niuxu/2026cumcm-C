# -*- coding: utf-8 -*-
"""Scaled Fourier predictor that normalizes weekend load to weekday level.

The idea:
1. Compute the ratio of weekday/weekend average net load from history
2. Scale weekend data up by this ratio during training
3. Fit Fourier on the scaled data (all days, but weekend normalized)
4. Scale predictions back down for weekend days
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BasePredictor
from .fourier import build_design, N_SLOTS


class ScaledFourierPredictor(BasePredictor):
    """Fourier predictor with weekend load scaling."""

    def __init__(
        self,
        dates_load,
        load_data,
        dates_pv,
        pv_data,
        *,
        window: int | None = None,
        n_min_days: int = 14,
    ):
        super().__init__(dates_load, load_data, dates_pv, pv_data)
        self.window = window
        self.n_min_days = n_min_days

        # Compute net load
        self.net = np.asarray(self.load_data, dtype=float) - np.asarray(
            self.pv_data, dtype=float
        )

        # Build date lookup and weekday array
        self._row_of = {}
        self._weekday = np.empty(len(self.dates_load), dtype=int)
        self._is_weekend = np.zeros(len(self.dates_load), dtype=bool)
        for i, d in enumerate(self.dates_load):
            ts = pd.Timestamp(d).normalize()
            self._row_of[ts] = i
            wd = ts.weekday()
            self._weekday[i] = wd
            self._is_weekend[i] = wd >= 5  # Saturday=5, Sunday=6

        # Compute scaling factor: weekday_avg / weekend_avg (per slot)
        self._compute_scale_factor()

        # Create scaled net load (weekend scaled up to weekday level)
        self.net_scaled = self.net.copy()
        weekend_mask = self._is_weekend
        self.net_scaled[weekend_mask] *= self.scale_factor  # scale up weekend

    def _compute_scale_factor(self):
        """Compute the ratio of weekday to weekend average net load per slot."""
        weekday_net = self.net[~self._is_weekend]
        weekend_net = self.net[self._is_weekend]

        weekday_avg = weekday_net.mean(axis=0)  # (144,)
        weekend_avg = weekend_net.mean(axis=0)   # (144,)

        # Scale factor per slot: weekday / weekend (to scale weekend UP to weekday level)
        self.scale_factor = np.where(
            weekend_avg > 1e-6,
            weekday_avg / weekend_avg,
            1.0
        )  # (144,)

        print(f"[ScaledFourier] Weekday avg net load (total): {weekday_avg.sum() * 10/60:.2f} kWh/day")
        print(f"[ScaledFourier] Weekend avg net load (total): {weekend_avg.sum() * 10/60:.2f} kWh/day")
        print(f"[ScaledFourier] Avg scale factor (weekday/weekend): {self.scale_factor.mean():.4f}")

    def _row(self, date_str: str) -> int:
        ts = pd.Timestamp(date_str).normalize()
        try:
            return self._row_of[ts]
        except KeyError as exc:
            raise ValueError(f"Date {date_str} not found in data") from exc

    def predict(self, target_date: str) -> tuple:
        """Predict net load for target_date; returns (N_hat, zeros(144))."""
        idx = self._row(target_date)

        start = 0 if self.window is None else max(0, idx - self.window)
        train = np.arange(start, idx)
        if train.size == 0:
            raise ValueError(f"Cannot predict {target_date}: no history available")

        if train.size < self.n_min_days:
            # Too little data: use mean of scaled history
            pred = self.net_scaled[train].mean(axis=0)
        else:
            # Fit on scaled data
            beta = self._fit_scaled(train)
            pred = self._design_for(idx) @ beta

        # Scale back down if target is weekend
        if self._is_weekend[idx]:
            pred = pred / self.scale_factor

        return np.asarray(pred, dtype=float), np.zeros(N_SLOTS)

    def _fit_scaled(self, rows: np.ndarray) -> np.ndarray:
        """OLS on scaled net load for all 144 slots."""
        X = build_design(rows, self._weekday[rows])
        Y = self.net_scaled[rows]
        if X.shape[0] < X.shape[1]:
            beta = np.linalg.pinv(X) @ Y
        else:
            beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        return beta

    def _design_for(self, idx: int) -> np.ndarray:
        return build_design(
            np.array([idx]), np.array([self._weekday[idx]])
        ).ravel()
