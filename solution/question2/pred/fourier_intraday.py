# -*- coding: utf-8 -*-
"""Fourier predictor with intraday (24h) harmonics.

Extends the base FourierPredictor by adding sin/cos terms for the within-day
cycle (period = 144 slots = 24 hours).

Feature vector x(d, t):
  - Day-level (N_DAY_FEATURES = 11): e_0..e_6,
                    sin(2pi*d/365), cos(2pi*d/365),
                    sin(4pi*d/365), cos(4pi*d/365)
  - Intraday (2*K): sin(2*pi*k*t/144), cos(2*pi*k*t/144) for k=1..K

Total features: N_DAY_FEATURES + 2*K

Training: flatten all (day, slot) pairs, single OLS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BasePredictor
from .fourier import build_design, N_SLOTS, PERIOD, N_FEATURES as N_DAY_FEATURES


class FourierIntradayPredictor(BasePredictor):
    """Fourier predictor with day-level + intraday harmonics."""

    def __init__(
        self,
        dates_load,
        load_data,
        dates_pv,
        pv_data,
        *,
        n_intraday: int = 6,
        window: int | None = None,
        n_min_days: int = 14,
    ):
        super().__init__(dates_load, load_data, dates_pv, pv_data)
        self.n_intraday = n_intraday  # K: number of intraday harmonics
        self.window = window
        self.n_min_days = n_min_days

        self.net = np.asarray(self.load_data, dtype=float) - np.asarray(
            self.pv_data, dtype=float
        )

        self._row_of = {}
        self._weekday = np.empty(len(self.dates_load), dtype=int)
        for i, d in enumerate(self.dates_load):
            ts = pd.Timestamp(d).normalize()
            self._row_of[ts] = i
            self._weekday[i] = ts.weekday()

        # Precompute intraday basis: (144, 2*K)
        self._intra_basis = self._build_intraday_basis()
        self.n_features = N_DAY_FEATURES + 2 * n_intraday

    def _build_intraday_basis(self) -> np.ndarray:
        """(144, 2*K) matrix of intraday sin/cos terms."""
        t = np.arange(N_SLOTS, dtype=float)
        basis = np.zeros((N_SLOTS, 2 * self.n_intraday))
        for k in range(1, self.n_intraday + 1):
            basis[:, 2 * (k - 1)] = np.sin(2 * np.pi * k * t / N_SLOTS)
            basis[:, 2 * (k - 1) + 1] = np.cos(2 * np.pi * k * t / N_SLOTS)
        return basis

    def _row(self, date_str: str) -> int:
        ts = pd.Timestamp(date_str).normalize()
        try:
            return self._row_of[ts]
        except KeyError as exc:
            raise ValueError(f"Date {date_str} not found in data") from exc

    def predict(self, target_date: str) -> tuple:
        idx = self._row(target_date)

        start = 0 if self.window is None else max(0, idx - self.window)
        train = np.arange(start, idx)
        if train.size == 0:
            raise ValueError(f"Cannot predict {target_date}: no history available")

        if train.size < self.n_min_days:
            pred = self.net[train].mean(axis=0)
            return pred, np.zeros(N_SLOTS)

        beta = self._fit(train)  # (n_features,)
        # pred(t) = day_features(idx) @ beta_day + intra_basis(t) @ beta_intra
        day_feat = self._day_features(idx)  # (N_DAY_FEATURES,)
        pred = (day_feat @ beta[:N_DAY_FEATURES]
                + self._intra_basis @ beta[N_DAY_FEATURES:])
        return np.asarray(pred, dtype=float), np.zeros(N_SLOTS)

    def _day_features(self, idx: int) -> np.ndarray:
        """(N_DAY_FEATURES,) day-level feature vector."""
        return build_design(np.array([idx]), np.array([self._weekday[idx]])).ravel()

    def _fit(self, rows: np.ndarray) -> np.ndarray:
        """OLS on flattened (day, slot) pairs."""
        n_days = rows.size
        n_feat = self.n_features

        # Build design matrix: (n_days * 144, n_feat)
        day_X = build_design(rows, self._weekday[rows])  # (n_days, N_DAY_FEATURES)

        # Full design: each row is [day_feat(d), intra_basis(t)]
        X = np.zeros((n_days * N_SLOTS, n_feat))
        for i in range(n_days):
            X[i * N_SLOTS:(i + 1) * N_SLOTS, :N_DAY_FEATURES] = day_X[i]
            X[i * N_SLOTS:(i + 1) * N_SLOTS, N_DAY_FEATURES:] = self._intra_basis

        Y = self.net[rows].ravel()  # (n_days * 144,)

        if X.shape[0] < X.shape[1]:
            beta = np.linalg.pinv(X) @ Y
        else:
            beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        return beta
