# -*- coding: utf-8 -*-
"""price_fourier implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd


PERIOD = 365.0          # days, annual harmonic
N_SLOTS = 144           # 10-minute slots per day
N_FEATURES = 11         # 7 weekday + 2*2 harmonics (no intercept)


def build_design(day_index: np.ndarray, weekday: np.ndarray) -> np.ndarray:
    """build_design implementation."""
    day_index = np.asarray(day_index, dtype=float)
    weekday = np.asarray(weekday, dtype=int)
    n = day_index.size

    X = np.zeros((n, N_FEATURES), dtype=float)
    X[np.arange(n), weekday] = 1.0
    X[:, 7] = np.sin(2 * np.pi * day_index / PERIOD)
    X[:, 8] = np.cos(2 * np.pi * day_index / PERIOD)
    X[:, 9] = np.sin(4 * np.pi * day_index / PERIOD)
    X[:, 10] = np.cos(4 * np.pi * day_index / PERIOD)
    return X


class PriceFourierPredictor:
    """PriceFourierPredictor implementation."""

    def __init__(
        self,
        dates_price: np.ndarray,
        price_data: np.ndarray,
        *,
        window: int | None = None,
        n_min_days: int = 14,
    ):
        self.dates_price = dates_price
        self.price_data = np.asarray(price_data, dtype=float)
        self.window = window
        self.n_min_days = n_min_days

        if self.price_data.shape[1] != N_SLOTS:
            raise ValueError(f"price_data must have {N_SLOTS} columns")

        # Fast, exact date -> row lookup
        self._row_of = {}
        self._weekday = np.empty(len(self.dates_price), dtype=int)
        for i, d in enumerate(self.dates_price):
            ts = pd.Timestamp(d).normalize()
            self._row_of[ts] = i
            self._weekday[i] = ts.weekday()

    def _row(self, date_str: str) -> int:
        ts = pd.Timestamp(date_str).normalize()
        try:
            return self._row_of[ts]
        except KeyError as exc:
            raise ValueError(f"Date {date_str} not found in data") from exc

    def predict(self, target_date: str) -> np.ndarray:
        """predict implementation."""
        idx = self._row(target_date)

        start = 0 if self.window is None else max(0, idx - self.window)
        train = np.arange(start, idx)
        if train.size == 0:
            raise ValueError(f"Cannot predict {target_date}: no history available")

        if train.size < self.n_min_days:
            # Too little data to identify 11 coefficients: use the plain mean.
            pred = self.price_data[train].mean(axis=0)
            return np.asarray(pred, dtype=float)

        beta = self._fit(train)
        pred = self._design_for(idx) @ beta
        # Ensure prices are non-negative
        pred = np.maximum(pred, 0.0)
        return np.asarray(pred, dtype=float)

    def _fit(self, rows: np.ndarray) -> np.ndarray:
        """_fit implementation."""
        X = build_design(rows, self._weekday[rows])
        Y = self.price_data[rows]
        if X.shape[0] < X.shape[1]:  # rank-deficient: minimum-norm solution
            beta = np.linalg.pinv(X) @ Y
        else:
            beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        return beta

    def _design_for(self, idx: int) -> np.ndarray:
        return build_design(
            np.array([idx]), np.array([self._weekday[idx]])
        ).ravel()
