# -*- coding: utf-8 -*-
"""Fourier (harmonic) regression predictor for electricity prices.

Design matrix for day index d (12 columns):

    x_d = [1, e_0..e_6, sin(2*pi*d/365), cos(2*pi*d/365),
           sin(4*pi*d/365), cos(4*pi*d/365)]

  * 1                : intercept
  * e_0..e_6         : weekday one-hot (Monday = 0), exactly one is 1
  * sin/cos(2*pi...) : annual harmonic
  * sin/cos(4*pi...) : semi-annual (2nd) harmonic

One OLS is solved per 10-minute slot t (144 independent regressions that
share the same design matrix, since the features depend only on the day):

    beta_t = argmin |X beta - p_t|^2  ->  P_hat(d,t) = x_d^T beta_t

Only data strictly before the target day is used (no look-ahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


PERIOD = 365.0          # days, annual harmonic
N_SLOTS = 144           # 10-minute slots per day
N_FEATURES = 12         # 1 intercept + 7 weekday + 2*2 harmonics


def build_design(day_index: np.ndarray, weekday: np.ndarray) -> np.ndarray:
    """Return the (n, 12) Fourier design matrix for the given days.

    Args:
        day_index: 0-based day index used for the harmonic terms.
        weekday:   Monday-based weekday (0..6) used for the one-hot block.
    """
    day_index = np.asarray(day_index, dtype=float)
    weekday = np.asarray(weekday, dtype=int)
    n = day_index.size

    X = np.zeros((n, N_FEATURES), dtype=float)
    X[:, 0] = 1.0
    X[np.arange(n), 1 + weekday] = 1.0
    X[:, 8] = np.sin(2 * np.pi * day_index / PERIOD)
    X[:, 9] = np.cos(2 * np.pi * day_index / PERIOD)
    X[:, 10] = np.sin(4 * np.pi * day_index / PERIOD)
    X[:, 11] = np.cos(4 * np.pi * day_index / PERIOD)
    return X


class PriceFourierPredictor:
    """Harmonic regression on electricity price with a rolling expanding window.

    Args:
        dates_price: array of date strings
        price_data: array of shape (n_days, 144) with price values in yuan/kWh
        window: number of most recent days used for training
                (None = use every day from the start of the series).
        n_min_days: below this many training days fall back to the mean of
                    whatever history exists instead of raising.
    """

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
        """Predict price for ``target_date``; returns array of shape (144,)."""
        idx = self._row(target_date)

        start = 0 if self.window is None else max(0, idx - self.window)
        train = np.arange(start, idx)
        if train.size == 0:
            raise ValueError(f"Cannot predict {target_date}: no history available")

        if train.size < self.n_min_days:
            # Too little data to identify 12 coefficients: use the plain mean.
            pred = self.price_data[train].mean(axis=0)
            return np.asarray(pred, dtype=float)

        beta = self._fit(train)
        pred = self._design_for(idx) @ beta
        # Ensure prices are non-negative
        pred = np.maximum(pred, 0.0)
        return np.asarray(pred, dtype=float)

    def _fit(self, rows: np.ndarray) -> np.ndarray:
        """OLS for all 144 slots at once -> (12, 144) coefficient matrix."""
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
