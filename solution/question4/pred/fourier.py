# -*- coding: utf-8 -*-
"""fourier implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BasePredictor

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


class FourierPredictor(BasePredictor):
    """FourierPredictor implementation."""

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

        if self.load_data.shape != self.pv_data.shape:
            raise ValueError("load and PV arrays must have the same shape")
        self.net = np.asarray(self.load_data, dtype=float) - np.asarray(
            self.pv_data, dtype=float
        )

        # Fast, exact date -> row lookup (the inherited implementation retries
        # a broken `array == str` mask and then scans every row with
        # pd.Timestamp, which is both slow and fragile).
        self._row_of = {}
        self._weekday = np.empty(len(self.dates_load), dtype=int)
        for i, d in enumerate(self.dates_load):
            ts = pd.Timestamp(d).normalize()
            self._row_of[ts] = i
            self._weekday[i] = ts.weekday()

    def _row(self, date_str: str) -> int:
        ts = pd.Timestamp(date_str).normalize()
        try:
            return self._row_of[ts]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Date {date_str} not found in data") from exc

    def predict(self, target_date: str) -> tuple:
        """predict implementation."""
        idx = self._row(target_date)

        start = 0 if self.window is None else max(0, idx - self.window)
        train = np.arange(start, idx)
        if train.size == 0:
            raise ValueError(f"Cannot predict {target_date}: no history available")

        if train.size < self.n_min_days:
            # Too little data to identify 11 coefficients: use the plain mean.
            pred = self.net[train].mean(axis=0)
            return pred, np.zeros(N_SLOTS)

        beta = self._fit(train)
        pred = self._design_for(idx) @ beta
        return np.asarray(pred, dtype=float), np.zeros(N_SLOTS)

    # ------------------------------------------------------------------ #
    def _fit(self, rows: np.ndarray) -> np.ndarray:
        """_fit implementation."""
        X = build_design(rows, self._weekday[rows])
        Y = self.net[rows]
        if X.shape[0] < X.shape[1]:  # rank-deficient: minimum-norm solution
            beta = np.linalg.pinv(X) @ Y
        else:
            beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        return beta

    def _design_for(self, idx: int) -> np.ndarray:
        return build_design(
            np.array([idx]), np.array([self._weekday[idx]])
        ).ravel()


class FourierQuantilePredictor(FourierPredictor):
    """FourierQuantilePredictor implementation."""

    def __init__(
        self,
        dates_load,
        load_data,
        dates_pv,
        pv_data,
        *,
        quantile: float = 5 / 6,
        residual_days: int = 60,
        window: int | None = None,
        n_min_days: int = 14,
    ):
        super().__init__(
            dates_load,
            load_data,
            dates_pv,
            pv_data,
            window=window,
            n_min_days=n_min_days,
        )
        self.quantile = quantile
        self.residual_days = residual_days
        self._residuals = None

    def rolling_residuals(self) -> np.ndarray:
        """rolling_residuals implementation."""
        if self._residuals is None:
            n_days = self.net.shape[0]
            out = np.full(self.net.shape, np.nan)
            for i in range(self.n_min_days, n_days):
                beta = self._fit(np.arange(0, i))
                out[i] = self.net[i] - self._design_for(i) @ beta
            self._residuals = out
        return self._residuals

    def predict(self, target_date: str) -> tuple:
        pred, _ = super().predict(target_date)
        idx = self._row(target_date)
        residuals = self.rolling_residuals()
        start = max(0, idx - self.residual_days)
        past = residuals[start:idx]
        past = past[~np.isnan(past).any(axis=1)]
        if past.shape[0] > 0:
            pred = pred + np.quantile(past, self.quantile, axis=0)
        return pred, np.zeros(N_SLOTS)


def fit_with_lags(
    net: np.ndarray,
    weekday: np.ndarray,
    rows: np.ndarray,
    target_idx: int,
    lags: tuple,
) -> np.ndarray:
    """fit_with_lags implementation."""
    n_features = N_FEATURES + len(lags)
    keep = rows[rows - max(lags) >= 0]

    if keep.size < n_features + 2:
        # Not enough lagged history yet: fall back to the pure Fourier fit.
        X = build_design(rows, weekday[rows])
        beta, *_ = np.linalg.lstsq(X, net[rows], rcond=None)
        xb = build_design(np.array([target_idx]), np.array([weekday[target_idx]]))
        return (xb @ beta).ravel()

    xb = build_design(
        np.array([target_idx]), np.array([weekday[target_idx]])
    ).ravel()
    Xb = build_design(keep, weekday[keep])
    ltrain = np.stack([net[keep - l] for l in lags], axis=2)        # (n, 144, k)
    ltarget = np.stack([net[target_idx - l] for l in lags], axis=1)  # (144, k)

    pred = np.empty(N_SLOTS)
    for t in range(N_SLOTS):
        X = np.column_stack([Xb, ltrain[:, t, :]])
        beta, *_ = np.linalg.lstsq(X, net[keep, t], rcond=None)
        pred[t] = np.concatenate([xb, ltarget[t]]) @ beta
    return pred


class FourierLagQuantilePredictor(BasePredictor):
    """FourierLagQuantilePredictor implementation."""

    def __init__(
        self,
        dates_load,
        load_data,
        dates_pv,
        pv_data,
        *,
        lags: tuple = (1, 2, 3),
        quantile: float = 0.75,
        residual_days: int = 30,
        n_min_days: int = 14,
    ):
        super().__init__(dates_load, load_data, dates_pv, pv_data)
        self.lags = tuple(int(l) for l in lags)
        self.quantile = float(quantile)
        self.residual_days = int(residual_days)
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
        self._residuals = None

    def _row(self, date_str: str) -> int:
        ts = pd.Timestamp(date_str).normalize()
        try:
            return self._row_of[ts]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Date {date_str} not found in data") from exc

    def _point(self, idx: int) -> np.ndarray:
        """_point implementation."""
        rows = np.arange(0, idx)
        if rows.size < self.n_min_days:
            raise ValueError(f"Cannot predict row {idx}: need >= {self.n_min_days} days")
        return fit_with_lags(self.net, self._weekday, rows, idx, self.lags)

    def rolling_residuals(self) -> np.ndarray:
        """rolling_residuals implementation."""
        if self._residuals is None:
            n_days = self.net.shape[0]
            out = np.full(self.net.shape, np.nan)
            for i in range(self.n_min_days, n_days):
                out[i] = self.net[i] - self._point(i)
            self._residuals = out
        return self._residuals

    def predict(self, target_date: str) -> tuple:
        """predict implementation."""
        idx = self._row(target_date)
        pred = self._point(idx)
        residuals = self.rolling_residuals()
        past = residuals[max(0, idx - self.residual_days):idx]
        past = past[~np.isnan(past).any(axis=1)]
        if past.shape[0] > 0:
            pred = pred + np.quantile(past, self.quantile, axis=0)
        return pred, np.zeros(N_SLOTS)
