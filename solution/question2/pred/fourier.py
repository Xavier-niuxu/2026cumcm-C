# -*- coding: utf-8 -*-
"""Fourier (harmonic) regression predictor for Question 2.

Design matrix for day index d (12 columns):

    x_d = [1, e_0..e_6, sin(2*pi*d/365), cos(2*pi*d/365),
           sin(4*pi*d/365), cos(4*pi*d/365)]

  * 1                : intercept
  * e_0..e_6         : weekday one-hot (Monday = 0), exactly one is 1
  * sin/cos(2*pi...) : annual harmonic
  * sin/cos(4*pi...) : semi-annual (2nd) harmonic

One OLS is solved per 10-minute slot t (144 independent regressions that
share the same design matrix, since the features depend only on the day):

    beta_t = argmin |X beta - y_t|^2  ->  N_hat(d,t) = x_d^T beta_t

Only data strictly before the target day is used (no look-ahead), so the
predictor can be rolled forward day by day.

Target variable is the NET load N = L - P.  The day-ahead LP of Question 2
only ever sees ``load_forecast - pv_forecast``, so the predictor returns
``(N_hat, zeros)`` and is a drop-in replacement for the (L, P) predictors.

``FourierQuantilePredictor`` adds a newsboy-style safety margin on top of the
point forecast, using the quantile of the model's own past rolling
out-of-sample residuals.  Under Question 2's asymmetric penalty (emergency
power costs 5x the normal price) the cost-optimal target is the
5/(5+1) = 83.3% quantile of the net-load distribution, not its mean.

Note on conditioning: the harmonic columns are nearly collinear with the
intercept over short windows, so a fit is only reliable for a *short*
extrapolation.  In rolling mode the target day is one day past the training
window, which is fine; refitting on a truncated window and extrapolating
further ahead is not, and is deliberately avoided here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BasePredictor

PERIOD = 365.0          # days, annual harmonic
N_SLOTS = 144           # 10-minute slots per day
N_FEATURES = 11         # 7 weekday + 2*2 harmonics (no intercept to avoid dummy variable trap)


def build_design(day_index: np.ndarray, weekday: np.ndarray) -> np.ndarray:
    """Return the (n, 11) Fourier design matrix for the given days.

    Args:
        day_index: 0-based day index used for the harmonic terms.
        weekday:   Monday-based weekday (0..6) used for the one-hot block.
    
    Note:
        Intercept column is omitted to avoid the dummy variable trap
        (weekday one-hot columns sum to 1, which is collinear with intercept).
    """
    day_index = np.asarray(day_index, dtype=float)
    weekday = np.asarray(weekday, dtype=int)
    n = day_index.size

    X = np.zeros((n, N_FEATURES), dtype=float)
    X[np.arange(n), weekday] = 1.0  # columns 0-6: weekday one-hot
    X[:, 7] = np.sin(2 * np.pi * day_index / PERIOD)
    X[:, 8] = np.cos(2 * np.pi * day_index / PERIOD)
    X[:, 9] = np.sin(4 * np.pi * day_index / PERIOD)
    X[:, 10] = np.cos(4 * np.pi * day_index / PERIOD)
    return X


class FourierPredictor(BasePredictor):
    """Harmonic regression on net load with a rolling expanding window.

    Args:
        window:        number of most recent days used for training
                       (None = use every day from the start of the series).
        n_min_days:    below this many training days fall back to the mean of
                       whatever history exists instead of raising.
    """

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
        """Predict net load for ``target_date``; returns (N_hat, zeros(144))."""
        idx = self._row(target_date)

        start = 0 if self.window is None else max(0, idx - self.window)
        train = np.arange(start, idx)
        if train.size == 0:
            raise ValueError(f"Cannot predict {target_date}: no history available")

        if train.size < self.n_min_days:
            # Too little data to identify 12 coefficients: use the plain mean.
            pred = self.net[train].mean(axis=0)
            return pred, np.zeros(N_SLOTS)

        beta = self._fit(train)
        pred = self._design_for(idx) @ beta
        return np.asarray(pred, dtype=float), np.zeros(N_SLOTS)

    # ------------------------------------------------------------------ #
    def _fit(self, rows: np.ndarray) -> np.ndarray:
        """OLS for all 144 slots at once -> (12, 144) coefficient matrix."""
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
    """Fourier point forecast + per-slot newsboy safety margin.

    The margin is the ``quantile`` of the model's own rolling
    *out-of-sample* residuals of the past ``residual_days`` days (all of them
    strictly earlier than the target day, so there is no look-ahead).  With
    Question 2's settlement (normal price p, emergency price 5p) the
    cost-minimising target is the C_u/(C_u+C_o) = 5/6 quantile of the net
    load, i.e. ``quantile=5/6``.
    """

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
        """(n_days, 144) matrix of one-day-ahead errors; NaN where unknown."""
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
