# -*- coding: utf-8 -*-
"""Prediction module with different forecasting strategies."""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BasePredictor(ABC):
    """Base class for load and PV predictors."""
    
    def __init__(self, dates_load: np.ndarray, load_data: np.ndarray,
                 dates_pv: np.ndarray, pv_data: np.ndarray):
        self.dates_load = dates_load
        self.load_data = load_data
        self.dates_pv = dates_pv
        self.pv_data = pv_data
    
    @abstractmethod
    def predict(self, target_date: str) -> tuple:
        """Predict load and PV for target date.
        
        Returns:
            load_pred: array of shape (144,) predicted load in kW
            pv_pred: array of shape (144,) predicted PV power in kW
        """
        pass
    
    def _get_date_index(self, date_str: str) -> int:
        mask = self.dates_load == date_str
        if mask.any():
            return int(np.where(mask)[0][0])
        target = pd.Timestamp(date_str)
        for i, d in enumerate(self.dates_load):
            try:
                if pd.Timestamp(d) == target:
                    return i
            except:
                continue
        raise ValueError(f"Date {date_str} not found in data")
    
    def _get_same_weekday_indices(self, target_idx: int, n_weeks: int = 4) -> list:
        target_dt = pd.Timestamp(self.dates_load[target_idx])
        target_weekday = target_dt.weekday()
        indices = []
        for i in range(target_idx - 1, -1, -1):
            if pd.Timestamp(self.dates_load[i]).weekday() == target_weekday:
                indices.append(i)
                if len(indices) >= n_weeks:
                    break
        return indices


class WeightedAveragePredictor(BasePredictor):
    """Weighted average of past 4 same-weekday days. Weights: 4,3,2,1."""
    
    def predict(self, target_date: str) -> tuple:
        target_idx = self._get_date_index(target_date)
        indices = self._get_same_weekday_indices(target_idx, n_weeks=4)
        if len(indices) < 4:
            raise ValueError(
                f"Cannot predict {target_date}: need at least 4 same-weekday days, "
                f"found {len(indices)}"
            )
        weights = np.array([4, 3, 2, 1])
        weight_sum = weights.sum()
        load_pred = np.zeros(144)
        pv_pred = np.zeros(144)
        for i, idx in enumerate(indices):
            load_pred += weights[i] * self.load_data[idx]
            pv_pred += weights[i] * self.pv_data[idx]
        load_pred /= weight_sum
        pv_pred /= weight_sum
        return load_pred, pv_pred


class LastWeekPredictor(BasePredictor):
    """Use data from exactly 7 days ago."""
    
    def predict(self, target_date: str) -> tuple:
        target_idx = self._get_date_index(target_date)
        last_week_idx = target_idx - 7
        if last_week_idx < 0:
            raise ValueError(
                f"Cannot predict {target_date}: need at least 7 days of historical data"
            )
        return self.load_data[last_week_idx].copy(), self.pv_data[last_week_idx].copy()


class LinearFitPredictor(BasePredictor):
    """Linear regression on past 4 same-weekday days."""
    
    def predict(self, target_date: str) -> tuple:
        target_idx = self._get_date_index(target_date)
        indices = self._get_same_weekday_indices(target_idx, n_weeks=4)
        if len(indices) < 4:
            raise ValueError(
                f"Cannot predict {target_date}: need at least 4 same-weekday days, "
                f"found {len(indices)}"
            )
        x = np.array([1, 2, 3, 4])
        x_target = 5.0
        indices_reversed = indices[::-1]
        load_pred = np.zeros(144)
        pv_pred = np.zeros(144)
        for t in range(144):
            y_load = np.array([self.load_data[idx, t] for idx in indices_reversed])
            y_pv = np.array([self.pv_data[idx, t] for idx in indices_reversed])
            x_mean = x.mean()
            y_load_mean = y_load.mean()
            y_pv_mean = y_pv.mean()
            denom = np.sum((x - x_mean) ** 2)
            a_load = np.sum((x - x_mean) * (y_load - y_load_mean)) / denom if denom > 0 else 0
            b_load = y_load_mean - a_load * x_mean
            a_pv = np.sum((x - x_mean) * (y_pv - y_pv_mean)) / denom if denom > 0 else 0
            b_pv = y_pv_mean - a_pv * x_mean
            load_pred[t] = max(0, a_load * x_target + b_load)
            pv_pred[t] = max(0, a_pv * x_target + b_pv)
        return load_pred, pv_pred


# FourierPredictor lives in the pred/ package (pred/fourier.py); re-export it
# here so `from prediction import FourierPredictor` keeps working, since
# main.py imports the flat module rather than the package.
from pred.fourier import FourierPredictor, FourierQuantilePredictor  # noqa: E402,F401
from pred.fourier_intraday import FourierIntradayPredictor  # noqa: E402,F401
