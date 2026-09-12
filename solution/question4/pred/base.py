# -*- coding: utf-8 -*-
"""Base predictor class for load and PV forecasting."""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BasePredictor(ABC):
    """Base class for load and PV predictors."""
    
    def __init__(self, dates_load: np.ndarray, load_data: np.ndarray,
                 dates_pv: np.ndarray, pv_data: np.ndarray):
        """Initialize predictor with historical data.
        
        Args:
            dates_load: array of date strings for load data
            load_data: array of shape (n_days, 144) with load values in kW
            dates_pv: array of date strings for PV data
            pv_data: array of shape (n_days, 144) with PV values in kW
        """
        self.dates_load = dates_load
        self.load_data = load_data
        self.dates_pv = dates_pv
        self.pv_data = pv_data
    
    @abstractmethod
    def predict(self, target_date: str) -> tuple:
        """Predict load and PV for target date.
        
        Args:
            target_date: target date string (e.g., "2025-02-01")
        
        Returns:
            load_pred: array of shape (144,) predicted load in kW
            pv_pred: array of shape (144,) predicted PV power in kW
        """
        pass
    
    def _get_date_index(self, date_str: str) -> int:
        """Get the index of a date in the dates array."""
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
        """Get indices of past n_weeks same-weekday days.
        
        Args:
            target_idx: index of target date
            n_weeks: number of weeks to look back
        
        Returns:
            list of indices, sorted from most recent to oldest
        """
        target_dt = pd.Timestamp(self.dates_load[target_idx])
        target_weekday = target_dt.weekday()
        
        same_weekday_indices = []
        for i in range(target_idx - 1, -1, -1):
            if pd.Timestamp(self.dates_load[i]).weekday() == target_weekday:
                same_weekday_indices.append(i)
                if len(same_weekday_indices) >= n_weeks:
                    break
        
        return same_weekday_indices
