# -*- coding: utf-8 -*-
"""base implementation."""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BasePredictor(ABC):
    """BasePredictor implementation."""
    
    def __init__(self, dates_load: np.ndarray, load_data: np.ndarray,
                 dates_pv: np.ndarray, pv_data: np.ndarray):
        """__init__ implementation."""
        self.dates_load = dates_load
        self.load_data = load_data
        self.dates_pv = dates_pv
        self.pv_data = pv_data
    
    @abstractmethod
    def predict(self, target_date: str) -> tuple:
        """predict implementation."""
        pass
    
    def _get_date_index(self, date_str: str) -> int:
        """_get_date_index implementation."""
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
        """_get_same_weekday_indices implementation."""
        target_dt = pd.Timestamp(self.dates_load[target_idx])
        target_weekday = target_dt.weekday()
        
        same_weekday_indices = []
        for i in range(target_idx - 1, -1, -1):
            if pd.Timestamp(self.dates_load[i]).weekday() == target_weekday:
                same_weekday_indices.append(i)
                if len(same_weekday_indices) >= n_weeks:
                    break
        
        return same_weekday_indices
