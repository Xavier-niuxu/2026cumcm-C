# -*- coding: utf-8 -*-
"""linear_fit implementation."""

import numpy as np
from .base import BasePredictor


class LinearFitPredictor(BasePredictor):
    """LinearFitPredictor implementation."""
    
    def predict(self, target_date: str) -> tuple:
        """predict implementation."""
        target_idx = self._get_date_index(target_date)
        same_weekday_indices = self._get_same_weekday_indices(target_idx, n_weeks=4)
        
        if len(same_weekday_indices) < 4:
            raise ValueError(
                f"Cannot predict {target_date}: need at least 4 same-weekday days, "
                f"found {len(same_weekday_indices)}"
            )
        
        # x values: 1, 2, 3, 4 (oldest to newest)
        # Target is at x = 5 (next week)
        x = np.array([1, 2, 3, 4])
        x_target = 5.0
        
        load_pred = np.zeros(144)
        pv_pred = np.zeros(144)
        
        # same_weekday_indices is sorted from newest to oldest
        # Reverse to get oldest to newest
        indices_reversed = same_weekday_indices[::-1]
        
        for t in range(144):
            y_load = np.array([self.load_data[idx, t] for idx in indices_reversed])
            y_pv = np.array([self.pv_data[idx, t] for idx in indices_reversed])
            
            # Linear regression: y = a*x + b
            # Using least squares formula
            x_mean = x.mean()
            y_load_mean = y_load.mean()
            y_pv_mean = y_pv.mean()
            
            # Slope: a = sum((x - x_mean) * (y - y_mean)) / sum((x - x_mean)^2)
            numerator_load = np.sum((x - x_mean) * (y_load - y_load_mean))
            numerator_pv = np.sum((x - x_mean) * (y_pv - y_pv_mean))
            denominator = np.sum((x - x_mean) ** 2)
            
            a_load = numerator_load / denominator if denominator > 0 else 0
            b_load = y_load_mean - a_load * x_mean
            
            a_pv = numerator_pv / denominator if denominator > 0 else 0
            b_pv = y_pv_mean - a_pv * x_mean
            
            # Predict at x_target = 5
            load_pred[t] = a_load * x_target + b_load
            pv_pred[t] = a_pv * x_target + b_pv
        
        # Ensure non-negative predictions
        load_pred = np.maximum(load_pred, 0)
        pv_pred = np.maximum(pv_pred, 0)
        
        return load_pred, pv_pred
