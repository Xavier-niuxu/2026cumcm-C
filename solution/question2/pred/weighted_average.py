# -*- coding: utf-8 -*-
"""weighted_average implementation."""

import numpy as np
from .base import BasePredictor


class WeightedAveragePredictor(BasePredictor):
    """WeightedAveragePredictor implementation."""
    
    def predict(self, target_date: str) -> tuple:
        """predict implementation."""
        target_idx = self._get_date_index(target_date)
        same_weekday_indices = self._get_same_weekday_indices(target_idx, n_weeks=4)
        
        if len(same_weekday_indices) < 4:
            raise ValueError(
                f"Cannot predict {target_date}: need at least 4 same-weekday days, "
                f"found {len(same_weekday_indices)}"
            )
        
        # Weights: 4, 3, 2, 1 (most recent to oldest)
        weights = np.array([4, 3, 2, 1])
        weight_sum = weights.sum()  # 10
        
        load_pred = np.zeros(144)
        pv_pred = np.zeros(144)
        
        for i, idx in enumerate(same_weekday_indices):
            load_pred += weights[i] * self.load_data[idx]
            pv_pred += weights[i] * self.pv_data[idx]
        
        load_pred /= weight_sum
        pv_pred /= weight_sum
        
        return load_pred, pv_pred
