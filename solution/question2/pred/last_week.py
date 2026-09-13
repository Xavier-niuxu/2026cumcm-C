# -*- coding: utf-8 -*-
"""last_week implementation."""

import numpy as np
from .base import BasePredictor


class LastWeekPredictor(BasePredictor):
    """LastWeekPredictor implementation."""
    
    def predict(self, target_date: str) -> tuple:
        """predict implementation."""
        target_idx = self._get_date_index(target_date)
        
        # Find exactly 7 days ago
        last_week_idx = target_idx - 7
        
        if last_week_idx < 0:
            raise ValueError(
                f"Cannot predict {target_date}: need at least 7 days of historical data"
            )
        
        load_pred = self.load_data[last_week_idx].copy()
        pv_pred = self.pv_data[last_week_idx].copy()
        
        return load_pred, pv_pred
