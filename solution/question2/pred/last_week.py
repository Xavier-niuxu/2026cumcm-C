# -*- coding: utf-8 -*-
"""Last week predictor: use data from exactly 7 days ago."""

import numpy as np
from .base import BasePredictor


class LastWeekPredictor(BasePredictor):
    """Predict using data from exactly 7 days ago (last week same day)."""
    
    def predict(self, target_date: str) -> tuple:
        """Predict load and PV for target date using last week's data.
        
        Args:
            target_date: target date string (e.g., "2025-02-01")
        
        Returns:
            load_pred: array of shape (144,) predicted load in kW
            pv_pred: array of shape (144,) predicted PV power in kW
        """
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
