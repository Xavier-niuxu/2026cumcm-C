# -*- coding: utf-8 -*-
"""Prediction module with different forecasting strategies."""

from .base import BasePredictor
from .weighted_average import WeightedAveragePredictor
from .last_week import LastWeekPredictor
from .linear_fit import LinearFitPredictor

__all__ = [
    "BasePredictor",
    "WeightedAveragePredictor",
    "LastWeekPredictor",
    "LinearFitPredictor",
]
