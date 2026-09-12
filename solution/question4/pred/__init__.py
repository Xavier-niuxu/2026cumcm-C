# -*- coding: utf-8 -*-
"""Prediction module with different forecasting strategies."""

from .base import BasePredictor
from .weighted_average import WeightedAveragePredictor
from .last_week import LastWeekPredictor
from .linear_fit import LinearFitPredictor
from .fourier import FourierPredictor, FourierQuantilePredictor
from .fourier_scaled import ScaledFourierPredictor
from .fourier_intraday import FourierIntradayPredictor

__all__ = [
    "BasePredictor",
    "WeightedAveragePredictor",
    "LastWeekPredictor",
    "LinearFitPredictor",
    "FourierPredictor",
    "FourierQuantilePredictor",
    "ScaledFourierPredictor",
    "FourierIntradayPredictor",
]
