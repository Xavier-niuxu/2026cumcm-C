# -*- coding: utf-8 -*-
"""__init__ implementation."""

from .base import BasePredictor
from .weighted_average import WeightedAveragePredictor
from .last_week import LastWeekPredictor
from .linear_fit import LinearFitPredictor
from .fourier import FourierPredictor, FourierQuantilePredictor, FourierLagQuantilePredictor
from .fourier_intraday import FourierIntradayPredictor

__all__ = [
    "BasePredictor",
    "WeightedAveragePredictor",
    "LastWeekPredictor",
    "LinearFitPredictor",
    "FourierPredictor",
    "FourierQuantilePredictor",
    "FourierLagQuantilePredictor",
    "FourierIntradayPredictor",
]
