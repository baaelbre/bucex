"""Compatibility imports; observation families live in bucex.observations."""
from .observations import Gaussian, GEV, SeasonalScale, contrasts, ObservationFamily
__all__ = ['Gaussian', 'GEV', 'SeasonalScale', 'contrasts', 'ObservationFamily']
