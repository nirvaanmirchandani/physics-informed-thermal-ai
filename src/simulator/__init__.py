"""
Physics-informed thermal prediction system for datacenter cooling optimization.

Stage 1: Lumped-parameter thermal simulator for generating synthetic training data.
"""

from .thermal_model import ThermalModel

__all__ = ['ThermalModel']
__version__ = '0.1.0'
