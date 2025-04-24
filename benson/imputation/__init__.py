"""
Benson imputation module.

This module provides various imputation strategies for handling missing data,
with a focus on preserving the statistical properties of the original data.
"""

from .config import ImputationConfig, PreprocessingConfig
from .distribution import DistributionImputer

__all__ = ["DistributionImputer", "ImputationConfig", "PreprocessingConfig"]
