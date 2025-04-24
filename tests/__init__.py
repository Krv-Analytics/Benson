"""
Benson test suite.

This module provides comprehensive test coverage for Benson's functionality,
including imputation strategies, magic methods, and utility functions.
"""

from tests.phil import (
    TestPhilColumnBehavior,
    TestPhilConfigBehavior,
    TestPhilDescriptorBehavior,
    TestPhilImputationBehavior,
    TestPhilInitializationBehavior,
    TestPhilTransformBehavior,
)
from tests.imputation import TestDistributionImputer
from tests.magic import TestMagic, TestECT

__all__ = [
    # Phil tests
    "TestPhilColumnBehavior",
    "TestPhilConfigBehavior",
    "TestPhilDescriptorBehavior",
    "TestPhilImputationBehavior",
    "TestPhilInitializationBehavior",
    "TestPhilTransformBehavior",
    # Other tests
    "TestDistributionImputer",
    "TestMagic",
    "TestECT",
]
