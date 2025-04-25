"""Phil test suite.

This module provides comprehensive test coverage for Phil's functionality,
including initialization, imputation, transformation, and configuration.
"""

from tests.phil.test_columns import TestPhilColumnBehavior
from tests.phil.test_config import TestPhilConfigBehavior
from tests.phil.test_descriptors import TestPhilDescriptorBehavior
from tests.phil.test_imputation import TestPhilImputationBehavior
from tests.phil.test_initialization import TestPhilInitializationBehavior
from tests.phil.test_fit import TestPhilFitBehavior

__all__ = [
    "TestPhilColumnBehavior",
    "TestPhilConfigBehavior",
    "TestPhilDescriptorBehavior",
    "TestPhilImputationBehavior",
    "TestPhilInitializationBehavior",
    "TestPhilFitBehavior",
]
