"""
Base classes and interfaces for using topological and geometric representation learning in Benson.

This module provides the foundational classes for implementing topological methods
used in analyzing and comparing imputed datasets.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel


class Magic(ABC):
    """
    Abstract base class for topology- and geometry-based representation learning methods.

    Magic methods generate low-dimensional descriptors of datasets, enabling
    comparison of different imputed versions to select a representative one.

    Parameters
    ----------
    config : BaseModel
        Configuration object specific to the magic method.

    Notes
    -----
    To implement a new magic method:
    1. Create a configuration class inheriting from BaseModel
    2. Create a magic class inheriting from this base class
    3. Implement the generate() method
    4. Register the method in the MagicGallery

    Examples
    --------
    >>> class MyMagicConfig(BaseModel):
    ...     param1: int = 42
    ...     param2: float = 0.5
    ...
    >>> class MyMagic(Magic):
    ...     def __init__(self, config: MyMagicConfig):
    ...         super().__init__(config)
    ...
    ...     def generate(self, data: np.ndarray) -> np.ndarray:
    ...         # Implementation here
    ...         return descriptor
    """

    def __init__(self, config: BaseModel):
        """
        Initialize the magic method with configuration.

        Parameters
        ----------
        config : BaseModel
            Configuration object containing method-specific parameters.
        """
        self.config = config

    @abstractmethod
    def generate(self, data: List[np.ndarray]) -> np.ndarray:
        """
        Generate a topological descriptor for the input data.

        Parameters
        ----------
        data : np.ndarray
            Input data array for which to generate the descriptor.

        Returns
        -------
        np.ndarray
            A descriptor array that captures topological features of the input.

        Notes
        -----
        The descriptor should capture meaningful topological or geometric
        properties that can be used to compare different datasets.
        """
        pass
