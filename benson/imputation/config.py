"""
Configuration models for Benson's imputation strategies.

This module provides configuration classes that define how imputation methods
are applied, including parameter grids, preprocessing configurations, and
gallery collections of predefined settings.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field
from sklearn.model_selection import ParameterGrid


class ImputationConfig(BaseModel):
    """
    Configuration for imputation methods and their parameter grids.

    This class defines which imputation methods to use and their associated
    hyperparameter search spaces. It supports both scikit-learn imputers and
    custom imputation methods.

    Parameters
    ----------
    methods : List[str]
        Names of imputation methods to use (e.g., "SimpleImputer", "KNNImputer").

    modules : List[str]
        Python modules where the methods can be found (e.g., "sklearn.impute").

    grids : List[ParameterGrid]
        Parameter grids for each method, defining hyperparameter search spaces.

    Examples
    --------
    >>> from sklearn.model_selection import ParameterGrid
    >>> grid = ImputationConfig(
    ...     methods=["SimpleImputer", "KNNImputer"],
    ...     modules=["sklearn.impute", "sklearn.impute"],
    ...     grids=[
    ...         ParameterGrid({"strategy": ["mean", "median"]}),
    ...         ParameterGrid({"n_neighbors": [3, 5]})
    ...     ]
    ... )
    """

    model_config = {"arbitrary_types_allowed": True}

    methods: List[str] = Field(..., description="Names of imputation methods")
    modules: List[str] = Field(..., description="Python modules containing methods")
    grids: List[ParameterGrid] = Field(..., description="Parameter grids for methods")


class PreprocessingConfig(BaseModel):
    """
    Configuration for data preprocessing steps.

    This class defines how data should be preprocessed before imputation,
    including scaling, encoding, and other transformations.

    Parameters
    ----------
    method : str
        Name of the preprocessing method (e.g., "StandardScaler").
    module : str
        Name of the module where the method can be found (e.g., "sklearn.preprocessing").
    params : Dict[str, List[Any]], optional
        Parameter grid for the preprocessing method.

    Examples
    --------
    >>> config = PreprocessingConfig(
    ...     method="StandardScaler",
    ...     params={"with_mean": [True, False]}
    ... )
    """

    method: str
    module: str = "sklearn.preprocessing"  # Default to sklearn.preprocessing
    params: Dict[str, Any] = Field(default_factory=dict)
