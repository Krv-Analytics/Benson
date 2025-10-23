"""Collection of predefined configurations for Benson.

This module provides galleries of predefined configurations for imputation,
preprocessing, and magic methods. These galleries make it easy to use
domain-specific settings for different types of data and industries.
"""

from typing import Dict, Any
from pydantic import BaseModel
from sklearn.model_selection import ParameterGrid
import numpy as np

from benson.imputation import ImputationConfig, PreprocessingConfig
from benson.magic import *


class GridGallery:
    """
    Collection of predefined imputation parameter grids.

    This class provides industry-specific imputation configurations optimized
    for different domains like finance, healthcare, marketing, etc.

    Available Grids
    --------------
    * `default` - General-purpose configuration using a mix of methods.
    * `finance` - Optimized for financial data with emphasis on robustness.
    * `healthcare` - Focused on preserving distributions in medical data.
    * `marketing` - Handles mixed-type data common in marketing.
    * `engineering` - Configured for technical and sensor data.
    * `risk_analysis` - Conservative settings for risk-sensitive data.
    """

    _grids = {
        "default": ImputationConfig(
            methods=[
                "BayesianRidge",
                "DecisionTreeRegressor",
                "RandomForestRegressor",
                "GradientBoostingRegressor",
            ],
            modules=[
                "sklearn.linear_model",
                "sklearn.tree",
                "sklearn.ensemble",
                "sklearn.ensemble",
            ],
            grids=[
                ParameterGrid({"alpha": [1.0, 0.1, 0.01]}),
                ParameterGrid(
                    {"max_depth": [None, 5, 10], "min_samples_split": [2, 5]}
                ),
                ParameterGrid({"n_estimators": [10, 50], "max_depth": [None, 5]}),
                ParameterGrid(
                    {"learning_rate": [0.1, 0.01], "n_estimators": [50, 100]}
                ),
            ],
        ),
        "sampling": ImputationConfig(
            methods=[
                "DistributionImputer",
            ],
            modules=[
                "benson.imputation",
            ],
            grids=[
                ParameterGrid({"random_state": np.arange(0, 100, 1)}),
            ],
        ),
        "finance": ImputationConfig(
            methods=["IterativeImputer", "KNNImputer", "SimpleImputer"],
            modules=["sklearn.impute"] * 3,
            grids=[
                ParameterGrid({"estimator": ["BayesianRidge"], "max_iter": [10, 50]}),
                ParameterGrid(
                    {
                        "n_neighbors": [3, 5, 10],
                        "weights": ["uniform", "distance"],
                    }
                ),
                ParameterGrid({"strategy": ["mean", "median"]}),
            ],
        ),
        "healthcare": ImputationConfig(
            methods=["KNNImputer", "SimpleImputer", "IterativeImputer"],
            modules=["sklearn.impute"] * 3,
            grids=[
                ParameterGrid({"n_neighbors": [5, 10], "weights": ["distance"]}),
                ParameterGrid({"strategy": ["median", "most_frequent"]}),
                ParameterGrid(
                    {
                        "estimator": ["RandomForestRegressor"],
                        "max_iter": [10, 20],
                    }
                ),
            ],
        ),
        "marketing": ImputationConfig(
            methods=["SimpleImputer", "KNNImputer", "IterativeImputer"],
            modules=["sklearn.impute"] * 3,
            grids=[
                ParameterGrid(
                    {
                        "strategy": ["most_frequent", "constant"],
                        "fill_value": ["unknown"],
                    }
                ),
                ParameterGrid({"n_neighbors": [3, 5], "weights": ["uniform"]}),
                ParameterGrid(
                    {
                        "estimator": ["GradientBoostingRegressor"],
                        "max_iter": [10, 30],
                    }
                ),
            ],
        ),
        "engineering": ImputationConfig(
            methods=["SimpleImputer", "KNNImputer", "IterativeImputer"],
            modules=["sklearn.impute"] * 3,
            grids=[
                ParameterGrid({"strategy": ["mean", "median"]}),
                ParameterGrid({"n_neighbors": [3, 5, 7], "weights": ["distance"]}),
                ParameterGrid(
                    {
                        "estimator": ["DecisionTreeRegressor"],
                        "max_iter": [10, 20],
                    }
                ),
            ],
        ),
    }

    @classmethod
    def get(cls, name: str) -> ImputationConfig:
        """
        Retrieve a predefined parameter grid.

        Parameters
        ----------
        name : str
            Name of the grid to retrieve (e.g., "default", "finance").

        Returns
        -------
        ImputationConfig
            The requested parameter grid configuration.
        """
        return cls._grids.get(name, cls._grids["default"])


class ProcessingGallery:
    """
    Collection of predefined preprocessing configurations.

    This class provides domain-specific preprocessing settings optimized
    for different types of data and industries.

    Available Configurations
    ----------------------
    default
        StandardScaler for general-purpose scaling.
    finance
        MinMaxScaler optimized for financial metrics.
    healthcare
        RobustScaler for handling medical outliers.
    marketing
        PowerTransformer for customer behavior data.
    engineering
        StandardScaler for technical measurements.
    """

    _numeric_methods = {
        "default": PreprocessingConfig(method="StandardScaler"),
        "finance": PreprocessingConfig(
            method="MinMaxScaler", params={"feature_range": [(0, 1)]}
        ),
        "healthcare": PreprocessingConfig(method="RobustScaler"),
        "marketing": PreprocessingConfig(
            method="PowerTransformer", params={"method": ["yeo-johnson"]}
        ),
        "engineering": PreprocessingConfig(method="StandardScaler"),
    }

    _categorical_methods = {
        "default": PreprocessingConfig(method="OneHotEncoder"),
        "finance": PreprocessingConfig(
            method="OneHotEncoder", params={"handle_unknown": ["ignore"]}
        ),
        "healthcare": PreprocessingConfig(
            method="OrdinalEncoder",
            params={"handle_unknown": ["use_encoded_value"]},
        ),
        "marketing": PreprocessingConfig(
            method="OneHotEncoder",
            params={"sparse": [False], "handle_unknown": ["ignore"]},
        ),
    }

    @classmethod
    def get(cls, name: str = "default") -> Dict[str, PreprocessingConfig]:
        """
        Get preprocessing configurations for numerical and categorical data.

        Parameters
        ----------
        name : str, default="default"
            Name of the configuration set (e.g., "finance", "healthcare").

        Returns
        -------
        Dict[str, PreprocessingConfig]
            Dictionary with "num" and "cat" preprocessing configurations.
        """
        return {
            "num": cls._numeric_methods.get(name, cls._numeric_methods["default"]),
            "cat": cls._categorical_methods.get(
                name, cls._categorical_methods["default"]
            ),
        }


class MagicGallery:
    """
    Collection of predefined magic method configurations.

    This class provides configurations for different topological analysis
    methods used to compare and select imputed datasets.

    Available Methods
    ---------------
    ECT
        Euler Characteristic Transform configurations.
    """

    @staticmethod
    def get(method: str) -> BaseModel:
        """
        Get configuration for a magic method.

        Parameters
        ----------
        method : str
            Name of the magic method (e.g., "ECT").

        Returns
        -------
        BaseModel
            Configuration object for the requested method.

        Raises
        ------
        ValueError
            If the requested method is not found.
        """
        if method == "ECT":
            return ECTConfig(
                num_thetas=64,
                radius=1.0,
                resolution=100,
                scale=500,
                ect_fn="scaled_sigmoid",
                seed=42,
            )
        raise ValueError(f"Unknown magic method: {method}")
