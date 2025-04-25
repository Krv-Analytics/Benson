"""
Scikit-learn compatible transformers for Benson.

This module provides scikit-learn compatible transformers for Benson components,
allowing them to be easily integrated into scikit-learn pipelines.
"""

from typing import Optional, Union, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from benson.phil import Phil
from benson import ImputationConfig


class PhilTransformer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer wrapper for the Phil imputation tool.

    This transformer allows using Phil in scikit-learn pipelines for imputing missing values
    in datasets.

    Parameters
    ----------
    samples : int, default=30
        Number of imputations to sample from the parameter grid.
    param_grid : str or ImputationConfig, default="default"
        Imputation parameter grid identifier or configuration. Can be:
        - "default": Uses default imputation strategies
        - ImputationConfig object: Custom imputation configuration
        - str: Predefined grid from GridGallery (e.g., "finance", "healthcare")
    magic : str, default="ECT"
        Representation learning method to use for quality assessment.
    config : dict or None, default=None
        Configuration for the chosen magic method.
    random_state : int or None, default=None
        Seed for reproducibility.
    max_iter : int, default=5
        Maximum number of iterations for the IterativeImputer.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.pipeline import Pipeline
    >>> from benson.transformers import PhilTransformer
    >>>
    >>> # Create a dataframe with missing values
    >>> df = pd.DataFrame({
    ...     'num': [1.0, None, 3.0, None],
    ...     'cat': ['a', 'b', None, 'd']
    ... })
    >>>
    >>> # Create a pipeline with PhilTransformer
    >>> pipeline = Pipeline([
    ...     ('imputer', PhilTransformer(samples=10))
    ... ])
    >>>
    >>> # Fit and transform the data
    >>> imputed_df = pipeline.fit_transform(df)
    """

    def __init__(
        self,
        samples: int = 30,
        param_grid: Union[str, ImputationConfig] = "default",
        magic: str = "ECT",
        config: Optional[dict] = None,
        random_state: Optional[int] = None,
        max_iter: int = 5,
    ):
        self.samples = samples
        self.param_grid = param_grid
        self.magic = magic
        self.config = config
        self.random_state = random_state
        self.max_iter = max_iter
        self.phil = None

    def fit(self, X: pd.DataFrame, y: Any = None) -> "PhilTransformer":
        """
        Fit the transformer on the input data.

        Parameters
        ----------
        X : pandas.DataFrame
            Input data to fit the imputer on.
        y : Any, default=None
            Ignored. Present for scikit-learn API compatibility.

        Returns
        -------
        self : PhilTransformer
            Returns self for method chaining.
        """
        # Store input feature names (sklearn convention)
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)

        self.phil = Phil(
            samples=self.samples,
            param_grid=self.param_grid,
            magic=self.magic,
            config=self.config,
            random_state=self.random_state,
        )
        self.phil.fit(X, max_iter=self.max_iter)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted imputation on the input data.

        Parameters
        ----------
        X : pandas.DataFrame
            Input data to impute.

        Returns
        -------
        pandas.DataFrame
            Imputed data.

        Raises
        ------
        RuntimeError
            If the transformer is used before calling fit.
        """
        if self.phil is None:
            raise RuntimeError(
                "This PhilTransformer instance is not fitted yet. "
                "Call 'fit' before using this estimator."
            )
        return self.phil.transform(X, max_iter=self.max_iter)
