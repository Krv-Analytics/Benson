"""
Distribution-preserving imputation strategies.

This module provides imputation methods that maintain the statistical properties
of the original data distribution, ensuring that imputed values reflect the
natural variability and patterns in the data.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class DistributionImputer(BaseEstimator):
    """
    A custom imputer that preserves data distributions by sampling from empirical distributions.

    This imputer maintains the statistical properties of the original data by randomly
    sampling from observed values instead of using summary statistics (mean, median, etc.).
    It is particularly useful when preserving the shape, variance, and outlier patterns
    of the original data distribution is important.

    The imputer can be used standalone or as part of sklearn's IterativeImputer for
    multivariate imputation by chaining (MICE).

    Parameters
    ----------
    missing_values : scalar, default=np.nan
        The placeholder for the missing values. All occurrences of `missing_values`
        will be imputed, in addition to np.nan.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the sampling process. Pass an int for
        reproducible results across multiple function calls.

    threshold : float, default=1.0
        Maximum fraction of missing values allowed for imputation. If the fraction
        of missing values in y exceeds this threshold, all predictions will be
        missing values. Must be in the range [0, 1].

    Attributes
    ----------
    distribution_ : ndarray
        The array of non-missing values in y collected during fit.

    skip_imputation_ : bool
        True if the fraction of missing values in y exceeds threshold.

    rng_ : RandomState
        The random number generator instance used for sampling.

    dtype_ : type
        The inferred data type of the feature being imputed.

    is_categorical_ : bool
        True if the feature being imputed contains categorical data.

    Examples
    --------
    >>> import numpy as np
    >>> from benson.imputation import DistributionImputer
    >>>
    >>> # Numeric data example
    >>> X = np.ones((5, 1))  # Dummy features
    >>> y = np.array([1.0, np.nan, 3.0, 4.0, np.nan])
    >>> imputer = DistributionImputer(random_state=42)
    >>> imputer.fit(X, y)
    >>> predictions = imputer.predict(X)
    >>>
    >>> # Categorical data example
    >>> y_cat = np.array(['a', None, 'b', 'c', None])
    >>> imputer = DistributionImputer()
    >>> imputer.fit(X, y_cat)
    >>> predictions = imputer.predict(X)

    Notes
    -----
    - The imputer handles both numeric and categorical data automatically.
    - For categorical data, it samples from the set of unique observed categories.
    - For numeric data, it samples from the empirical distribution of observed values.
    - If threshold is exceeded, the imputer returns missing values (np.nan for numeric,
      None for categorical).

    See Also
    --------
    sklearn.impute.IterativeImputer : For using this imputer in MICE.
    """

    def __init__(self, missing_values=np.nan, random_state=None, threshold=1.0):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self.missing_values = missing_values
        self.random_state = random_state
        self.threshold = threshold

    def fit(self, X, y):
        """
        Fit the imputer by collecting the distribution of non-missing values
        from y. If the fraction of missing values exceeds 'threshold', set the
        imputer to skip imputation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Not used, present here for API consistency.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        if not isinstance(y, (np.ndarray, pd.Series)):
            y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError("DistributionImputer only supports 1D y.")

        # Infer and store the data type
        self.dtype_ = y.dtype
        self.is_categorical_ = y.dtype.kind in "OSU"  # Object, String, Unicode

        # Convert to object dtype to handle mixed numeric/string data
        if not self.is_categorical_:
            y = y.astype(float, copy=True)
        else:
            y = y.astype(object, copy=True)

        # Identify missing values in y
        missing_mask = (y == self.missing_values) | pd.isnull(y)
        fraction_missing = missing_mask.sum() / y.size

        if fraction_missing == 1.0:
            self.skip_imputation_ = True
            self.distribution_ = np.array([], dtype=self.dtype_)
        else:
            # Check threshold
            self.skip_imputation_ = fraction_missing > self.threshold

            # Store distribution
            if not self.skip_imputation_:
                self.distribution_ = y[~missing_mask]
            else:
                self.distribution_ = np.array([], dtype=self.dtype_)

        # Initialize random state
        if isinstance(self.random_state, np.random.RandomState):
            self.rng_ = self.random_state
        else:
            self.rng_ = np.random.RandomState(self.random_state)

        return self

    def predict(self, X):
        """
        Predict by sampling from the stored distribution for each sample in X.
        If skip_imputation_ is True or the distribution is empty, return NaN
        for numeric data or None for categorical data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Returns sampled values from the empirical distribution.
        """
        if not hasattr(self, "distribution_"):
            raise RuntimeError("Call fit before predict")

        n_samples = X.shape[0]

        if self.skip_imputation_ or self.distribution_.size == 0:
            if self.is_categorical_:
                return np.full(n_samples, None, dtype=object)
            else:
                return np.full(n_samples, np.nan, dtype=float)

        predictions = self.rng_.choice(self.distribution_, size=n_samples, replace=True)

        # Ensure predictions match the inferred data type
        return predictions.astype(self.dtype_)
