"""
Benson: Advanced Data Imputation Framework
========================================

Benson is a powerful Python library for intelligent data imputation, designed to handle
complex missing data scenarios in high-dimensional datasets. It combines advanced statistical
methods with topological data analysis to provide robust and accurate imputations.

Key Components
-------------
* Phil: Progressive High-dimensional Imputation Lab
  The core engine that combines iterative imputation with topological analysis.

* Distribution-Preserving Imputation
  Statistical methods that maintain the original data distribution properties.

* Magic Methods
  Topological data analysis tools for evaluating imputation quality.

Main Features
------------
- Multiple imputation strategies with automatic selection
- Distribution-preserving imputation methods
- Topological data analysis for quality assessment
- Support for both numerical and categorical data
- Scalable to high-dimensional datasets

Example
-------
>>> from benson import Phil
>>> phil = Phil()
>>> imputed_df = phil.fit_transform(df)

The library automatically handles:
- Missing value detection
- Data type inference
- Imputation strategy selection
- Quality assessment
- Representative imputation selection

See Also
--------
* Documentation: <link to documentation>
* Source Code: https://github.com/<organization>/benson
"""

from .imputation import (
    DistributionImputer,
    ImputationConfig,
    PreprocessingConfig,
)
from .phil import Phil
from .magic import Magic, ECT, ECTConfig

__version__ = "0.1.0"
__all__ = [
    "Phil",
    "DistributionImputer",
    "ImputationConfig",
    "PreprocessingConfig",
    "Magic",
    "ECT",
    "ECTConfig",
]
