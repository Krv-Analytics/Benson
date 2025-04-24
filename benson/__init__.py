"""
Benson is the powerhouse behind `Phil`, the advanced imputation engine designed to intelligently handle missing data in complex datasets. Whether you're dealing with high-dimensional gaps, inconsistent encodings, or stubborn anomalies, Benson ensures your data gets the cleanup it deserves—efficiently and at scale.

🔥 **Capabilities**

• Phil 🧩: PHIL: a Progressive High-Dimensional Imputation Lab.

👉 Phil is an advanced data imputation tool that combines scikit-learn's IterativeImputer with topological methods to generate and analyze multiple versions of a dataset. It allows users to impute missing data using various techniques, generate representations of imputed datasets, and democratically select a representative version.

🚀 **Coming Soon**

• Bob 🛠️: A structured data repair module that cleans, normalizes, and reconciles inconsistencies in your datasets.

• AgentBenson 🤖: A seamless integration layer for popular agentic frameworks, enabling automated data cleaning and repair with minimal intervention.


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

"""

from benson.imputation import (
    ImputationConfig,
    PreprocessingConfig,
    DistributionImputer,
)

from benson.magic import ECT
from benson.phil import Phil

from benson.gallery import GridGallery


__version__ = "0.1.0"
__all__ = [
    "Phil",
    "GridGallery",
    "ECT",
]
