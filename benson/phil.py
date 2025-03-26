import importlib
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from benson import ImputationGrid
from benson.gallery import GridGallery, MagicGallery
from benson.magic import Magic
import benson.magic as METHODS


class Phil:
    """
    Phil your missing data with confidence! Avoid analysis paralysis by combining scikit-learn's IterativeImputer with topological methods to handle hundreds of possible versions of your data with ease.
    """

    def __init__(self, param_grid="default", magic="ECT", config=None, random_state=None):
        self.config, self.magic = self._configure_magic_method(magic=magic, config=config)
        self.param_grid = self._configure_param_grid(param_grid)
        self.random_state = random_state
        self.representations = []
        self.magic_descriptors = []


    
    def impute(self, df: pd.DataFrame, max_iter: int = 10) -> List[pd.DataFrame]:
        """
        Runs IterativeImputer over all parameter combinations using pipelines and returns a list of imputed DataFrames.
        """
        results = []
        methods = self.param_grid.methods
        modules = self.param_grid.modules
        grids = self.param_grid.grids
        
        # Create a pipeline with preprocessing and imputation
        # Identify categorical and numerical columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        numerical_columns = df.select_dtypes(include=['number']).columns

        preprocessor = self._configure_preprocessor(categorical_columns, numerical_columns)
        
        imputers = []

        for method, module, params in zip(methods, modules, grids):
            imported_module = importlib.import_module(module)
            model = getattr(imported_module, method)
            for param_vals in params:
                compatible_params = {k: v for k, v in param_vals.items() if k in model.__init__.__code__.co_varnames}
                estimator = model(**compatible_params)

                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('imputer', IterativeImputer(estimator=estimator, random_state=self.random_state, max_iter=max_iter))
                ])
                imputers.append(pipeline)

        # Apply pipelines to the data
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            results = [pd.DataFrame(imputer.fit_transform(df), columns=df.columns) for imputer in imputers]

        return results

    def generate_descriptors(self) -> List[np.ndarray]:
        """
        Generates representations using the magic method.
        """
        return [self.magic.generate(imputed_df.values) for imputed_df in self.representations]

    def fit_transform(self, df: pd.DataFrame, max_iter: int = 5) -> pd.DataFrame:
        """
        Imputes missing data, generates representations, and selects the best representation.
        """
        self.representations = self.impute(df, max_iter)
        self.magic_descriptors = self.generate_descriptors()
        self.closest_index = self._select_representative(self.magic_descriptors)
        return self.representations[self.closest_index]

    @staticmethod
    def _select_representative(descriptors: List[np.ndarray]) -> int:
        """
        Selects the representation closest to the average topological descriptor.
        """
        avg_descriptor = np.mean(descriptors, axis=0)
        closest_index = np.argmin([np.linalg.norm(descriptor - avg_descriptor) for descriptor in descriptors])
        return closest_index

    @staticmethod
    def _compute_representation_variability(descriptors: List) -> float:
        """
        Computes the variability of the representations.
        """
        return np.var(descriptors)

    @staticmethod
    def _configure_magic_method(magic: str, config) -> Tuple[BaseModel, Magic]:
        """
        Configures the magic method based on the magic string and configuration dictionary.
        """
        magic_method = getattr(METHODS, magic, None)
        if magic_method is None:
            raise ValueError(f"Magic method '{magic}' not found in the magic submodule. "
                             f"Defaulting to Euler Characteristic Transform (ECT).")

        if not isinstance(config, BaseModel):
            config = MagicGallery.get(magic)

        magic_instance = magic_method(config=config)
        return config, magic_instance

    @staticmethod
    def _configure_param_grid(param_grid) -> ImputationGrid:
        """
        Configures the parameter grid based on the parameter grid string or BaseModel.
        """
        if isinstance(param_grid, str):
            return GridGallery.get(param_grid)
        if isinstance(param_grid, BaseModel):
            return param_grid.model_dump()
        raise ValueError("Invalid parameter grid type. Must be a string or a BaseModel instance.")
    
    @staticmethod
    def _configure_preprocessor(categorical_columns, numerical_columns) -> ColumnTransformer:
        """
        Configures the preprocessor for the pipeline.
        """
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_columns),
                ('cat', OneHotEncoder(), categorical_columns)
            ]
        )
        return preprocessor


if __name__ == "__main__":
    from sklearn.datasets import load_diabetes
    import time

    # Example usage
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)

    # Introduce missing values
    np.random.seed(42)
    mask = np.random.rand(*df.shape) < 0.2
    df[mask] = np.nan

    phil = Phil()
    start = time.time()
    best_imputed = phil.fit_transform(df)
    print("Best imputed DataFrame:")
    print(best_imputed.head())
    print(best_imputed.isnull().sum())
    print("Variability of representations:", phil._compute_representation_variability(phil.representations))
    print(f"Time taken: {time.time() - start}")
    
    print("Phil your missing data with confidence!")
    print("Avoid analysis paralysis by combining scikit-learn's IterativeImputer with topological methods to handle hundreds of possible versions of your data with ease.")

