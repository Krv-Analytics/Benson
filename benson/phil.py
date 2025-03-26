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
    Phil is an advanced data imputation tool that combines scikit-learn's IterativeImputer 
    with topological methods to generate and analyze multiple versions of a dataset.
    
    This class allows users to impute missing data using various imputation techniques, 
    generate representations of imputed datasets, and select the most representative version.
    """

    def __init__(self, samples: int = 30, param_grid: str = "default", magic: str = "ECT", config=None, random_state=None):
        """
            Parameters
            ----------
            samples : int, optional
                Number of imputations to sample from the parameter grid. Default is 30.
            param_grid : str, optional
                Imputation parameter grid identifier or configuration. Default is "default".
            magic : str, optional
                Topological data analysis method to use. Default is "ECT".
            config : dict or None, optional
                Configuration for the chosen magic method. Default is None.
            random_state : int or None, optional
                Seed for reproducibility. Default is None.

            Attributes
            ----------
            config : dict
                Configuration for the chosen magic method.
            magic : str
                Topological data analysis method to use.
            samples : int
                Number of imputations to sample from the parameter grid.
            param_grid : str
                Imputation parameter grid identifier or configuration.
            random_state : int or None
                Seed for reproducibility.
            representations : list
                List to store representations generated during imputation.
            magic_descriptors : list
                List to store descriptors for the chosen magic method.
        """
        self.config, self.magic = self._configure_magic_method(magic=magic, config=config)
        self.samples = samples
        self.param_grid = self._configure_param_grid(param_grid)
        self.random_state = random_state
        self.representations = []
        self.magic_descriptors = []
    
    def impute(self, df: pd.DataFrame, max_iter: int = 10) -> List[pd.DataFrame]:
        """
            Parameters
            ----------
            df : pandas.DataFrame
                DataFrame containing missing values to be imputed.
            max_iter : int, optional
                Maximum number of iterations for the IterativeImputer, by default 10.

            Returns
            -------
            list of pandas.DataFrame
                A list of DataFrames with imputed values.

            Notes
            -----
            This method identifies categorical and numerical columns, configures a preprocessing
            pipeline, creates multiple imputers with different parameter settings, selects the
            most appropriate imputations, and applies them to the input DataFrame.
        """
        categorical_columns, numerical_columns = self._identify_column_types(df)
        preprocessor = self._configure_preprocessor(categorical_columns, numerical_columns)
        imputers = self._create_imputers(preprocessor, max_iter)
        selected_imputers = self._select_imputations(imputers)
        return self._apply_imputations(df, selected_imputers)

    @staticmethod
    def _identify_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
            Parameters
            ----------
            df : pandas.DataFrame
                Input DataFrame containing the data to analyze.

            Returns
            -------
            tuple of list of str
                A tuple containing two lists:
                - The first list contains the names of categorical columns.
                - The second list contains the names of numerical columns.

            Examples
            --------
            >>> import pandas as pd
            >>> data = {'col1': [1, 2, 3], 'col2': ['a', 'b', 'c'], 'col3': [1.1, 2.2, 3.3]}
            >>> df = pd.DataFrame(data)
            >>> _identify_column_types(df)
            (['col2'], ['col1', 'col3'])
        """
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
        return categorical_columns, numerical_columns

    def _create_imputers(self, preprocessor: ColumnTransformer, max_iter: int) -> List[Pipeline]:
        """
        Constructs a list of imputation pipelines with various parameter configurations.
        Parameters
        ----------
        preprocessor : ColumnTransformer
            A scikit-learn ColumnTransformer object used for data preprocessing.
        max_iter : int
            The maximum number of iterations for the IterativeImputer.
        Returns
        -------
        List[Pipeline]
            A list of scikit-learn Pipeline objects, each containing a preprocessing
            step and an imputation model configured with different parameter settings.
        Notes
        -----
        - The method dynamically imports and initializes models based on the parameter
          grid defined in `self.param_grid`.
        - Only parameters compatible with the model's constructor are passed during
          initialization.
        """
        imputers = []
        for method, module, params in zip(self.param_grid.methods, self.param_grid.modules, self.param_grid.grids):
            model = self._import_model(module, method)
            for param_vals in params:
                compatible_params = {k: v for k, v in param_vals.items() if k in model.__init__.__code__.co_varnames}
                estimator = model(**compatible_params)
                imputers.append(self._build_pipeline(preprocessor, estimator, max_iter))
        return imputers

    def _import_model(self, module: str, method: str):
        """Dynamically imports a model from a specified module."""
        imported_module = importlib.import_module(module)
        return getattr(imported_module, method)

    def _build_pipeline(self, preprocessor: ColumnTransformer, estimator, max_iter: int) -> Pipeline:
        """Builds an imputation pipeline with a given estimator."""
        return Pipeline([
            ('preprocessor', preprocessor),
            ('imputer', IterativeImputer(estimator=estimator, random_state=self.random_state, max_iter=max_iter))
        ])

    def _select_imputations(self, imputers: List[Pipeline]) -> List[Pipeline]:
        """Randomly selects a subset of imputers to run."""
        np.random.seed(self.random_state)
        selected_idxs = np.random.choice(range(len(imputers)), min(self.samples, len(imputers)), replace=False)
        return [imputers[idx] for idx in selected_idxs]

    def _apply_imputations(self, df: pd.DataFrame, imputers: List[Pipeline]) -> List[pd.DataFrame]:
        """Applies imputers to the dataset."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            return [pd.DataFrame(imputer.fit_transform(df), columns=df.columns) for imputer in imputers]

    def generate_descriptors(self) -> List[np.ndarray]:
        """Generates topological descriptors for imputed datasets."""
        return [self.magic.generate(imputed_df.values) for imputed_df in self.representations]

    def fit_transform(self, df: pd.DataFrame, max_iter: int = 5) -> pd.DataFrame:
        """
        Perform imputation, generate feature representations, and select the most representative dataset.
        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing missing values to be imputed.
        max_iter : int, optional
            Maximum number of iterations for the imputation process (default is 5).
        Returns
        -------
        pandas.DataFrame
            The best-imputed DataFrame selected based on generated descriptors.
        Notes
        -----
        This method performs the following steps:
        1. Imputes missing values in the input DataFrame.
        2. Generates feature representations (magic descriptors).
        3. Selects the most representative dataset based on the descriptors.
        """
        self.representations = self.impute(df, max_iter)
        self.magic_descriptors = self.generate_descriptors()
        self.closest_index = self._select_representative(self.magic_descriptors)
        return self.representations[self.closest_index]

    @staticmethod
    def _select_representative(descriptors: List[np.ndarray]) -> int:
        """Finds the descriptor closest to the mean representation."""
        avg_descriptor = np.mean(descriptors, axis=0)
        return np.argmin([np.linalg.norm(descriptor - avg_descriptor) for descriptor in descriptors])

    @staticmethod
    def _configure_magic_method(magic: str, config) -> Tuple[BaseModel, Magic]:
        """Configures the topological method."""
        magic_method = getattr(METHODS, magic, None)
        if magic_method is None:
            raise ValueError(f"Magic method '{magic}' not found.")
        if not isinstance(config, BaseModel):
            config = MagicGallery.get(magic)
        return config, magic_method(config=config)

    @staticmethod
    def _configure_param_grid(param_grid) -> ImputationGrid:
        """Retrieves the imputation parameter grid."""
        if isinstance(param_grid, str):
            return GridGallery.get(param_grid)
        if isinstance(param_grid, BaseModel):
            return param_grid.model_dump()
        raise ValueError("Invalid parameter grid type.")
    
    @staticmethod
    def _configure_preprocessor(categorical_columns: List[str], numerical_columns: List[str]) -> ColumnTransformer:
        """Configures the preprocessing pipeline."""
        return ColumnTransformer([
            ('num', StandardScaler(), numerical_columns),
            ('cat', OneHotEncoder(), categorical_columns)
        ])
    