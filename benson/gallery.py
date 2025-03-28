from benson import ImputationGrid,PreprocessingConfig
from benson.magic import *
from sklearn.model_selection import ParameterGrid
from typing import Dict


class GridGallery:
    """
    A collection of predefined parameter grids for imputation models.

    This class provides predefined hyperparameter grids for various imputation models,
    allowing an agent to quickly select and apply a suitable configuration based on 
    industry-specific needs.

    Attributes
    ----------
    _grids : dict
        A dictionary mapping grid names to ImputationGrid instances, each containing
        methods, modules, and parameter grids for specific imputation models.

    Methods
    -------
    get(name: str) -> ImputationGrid
        Retrieve a predefined parameter grid by name.
    """

    _grids = {
        "default": ImputationGrid(
            methods=[
                'BayesianRidge',
                'DecisionTreeRegressor',
                'RandomForestRegressor',
                'GradientBoostingRegressor',
            ],
            modules=[
                'sklearn.linear_model',
                'sklearn.tree',
                'sklearn.ensemble',
                'sklearn.ensemble',
            ],
            grids=[
                ParameterGrid({'alpha': [1.0, 0.1, 0.01]}),
                ParameterGrid({'max_depth': [None, 5, 10], 'min_samples_split': [2, 5]}),
                ParameterGrid({'n_estimators': [10, 50], 'max_depth': [None, 5]}),
                ParameterGrid({'learning_rate': [0.1, 0.01], 'n_estimators': [50, 100]}),
            ]
        ),

        "finance": ImputationGrid(
            methods=[
                'IterativeImputer',
                'KNNImputer',
                'SimpleImputer',
            ],
            modules=[
                'sklearn.impute',
                'sklearn.impute',
                'sklearn.impute',
            ],
            grids=[
                ParameterGrid({'estimator': ['BayesianRidge'], 'max_iter': [10, 50]}),
                ParameterGrid({'n_neighbors': [3, 5, 10], 'weights': ['uniform', 'distance']}),
                ParameterGrid({'strategy': ['mean', 'median']}),
            ]
        ),

        "healthcare": ImputationGrid(
            methods=[
                'KNNImputer',
                'SimpleImputer',
                'IterativeImputer',
            ],
            modules=[
                'sklearn.impute',
                'sklearn.impute',
                'sklearn.impute',
            ],
            grids=[
                ParameterGrid({'n_neighbors': [5, 10], 'weights': ['distance']}),
                ParameterGrid({'strategy': ['median', 'most_frequent']}),
                ParameterGrid({'estimator': ['RandomForestRegressor'], 'max_iter': [10, 20]}),
            ]
        ),

        "marketing": ImputationGrid(
            methods=[
                'SimpleImputer',
                'KNNImputer',
                'IterativeImputer',
            ],
            modules=[
                'sklearn.impute',
                'sklearn.impute',
                'sklearn.impute',
            ],
            grids=[
                ParameterGrid({'strategy': ['most_frequent', 'constant'], 'fill_value': ['unknown']}),
                ParameterGrid({'n_neighbors': [3, 5], 'weights': ['uniform']}),
                ParameterGrid({'estimator': ['GradientBoostingRegressor'], 'max_iter': [10, 30]}),
            ]
        ),

        "engineering": ImputationGrid(
            methods=[
                'SimpleImputer',
                'KNNImputer',
                'IterativeImputer',
            ],
            modules=[
                'sklearn.impute',
                'sklearn.impute',
                'sklearn.impute',
            ],
            grids=[
                ParameterGrid({'strategy': ['mean', 'median']}),
                ParameterGrid({'n_neighbors': [3, 5, 7], 'weights': ['distance']}),
                ParameterGrid({'estimator': ['DecisionTreeRegressor'], 'max_iter': [10, 20]}),
            ]
        ),

        "risk_analysis": ImputationGrid(
            methods=[
                'IterativeImputer',
                'SimpleImputer',
            ],
            modules=[
                'sklearn.impute',
                'sklearn.impute',
            ],
            grids=[
                ParameterGrid({'estimator': ['BayesianRidge'], 'max_iter': [50, 100]}),
                ParameterGrid({'strategy': ['median', 'most_frequent']}),
            ]
        ),
    }

    @classmethod
    def get(cls, name: str) -> ImputationGrid:
        """
        Retrieve a predefined parameter grid by name.

        Parameters
        ----------
        name : str
            The name of the desired parameter grid.

        Returns
        -------
        ImputationGrid
            An instance of ImputationGrid containing methods, modules, and parameter grids.
        """
        return cls._grids.get(name, cls._grids["default"])
    
    
class ProcessingGallery:
    """
    A collection of predefined parameter grids for data preprocessing.

    This class provides predefined hyperparameter grids for various imputation
    and transformation strategies tailored to different industries. It includes
    methods for both numeric and categorical data preprocessing, utilizing
    scikit-learn transformers and custom configurations for specific domains
    such as finance, healthcare, marketing, engineering, and risk analysis.

    Methods
    -------
    get(name, numeric_columns, categorical_columns)
        Retrieve a predefined preprocessing pipeline by name, returning a
        ColumnTransformer configured with the appropriate transformers for
        the specified numeric and categorical columns.
    """

    _numeric_methods = {
        "default": PreprocessingConfig(
            method='StandardScaler',
        ),
        "finance": PreprocessingConfig(
            method='MinMaxScaler',
            params={'feature_range': [(0, 1)]}
        ),
        "healthcare": PreprocessingConfig(
            method='RobustScaler',
        ),
        "marketing": PreprocessingConfig(
            method='PowerTransformer',
            params={'method': ['yeo-johnson']}
        ),
        "engineering": PreprocessingConfig(
            method='StandardScaler'
        ),
        "risk_analysis": PreprocessingConfig(
            method='QuantileTransformer',
            params={'n_quantiles': [100], 'output_distribution': ['normal']}
        )
    }

    _categorical_methods = {
        "default": PreprocessingConfig(
            method='OneHotEncoder',
        ),
        "finance": PreprocessingConfig(
            method='OrdinalEncoder',
            params={'handle_unknown': 'use_encoded_value', 'unknown_value': -1}
        ),
        "healthcare": PreprocessingConfig(
            method='OneHotEncoder',
            params={'sparse_output': [False]}
        ),
        "marketing": PreprocessingConfig(
            method='TargetEncoder',
            module='category_encoders',
            params={'smoothing': [0.5], 'min_samples_leaf': [10]}
        ),
        "engineering": PreprocessingConfig(
            method='OrdinalEncoder'
        ),
        "risk_analysis": PreprocessingConfig(
            method='OneHotEncoder',
            params={'drop': 'first'}
        )
    }

    @classmethod
    def get(cls, name: str) -> Dict[str,PreprocessingConfig]:
        """
        Retrieve a predefined preprocessing pipeline by name.

        Parameters
        ----------
        name : str
            The name of the preprocessing pipeline.
        numeric_columns : list
            List of numeric column names.
        categorical_columns : list
            List of categorical column names.

        Returns
        -------
        ColumnTransformer
            An instance of ColumnTransformer with configured transformers.
        """
        numeric = cls._numeric_methods.get(name, cls._numeric_methods["default"])
        categorical = cls._categorical_methods.get(name, cls._categorical_methods["default"])
        return {'num': numeric, 'cat': categorical}

class MagicGallery:
    """
    A collection of predefined magic configurations for data representations.
    
    This class provides predefined configurations for different topological data analysis methods,
    allowing an agent to quickly apply a chosen method for feature transformation and data repair.
    """
    
    _methods = {
        "ECT": ECTConfig(
            num_thetas=64,
            radius=1.0,
            resolution=64,
            scale=500,
            ect_fn="scaled_sigmoid",
            seed=42
        ),
    }
    
    @classmethod
    def get(cls, name: str) -> Magic:
        """
        Retrieve a predefined magic method configuration by name.
        
        Parameters
        ----------
        name : str
            The name of the desired magic method.
        
        Returns
        -------
        Magic
            A configured instance of the corresponding magic method.
        """
        return cls._methods.get(name, ECT)
