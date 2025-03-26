from benson import ImputationGrid
from benson.magic import *
from sklearn.model_selection import ParameterGrid

class GridGallery:
    """
    A collection of predefined parameter grids for imputation models.
    
    This class provides predefined hyperparameter grids for various imputation models,
    allowing an agent to quickly select and apply a suitable configuration.
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
        )
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
