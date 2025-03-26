from benson import ImputationGrid
from benson.magic import *

from sklearn.model_selection import ParameterGrid


class GridGallery:
    """
    Stores predefined parameter grids for fallback and curated 'gallery' options.
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
                ParameterGrid(
                    {'alpha': [1.0, 0.1, 0.01]}
                ),
                ParameterGrid(
                    {'max_depth': [None, 5, 10], 'min_samples_split': [2, 5]}
                ),
                ParameterGrid(
                    {'n_estimators': [10, 50], 'max_depth': [None, 5]}
                ),
                ParameterGrid(
                    {'learning_rate': [0.1, 0.01], 'n_estimators': [50, 100]}
                ),
            ]
        )
    
    }
    

    @classmethod
    def get(cls, name: str) -> ParameterGrid:
        """Retrieve a predefined parameter grid by name."""
        return cls._grids.get(name, cls._grids["default"])
    
    
    
class MagicGallery:
    """
    Stores predefined magic configurations for fallback and curated 'gallery' options.
    """
    _methods = {
        "ECT": ECTConfig(num_thetas= 64,
    radius= 1.0,
    resolution= 64,
    scale= 500,
    ect_fn = "scaled_sigmoid",seed=42),
    }

    @classmethod
    def get(cls, name: str) -> Magic:
        """Retrieve a predefined magic method by name."""
        return cls._methods.get(name, ECT)