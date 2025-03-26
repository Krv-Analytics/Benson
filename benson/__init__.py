from pydantic import BaseModel, Field
from typing import Dict, List, Any
from itertools import product
from sklearn.model_selection import ParameterGrid

class ImputationGrid(BaseModel):
    """
    Represents a grid of parameters for hyperparameter tuning.

    This class is designed to facilitate the organization and management of hyperparameter grids 
    for different machine learning models. It uses `ParameterGrid` from scikit-learn to define 
    parameter combinations and provides a structure to associate these grids with specific models.

    Attributes:
        methods (List[str]): 
            A list of strings representing sklearn-compatible model names. The index of each model 
            corresponds to the index of its associated parameter grid in the `grids` attribute and 
            its module in the `modules` attribute.
        grids (List[ParameterGrid]): 
            A list of `ParameterGrid` objects containing hyperparameter combinations for different models. 
            The index of each grid corresponds to the index of its associated model in the `methods` 
            attribute and its module in the `modules` attribute.
        modules (List[str]): 
            A list of strings representing the names of sklearn modules that can be loaded dynamically 
            to access the corresponding models. The index of each module corresponds to the index of 
            its associated model in the `methods` attribute and its parameter grid in the `grids` attribute.
    """
    methods: List[str] = Field(
        default_factory=list,
        description="A list of sklearn model names, where the index maps to the corresponding ParameterGrid and module."
    )
    modules: List[str] = Field(
        default_factory=list,
        description="A list of sklearn module names, where the index maps to the corresponding model and ParameterGrid."
    )
    grids: List[ParameterGrid] = Field(
        default_factory=list,
        description="A list of ParameterGrid objects, where the index maps to the corresponding model and module."
    )

    class Config:
        arbitrary_types_allowed = True
